"""
AI-Powered Resume Analyzer & Job Matching System
=================================================
Single-file version: everything (PDF parsing, prompts, Gemini API calls,
and the Streamlit dashboard) lives in this one app.py for simplicity.

Setup:
    1. pip install -r requirements.txt
    2. Get a free key at https://aistudio.google.com/apikey
    3. Paste it into the sidebar when the app runs (or set GEMINI_API_KEY in a .env file)

Run with:
    streamlit run app.py
"""

import io
import json
import os
import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader
from google import genai
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env is optional; sidebar input still works without it

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # much higher free-tier daily quota
FALLBACK_MODEL = "gemini-3.6-flash"  # higher quality, but only 20 free requests/day


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages_text)

    full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]{2,}", " ", full_text)
    return full_text.strip()


def is_text_extractable(text: str, min_chars: int = 50) -> bool:
    return len(text.strip()) >= min_chars


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("A Gemini API key is required.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def get_json_response(self, prompt: str) -> dict:
        import time

        last_error = None
        # Try the primary model twice (transient 503s clear up fast), then
        # fall back to a second model if it's still overloaded.
        attempts = [
            (self.model, 2),
            (FALLBACK_MODEL, 1),
        ]
        for model_name, tries in attempts:
            for attempt in range(tries):
                try:
                    response = self.client.models.generate_content(model=model_name, contents=prompt)
                    return self._parse_json(response.text or "")
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        break  # this model's quota is used up for now; move straight to fallback
                    if "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower():
                        time.sleep(3)  # brief pause before retrying the same model
                        continue
                    raise RuntimeError(f"Gemini API request failed: {e}") from e
        raise RuntimeError(
            f"Gemini API is currently overloaded on all models tried. "
            f"Please wait a minute and try again. (Last error: {last_error})"
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse Gemini's response as JSON: {e}\n\nRaw:\n{text}") from e
        raise ValueError(f"Gemini's response did not contain JSON:\n\n{text}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
RESUME_ANALYSIS_PROMPT = """You are an expert technical recruiter and resume reviewer.
Analyze the resume below and return your assessment.

<resume>
{resume_text}
</resume>

Return ONLY valid JSON (no markdown fences, no commentary before or after) matching
exactly this shape:

{{
  "overall_summary": "2-3 sentence summary of the candidate's profile",
  "strengths": ["strength 1", "strength 2", "..."],
  "weaknesses": ["weakness 1", "weakness 2", "..."],
  "key_skills": ["skill 1", "skill 2", "..."],
  "years_of_experience_estimate": "e.g. '3-5 years'",
  "suggested_job_titles": ["title 1", "title 2", "title 3"],
  "formatting_issues": ["issue 1", "issue 2"],
  "improvement_recommendations": [
    {{"area": "short area name", "suggestion": "concrete, actionable suggestion"}}
  ]
}}

Give 3-6 items for list fields where relevant. Be specific and evidence-based -
refer to what's actually in the resume, not generic advice."""


ATS_SCORE_PROMPT = """You are an Applicant Tracking System (ATS) simulator and resume
scoring engine. Score the resume below as an ATS parser would, then evaluate
formatting/parseability issues that could hurt a candidate.

<resume>
{resume_text}
</resume>

{job_description_block}

Return ONLY valid JSON (no markdown fences, no commentary) matching exactly this shape:

{{
  "ats_score": 0,
  "score_breakdown": {{
    "keyword_match": 0,
    "formatting": 0,
    "structure": 0,
    "measurable_impact": 0
  }},
  "matched_keywords": ["keyword 1", "keyword 2"],
  "missing_keywords": ["keyword 1", "keyword 2"],
  "parseability_issues": ["issue 1", "issue 2"],
  "quick_wins": ["specific fix 1", "specific fix 2", "specific fix 3"]
}}

Scoring rules:
- "ats_score" is 0-100 overall.
- Each "score_breakdown" value is 0-100 for that sub-category.
- If a job description was provided, base "matched_keywords" / "missing_keywords" on it.
  If no job description was provided, base keywords on what's standard for the
  candidate's apparent target role, and say so implicitly by keeping the list realistic."""


SKILL_GAP_PROMPT = """You are a career coach specializing in skill-gap analysis.
Compare the candidate's resume against the target job description and identify gaps.

<resume>
{resume_text}
</resume>

<target_job_description>
{job_description}
</target_job_description>

Return ONLY valid JSON (no markdown fences, no commentary) matching exactly this shape:

{{
  "match_percentage": 0,
  "matching_skills": ["skill 1", "skill 2"],
  "missing_skills": [
    {{"skill": "skill name", "importance": "high|medium|low", "how_to_acquire": "short suggestion"}}
  ],
  "transferable_skills": ["skill the candidate has that partially covers a gap"],
  "recommended_learning_path": ["step 1", "step 2", "step 3"],
  "verdict": "1-2 sentence honest verdict on how strong a fit this is right now"
}}

Be honest and specific. "match_percentage" should reflect a realistic hiring-manager
judgment, not just keyword overlap."""


JOB_MATCHING_PROMPT = """You are a career advisor. Based on the candidate's resume,
suggest job roles that would be a strong fit, independent of any single job description.

<resume>
{resume_text}
</resume>

Return ONLY valid JSON (no markdown fences, no commentary) matching exactly this shape:

{{
  "recommended_roles": [
    {{
      "title": "job title",
      "fit_score": 0,
      "reasoning": "1-2 sentences on why this fits",
      "typical_seniority": "e.g. 'Mid-level'",
      "industries": ["industry 1", "industry 2"]
    }}
  ],
  "career_trajectory_suggestion": "1-2 sentences on a plausible next-step career path"
}}

Provide 3-5 roles in "recommended_roles", ordered by fit_score descending (0-100)."""


COVER_LETTER_PROMPT = """You are an expert cover letter writer. Write a compelling,
specific, non-generic cover letter using the candidate's actual resume content and
the target job description below. Avoid cliches like "I am writing to express my
interest". Keep it to 3-4 short paragraphs. Do not invent facts not supported by
the resume.

<resume>
{resume_text}
</resume>

<job_description>
{job_description}
</job_description>

<tone>
{tone}
</tone>

Return ONLY valid JSON (no markdown fences, no commentary) matching exactly this shape:

{{
  "cover_letter": "the full cover letter as plain text, with \\n\\n between paragraphs",
  "notes": ["any assumptions you made, e.g. missing company name placeholder"]
}}"""


def build_job_description_block(job_description: str) -> str:
    if job_description and job_description.strip():
        return f"<target_job_description>\n{job_description.strip()}\n</target_job_description>"
    return "<target_job_description>\n(none provided - score generically for the candidate's apparent target role)\n</target_job_description>"


# ---------------------------------------------------------------------------
# PDF report export
# ---------------------------------------------------------------------------
def build_pdf_report(analysis, ats_result, skill_gap, job_matches, cover_letter) -> bytes:
    """Compile whichever results are available into a single downloadable PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=10, textColor=colors.HexColor("#1e293b"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#2563eb"))
    body = ParagraphStyle("Body", parent=styles["Normal"], spaceAfter=6, leading=14)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=9, textColor=colors.grey)

    story = []
    story.append(Paragraph("Resume Analysis Report", h1))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')}", small))
    story.append(Spacer(1, 12))

    def bullet_list(items):
        return ListFlowable(
            [ListItem(Paragraph(str(item), body)) for item in items],
            bulletType="bullet", start="•",
        )

    if analysis:
        story.append(Paragraph("Resume Analysis", h2))
        story.append(Paragraph(f"<b>Summary:</b> {analysis.get('overall_summary', '')}", body))
        if analysis.get("strengths"):
            story.append(Paragraph("Strengths", styles["Heading3"]))
            story.append(bullet_list(analysis["strengths"]))
        if analysis.get("weaknesses"):
            story.append(Paragraph("Weaknesses", styles["Heading3"]))
            story.append(bullet_list(analysis["weaknesses"]))
        if analysis.get("key_skills"):
            story.append(Paragraph(f"<b>Key Skills:</b> {', '.join(analysis['key_skills'])}", body))
        if analysis.get("years_of_experience_estimate"):
            story.append(Paragraph(f"<b>Estimated Experience:</b> {analysis['years_of_experience_estimate']}", body))
        if analysis.get("suggested_job_titles"):
            story.append(Paragraph(f"<b>Suggested Job Titles:</b> {', '.join(analysis['suggested_job_titles'])}", body))
        if analysis.get("improvement_recommendations"):
            story.append(Paragraph("Improvement Recommendations", styles["Heading3"]))
            story.append(bullet_list([
                f"<b>{r.get('area', '')}:</b> {r.get('suggestion', '')}"
                for r in analysis["improvement_recommendations"]
            ]))

    if ats_result:
        story.append(Paragraph("ATS Score", h2))
        story.append(Paragraph(f"<b>Overall Score:</b> {ats_result.get('ats_score', 0)}/100", body))
        breakdown = ats_result.get("score_breakdown", {})
        if breakdown:
            table_data = [["Category", "Score"]] + [[k.replace("_", " ").title(), f"{v}/100"] for k, v in breakdown.items()]
            t = Table(table_data, colWidths=[3 * inch, 1.5 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 8))
        if ats_result.get("matched_keywords"):
            story.append(Paragraph(f"<b>Matched Keywords:</b> {', '.join(ats_result['matched_keywords'])}", body))
        if ats_result.get("missing_keywords"):
            story.append(Paragraph(f"<b>Missing Keywords:</b> {', '.join(ats_result['missing_keywords'])}", body))
        if ats_result.get("quick_wins"):
            story.append(Paragraph("Quick Wins", styles["Heading3"]))
            story.append(bullet_list(ats_result["quick_wins"]))

    if skill_gap:
        story.append(Paragraph("Skill Gap Analysis", h2))
        story.append(Paragraph(f"<b>Match Percentage:</b> {skill_gap.get('match_percentage', 0)}%", body))
        story.append(Paragraph(f"<b>Verdict:</b> {skill_gap.get('verdict', '')}", body))
        if skill_gap.get("matching_skills"):
            story.append(Paragraph("Matching Skills", styles["Heading3"]))
            story.append(bullet_list(skill_gap["matching_skills"]))
        if skill_gap.get("missing_skills"):
            story.append(Paragraph("Missing Skills", styles["Heading3"]))
            story.append(bullet_list([
                f"[{g.get('importance', 'medium').upper()}] {g.get('skill', '')} — {g.get('how_to_acquire', '')}"
                for g in skill_gap["missing_skills"]
            ]))
        if skill_gap.get("recommended_learning_path"):
            story.append(Paragraph("Recommended Learning Path", styles["Heading3"]))
            story.append(bullet_list(skill_gap["recommended_learning_path"]))

    if job_matches:
        story.append(Paragraph("Job Role Matching", h2))
        for role in job_matches.get("recommended_roles", []):
            story.append(Paragraph(
                f"<b>{role.get('title', '')}</b> — Fit: {role.get('fit_score', 0)}% "
                f"({role.get('typical_seniority', 'N/A')})", body
            ))
            story.append(Paragraph(role.get("reasoning", ""), small))
            story.append(Spacer(1, 4))
        if job_matches.get("career_trajectory_suggestion"):
            story.append(Paragraph(f"<b>Career Trajectory:</b> {job_matches['career_trajectory_suggestion']}", body))

    if cover_letter and cover_letter.get("cover_letter"):
        story.append(Paragraph("Cover Letter", h2))
        for para in cover_letter["cover_letter"].split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), body))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()



for key in ["resume_text", "analysis", "ats_result", "skill_gap", "job_matches", "cover_letter", "_pdf_report"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Resume Analyzer")
    st.caption("Powered by Google Gemini")

    api_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key at aistudio.google.com/apikey. Stored only for this session.",
    )

    st.divider()
    uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    job_description = st.text_area(
        "Target job description (optional, required for skill-gap & cover letter)",
        height=200,
        placeholder="Paste the job posting text here...",
    )

    st.divider()
    if st.button("🔄 Reset session"):
        for key in ["resume_text", "analysis", "ats_result", "skill_gap", "job_matches", "cover_letter", "_pdf_report"]:
            st.session_state[key] = None
        st.rerun()


def get_client():
    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar to continue.")
        return None
    return GeminiClient(api_key=api_key)


def ensure_resume_text() -> bool:
    if uploaded_file is None:
        st.info("Upload a resume PDF in the sidebar to get started.")
        return False

    if st.session_state.resume_text is None or st.session_state.get("_uploaded_name") != uploaded_file.name:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file.read())
        if not is_text_extractable(text):
            st.error("Couldn't extract readable text from this PDF. Try a text-based PDF export instead.")
            return False
        st.session_state.resume_text = text
        st.session_state._uploaded_name = uploaded_file.name
        for key in ["analysis", "ats_result", "skill_gap", "job_matches", "cover_letter", "_pdf_report"]:
            st.session_state[key] = None

    return True


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
st.title("AI-Powered Resume Analyzer & Job Matching")

tabs = st.tabs(["🔍 Resume Analysis", "🎯 ATS Score", "📊 Skill Gap", "💼 Job Matching", "✉️ Cover Letter", "📥 Export Report"])

with tabs[0]:
    st.subheader("Resume Analysis")
    st.caption("Strengths, weaknesses, key skills, and improvement recommendations.")

    if ensure_resume_text():
        if st.button("Run Analysis", key="btn_analysis"):
            client = get_client()
            if client:
                with st.spinner("Analyzing resume..."):
                    try:
                        prompt = RESUME_ANALYSIS_PROMPT.format(resume_text=st.session_state.resume_text)
                        st.session_state.analysis = client.get_json_response(prompt)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        result = st.session_state.analysis
        if result:
            st.markdown(f"**Summary:** {result.get('overall_summary', '')}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Strengths**")
                for s in result.get("strengths", []):
                    st.markdown(f"- {s}")
            with col2:
                st.markdown("**⚠️ Weaknesses**")
                for w in result.get("weaknesses", []):
                    st.markdown(f"- {w}")
            st.markdown("**🛠 Key Skills**")
            st.write(", ".join(result.get("key_skills", [])))
            st.markdown(f"**Estimated Experience:** {result.get('years_of_experience_estimate', 'N/A')}")
            st.markdown("**Suggested Job Titles**")
            st.write(", ".join(result.get("suggested_job_titles", [])))
            if result.get("formatting_issues"):
                st.markdown("**📄 Formatting Issues**")
                for f in result.get("formatting_issues", []):
                    st.markdown(f"- {f}")
            st.markdown("**💡 Improvement Recommendations**")
            for rec in result.get("improvement_recommendations", []):
                st.markdown(f"- **{rec.get('area', '')}**: {rec.get('suggestion', '')}")

with tabs[1]:
    st.subheader("ATS Score")
    st.caption("Simulated Applicant Tracking System scoring and keyword matching.")

    if ensure_resume_text():
        if st.button("Run ATS Scoring", key="btn_ats"):
            client = get_client()
            if client:
                with st.spinner("Scoring resume..."):
                    try:
                        prompt = ATS_SCORE_PROMPT.format(
                            resume_text=st.session_state.resume_text,
                            job_description_block=build_job_description_block(job_description),
                        )
                        st.session_state.ats_result = client.get_json_response(prompt)
                    except Exception as e:
                        st.error(f"ATS scoring failed: {e}")

        result = st.session_state.ats_result
        if result:
            score = result.get("ats_score", 0)
            col1, col2 = st.columns([1, 2])
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=score, title={"text": "ATS Score"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"},
                           "steps": [{"range": [0, 50], "color": "#fecaca"},
                                     {"range": [50, 75], "color": "#fef08a"},
                                     {"range": [75, 100], "color": "#bbf7d0"}]},
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                breakdown = result.get("score_breakdown", {})
                if breakdown:
                    df = pd.DataFrame({"Category": list(breakdown.keys()), "Score": list(breakdown.values())})
                    st.bar_chart(df.set_index("Category"))
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**✅ Matched Keywords**")
                st.write(", ".join(result.get("matched_keywords", [])) or "—")
            with col4:
                st.markdown("**❌ Missing Keywords**")
                st.write(", ".join(result.get("missing_keywords", [])) or "—")
            if result.get("parseability_issues"):
                st.markdown("**⚠️ Parseability Issues**")
                for p in result.get("parseability_issues", []):
                    st.markdown(f"- {p}")
            st.markdown("**⚡ Quick Wins**")
            for q in result.get("quick_wins", []):
                st.markdown(f"- {q}")

with tabs[2]:
    st.subheader("Skill Gap Analysis")
    st.caption("Requires a job description in the sidebar.")

    if ensure_resume_text():
        if not job_description.strip():
            st.info("Paste a target job description in the sidebar to run this analysis.")
        else:
            if st.button("Run Skill Gap Analysis", key="btn_gap"):
                client = get_client()
                if client:
                    with st.spinner("Comparing resume to job description..."):
                        try:
                            prompt = SKILL_GAP_PROMPT.format(
                                resume_text=st.session_state.resume_text,
                                job_description=job_description,
                            )
                            st.session_state.skill_gap = client.get_json_response(prompt)
                        except Exception as e:
                            st.error(f"Skill gap analysis failed: {e}")

            result = st.session_state.skill_gap
            if result:
                st.metric("Match Percentage", f"{result.get('match_percentage', 0)}%")
                st.markdown(f"**Verdict:** {result.get('verdict', '')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Matching Skills**")
                    for s in result.get("matching_skills", []):
                        st.markdown(f"- {s}")
                    st.markdown("**🔄 Transferable Skills**")
                    for s in result.get("transferable_skills", []):
                        st.markdown(f"- {s}")
                with col2:
                    st.markdown("**❌ Missing Skills**")
                    for gap in result.get("missing_skills", []):
                        importance = gap.get("importance", "medium")
                        badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(importance, "⚪")
                        st.markdown(f"{badge} **{gap.get('skill', '')}** — {gap.get('how_to_acquire', '')}")
                st.markdown("**📚 Recommended Learning Path**")
                for i, step in enumerate(result.get("recommended_learning_path", []), 1):
                    st.markdown(f"{i}. {step}")

with tabs[3]:
    st.subheader("Job Role Matching")
    st.caption("Roles that fit this resume, based purely on the candidate's background.")

    if ensure_resume_text():
        if st.button("Find Matching Roles", key="btn_match"):
            client = get_client()
            if client:
                with st.spinner("Finding matching roles..."):
                    try:
                        prompt = JOB_MATCHING_PROMPT.format(resume_text=st.session_state.resume_text)
                        st.session_state.job_matches = client.get_json_response(prompt)
                    except Exception as e:
                        st.error(f"Job matching failed: {e}")

        result = st.session_state.job_matches
        if result:
            for role in result.get("recommended_roles", []):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{role.get('title', '')}**")
                        st.caption(role.get("reasoning", ""))
                        st.caption(f"Seniority: {role.get('typical_seniority', 'N/A')}  •  Industries: {', '.join(role.get('industries', []))}")
                    with c2:
                        st.metric("Fit", f"{role.get('fit_score', 0)}%")
            if result.get("career_trajectory_suggestion"):
                st.markdown("**🚀 Career Trajectory**")
                st.write(result["career_trajectory_suggestion"])

with tabs[4]:
    st.subheader("Cover Letter Generator")
    st.caption("Requires a job description in the sidebar.")

    if ensure_resume_text():
        tone = st.selectbox("Tone", ["Professional", "Enthusiastic", "Concise", "Warm/personable"])

        if not job_description.strip():
            st.info("Paste a target job description in the sidebar to generate a cover letter.")
        else:
            if st.button("Generate Cover Letter", key="btn_cover"):
                client = get_client()
                if client:
                    with st.spinner("Writing cover letter..."):
                        try:
                            prompt = COVER_LETTER_PROMPT.format(
                                resume_text=st.session_state.resume_text,
                                job_description=job_description,
                                tone=tone,
                            )
                            st.session_state.cover_letter = client.get_json_response(prompt)
                        except Exception as e:
                            st.error(f"Cover letter generation failed: {e}")

            result = st.session_state.cover_letter
            if result:
                st.text_area("Cover Letter", value=result.get("cover_letter", ""), height=350)
                st.download_button("⬇️ Download as .txt", data=result.get("cover_letter", ""),
                                    file_name="cover_letter.txt", mime="text/plain")
                if result.get("notes"):
                    st.caption("Notes: " + "; ".join(result["notes"]))

with tabs[5]:
    st.subheader("Export Full Report")
    st.caption("Compiles every analysis you've run so far into one downloadable PDF.")

    results_available = any([
        st.session_state.analysis, st.session_state.ats_result, st.session_state.skill_gap,
        st.session_state.job_matches, st.session_state.cover_letter,
    ])

    if not results_available:
        st.info("Run at least one analysis in the other tabs first — this report includes whatever you've generated so far.")
    else:
        included = []
        if st.session_state.analysis: included.append("Resume Analysis")
        if st.session_state.ats_result: included.append("ATS Score")
        if st.session_state.skill_gap: included.append("Skill Gap")
        if st.session_state.job_matches: included.append("Job Matching")
        if st.session_state.cover_letter: included.append("Cover Letter")
        st.write("**This report will include:** " + ", ".join(included))

        if st.button("📄 Generate PDF Report"):
            with st.spinner("Building your report..."):
                try:
                    pdf_bytes = build_pdf_report(
                        st.session_state.analysis, st.session_state.ats_result,
                        st.session_state.skill_gap, st.session_state.job_matches,
                        st.session_state.cover_letter,
                    )
                    st.session_state["_pdf_report"] = pdf_bytes
                    st.success("Report ready!")
                except Exception as e:
                    st.error(f"Couldn't build the PDF: {e}")

        if st.session_state.get("_pdf_report"):
            st.download_button(
                "⬇️ Download PDF Report",
                data=st.session_state["_pdf_report"],
                file_name="resume_analysis_report.pdf",
                mime="application/pdf",
            )