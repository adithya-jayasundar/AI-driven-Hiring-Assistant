import streamlit as st
import json
from io import BytesIO
from reportlab.pdfgen import canvas

# IMPORT BACKEND
from backend import (
    parse_with_llm,
    extract_pdf,
    JobPostingSchema,
    ResumeSchema,
    index_resume,
    index_jd,
    hybrid_search,
    score_candidate,
    rank_candidates,
    explain_candidate,
)


# PAGE CONFIG


st.set_page_config(page_title="AI Hiring Engine", layout="wide")
st.title("AI Hiring & Candidate Ranking System")


# SESSION STATE


if "jd" not in st.session_state:
    st.session_state.jd = None

if "resumes" not in st.session_state:
    st.session_state.resumes = []

if "raw_texts" not in st.session_state:
    st.session_state.raw_texts = []

if "ranked" not in st.session_state:
    st.session_state.ranked = []

if "explanations" not in st.session_state:
    st.session_state.explanations = {}



# JOB DESCRIPTION


st.header(" Step 1: Paste Job Description")

jd_text = st.text_area("Paste JD here", height=250)

if st.button("Parse Job Description"):
    if not jd_text.strip():
        st.error("Please enter a JD.")
    else:
        with st.spinner("Parsing JD..."):
            jd = parse_with_llm(jd_text, JobPostingSchema)

        if jd:
            st.session_state.jd = jd
            index_jd(jd, jd_text)
            st.success("JD parsed and indexed!")
            st.json(json.loads(jd.model_dump_json(indent=2)))
        else:
            st.error("❌ LLM failed to parse JD.")



# UPLOAD RESUMES


st.header("Step 2: Upload Resumes (PDF)")

uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Process Resumes"):
    if st.session_state.jd is None:
        st.error("Parse JD first.")
    else:
        st.session_state.resumes = []
        st.session_state.raw_texts = []

        for pdf in uploads:
            st.write(f"Processing: {pdf.name}")

            with st.spinner("Extracting text..."):
                raw = extract_pdf(pdf)

            with st.spinner("Parsing resume..."):
                resume = parse_with_llm(raw, ResumeSchema)

            if resume:
                st.session_state.resumes.append(resume)
                st.session_state.raw_texts.append(raw)

                index_resume(resume, raw)

                st.success(f"Indexed {resume.candidate_name}")
            else:
                st.error(f"❌ Could not parse {pdf.name}")



# RANKING


if st.session_state.resumes:
    st.header("Ranked Candidates")

    ranked = rank_candidates(st.session_state.jd, st.session_state.resumes)
    st.session_state.ranked = ranked

    st.table([
        {
            "Rank": r["rank"],
            "Candidate": r["candidate_name"],
            "Score": r["final_score"],
        }
        for r in ranked
    ])

    st.subheader(" Explainability")

    for r in ranked[:4]:
        name = r["candidate_name"]
        resume_obj = next(x for x in st.session_state.resumes if x.candidate_name == name)
        sc = r["final_score"]

        with st.expander(f"{name} — Explanation"):
            exp_text = explain_candidate(st.session_state.jd, resume_obj, sc, r["rank"])
            st.session_state.explanations[name] = exp_text
            st.text(exp_text)



# DOWNLOAD PDF


if st.session_state.explanations:
    st.header("⬇ Download Explainability Report")

    def build_pdf():
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        y = 800
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, y, "AI Hiring Engine — Explainability Report")
        y -= 40

        c.setFont("Helvetica", 10)
        for name, txt in st.session_state.explanations.items():
            c.drawString(30, y, f"Candidate: {name}")
            y -= 20

            for line in txt.split("\n"):
                c.drawString(30, y, line[:110])
                y -= 14
                if y < 40:
                    c.showPage()
                    y = 800
            y -= 20

        c.save()
        buffer.seek(0)
        return buffer

    pdf = build_pdf()

    st.download_button(
        "Download PDF",
        data=pdf,
        file_name="Explainability_Report.pdf",
        mime="application/pdf"
    )


# HYBRID SEARCH SECTION

st.header("Hybrid Vector Search")

query = st.text_input("Enter search text")

if st.button("Run Hybrid Search"):
    if not query.strip():
        st.error("Type a query.")
    else:
        with st.spinner("Searching..."):
            results = hybrid_search(query, top_k=10)

        st.subheader("Search Results")

        if not results:
            st.info("No matches.")
        else:
            st.table([
                {
                    "Candidate": r["meta"].get("candidate_name", "Unknown"),
                    "Doc ID": r["doc_id"],
                    "Score": r["score"],
                    "Excerpt": r["text"][:160] + "..."
                }
                for r in results
            ])
