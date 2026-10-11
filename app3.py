
import streamlit as st
from utils import extract_text_from_pdf,extract_skills,calculate_match_score,missing_skills,ai_suggestions

st.set_page_config(page_title="AI Resume Analyzer",layout="wide")
st.title("AI Resume Analyzer")
uploaded_file = st.file_uploader("Upload Resume PDF",type=["pdf"])
job_description = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if uploaded_file and job_description:
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description)
        score = calculate_match_score(resume_text,job_description)
        missing = missing_skills(resume_skills,jd_skills)
        
        st.subheader("Match Score")
        st.metric("Compatibility", f"{score:.1f}%")
        st.subheader("Resume Skills")
        st.write(resume_skills)
        st.subheader("Job Skills")
        st.write(jd_skills)
        st.subheader("Missing Skills")
        st.write(missing)

        with st.spinner("Generating AI feedback..."):
            feedback = ai_suggestions(resume_text, job_description)
            st.subheader("AI Suggestions")
            st.write(feedback)
