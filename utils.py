
import pdfplumber
from transformers import pipeline
from skills import SKILLS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):

    """
    Extract text from uploaded PDF.
    """

    text = ""

    try:
        with pdfplumber.open(pdf_file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"PDF Error: {e}")

    return text


def extract_skills(text):
    """
    Extract skills from resume/job description.
    """

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        if skill.lower() in text:
            found_skills.append(skill)

    return sorted(list(set(found_skills)))


def calculate_match_score(resume_text,job_description):
    """
    Calculate similarity score between
    resume and job description.
    """

    embeddings = model.encode([resume_text, job_description])

    similarity = cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]

    score = round(similarity * 100, 2)

    return score


def missing_skills( resume_skills,jd_skills):
    """
    Find skills present in JD but
    absent in resume.
    """

    missing = list(set(jd_skills) - set(resume_skills))
    return sorted(missing)


def common_skills(resume_skills,jd_skills):
    """
    Find matching skills.
    """

    common = list(set(resume_skills) & set(jd_skills))

    return sorted(common)


def ai_suggestions(resume_text, job_description):
    """
    Generate AI feedback using  Hugging Face model.
    """

    # Load a free text-generation model
    generator = pipeline("text-generation", model="gpt2")

    prompt = f"""
    You are an ATS Resume Expert.

    Analyze the following resume and compare it with the job description.

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}

    Provide:
    1. Resume Match Summary
    2. Strengths
    3. Weaknesses
    4. Missing Skills
    5. ATS Improvement Suggestions
    6. Recommended Resume Changes
    7. Final ATS Readiness Score out of 100
    """

    try:
        result = generator(prompt, max_length=400, num_return_sequences=1)
        return result[0]["generated_text"]
    except Exception as e:  
        return f"Error generating AI feedback: {str(e)}"
    


def calculate_ats_score(resume_skills,jd_skills):
    """
    Simple ATS score based on
    matching skills.
    """

    if len(jd_skills) == 0:
        return 0

    matched = len(set(resume_skills) & set(jd_skills))

    score = (matched /len(jd_skills)) * 100

    return round(score, 2)


def analyze_resume(resume_text,job_description):
    """
    Complete analysis function.
    """

    resume_skills = extract_skills(resume_text)

    jd_skills = extract_skills(job_description)

    match_score = calculate_match_score(resume_text,job_description)

    ats_score = calculate_ats_score(resume_skills,jd_skills)

    missing = missing_skills(resume_skills,jd_skills)

    common = common_skills(resume_skills,jd_skills)

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "match_score": match_score,
        "ats_score": ats_score,
        "missing_skills": missing,
        "common_skills": common
    }