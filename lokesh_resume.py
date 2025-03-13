import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import time
import base64
import re
from PIL import Image
import pytesseract
import numpy as np

# Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save("model/")

# --- Page Configuration --- #
st.set_page_config(
    page_title="Lokesh🖤 | Resume Screening App",
    page_icon="msl.png",
    layout="wide"
)

# --- Functions --- #
def preprocess_text(text):
    """Simple text preprocessing without external libraries."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    pdf = PdfReader(file)
    text = "".join([page.extract_text() or "" for page in pdf.pages])
    return text

def extract_text_from_image(image):
    """Extracts text from an image file using OCR."""
    return pytesseract.image_to_string(image)

def extract_text_from_txt(file):
    """Extracts text from a plain text file."""
    return file.read().decode("utf-8")

def extract_experience_keywords(text, job_description):
    """
    Identifies relevant experience-related keywords in the resume
    that match terms in the job description.
    """
    job_words = set(job_description.lower().split())
    resume_words = set(text.lower().split())

    experience_keywords = ["years of experience", "worked at", "background in", "skills in", "expertise in"]
    matched_keywords = job_words.intersection(resume_words)
    experience_found = any(keyword in text.lower() for keyword in experience_keywords)

    return len(matched_keywords), experience_found  # Number of skill matches & experience presence

def rank_resumes(job_description, resumes, file_names):
    """Ranks resumes based on similarity & experience matching."""
    job_description_cleaned = preprocess_text(job_description)
    resumes_cleaned = [preprocess_text(resume) for resume in resumes]
    
    job_embedding = model.encode(job_description_cleaned, convert_to_tensor=True)
    resume_embeddings = model.encode(resumes_cleaned, convert_to_tensor=True)

    cosine_similarities = util.pytorch_cos_sim(job_embedding, resume_embeddings)[0].cpu().numpy()
    absolute_scores = (cosine_similarities * 100).clip(0, 100)

    experience_scores = []
    for resume in resumes:
        skill_match_count, has_experience = extract_experience_keywords(resume, job_description)
        exp_score = skill_match_count * 2
        if has_experience:
            exp_score += 10
        experience_scores.append(exp_score)
    
    experience_weight = 0.75 if "experience" in job_description.lower() else 0.5
    similarity_weight = 1 + experience_weight
    final_scores = [(similarity_weight * sim) + (experience_weight * exp) for sim, exp in zip(absolute_scores, experience_scores)]
    
    results = pd.DataFrame({
        "Resume": file_names,
        "Similarity Score (%)": [round(score, 2) for score in absolute_scores],
        "Final Score (%)": [round(score, 2) for score in final_scores]
    })
    
    results = results.sort_values(by="Final Score (%)", ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", range(1, len(results) + 1))
    return results

# --- Main App --- #
st.title("🔍 AI Resume Screening & Ranking System")

col1, col2 = st.columns(2)

with col1:
    st.header("📄 Job Description")
    job_description = st.text_area("Enter Job Description Here", height=300)

with col2:
    st.header("📂 Upload Resumes (PDF, Image, or TXT)")
    uploaded_files = st.file_uploader("Accept multiple files", type=["pdf", "jpg", "png", "jpeg", "txt"], accept_multiple_files=True)

    if uploaded_files:
        resumes = []
        file_names = []
        
        for i, file in enumerate(uploaded_files):
            time.sleep(0.5)
            st.progress((i + 1) / len(uploaded_files), f"Processing {file.name}...")

            try:
                if file.type == "application/pdf":
                    text = extract_text_from_pdf(file)
                elif file.type in ["image/jpeg", "image/png"]:
                    image = Image.open(file) 
                    text = extract_text_from_image(image)
                elif file.type == "text/plain":
                    text = extract_text_from_txt(file)
                else:
                    st.warning(f"Unsupported file type: {file.type}")
                    continue 

                resumes.append(text)
                file_names.append(file.name)
            
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        if job_description and resumes:
            st.header("📊 Ranked Results")
            results = rank_resumes(job_description, resumes, file_names)
            st.dataframe(results)
            
            csv = results.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="ranked_resumes.csv">📥 Download Results (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
