import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2 


@st.cache_data
def load_data():
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    return df


def cleanresume(text):
    cleantext = re.sub(r'http\S+\s', ' ', text)  
    cleantext = re.sub(r'@\S+', ' ', cleantext) 
    cleantext = re.sub(r'#\S+', ' ', cleantext)  
    cleantext = re.sub(r'@\S+', ' ', cleantext) 
    cleantext = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleantext)  # Raw escape
    cleantext = re.sub(r'[^\x00-\x7f]', r' ', cleantext)  
    cleantext = re.sub(r'\s+', ' ', cleantext) 
    return cleantext  


def encode_categories(df):
    le = LabelEncoder()
    df['Category'] = le.fit_transform(df['Category'])
    return df, le


def vectorize_text(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    required_text = vectorizer.fit_transform(df['Resume'])
    return required_text, vectorizer


@st.cache_data
def load_model_and_resumes():
    clf = pickle.load(open('clf.pkl', 'rb'))
    ranked_resumes = pickle.load(open('ranked_resumes.pkl', 'rb'))
    return clf, ranked_resumes


def predict_category(clf, vectorizer, resume_text):
    cleaned_resume = cleanresume(resume_text)
    input_feature = vectorizer.transform([cleaned_resume])
    prediction_id = clf.predict(input_feature)[0]
    return prediction_id


def get_category_name(prediction_id):
    Category_mapping = {
        15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
        24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
        18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
        1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
        19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
        17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
    }
    return Category_mapping.get(prediction_id, "Unknown")

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    return text


def main():
    st.title("AI Resume Screening & Ranking System")
 
    df = load_data()
    
   
    df['Resume'] = df['Resume'].apply(lambda x: cleanresume(x))
    df, le = encode_categories(df)
    required_text, vectorizer = vectorize_text(df)
    
   
    clf, ranked_resumes = load_model_and_resumes()
    

    st.write("**Enter a brief description of your resume (optional):**")
    job_description = st.text_area("", placeholder="Data Science is a multidisciplinary field that combines...")
    
   
    st.write("**Upload Resume**")
    uploaded_file = st.file_uploader("", type=["pdf", "txt"], help="Drag and drop file here. Limit 200MB per file: TXT, PDF")
    
    if st.button("Submit"):
        if uploaded_file:
            # Extract text from uploaded file
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                resume_text = extract_text_from_txt(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a PDF or TXT file.")
                return
            
          
            job_vec = vectorizer.transform([job_description])
            
            
            similarities = cosine_similarity(job_vec, required_text)
            
        
            ranked_resumes = sorted(zip(similarities.flatten(), df['Resume']), reverse=True)
            
         
            prediction_id = predict_category(clf, vectorizer, resume_text)
            category_name = get_category_name(prediction_id)
            st.write("## Predicted Category:")
            st.write(f"**{category_name}**")
            
         
            total_score = similarities[0][0]  # Assuming the first score is for the uploaded resume
            st.write("## Total Score:")
            st.write(f"**{total_score:.4f}**")
        else:
            st.warning("Please upload a resume.")

if __name__ == "__main__":
    main()