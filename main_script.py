import os
import json
import fitz  # PyMuPDF
import gspread
import pandas as pd
import requests
import spacy
from collections import Counter
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer, util
from jobspy import scrape_jobs

# --- 1. SETUP & AUTH ---
def get_google_client():
    secret_json = os.getenv('GOOGLE_SHEET_CREDENTIALS')
    creds_dict = json.loads(secret_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

# Load AI Model (Global)
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_match(resume_text, job_blob):
    embeddings = model.encode([resume_text, job_blob], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(score) * 100, 2)

def extract_resume_text():
    resume_path = "resume.pdf" # Make sure your file is named exactly this in GitHub
    text = ""
    with fitz.open(resume_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# --- 2. SCRAPERS ---
def fetch_adzuna(app_id, app_key):
    print("📡 Fetching Adzuna...")
    url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"
    params = {'app_id': app_id, 'app_key': app_key, 'what': 'data', 'where': 'UK', 'results_per_page': 50}
    try:
        res = requests.get(url, params=params).json()
        return res.get('results', [])
    except:
        return []

def fetch_linkedin():
    print("🛰️ Fetching LinkedIn via JobSpy...")
    try:
        jobs = scrape_jobs(
            site_name=["linkedin"],
            search_term="data",
            location="United Kingdom",
            results_wanted=30,
            hours_old=24,
            linkedin_fetch_description=True
        )
        return jobs
    except:
        return pd.DataFrame()

# --- 3. MAIN EXECUTION ---
def main():
    # 1. Prepare Data
    resume_text = extract_resume_text()
    adz_id = os.getenv('ADZUNA_APP_ID')
    adz_key = os.getenv('ADZUNA_APP_KEY')
    
    # 2. Collect Jobs
    adz_jobs = fetch_adzuna(adz_id, adz_key)
    li_jobs = fetch_linkedin()

    scored_list = []

    # Score Adzuna
    for j in adz_jobs:
        score = calculate_match(resume_text, f"{j.get('title')} {j.get('description')}")
        scored_list.append([score, 'Adzuna', j.get('title'), j.get('company', {}).get('display_name'), j.get('redirect_url')])

    # Score LinkedIn
    for _, row in li_jobs.iterrows():
        score = calculate_match(resume_text, f"{row['title']} {row['description']}")
        scored_list.append([score, 'LinkedIn', row['title'], row['company'], row['job_url']])

    # 3. Filter & Upload
    df_new = pd.DataFrame(scored_list, columns=['Score', 'Source', 'Title', 'Company', 'Link'])
    df_new = df_new[df_new['Score'] > 40] # Only keep decent matches

    # Connect to Sheets
    gc = get_google_client()
    sh = gc.open('JobTracker_2026').sheet1 # MAKE SURE THIS NAME MATCHES YOUR SHEET
    
    # Simple Deduplication
    existing_links = set(sh.col_values(5)) # Column 5 is Link
    unique_jobs = [row for row in df_new.values.tolist() if row[4] not in existing_links]

    if unique_jobs:
        sh.append_rows(unique_jobs)
        print(f"✅ Success! Added {len(unique_jobs)} new jobs.")
    else:
        print("No new unique jobs found.")

if __name__ == "__main__":
    main()

