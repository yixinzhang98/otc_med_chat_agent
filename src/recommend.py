import pandas as pd
import re
import spacy

df = pd.read_csv("data/otc_medications.csv")
nlp = spacy.load("en_core_web_sm")

def extract_symptoms(user_prompt, symptom_list):
    """
    Extracts symptoms mentioned in the user prompt by matching against a list of known symptoms.
    Uses spaCy for NLP-based matching and regex for fallback.
    Returns a list of detected symptoms.
    """
    detected = []
    prompt = user_prompt.lower()
    doc = nlp(prompt)
    # Try to match using NLP phrase similarity
    for symptom in symptom_list:
        symptom_doc = nlp(symptom.lower())
        for chunk in doc.noun_chunks:
            if symptom_doc.similarity(chunk) > 0.85:
                detected.append(symptom)
                break
        else:
            # Fallback to regex word boundary match
            pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
            if re.search(pattern, prompt):
                detected.append(symptom)
    return list(set(detected))

def recommend_medication(user_prompt):
    # Build a list of all unique symptoms from the dataset
    all_symptoms = set()
    for uses in df["Uses"]:
        for s in uses.split(","):
            all_symptoms.add(s.strip())
    # Extract symptoms from user prompt
    symptoms = extract_symptoms(user_prompt, list(all_symptoms))
    recommendations = []
    for _, row in df.iterrows():
        for symptom in symptoms:
            if symptom.lower() in row["Uses"].lower():
                recommendations.append({
                    "name": row["Medication"],
                    "purpose": row["Uses"],
                    "notes": row["Notes"]
                })
    return recommendations
