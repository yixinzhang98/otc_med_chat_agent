import pandas as pd

df = pd.read_csv("data/otc_medications.csv")

def recommend_medication(symptoms):
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
