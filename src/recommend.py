import pandas as pd
import spacy

# Load medication dataset and spaCy model once
df = pd.read_csv("data/otc_medications.csv")
nlp = spacy.load("en_core_web_sm")


def extract_symptoms(prompt, all_symptoms):
    prompt_lower = prompt.lower()
    extracted = []
    for symptom in all_symptoms:
        if symptom.lower() in prompt_lower:
            extracted.append(symptom)
    return extracted


class ChatSession:
    def __init__(self):
        self.state = "ask_symptom"
        self.symptoms = []
        self.durations = []
        self.other_meds = ""
        self.current_symptom = ""
        self.answers = []

        self.df = df
        self.all_symptoms = set()
        self.row_symptom_sets = []

        # Preprocess symptom sets
        for uses in self.df["Uses"]:
            symptom_set = set()
            for s in uses.split(","):
                clean_symptom = s.strip().lower()
                self.all_symptoms.add(clean_symptom)
                symptom_set.add(clean_symptom)
            self.row_symptom_sets.append(symptom_set)

    def get_next_question(self):
        if self.state == "ask_symptom":
            return "What symptom are you experiencing?"
        elif self.state == "ask_duration":
            return f"How long have you had '{self.current_symptom}'?"
        elif self.state == "ask_more":
            return "Are you experiencing any other symptoms? (yes/no)"
        elif self.state == "ask_medications":
            return "Are you taking any other medications?"
        else:
            return None

    def record_answer(self, answer):
        self.answers.append(answer)
        if self.state == "ask_symptom":
            self.current_symptom = answer
            self.state = "ask_duration"
        elif self.state == "ask_duration":
            self.symptoms.append(self.current_symptom)
            self.durations.append(answer)
            self.state = "ask_more"
        elif self.state == "ask_more":
            if answer.strip().lower() in ["yes", "y"]:
                self.state = "ask_symptom"
            else:
                self.state = "ask_medications"
        elif self.state == "ask_medications":
            self.other_meds = answer
            self.state = "done"

    def has_more_questions(self):
        return self.state != "done"

    def get_collected_prompt(self):
        prompt = ""
        for s, d in zip(self.symptoms, self.durations):
            prompt += f"Symptom: {s}, Duration: {d}. "
        prompt += f"Other medications: {self.other_meds}"
        return prompt

    def get_recommendations(self):
        prompt = self.get_collected_prompt()
        symptoms = extract_symptoms(prompt, list(self.all_symptoms))
        recommendations = []

        if not symptoms:
            return recommendations

        doc_prompt = nlp(prompt)

        scored_rows = []
        for idx, (_, row) in enumerate(self.df.iterrows()):
            uses = row["Uses"]
            notes = row["Notes"]
            uses_doc = nlp(uses)

            # Count how many symptoms match exactly
            match_count = 0
            med_symptoms = self.row_symptom_sets[idx]
            for symptom in symptoms:
                if symptom.lower() in med_symptoms:
                    match_count += 1

            # NLP similarity (scaled down)
            similarity = doc_prompt.similarity(uses_doc)

            # Weighted score: prioritize symptom match more heavily
            total_score = (match_count * 1.5) + (similarity * 0.5)
            scored_rows.append((total_score, match_count, similarity, row))

        # Sort by weighted score
        scored_rows.sort(reverse=True, key=lambda x: x[0])

        for total_score, match_count, similarity, row in scored_rows[:10]:
            recommendations.append({
                "name": row["Medication"],
                "purpose": row["Uses"],
                "notes": row["Notes"],
                "match_count": match_count,
                "similarity_score": round(similarity, 2),
                "combined_score": round(total_score, 2)
            })

        return recommendations