import pandas as pd
import re
import spacy

df = pd.read_csv("data/otc_medications.csv")
nlp = spacy.load("en_core_web_sm")



def ask_initial_questions():
    # You can customize these questions as needed
    return [
        "What symptoms are you experiencing?",
        "How long have you had these symptoms?",
        "Are you taking any other medications?"
    ]

def extract_symptoms(prompt, all_symptoms):
    """
    Extracts symptoms from the user's prompt by matching known symptoms.
    """
    prompt_lower = prompt.lower()
    extracted = []
    for symptom in all_symptoms:
        if symptom.lower() in prompt_lower:
            extracted.append(symptom)
    return extracted


class ChatSession:
    """
    Manages the state of a chat session, including asking initial questions,
    collecting user responses, and determining the most relevant use case.
    """
    def __init__(self):
        self.questions = ask_initial_questions()
        self.answers = []
        self.current_question_idx = 0
        # Build a list of all unique symptoms from the dataset
        self.all_symptoms = set()
        # Preprocess: create a list of sets of symptoms for each row
        self.row_symptom_sets = []
        for uses in df["Uses"]:
            symptom_set = set()
            for s in uses.split(","):
                clean_symptom = s.strip().lower()
                self.all_symptoms.add(clean_symptom)
                symptom_set.add(clean_symptom)
            self.row_symptom_sets.append(symptom_set)

    def has_more_questions(self):
        return self.current_question_idx < len(self.questions)

    def get_next_question(self):
        if self.has_more_questions():
            return self.questions[self.current_question_idx]
        return None

    def record_answer(self, answer):
        self.answers.append(answer)
        self.current_question_idx += 1

    def get_collected_prompt(self):
        # Combine all answers into a single prompt for symptom extraction
        return " ".join(self.answers)

    def get_recommendations(self):
        """
        Use all collected answers to extract symptoms and recommend medication.

        Returns:
            list: A list of recommendation dicts. Each dict contains:
                - name: Medication name
                - purpose: Medication uses
                - notes: Additional notes
                - similarity_score: spaCy similarity score (float, 0 to 1, higher means more similar)
                between the user's prompt and the medication's uses description.
        """
        prompt = self.get_collected_prompt()
        symptoms = extract_symptoms(prompt, list(self.all_symptoms))
        recommendations = []
        # Lowercase symptoms for matching
        symptoms_lower = [s.lower() for s in symptoms]
        if symptoms:
            doc_prompt = nlp(prompt)
            # Precompute spaCy Doc objects for "Uses" column
            uses_docs = [(row, nlp(row["Uses"])) for _, row in df.iterrows()]
            scored_rows = []
            for row, uses_doc in uses_docs:
                score = doc_prompt.similarity(uses_doc)
                scored_rows.append((score, row))
            # Sort by similarity score descending
            scored_rows.sort(reverse=True, key=lambda x: x[0])
            for score, row in scored_rows:
                recommendations.append({
                    "name": row["Medication"],
                    "purpose": row["Uses"],
                    "notes": row["Notes"],
                    "similarity_score": score
                })
            return recommendations
        else:
            # Fallback: recommend the most common medication for the first symptom
            if symptoms:
                first_symptom = symptoms[0]
                for idx, (_, row) in enumerate(df.iterrows()):
                    if first_symptom.lower() in self.row_symptom_sets[idx]:
                        recommendations.append({
                            "name": row["Medication"],
                            "purpose": row["Uses"],
                            "notes": row["Notes"],
                            "fallback_reason": "Prompt too short for similarity, matched by symptom"
                        })
                        break
            return recommendations

# Example usage:
# session = ChatSession()
# while session.has_more_questions():
#     question = session.get_next_question()
#     print(question)
#     user_input = input()
#     session.record_answer(user_input)
# recs = session.get_recommendations()
# print(recs)
