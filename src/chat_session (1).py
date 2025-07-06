
import spacy
from symptom_map import SYMPTOM_SYNONYMS

class ChatSession:
    def __init__(self, df, nlp=None):
        self.state = "ask_symptom"
        self.symptoms = []
        self.durations = []
        self.other_meds = ""
        self.current_symptom = ""
        self.answers = []

        self.df = df
        self.nlp = nlp or spacy.load("en_core_web_sm")

        self.synonym_to_symptom = {}
        for canonical, variants in SYMPTOM_SYNONYMS.items():
            self.synonym_to_symptom[canonical] = canonical
            for v in variants:
                self.synonym_to_symptom[v.lower()] = canonical

    def normalize_symptoms(self, prompt):
        prompt = prompt.lower()
        matched = set()
        for phrase, canonical in self.synonym_to_symptom.items():
            if phrase in prompt:
                matched.add(canonical)
        return list(matched)

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
            normalized = self.normalize_symptoms(answer)
            self.current_symptom = normalized[0] if normalized else answer
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
        doc_prompt = self.nlp(prompt)

        scored_rows = []
        for _, row in self.df.iterrows():
            uses_doc = self.nlp(row["Uses"])
            score = doc_prompt.similarity(uses_doc)
            scored_rows.append((score, row))

        scored_rows.sort(reverse=True, key=lambda x: x[0])
        recommendations = []
        for score, row in scored_rows[:10]:
            recommendations.append({
                "name": row["Medication"],
                "purpose": row["Uses"],
                "notes": row["Notes"],
                "similarity_score": score
            })
        return recommendations
