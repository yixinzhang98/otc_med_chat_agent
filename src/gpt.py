import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set in terminal or .env

def gpt_recommendation(symptom_text):
    prompt = f'''
You are a helpful medical assistant. Based on the user's symptoms below, suggest over-the-counter medications, explain why they help, and include safety tips.

Symptoms: {symptom_text}

Format:
Medication Name: ...
Why it's helpful: ...
Precautions: ...
'''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=400
    )
    return response["choices"][0]["message"]["content"]
