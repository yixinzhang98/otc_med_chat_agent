
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from recommend import ChatSession

import streamlit as st

st.set_page_config(page_title="OTC Medication Chat Agent", page_icon="💊")
st.title("💊 OTC Medication Chat Agent")

# Initialize or resume chat session
if "chat_session" not in st.session_state:
    st.session_state.chat_session = ChatSession()

chat_session = st.session_state.chat_session

# Display historical Q&A
for i in range(len(chat_session.answers)):
    question = None
    if i < len(chat_session.answers):
        # Infer question from state history
        if i == 0 or chat_session.answers[i-1].strip().lower() in ["yes", "y"]:
            question = "What symptom are you experiencing?"
        elif i > 0 and chat_session.answers[i-1] == chat_session.current_symptom:
            question = f"How long have you had '{chat_session.current_symptom}'?"
        elif chat_session.answers[i-1].strip().lower() not in ["yes", "no", "y", "n"]:
            question = "Are you experiencing any other symptoms? (yes/no)"
        elif i == len(chat_session.answers) - 1:
            question = "Are you taking any other medications?"

    if question:
        st.markdown(f"**Q{i+1}: {question}**")
    st.markdown(f"**A{i+1}: {chat_session.answers[i]}**")

# Ask the next question if applicable
if chat_session.has_more_questions():
    question = chat_session.get_next_question()
    user_input = st.text_input(question, key=f"q_{len(chat_session.answers)}")
    if user_input:
        chat_session.record_answer(user_input)
        st.rerun()
else:
    st.subheader("🧾 Recommendations Based on Your Symptoms")
    recs = chat_session.get_recommendations()
    if recs:
        for rec in recs[:3]:  # Show top 3
            st.subheader(rec["name"])
            st.write(f"**Purpose:** {rec['purpose']}")
            st.write(f"**Notes:** {rec['notes']}")
            if "similarity_score" in rec:
                st.caption(f"Similarity Score: {rec['similarity_score']:.2f}")
    else:
        st.warning("No recommendations found. Please try again with clearer symptoms.")

    st.markdown("> ⚠️ This tool is for informational use only. Always consult a healthcare professional.")

    if st.button("🔄 Start Over"):
        del st.session_state.chat_session
        st.rerun()
