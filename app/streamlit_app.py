import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from recommend import ChatSession

import streamlit as st

st.title("OTC Medication Chat Agent")

# Initialize chat session in Streamlit session state
if "chat_session" not in st.session_state:
    st.session_state.chat_session = ChatSession()

chat_session = st.session_state.chat_session

# Show chat history
for idx, (q, a) in enumerate(zip(chat_session.questions, chat_session.answers)):
    st.markdown(f"**Q{idx+1}: {q}**")
    st.markdown(f"**A{idx+1}: {a}**")

# Get the current question to ask
current_question = chat_session.get_next_question()

if current_question:
    user_input = st.text_input(current_question, key=f"q_{chat_session.current_question_idx}")
    if user_input:
        chat_session.record_answer(user_input)
        st.rerun()
else:
    # All questions answered, show recommendations
    results = chat_session.get_recommendations()
    if results:
        st.success("Recommended medication(s):")
        # Show top 3 recommendations, or all if fewer
        for med in results[:3]:
            st.subheader(med["name"])
            st.write(f"**Purpose:** {med['purpose']}")
            st.write(f"**Notes:** {med['notes']}")
            if "similarity_score" in med:
                st.caption(f"Similarity score: {med['similarity_score']:.2f}")
            if "fallback_reason" in med:
                st.caption(f"Reason: {med['fallback_reason']}")
    else:
        st.warning("No recommendations found. Try describing your symptoms more clearly.")

    # Option to restart the chat
    if st.button("Start Over"):
        del st.session_state.chat_session
        st.rerun()