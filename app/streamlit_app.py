import streamlit as st
from src.recommend import recommend

st.title("OTC Medication Chat Agent")
user_input = st.text_input("Describe your symptoms:")

if user_input:
    symptoms = user_input.lower().split(", ")
    results = recommend(symptoms)

    if results:
        for med in results:
            st.subheader(med["name"])
            st.write(f"**Purpose:** {med['purpose']}")
            st.write(f"**Notes:** {med['notes']}")
    else:
        st.warning("No recommendations found. Try describing your symptoms more clearly.")
