"""Age & Gender Detector - main Streamlit app.

Run locally:   streamlit run app.py
Each internship task is a page in the sidebar; all of them reuse the base model in utils/face_utils.py.
"""
import streamlit as st

st.set_page_config(page_title="Age & Gender Detector", page_icon="🧑", layout="wide")

pages = {
    "Base model": [
        st.Page("app_pages/home.py", title="Age, gender & ethnicity", icon="🏠", default=True),
    ],
    "Internship tasks": [
        st.Page("app_pages/task1_car_colour.py", title="1 · Car colour detection", icon="🚗"),
        st.Page("app_pages/task2_sign_language.py", title="2 · Sign language", icon="🤟"),
        st.Page("app_pages/task3_nationality.py", title="3 · Nationality detection", icon="🌍"),
        st.Page("app_pages/task4_long_hair.py", title="4 · Long hair identification", icon="💇"),
        st.Page("app_pages/task5_voice_age.py", title="5 · Age from voice", icon="🎙️"),
        st.Page("app_pages/task6_senior_citizen.py", title="6 · Senior citizen identification", icon="👴"),
    ],
}

st.navigation(pages).run()
