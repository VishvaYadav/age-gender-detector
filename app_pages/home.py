"""Home page: the base model on its own - upload a photo, see every face's age, gender and ethnicity."""
import pandas as pd
import streamlit as st

from app_pages.common import image_input, load_analyzer_or_stop
from utils.face_utils import bgr_to_rgb, draw_results

st.title("🧑 Age, Gender & Ethnicity Detector")
st.write("A CNN trained on the **UTKFace** dataset (23,705 faces). Upload a photo or use your camera: "
         "every face is detected and its age, gender and ethnicity are predicted.")

analyzer = load_analyzer_or_stop()
bgr, _ = image_input("home")

if bgr is not None:
    with st.spinner("Analysing…"):
        results = analyzer.analyze(bgr, whole_image_if_no_face=False)

    col_img, col_out = st.columns([3, 2])
    with col_img:
        st.subheader("Preview")
        st.image(bgr_to_rgb(draw_results(bgr, results)), width="stretch")
    with col_out:
        st.subheader("Results")
        if not results:
            st.warning("No face found. Try a clearer, front-facing photo.")
        else:
            st.metric("Faces found", len(results))
            st.dataframe(pd.DataFrame([{
                "Face": i + 1,
                "Gender": f"{r['gender']} ({r['gender_conf']:.0%})",
                "Age": r["age"],
                "Ethnicity": f"{r['ethnicity']} ({r['ethnicity_conf']:.0%})",
            } for i, r in enumerate(results)]), hide_index=True, width="stretch")
            with st.expander("Ethnicity probabilities"):
                for i, r in enumerate(results):
                    st.write(f"**Face {i + 1}**")
                    st.bar_chart(pd.Series(r["ethnicity_probs"]), horizontal=True, height=180)

with st.expander("ℹ️ About the model"):
    st.markdown("""
- **Face detection:** OpenCV DNN (res10 SSD)
- **Input:** each face cropped with a 25% margin, grayscale, 128×128
- **Network:** 5 convolution blocks shared by three heads: gender (sigmoid), ethnicity (5-class softmax), age (regression)
- **Training notebook:** `notebooks/01_base_model_training.ipynb`
""")
