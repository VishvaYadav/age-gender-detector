"""Helpers shared by every page of the Streamlit app."""
import streamlit as st
from PIL import Image

from utils.face_utils import FaceAnalyzer, pil_to_bgr


@st.cache_resource(show_spinner="Loading the age & gender model…")
def get_analyzer() -> FaceAnalyzer:
    """Load the face detector and base model once and share them across pages and reruns."""
    return FaceAnalyzer()


def load_analyzer_or_stop() -> FaceAnalyzer:
    try:
        return get_analyzer()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


def image_input(key: str, label: str = "Upload a photo", allow_camera: bool = True):
    """Upload or camera snapshot -> (BGR array, PIL image), or (None, None) if nothing given yet."""
    tabs = st.tabs(["📁 Upload image", "📷 Camera"] if allow_camera else ["📁 Upload image"])
    file = None
    with tabs[0]:
        file = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=f"{key}_upload")
    if allow_camera:
        with tabs[1]:
            snap = st.camera_input("Take a photo", key=f"{key}_camera")
            file = file or snap
    if file is None:
        return None, None
    pil = Image.open(file)
    return pil_to_bgr(pil), pil


def coming_soon(title: str, description: str):
    st.title(title)
    st.info(description)
    st.caption("🚧 This task is under construction.")
