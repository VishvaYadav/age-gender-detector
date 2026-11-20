"""Task 6 page: senior citizen identification on a store video or camera snapshots."""
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app_pages.common import image_input, load_analyzer_or_stop
from tasks.task6_senior_citizen.senior_tracker import CSV_COLUMNS, SeniorCitizenTracker, process_video
from utils.face_utils import SENIOR_AGE, bgr_to_rgb

LOG_PATH = Path("outputs/visits.csv")

st.title("👴 Task 6 – Senior Citizen Identification")
st.write(f"Detects every visitor in a store video or camera photo, predicts **age and gender**, marks anyone "
         f"**over {SENIOR_AGE} as a senior citizen** (red box), and logs age, gender and time of visit to a CSV file.")

analyzer = load_analyzer_or_stop()
if "visit_log" not in st.session_state:
    st.session_state.visit_log = []


def add_to_log(records):
    """Give records ids that continue from the existing log, keep them in the session and in the CSV."""
    start = len(st.session_state.visit_log)
    for k, r in enumerate(records, 1):
        r["visitor_id"] = start + k
    st.session_state.visit_log.extend(records)
    LOG_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(records, columns=CSV_COLUMNS).to_csv(LOG_PATH, mode="a", index=False,
                                                       header=not LOG_PATH.exists())


def show_summary(records):
    if not records:
        st.warning("No visitors detected. Try a video where faces are clearly visible.")
        return
    df = pd.DataFrame(records, columns=CSV_COLUMNS)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Visitors", len(df))
    c2.metric("Senior citizens", int((df.senior_citizen == "Yes").sum()))
    c3.metric("Male", int((df.gender == "Male").sum()))
    c4.metric("Female", int((df.gender == "Female").sum()))
    st.dataframe(df.style.apply(lambda row: ["background-color: rgba(255,0,0,0.15)"] * len(row)
                                if row.senior_citizen == "Yes" else [""] * len(row), axis=1),
                 hide_index=True, width="stretch")


tab_video, tab_photo = st.tabs(["🎬 Store video", "📷 Camera / photo"])

with tab_video:
    video = st.file_uploader("Upload a video (mp4, avi, mov)", type=["mp4", "avi", "mov", "mkv"])
    every_n = st.slider("Analyse 1 in every N frames (higher = faster, lower = more accurate)", 1, 15, 5)
    if video and st.button("▶ Analyse video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.name).suffix) as tmp:
            tmp.write(video.read())
        preview, progress = st.empty(), st.progress(0.0, text="Analysing video…")

        def on_frame(frame, p):
            preview.image(bgr_to_rgb(frame), caption="Live preview (red = senior citizen)", width="stretch")
            progress.progress(p, text=f"Analysing video… {p:.0%}")

        tracker = process_video(tmp.name, analyzer, every_n_frames=every_n, start_time=datetime.now(),
                                on_frame=on_frame)
        progress.progress(1.0, text="Done")
        Path(tmp.name).unlink(missing_ok=True)
        add_to_log(tracker.records)
        st.session_state.last_video_records = tracker.records

    if st.session_state.get("last_video_records") is not None:
        st.subheader("Visitors in this video")
        show_summary(st.session_state.last_video_records)

with tab_photo:
    st.caption("Each photo is logged as a visit at the current time. For a live webcam feed, run "
               "`python -m tasks.task6_senior_citizen.run_video 0` on your computer.")
    bgr, _ = image_input("task6", allow_camera=True)
    if bgr is not None:
        tracker = SeniorCitizenTracker(analyzer, min_frames=1)
        results = tracker.update(bgr, datetime.now())
        st.image(bgr_to_rgb(tracker.draw(bgr, results)), width="stretch")
        key = hash(bgr.tobytes())
        if st.session_state.get("last_photo_key") != key:          # log each photo only once
            st.session_state.last_photo_key = key
            st.session_state.last_photo_records = tracker.finish()
            add_to_log(st.session_state.last_photo_records)
        show_summary(st.session_state.last_photo_records)

st.divider()
st.subheader("📒 Visit log")
log = st.session_state.visit_log
if log:
    df = pd.DataFrame(log, columns=CSV_COLUMNS)
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("⬇ Download visit log (CSV)", df.to_csv(index=False), "visits.csv", "text/csv")
    if st.button("Clear log"):
        st.session_state.visit_log = []
        st.session_state.pop("last_video_records", None)
        st.rerun()
else:
    st.caption("No visits logged yet in this session.")
