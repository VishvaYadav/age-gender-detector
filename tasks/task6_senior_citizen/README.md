# Task 6 – Senior citizen identification

**Goal:** detect multiple people in a video or webcam feed of a mall / local store, predict each person's age and gender, mark anyone **over 60 as a senior citizen**, and store age, gender and time of visit in a CSV file.

**Status:** ✅ done

## How it builds on the base model
- Faces are found with the shared OpenCV DNN detector and passed to the **base UTKFace model** (`utils/face_utils.py`) for age and gender. No new model was trained: this task reuses and extends the training project.
- **Visitor tracking** (`senior_tracker.py`): each face is matched to the same visitor across frames using box overlap (IoU) and centre distance. Without this, one person would be logged on every frame.
- **Stable predictions:** a visitor's age is the **median** of all frames they were seen in, and gender is the average female probability. One blurry frame can't flip the result.
- **Noise filter:** faces seen in fewer than 3 analysed frames are ignored (usually false detections).
- **Logging:** when a visitor has been out of view for ~2 seconds (or the video ends) they are written to the log once.
- **Speed:** only 1 in every N frames is analysed (default 5, about 6 per second for 30 fps video), so it runs on a laptop CPU.

## Output
`outputs/visits.csv`:

| visitor_id | time_of_visit | age | gender | senior_citizen | frames_seen |
|---|---|---|---|---|---|
| 1 | 2026-11-20 10:15:02 | 67 | Male | Yes | 14 |

Time of visit = when the analysis started + the visitor's first appearance in the video (or the real time for webcam and photos).

## How to run
**App:** `streamlit run app.py` → sidebar → *6 · Senior citizen identification*. Upload a store video (live preview, red boxes = seniors) or use the camera, then download the CSV.

**Command line:**
```bash
python -m tasks.task6_senior_citizen.run_video store.mp4     # a video file
python -m tasks.task6_senior_citizen.run_video 0             # live webcam, Ctrl+C to stop
```

## Limitations
- The base model is less accurate above 60 because UTKFace has fewer older faces, so people near the threshold can be misclassified.
- A visitor who leaves and comes back after more than ~2 seconds is counted as a new visit.
