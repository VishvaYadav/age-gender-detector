"""Command-line version of Task 6.

    python -m tasks.task6_senior_citizen.run_video store.mp4
    python -m tasks.task6_senior_citizen.run_video 0              # 0 = webcam (press Ctrl+C to stop)

Writes the visit log to outputs/visits.csv (appends if it exists).
"""
import argparse
import time
from datetime import datetime

import cv2

from tasks.task6_senior_citizen.senior_tracker import SeniorCitizenTracker, process_video
from utils.face_utils import FaceAnalyzer


def run_webcam(analyzer, camera_index, every_n):
    cap = cv2.VideoCapture(camera_index)
    tracker = SeniorCitizenTracker(analyzer, max_missing=15)
    print("Webcam running - press Ctrl+C to stop.")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every_n == 0:
                for r in tracker.update(frame, datetime.now()):
                    print(f"  visitor #{r['visitor_id']}: {r['gender']}, {r['age']}"
                          f"{'  SENIOR' if r['is_senior'] else ''}")
            i += 1
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    cap.release()
    tracker.finish()
    return tracker


def main():
    p = argparse.ArgumentParser(description="Senior citizen identification on a video or webcam.")
    p.add_argument("source", help="video file path, or a webcam number such as 0")
    p.add_argument("--every", type=int, default=5, help="analyse 1 in N frames (default 5)")
    p.add_argument("--csv", default="outputs/visits.csv")
    args = p.parse_args()

    analyzer = FaceAnalyzer()
    if args.source.isdigit():
        tracker = run_webcam(analyzer, int(args.source), args.every)
    else:
        tracker = process_video(args.source, analyzer, every_n_frames=args.every)

    print(f"\nVisitors logged: {len(tracker.records)}")
    for r in tracker.records:
        print(f"  #{r['visitor_id']}  {r['time_of_visit']}  {r['gender']:6}  age {r['age']:3}  "
              f"senior: {r['senior_citizen']}")
    print(f"Saved to {tracker.save_csv(args.csv)}")


if __name__ == "__main__":
    main()
