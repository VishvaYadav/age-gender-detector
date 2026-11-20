"""Task 6 - Senior citizen identification for a mall / local store.

Detects every person in a video (or webcam frames), predicts age and gender with the base model,
marks anyone over 60 as a senior citizen, and logs age, gender and time of visit to a CSV file.

Why tracking: a person stays in view for many frames. Without tracking they would be logged on
every frame. Each face is matched to the same visitor across frames (by box overlap / distance),
predictions are averaged over the frames they were seen (more stable than one frame), and the
visitor is written to the log once.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from utils.face_utils import SENIOR_AGE, FaceAnalyzer, draw_results

CSV_COLUMNS = ["visitor_id", "time_of_visit", "age", "gender", "senior_citizen", "frames_seen"]
SENIOR_COLOR = (0, 0, 255)      # red (BGR)
OTHER_COLOR = (0, 180, 0)       # green


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1, x2, y2 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def center_distance(a, b) -> float:
    """Distance between box centres, relative to the box size (1.0 = one face-width apart)."""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    d = np.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))
    return d / max(1.0, (aw + bw) / 2)


@dataclass
class Visitor:
    id: int
    box: tuple
    first_seen: datetime
    last_seen: datetime
    ages: list = field(default_factory=list)
    female_probs: list = field(default_factory=list)
    missing: int = 0
    logged: bool = False

    @property
    def hits(self) -> int:
        return len(self.ages)

    @property
    def age(self) -> int:
        return int(round(median(self.ages)))

    @property
    def gender(self) -> str:
        return "Female" if np.mean(self.female_probs) > 0.5 else "Male"

    @property
    def is_senior(self) -> bool:
        return median(self.ages) > SENIOR_AGE

    def record(self) -> dict:
        return {
            "visitor_id": self.id,
            "time_of_visit": self.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
            "age": self.age,
            "gender": self.gender,
            "senior_citizen": "Yes" if self.is_senior else "No",
            "frames_seen": self.hits,
        }


class SeniorCitizenTracker:
    """Feed it frames with update(); it keeps track of visitors and builds the visit log."""

    def __init__(self, analyzer: FaceAnalyzer, min_frames: int = 3, max_missing: int = 10,
                 iou_threshold: float = 0.3, max_center_distance: float = 0.8):
        self.analyzer = analyzer
        self.min_frames = min_frames            # ignore faces seen in fewer frames (false detections)
        self.max_missing = max_missing          # frames a visitor may vanish before they count as gone
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.visitors: list[Visitor] = []
        self.records: list[dict] = []
        self._next_id = 1

    # ---- matching faces to visitors ------------------------------------------------------- #
    def _match(self, boxes):
        active = [v for v in self.visitors if not v.logged]
        pairs = []
        for di, b in enumerate(boxes):
            for v in active:
                o = iou(b, v.box)
                if o >= self.iou_threshold or center_distance(b, v.box) <= self.max_center_distance:
                    pairs.append((o - center_distance(b, v.box), di, v))
        pairs.sort(key=lambda p: p[0], reverse=True)       # best matches first
        used_d, used_v, matches = set(), set(), {}
        for _, di, v in pairs:
            if di not in used_d and v.id not in used_v:
                matches[di] = v
                used_d.add(di)
                used_v.add(v.id)
        return matches

    # ---- main step ------------------------------------------------------------------------ #
    def update(self, frame_bgr: np.ndarray, timestamp: datetime) -> list[dict]:
        """Process one frame. Returns this frame's faces, each with its visitor id."""
        results = self.analyzer.analyze(frame_bgr)
        matches = self._match([r["box"] for r in results])
        seen_ids = set()
        for di, r in enumerate(results):
            v = matches.get(di)
            if v is None:
                v = Visitor(self._next_id, r["box"], timestamp, timestamp)
                self._next_id += 1
                self.visitors.append(v)
            v.box, v.last_seen, v.missing = r["box"], timestamp, 0
            v.ages.append(r["age_raw"])
            v.female_probs.append(r["gender_conf"] if r["gender"] == "Female" else 1 - r["gender_conf"])
            seen_ids.add(v.id)
            # show the visitor's running (averaged) prediction, not just this frame's
            r.update(visitor_id=v.id, age=v.age, gender=v.gender, is_senior=v.is_senior)

        for v in self.visitors:
            if not v.logged and v.id not in seen_ids:
                v.missing += 1
                if v.missing > self.max_missing:
                    self._log(v)
        return results

    def _log(self, v: Visitor):
        v.logged = True
        if v.hits >= self.min_frames:
            self.records.append(v.record())

    def finish(self) -> list[dict]:
        """Call at the end of the video: logs everyone still in view."""
        for v in self.visitors:
            if not v.logged:
                self._log(v)
        return self.records

    # ---- output --------------------------------------------------------------------------- #
    @staticmethod
    def draw(frame_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
        def label(r):
            text = f"#{r['visitor_id']} {r['gender']}, {r['age']}"
            return text + "  SENIOR" if r["is_senior"] else text
        return draw_results(frame_bgr, results, label_fn=label,
                            color_fn=lambda r: SENIOR_COLOR if r["is_senior"] else OTHER_COLOR)

    def save_csv(self, path: str | Path, append: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or not append
        with path.open("a" if append else "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new_file:
                w.writeheader()
            w.writerows(self.records)
        return path


def process_video(video_path: str, analyzer: FaceAnalyzer, every_n_frames: int = 5,
                  start_time: datetime | None = None, on_frame=None) -> SeniorCitizenTracker:
    """Run the tracker over a video file.

    every_n_frames: analyse 1 frame in N (5 -> ~6 per second for 30 fps video), much faster on CPU.
    on_frame(annotated_bgr, progress_0_to_1): optional callback, used by the app for a live preview.
    Time of visit = start_time + position in the video (start_time defaults to now).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    start_time = start_time or datetime.now()
    # max_missing is in analysed frames: allow ~2 seconds out of view before a visitor counts as gone
    tracker = SeniorCitizenTracker(analyzer, max_missing=max(3, int(2 * fps / every_n_frames)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n_frames == 0:
            ts = start_time + timedelta(seconds=i / fps)
            results = tracker.update(frame, ts)
            if on_frame:
                on_frame(tracker.draw(frame, results), min(1.0, i / total))
        i += 1
    cap.release()
    tracker.finish()
    return tracker
