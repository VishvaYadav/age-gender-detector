"""Shared face detection and age / gender / ethnicity prediction.

Every internship task imports from here, so the base model trained in
notebooks/01_base_model_training.ipynb is loaded and used the same way everywhere.

Quick use:
    from utils.face_utils import FaceAnalyzer
    analyzer = FaceAnalyzer()
    results = analyzer.analyze(bgr_image)       # one dict per face
    annotated = draw_results(bgr_image, results)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

BASE_MODEL_PATH = MODELS_DIR / "base_age_gender_ethnicity.keras"
LABELS_PATH = MODELS_DIR / "labels.json"
DNN_PROTO = MODELS_DIR / "face_detector" / "deploy.prototxt"
DNN_WEIGHTS = MODELS_DIR / "face_detector" / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

SENIOR_AGE = 60


# --------------------------------------------------------------------------- #
# Face detection
# --------------------------------------------------------------------------- #
class FaceDetector:
    """Finds faces in a BGR image and returns (x, y, w, h) boxes.

    Uses OpenCV's DNN face detector (res10 SSD), which handles tilted faces,
    small faces and several people far better than the old Haar cascade.
    Falls back to the Haar cascade bundled with OpenCV if the DNN files are missing.
    """

    def __init__(self, confidence: float = 0.6, min_size: int = 24):
        self.confidence = confidence
        self.min_size = min_size
        # readNetFromCaffe exists in OpenCV 4.x (pinned in requirements.txt); OpenCV 5 removed it
        if DNN_PROTO.exists() and DNN_WEIGHTS.exists() and hasattr(cv2.dnn, "readNetFromCaffe"):
            self.kind = "dnn"
            self.net = cv2.dnn.readNetFromCaffe(str(DNN_PROTO), str(DNN_WEIGHTS))
        else:
            self.kind = "haar"
            cascade = Path(getattr(cv2, "data", None).haarcascades if hasattr(cv2, "data") else "") / \
                "haarcascade_frontalface_default.xml"
            self.net = cv2.CascadeClassifier(str(cascade))
            if self.net.empty():
                raise RuntimeError("No face detector available. Install OpenCV 4.x: "
                                   "pip install 'opencv-python-headless>=4.8,<5'")

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        boxes = []
        if self.kind == "dnn":
            blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            out = self.net.forward()[0, 0]                      # rows: [_, _, score, x1, y1, x2, y2]
            for _, _, score, x1, y1, x2, y2 in out:
                if score < self.confidence:
                    continue
                x1, y1 = int(max(0, x1 * w)), int(max(0, y1 * h))
                x2, y2 = int(min(w, x2 * w)), int(min(h, y2 * h))
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            found = self.net.detectMultiScale(gray, 1.1, 5, minSize=(self.min_size, self.min_size))
            boxes = [tuple(int(v) for v in b) for b in found]
        return [b for b in boxes if b[2] >= self.min_size and b[3] >= self.min_size]


def expand_box(box, image_shape, margin: float = 0.25):
    """Grow a tight face box so the crop looks like UTKFace (forehead, chin, a bit of hair)."""
    x, y, w, h = box
    H, W = image_shape[:2]
    side = int(max(w, h) * (1 + 2 * margin))
    cx, cy = x + w / 2, y + h / 2
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    return x1, y1, x2 - x1, y2 - y1


# --------------------------------------------------------------------------- #
# Age / gender / ethnicity prediction
# --------------------------------------------------------------------------- #
class AgeGenderPredictor:
    """Wraps the base multi-task CNN: face crop in, age / gender / ethnicity out."""

    def __init__(self, model_path: Path = BASE_MODEL_PATH, labels_path: Path = LABELS_PATH):
        import tensorflow as tf  # imported here so the detector can be used without TensorFlow

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Download base_age_gender_ethnicity.keras "
                "from the Kaggle notebook's Output tab and put it in the models/ folder.")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        labels = json.loads(Path(labels_path).read_text())
        self.img_size = int(labels.get("img_size", 128))
        self.gender_names = {int(k): v for k, v in labels["gender"].items()}
        self.ethnicity_names = {int(k): v for k, v in labels["ethnicity"].items()}

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Same preprocessing as training: grayscale, 128x128, scaled to 0-1."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
        gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return (gray.astype("float32") / 255.0)[..., None]

    def predict(self, face_crops: list[np.ndarray]) -> list[dict]:
        """Predict for a batch of face crops (BGR). Returns one dict per face."""
        if not face_crops:
            return []
        batch = np.stack([self.preprocess(f) for f in face_crops])
        p_gender, p_eth, p_age = self.model.predict(batch, verbose=0)
        results = []
        for g, e, a in zip(p_gender[:, 0], p_eth, p_age[:, 0]):
            gi, ei = int(g > 0.5), int(e.argmax())
            age = float(max(0.0, a))
            results.append({
                "age": int(round(age)),
                "age_raw": age,
                "gender": self.gender_names[gi],
                "gender_conf": float(g if gi == 1 else 1 - g),
                "ethnicity": self.ethnicity_names[ei],
                "ethnicity_conf": float(e[ei]),
                "ethnicity_probs": {self.ethnicity_names[i]: float(p) for i, p in enumerate(e)},
                "is_senior": age > SENIOR_AGE,
            })
        return results


class FaceAnalyzer:
    """Detector + predictor in one: give it an image, get every face's predictions."""

    def __init__(self, detector: FaceDetector | None = None, predictor: AgeGenderPredictor | None = None,
                 margin: float = 0.25):
        self.detector = detector or FaceDetector()
        self.predictor = predictor or AgeGenderPredictor()
        self.margin = margin

    def analyze(self, bgr: np.ndarray, whole_image_if_no_face: bool = False) -> list[dict]:
        boxes = self.detector.detect(bgr)
        if not boxes and whole_image_if_no_face:
            boxes = [(0, 0, bgr.shape[1], bgr.shape[0])]
        crops, kept = [], []
        for b in boxes:
            x, y, w, h = expand_box(b, bgr.shape, self.margin)
            crop = bgr[y:y + h, x:x + w]
            if crop.size:
                crops.append(crop)
                kept.append(b)
        results = self.predictor.predict(crops)
        for r, b in zip(results, kept):
            r["box"] = b
        return results


# --------------------------------------------------------------------------- #
# Drawing and image helpers
# --------------------------------------------------------------------------- #
def default_label(r: dict) -> str:
    text = f"{r['gender']}, {r['age']}"
    return text + " (Senior)" if r.get("is_senior") else text


def draw_results(bgr: np.ndarray, results: list[dict], label_fn=default_label,
                 color=(0, 200, 0), color_fn=None) -> np.ndarray:
    """Draw a box and label for each result. color_fn(r) -> BGR tuple overrides color."""
    out = bgr.copy()
    scale = max(0.5, min(out.shape[:2]) / 900)
    thick = max(1, int(round(2 * scale)))
    for r in results:
        x, y, w, h = r["box"]
        c = color_fn(r) if color_fn else color
        cv2.rectangle(out, (x, y), (x + w, y + h), c, thick + 1)
        text = label_fn(r)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thick)
        ty = y - 6 if y - th - 10 > 0 else y + h + th + 6
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 6, ty + base), c, -1)
        cv2.putText(out, text, (x + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), thick,
                    cv2.LINE_AA)
    return out


def pil_to_bgr(pil_image) -> np.ndarray:
    """Convert an uploaded PIL image (e.g. from Streamlit) to an OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
