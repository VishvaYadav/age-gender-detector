"""Test the shared face pipeline on an image from the command line.

    python -m utils.demo path/to/photo.jpg
    python -m utils.demo path/to/photo.jpg --out result.jpg

Prints every detected face's age, gender and ethnicity and saves an annotated copy.
"""
import argparse
from pathlib import Path

import cv2

from utils.face_utils import FaceAnalyzer, draw_results


def main():
    parser = argparse.ArgumentParser(description="Detect faces and predict age, gender and ethnicity.")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--out", help="where to save the annotated image (default: <name>_result.jpg)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    analyzer = FaceAnalyzer()
    print(f"Face detector: {analyzer.detector.kind}")
    results = analyzer.analyze(img)
    print(f"Faces found: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"  Face {i}: {r['gender']} ({r['gender_conf']:.0%}), age {r['age']}, "
              f"{r['ethnicity']} ({r['ethnicity_conf']:.0%}){'  -> SENIOR CITIZEN' if r['is_senior'] else ''}"
              f"  box={r['box']}")

    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_result.jpg"))
    cv2.imwrite(out, draw_results(img, results))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
