"""Headless end-to-end smoke test used by CI.

Runs the real detect -> annotate pipeline on a committed fixture image and
asserts that known objects are found. No camera or display required.
"""

import os
import sys

import cv2

# Make the project root importable when run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.detector import Detector  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "street.jpg")


def test_detects_objects_in_fixture():
    frame = cv2.imread(FIXTURE)
    assert frame is not None, f"could not read fixture: {FIXTURE}"

    detector = Detector(
        config.MODEL_NAME,
        config.CONFIDENCE_THRESHOLD,
        config.IOU_THRESHOLD,
        config.CLASS_FILTER,
    )
    result = detector.detect(frame)
    labels = [result.names[int(b.cls[0])] for b in result.boxes]

    print("Detected:", labels)
    assert len(result.boxes) > 0, "expected at least one detection"
    assert "person" in labels, f"expected a person, got {labels}"

    # annotation must return a same-shaped image without raising
    annotated = detector.annotate(frame, result)
    assert annotated.shape == frame.shape


if __name__ == "__main__":
    test_detects_objects_in_fixture()
    print("smoke test passed")
