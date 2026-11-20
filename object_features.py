"""Feature extraction for the live object classifier."""
from __future__ import annotations

import cv2
import numpy as np


# The object is expected to be held in the central guide area shown in the UI.
ROI_FRACTION = 0.72
FEATURE_SIZE = (96, 96)

_HOG = cv2.HOGDescriptor(
    _winSize=FEATURE_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def center_roi(frame: np.ndarray) -> np.ndarray:
    """Return a square crop around the center of a BGR frame."""
    if frame is None or frame.size == 0:
        raise ValueError("Frame kamera kosong.")

    height, width = frame.shape[:2]
    side = max(32, int(min(height, width) * ROI_FRACTION))
    x1 = max(0, (width - side) // 2)
    y1 = max(0, (height - side) // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Area objek tidak valid.")
    return crop


def extract_object_features(frame: np.ndarray) -> np.ndarray:
    """Create a compact HOG + HSV histogram feature vector for one frame."""
    crop = center_roi(frame)
    resized = cv2.resize(crop, FEATURE_SIZE, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    hog_features = _HOG.compute(gray).reshape(-1).astype(np.float32)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256]
    )
    color_hist = cv2.normalize(color_hist, color_hist).reshape(-1).astype(np.float32)

    return np.concatenate([hog_features, color_hist]).astype(np.float32)
