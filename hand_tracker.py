"""MediaPipe Tasks hand landmark detection and feature normalization."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def landmarks_to_feature(landmarks) -> np.ndarray:
    """Normalize 21 hand landmarks relative to wrist and hand size."""
    points = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
    )
    if points.shape != (21, 3):
        raise ValueError("Landmark tangan tidak berjumlah 21 titik.")

    # Make the wrist the origin and scale using the wrist-to-middle-MCP distance.
    points -= points[0].copy()
    scale = float(np.linalg.norm(points[9, :2]))
    if scale < 1e-5:
        scale = 1.0
    points /= scale
    return np.clip(points.reshape(-1), -5.0, 5.0).astype(np.float32)


class HandTracker:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model tangan belum ditemukan: {model_path}\n"
                "Jalankan: python download_hand_model.py"
            )

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe belum terpasang. Jalankan: pip install -r requirements.txt"
            ) from exc

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Return (normalized feature or None, pixel landmarks or None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None, None

        landmarks = result.hand_landmarks[0]
        feature = landmarks_to_feature(landmarks)
        height, width = frame_bgr.shape[:2]
        pixel_points = [
            (int(point.x * width), int(point.y * height)) for point in landmarks
        ]
        return feature, pixel_points

    def close(self) -> None:
        if getattr(self, "_detector", None) is not None:
            self._detector.close()
            self._detector = None
