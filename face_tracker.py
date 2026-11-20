"""MediaPipe Face Landmarker wrapper that returns 52 blendshape features."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# A small set of facial landmarks used only to draw a simple live overlay.
DISPLAY_LANDMARKS = (10, 152, 33, 263, 61, 291, 1, 199, 0, 17)


class FaceTracker:
    """Detect one face and expose its 52 blendshape coefficients."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model wajah belum ditemukan: {model_path}\n"
                "Jalankan: python download_models.py"
            )

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe belum terpasang. Jalankan: pip install -r requirements.txt"
            ) from exc

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Return (52 blendshape scores or None, selected pixel landmarks or None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.face_blendshapes or not result.face_landmarks:
            return None, None

        categories = result.face_blendshapes[0]
        # Sort by category index so feature ordering stays identical frame to frame.
        ordered = sorted(
            enumerate(categories),
            key=lambda item: item[1].index if item[1].index >= 0 else item[0],
        )
        feature = np.asarray([item.score for _, item in ordered], dtype=np.float32)
        if feature.size != 52:
            raise RuntimeError(
                f"MediaPipe mengembalikan {feature.size} blendshape; aplikasi membutuhkan 52."
            )

        landmarks = result.face_landmarks[0]
        height, width = frame_bgr.shape[:2]
        pixel_points = [
            (int(landmarks[index].x * width), int(landmarks[index].y * height))
            for index in DISPLAY_LANDMARKS
        ]
        return np.clip(feature, 0.0, 1.0), pixel_points

    def close(self) -> None:
        detector = getattr(self, "_detector", None)
        if detector is not None:
            detector.close()
            self._detector = None
