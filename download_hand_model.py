"""Download the official MediaPipe Hand Landmarker model asset."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"Model sudah tersedia: {MODEL_PATH}")
        return

    print("Mengunduh model Hand Landmarker resmi MediaPipe...")
    print("Pastikan komputer terhubung ke internet.")
    temporary_path = MODEL_PATH.with_suffix(".task.download")
    try:
        urlretrieve(MODEL_URL, temporary_path)
        if temporary_path.stat().st_size < 1_000_000:
            raise RuntimeError("File model terlalu kecil; unduhan mungkin tidak lengkap.")
        temporary_path.replace(MODEL_PATH)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print(f"Selesai. Model tersimpan di: {MODEL_PATH}")


if __name__ == "__main__":
    main()
