"""Download official MediaPipe Hand and Face Landmarker task assets."""
from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def download_file(filename: str, url: str) -> None:
    destination = MODEL_DIR / filename
    if destination.exists() and destination.stat().st_size > 500_000:
        print(f"Sudah tersedia: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".download")
    request = Request(url, headers={"User-Agent": "TeachableMachinePython/1.0"})
    print(f"Mengunduh {filename} ...")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temporary.stat().st_size < 500_000:
            raise RuntimeError(f"File {filename} terlalu kecil; unduhan tidak lengkap.")
        temporary.replace(destination)
        print(f"Selesai: {destination}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        download_file(filename, url)
    print("Semua model MediaPipe siap digunakan.")


if __name__ == "__main__":
    main()
