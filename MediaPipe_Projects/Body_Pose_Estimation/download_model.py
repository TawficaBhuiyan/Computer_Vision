"""One-time download of the MediaPipe Pose Landmarker model bundle.

Cross-platform (Windows/macOS/Linux). Run once after creating the env:
    python download_model.py
"""
import os
import urllib.request

DEST = os.path.join("models", "pose_landmarker_heavy.task")
URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)


def main() -> None:
    os.makedirs("models", exist_ok=True)
    if os.path.exists(DEST):
        print(f"Already present: {DEST}")
        return
    print("Downloading pose_landmarker_heavy.task ...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST}")


if __name__ == "__main__":
    main()
