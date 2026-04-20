#!/usr/bin/env python3
"""
Descarga el modelo Face Landmarker de MediaPipe desde Google Storage.
Uso: python download_model.py
"""

import os
import urllib.request
import sys
from pathlib import Path

# URL del modelo MediaPipe Face Landmarker
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
CHUNK_SIZE = 8192


def download_model() -> bool:
    """Descarga el modelo face_landmarker.task desde Google Storage."""
    MODEL_DIR.mkdir(exist_ok=True)

    if MODEL_PATH.exists():
        print(f"[OK] Modelo ya existe en {MODEL_PATH}")
        return True

    print(f"Descargando modelo desde {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[OK] Modelo descargado exitosamente en {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Error descargando modelo: {e}", file=sys.stderr)
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
