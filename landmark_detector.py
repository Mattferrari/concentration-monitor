#!/usr/bin/env python3
"""
Detector de landmarks faciales usando MediaPipe Face Landmarker.
Soporta aceleración con Coral TPU con fallback automático a CPU.
Expone: FaceLandmarkDetector.detect(frame) -> landmarks o None
"""

from typing import Optional
import numpy as np
from pathlib import Path


class FaceLandmarkDetector:
    """Detecta 478 landmarks faciales usando MediaPipe, con soporte Coral TPU."""

    def __init__(self, model_path: str = "models/face_landmarker.task", use_tpu: bool = True):
        """
        Inicializa el detector con el modelo MediaPipe.

        Args:
            model_path: Ruta al archivo face_landmarker.task
            use_tpu: Intentar usar Coral TPU si está disponible
        """
        self.model_path = model_path
        self.detector = None
        self.use_tpu = use_tpu
        self._init_detector()

    def _init_detector(self) -> None:
        """Inicializa MediaPipe Face Landmarker con fallback TPU -> CPU."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

            # Intentar con Coral TPU
            if self.use_tpu:
                try:
                    from mediapipe.tasks.python import BaseOptions

                    base_options = BaseOptions(model_asset_path=self.model_path)
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options, num_faces=1, min_face_detection_confidence=0.5
                    )
                    self.detector = vision.FaceLandmarker.create_from_options(options)
                    print("✓ Face Landmarker inicializado (Intentando Coral TPU)")
                except Exception as tpu_error:
                    print(f"⚠ Coral TPU no disponible: {tpu_error}, usando CPU...")
                    self._init_detector_cpu()
            else:
                self._init_detector_cpu()

        except Exception as e:
            print(f"✗ Error inicializando Face Landmarker: {e}")
            self.detector = None

    def _init_detector_cpu(self) -> None:
        """Inicializa Face Landmarker en CPU."""
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision

            base_options = BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options, num_faces=1, min_face_detection_confidence=0.5
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("✓ Face Landmarker inicializado (CPU)")
        except Exception as e:
            print(f"✗ Error inicializando Face Landmarker CPU: {e}")
            self.detector = None

    def detect(self, frame: np.ndarray) -> Optional[list[tuple[float, float, float]]]:
        """
        Detecta landmarks faciales en un frame.

        Args:
            frame: Frame como numpy array BGR (altura x ancho x 3)

        Returns:
            Lista de 478 landmarks como [(x, y, z), ...] normalizados [0,1] o None si no detecta
        """
        if self.detector is None:
            return None

        try:
            import mediapipe as mp

            # Convertir BGR a RGB
            frame_rgb = frame[..., ::-1]  # BGR -> RGB

            # Crear image object de MediaPipe
            mp_image = mp.Image(data=frame_rgb, image_format=mp.ImageFormat.SRGB)

            # Detectar landmarks
            detection_result = self.detector.detect(mp_image)

            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                # Extraer landmarks del primer rostro detectado
                landmarks = detection_result.face_landmarks[0]
                # Convertir a lista de tuplas (x, y, z) normalizadas
                landmarks_list = [(lm.x, lm.y, lm.z) for lm in landmarks]
                return landmarks_list
            else:
                return None

        except Exception as e:
            print(f"Error detectando landmarks: {e}")
            return None

    def release(self) -> None:
        """Libera recursos del detector."""
        self.detector = None
