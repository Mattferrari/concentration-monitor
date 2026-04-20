#!/usr/bin/env python3
"""
Captura de video desde cámara Raspberry Pi o USB usando picamera2 o OpenCV.
Expone: get_frame_generator() que devuelve frames en tiempo real como numpy arrays BGR.
"""

from typing import Generator, Optional
import numpy as np
import cv2


class FrameCapture:
    """Captura frames de la cámara con soporte para picamera2 (RPi) o OpenCV (USB)."""

    def __init__(self, resolution: tuple[int, int] = (640, 480), fps: int = 30):
        """
        Inicializa el capturador de frames.

        Args:
            resolution: Tupla (ancho, alto) de la resolución deseada
            fps: Fotogramas por segundo objetivo
        """
        self.resolution = resolution
        self.fps = fps
        self.camera = None
        self._init_camera()

    def _init_camera(self) -> None:
        """Intenta inicializar picamera2 (RPi), fallback a OpenCV USB."""
        try:
            # Intentar cargar picamera2 (Raspberry Pi)
            from picamera2 import Picamera2

            self.camera = Picamera2()
            config = self.camera.create_video_configuration(
                main={"format": "BGR888", "size": self.resolution}
            )
            self.camera.configure(config)
            self.camera.start()
            print(f"[OK] Cámara Raspberry Pi inicializada: {self.resolution} @ {self.fps}fps")
            self.use_picamera2 = True
        except Exception as e:
            print(f"[WARN] picamera2 no disponible ({e}), usando OpenCV...")
            try:
                # Fallback a OpenCV con cámara USB
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    raise RuntimeError("No se pudo abrir la cámara USB")

                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                print(f"[OK] Cámara USB inicializada: {self.resolution} @ {self.fps}fps")
                self.use_picamera2 = False
            except Exception as e2:
                print(f"[ERROR] Error inicializando cámara: {e2}")
                self.camera = None
                self.use_picamera2 = False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Captura un frame de la cámara.

        Returns:
            Frame como numpy array BGR o None si error
        """
        if self.camera is None:
            return None

        try:
            if self.use_picamera2:
                request = self.camera.capture_request()
                frame = request.make_array("main")
                request.release()
                return frame
            else:
                ret, frame = self.camera.read()
                return frame if ret else None
        except Exception as e:
            print(f"Error capturando frame: {e}")
            return None

    def get_frame_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generador infinito que devuelve frames en tiempo real.

        Yields:
            Frame como numpy array BGR
        """
        while True:
            frame = self.get_frame()
            if frame is not None:
                yield frame

    def release(self) -> None:
        """Libera los recursos de la cámara."""
        if self.camera is not None:
            try:
                if self.use_picamera2:
                    self.camera.stop()
                else:
                    self.camera.release()
                print("[OK] Cámara liberada")
            except Exception as e:
                print(f"Error liberando cámara: {e}")
