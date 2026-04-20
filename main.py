#!/usr/bin/env python3
"""
Orquestador principal del sistema de detección de concentración.
Integra captura, detección, cálculo de métricas, serie temporal y visualización.
"""

import argparse
import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np

from capture import FrameCapture
from landmark_detector import FaceLandmarkDetector
from concentration import (
    compute_eye_aspect_ratio,
    compute_blink_rate,
    compute_gaze_deviation,
    compute_head_yaw,
    compute_mouth_aspect_ratio,
    detect_yawn,
    compute_concentration_score,
)
from time_series import ConcentrationTimeSeries
from dashboard import ConcentrationDashboard


class ConcentrationMonitor:
    """Sistema integrado de monitoreo de concentración."""

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        fps_target: int = 30,
        interval_seconds: int = 10,
        use_tpu: bool = True,
    ):
        """
        Inicializa el monitor de concentración.

        Args:
            resolution: Resolución de captura (ancho, alto)
            fps_target: FPS objetivo de captura
            interval_seconds: Intervalo en segundos para loguear scores
            use_tpu: Intentar usar Coral TPU
        """
        self.resolution = resolution
        self.fps_target = fps_target
        self.interval_seconds = interval_seconds

        # Inicializar componentes
        print("Inicializando componentes...")
        self.capture = FrameCapture(resolution=resolution, fps=fps_target)
        self.detector = FaceLandmarkDetector(use_tpu=use_tpu)
        self.time_series = ConcentrationTimeSeries()
        self.dashboard = ConcentrationDashboard()

        # Series temporales para métricas
        self.ear_deque = deque(maxlen=900)  # 30s a 30fps
        self.mar_deque = deque(maxlen=900)

        # Logging a CSV
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.csv_path = self._init_csv_log()

        # Contadores
        self.frame_count = 0
        self.last_score_log = time.time()
        self.face_missing_since = None
        self.fps_accumulator = deque(maxlen=30)

    def _init_csv_log(self) -> Path:
        """
        Inicializa archivo CSV para logging.

        Returns:
            Ruta del archivo CSV
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = self.log_dir / f"session_{timestamp}.csv"

        # Escribir header
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "score", "ear", "gaze_dev", "head_yaw", "blink_rate", "recommendation"]
            )

        print(f"📝 Logging a: {csv_path}")
        return csv_path

    def _log_score(
        self,
        score: float,
        ear: float,
        gaze_dev: float,
        head_yaw: float,
        blink_rate: float,
        recommendation: str,
    ) -> None:
        """
        Loguea un score en el CSV.

        Args:
            score: Score de concentración
            ear: Eye Aspect Ratio
            gaze_dev: Desviación de mirada en grados
            head_yaw: Ángulo yaw en grados
            blink_rate: Parpadeos por minuto
            recommendation: Texto de recomendación
        """
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{score:.2f}",
                    f"{ear:.3f}",
                    f"{gaze_dev:.2f}",
                    f"{head_yaw:.2f}",
                    f"{blink_rate:.2f}",
                    recommendation,
                ]
            )

    def run(self) -> None:
        """Ejecuta el loop principal del sistema."""
        print("🚀 Iniciando monitor de concentración...")
        print("Presiona 'q' o Ctrl+C para salir\n")

        try:
            frame_generator = self.capture.get_frame_generator()

            while True:
                loop_start = time.time()

                # Capturar frame
                try:
                    frame = next(frame_generator)
                except StopIteration:
                    print("⚠ Captura de frames terminada")
                    break

                # Detectar landmarks
                landmarks = self.detector.detect(frame)

                # Calcular score
                score = None
                ear = None
                gaze_dev = None
                head_yaw = None
                blink_rate = None
                yawn = False

                if landmarks is not None:
                    # Reset contador de rostro faltante
                    self.face_missing_since = None

                    # Calcular métricas
                    ear_left, ear_right = compute_eye_aspect_ratio(landmarks)
                    ear = (ear_left + ear_right) / 2.0
                    self.ear_deque.append(ear)

                    gaze_dev = compute_gaze_deviation(landmarks)
                    head_yaw = compute_head_yaw(landmarks)

                    mar = compute_mouth_aspect_ratio(landmarks)
                    self.mar_deque.append(mar)

                    blink_rate = compute_blink_rate(self.ear_deque)
                    yawn = detect_yawn(self.mar_deque)

                    # Score final
                    score = compute_concentration_score(ear, gaze_dev, head_yaw, blink_rate, yawn)
                    self.time_series.add_score(score)

                else:
                    # Rostro no detectado
                    if self.face_missing_since is None:
                        self.face_missing_since = time.time()

                    # Si lleva > 5s sin detectar, resetear
                    if time.time() - self.face_missing_since > 5:
                        self.ear_deque.clear()
                        self.mar_deque.clear()
                        self.face_missing_since = None

                # Logging cada interval_seconds
                current_time = time.time()
                if current_time - self.last_score_log >= self.interval_seconds:
                    recommendation = self.time_series.get_recommendation()
                    if score is not None:
                        self._log_score(
                            score, ear or 0, gaze_dev or 0, head_yaw or 0, blink_rate or 0, recommendation
                        )
                    self.last_score_log = current_time

                # Renderizar dashboard
                moving_avg = self.time_series.get_moving_average()
                scores_history = self.time_series.get_score_history_last_n_minutes(5)
                recommendation = self.time_series.get_recommendation()

                # Calcular FPS
                loop_time = time.time() - loop_start
                if loop_time > 0:
                    fps = 1.0 / loop_time
                    self.fps_accumulator.append(fps)
                    avg_fps = sum(self.fps_accumulator) / len(self.fps_accumulator)
                else:
                    avg_fps = 0

                # Renderizar
                canvas = self.dashboard.render(
                    frame, landmarks, score, moving_avg, scores_history, recommendation, avg_fps
                )

                # Mostrar canvas
                cv2.imshow("Concentration Monitor", canvas)

                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\n⏹ Detenido por usuario")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Limpia recursos."""
        print("\n🧹 Limpiando recursos...")
        self.capture.release()
        self.detector.release()
        cv2.destroyAllWindows()
        print(f"✓ {self.frame_count} frames procesados")
        print(f"✓ Log guardado en: {self.csv_path}")
        print("✓ Salida limpia")


def main():
    """Función principal con argparse."""
    parser = argparse.ArgumentParser(
        description="Sistema de detección de concentración para Raspberry Pi 5 + Coral TPU"
    )
    parser.add_argument(
        "--no-tpu",
        action="store_true",
        help="Forzar CPU, no usar Coral TPU",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="640x480",
        help="Resolución de captura (ancho x alto, default: 640x480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS objetivo (default: 30)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Intervalo de logging en segundos (default: 10)",
    )

    args = parser.parse_args()

    # Parsear resolución
    try:
        width, height = map(int, args.resolution.split("x"))
        resolution = (width, height)
    except ValueError:
        print(f"✗ Formato de resolución inválido: {args.resolution}")
        print("Use: --resolution 640x480")
        sys.exit(1)

    # Crear y ejecutar monitor
    monitor = ConcentrationMonitor(
        resolution=resolution,
        fps_target=args.fps,
        interval_seconds=args.interval,
        use_tpu=not args.no_tpu,
    )
    monitor.run()


if __name__ == "__main__":
    main()
