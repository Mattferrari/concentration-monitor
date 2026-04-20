#!/usr/bin/env python3
"""
Dashboard visual para el sistema de detección de concentración.
Visualiza frame, landmarks, score, gráfica temporal y recomendaciones.
"""

from typing import Optional, List
import numpy as np
import cv2


class ConcentrationDashboard:
    """Renderiza visualización en vivo del sistema de concentración."""

    def __init__(self, canvas_width: int = 1280, canvas_height: int = 720):
        """
        Inicializa el dashboard.

        Args:
            canvas_width: Ancho del canvas en píxeles
            canvas_height: Alto del canvas en píxeles
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Colores BGR
        self.color_red = (0, 0, 255)
        self.color_green = (0, 255, 0)
        self.color_yellow = (0, 255, 255)
        self.color_white = (255, 255, 255)
        self.color_black = (0, 0, 0)
        self.color_gray = (128, 128, 128)

    def _draw_landmarks(
        self, frame: np.ndarray, landmarks: List[tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Dibuja landmarks relevantes sobre el frame.

        Solo dibuja ojos, nariz, mentón, boca e iris (no los 478).

        Args:
            frame: Frame original
            landmarks: 478 landmarks normalizados

        Returns:
            Frame con landmarks dibujados
        """
        h, w = frame.shape[:2]

        # Índices a dibujar
        relevant_indices = {
            "ojos_dch": [33, 133, 160, 158],
            "ojos_izq": [362, 263, 385, 387],
            "nariz": [1, 4],
            "menton": [152],
            "boca": [13, 14, 78, 308, 82, 312],
            "iris_izq": [468, 469, 470, 471, 472],
            "iris_dch": [473, 474, 475, 476, 477],
        }

        # Dibujar puntos
        for group, indices in relevant_indices.items():
            if "iris" in group:
                color = self.color_yellow
                radius = 3
            elif "boca" in group:
                color = self.color_green
                radius = 2
            else:
                color = self.color_white
                radius = 2

            for idx in indices:
                if idx < len(landmarks):
                    x = int(landmarks[idx][0] * w)
                    y = int(landmarks[idx][1] * h)
                    cv2.circle(frame, (x, y), radius, color, -1)

        return frame

    def _score_to_color(self, score: float) -> tuple[int, int, int]:
        """
        Mapea un score de concentración a color BGR.

        Args:
            score: Score 0-10

        Returns:
            Tupla color BGR
        """
        if score < 4.0:
            return self.color_red  # Rojo: bajo
        elif score < 7.0:
            return self.color_yellow  # Amarillo: medio
        else:
            return self.color_green  # Verde: alto

    def _draw_score_box(self, canvas: np.ndarray, score: float) -> np.ndarray:
        """
        Dibuja el score de concentración grande en la esquina inferior derecha.

        Args:
            canvas: Canvas del dashboard
            score: Score 0-10

        Returns:
            Canvas modificado
        """
        if score is None:
            score_text = "N/A"
            color = self.color_gray
        else:
            score_text = f"{score:.1f}"
            color = self._score_to_color(score)

        # Caja de fondo
        box_width = 150
        box_height = 100
        box_x = self.canvas_width - box_width - 20
        box_y = self.canvas_height - box_height - 20

        # Rectángulo de fondo semi-transparente
        overlay = canvas.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # Borde de color
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_width, box_y + box_height), color, 3)

        # Texto del score
        cv2.putText(
            canvas,
            "Score",
            (box_x + 15, box_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.color_white,
            1,
        )
        cv2.putText(
            canvas,
            score_text,
            (box_x + 20, box_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            color,
            2,
        )

        return canvas

    def _draw_graph(
        self, canvas: np.ndarray, scores_history: List[float], moving_avg: float
    ) -> np.ndarray:
        """
        Dibuja gráfica de series temporales (últimos 5 minutos).

        Args:
            canvas: Canvas del dashboard
            scores_history: Lista de scores recientes
            moving_avg: Media móvil actual

        Returns:
            Canvas modificado
        """
        if len(scores_history) == 0:
            return canvas

        # Área de gráfica: esquina inferior izquierda
        graph_width = 400
        graph_height = 150
        graph_x = 20
        graph_y = self.canvas_height - graph_height - 20

        # Fondo semi-transparente
        overlay = canvas.copy()
        cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        # Borde
        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), self.color_white, 1)

        # Escalar puntos al área de gráfica
        if len(scores_history) > 1:
            x_step = graph_width / (len(scores_history) - 1)
            points = []
            for i, score in enumerate(scores_history):
                x = graph_x + int(i * x_step)
                # Invertir Y (puntuación arriba = más arriba en pantalla)
                y = graph_y + graph_height - int((score / 10.0) * graph_height)
                points.append((x, y))

            # Dibujar línea de scores
            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i], self.color_green, 2)

            # Dibujar puntos
            for point in points:
                cv2.circle(canvas, point, 3, self.color_green, -1)

            # Dibujar línea de media móvil
            ma_y = graph_y + graph_height - int((moving_avg / 10.0) * graph_height)
            cv2.line(canvas, (graph_x, ma_y), (graph_x + graph_width, ma_y), self.color_yellow, 2)

        # Label
        cv2.putText(
            canvas,
            "Score (5 min)",
            (graph_x + 10, graph_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.color_white,
            1,
        )

        return canvas

    def _draw_fps(self, canvas: np.ndarray, fps: float) -> np.ndarray:
        """
        Dibuja FPS en la esquina superior izquierda.

        Args:
            canvas: Canvas del dashboard
            fps: Fotogramas por segundo

        Returns:
            Canvas modificado
        """
        cv2.putText(
            canvas,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.color_green,
            2,
        )
        return canvas

    def _draw_recommendation(self, canvas: np.ndarray, recommendation: str) -> np.ndarray:
        """
        Dibuja recomendación en el footer.

        Args:
            canvas: Canvas del dashboard
            recommendation: Texto de recomendación

        Returns:
            Canvas modificado
        """
        if not recommendation:
            return canvas

        # Área de recomendación (abajo, centrado)
        text_size = cv2.getTextSize(recommendation, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (self.canvas_width - text_size[0]) // 2
        text_y = self.canvas_height - 30

        # Fondo semi-transparente
        padding = 10
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (text_x - padding, text_y - text_size[1] - padding),
            (text_x + text_size[0] + padding, text_y + padding),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Texto
        cv2.putText(
            canvas,
            recommendation,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.color_white,
            2,
        )

        return canvas

    def render(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[tuple[float, float, float]]],
        score: Optional[float],
        moving_avg: float,
        scores_history: List[float],
        recommendation: str,
        fps: float,
    ) -> np.ndarray:
        """
        Renderiza el dashboard completo.

        Args:
            frame: Frame original de la cámara
            landmarks: 478 landmarks o None
            score: Score de concentración actual
            moving_avg: Media móvil
            scores_history: Histórico de últimos 5 minutos
            recommendation: Texto de recomendación
            fps: Fotogramas por segundo

        Returns:
            Canvas del dashboard listo para mostrar
        """
        # Canvas blanco base
        canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 240

        # Redimensionar y colocar frame original en esquina superior izquierda
        frame_height = 360
        frame_width = int((frame.shape[1] / frame.shape[0]) * frame_height)
        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        canvas[20 : 20 + frame_height, 20 : 20 + frame_width] = frame_resized

        # Dibujar landmarks si existen
        if landmarks is not None:
            frame_resized = self._draw_landmarks(frame_resized, landmarks)
            canvas[20 : 20 + frame_height, 20 : 20 + frame_width] = frame_resized

        # Dibujar componentes
        canvas = self._draw_fps(canvas, fps)
        canvas = self._draw_score_box(canvas, score)
        canvas = self._draw_graph(canvas, scores_history, moving_avg)
        canvas = self._draw_recommendation(canvas, recommendation)

        return canvas
