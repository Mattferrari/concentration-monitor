#!/usr/bin/env python3
"""
Gestión de series temporales de scores de concentración.
Calcula media móvil y genera recomendaciones de descanso.
"""

from collections import deque
from typing import Optional


class ConcentrationTimeSeries:
    """Mantiene histórico de scores y genera recomendaciones de descanso."""

    def __init__(
        self,
        window_size: int = 180,
        moving_avg_points: int = 18,
        critical_threshold: float = 4.0,
        warning_threshold: float = 6.0,
        optimal_threshold: float = 7.0,
    ):
        """
        Inicializa la serie temporal.

        Args:
            window_size: Número máximo de scores (180 = 30 min a 10s interval)
            moving_avg_points: Puntos para media móvil (18 = 3 min a 10s interval)
            critical_threshold: Score < esto = recomendación CRÍTICA
            warning_threshold: Score entre esto y optimal = AVISO
            optimal_threshold: Score > esto = ÓPTIMO
        """
        self.scores_deque = deque(maxlen=window_size)
        self.moving_avg_points = moving_avg_points
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.optimal_threshold = optimal_threshold
        self.critical_counter = 0  # Conteo de muestras bajo umbral crítico
        self.warning_counter = 0  # Conteo de muestras en rango de aviso

    def add_score(self, score: Optional[float]) -> None:
        """
        Añade un nuevo score a la serie temporal.

        Args:
            score: Score de concentración (0-10) o None si no se detectó cara
        """
        if score is not None:
            self.scores_deque.append(score)

            # Actualizar contadores
            if score < self.critical_threshold:
                self.critical_counter += 1
                self.warning_counter = 0
            elif score < self.warning_threshold:
                self.critical_counter = 0
                self.warning_counter += 1
            else:
                self.critical_counter = 0
                self.warning_counter = 0

    def get_moving_average(self) -> float:
        """
        Calcula la media móvil de los últimos N puntos.

        Returns:
            Media móvil o 0 si hay menos puntos de los necesarios
        """
        if len(self.scores_deque) < self.moving_avg_points:
            return 0.0

        recent_scores = list(self.scores_deque)[-self.moving_avg_points :]
        return sum(recent_scores) / len(recent_scores)

    def get_recommendation(self) -> str:
        """
        Genera una recomendación basada en el histórico de scores.

        Lógica:
        - Si media < 4.0 durante 2+ min (12 muestras): "¡Descansa AHORA!"
        - Si media 4.0-6.0 durante 5+ min (30 muestras): "Descansa en 10 min"
        - Si media > 7.0: Sin recomendación

        Returns:
            String con recomendación o vacío si no hay
        """
        moving_avg = self.get_moving_average()

        # Crítico: 12 muestras = 2 min a 10s interval
        if self.critical_counter >= 12 and moving_avg < self.critical_threshold:
            return "🚨 ¡DESCANSA AHORA! Concentración muy baja (< 4.0)"

        # Aviso: 30 muestras = 5 min a 10s interval
        if (
            self.warning_counter >= 30
            and self.critical_threshold <= moving_avg < self.warning_threshold
        ):
            return "⚠️ Descansa en 10 minutos (concentración 4.0-6.0)"

        # Óptimo: sin recomendación
        if moving_avg >= self.optimal_threshold:
            return "✓ Concentración óptima"

        # Rango normal: sin recomendación
        return ""

    def get_score_history_last_n_minutes(self, minutes: int = 5) -> list[float]:
        """
        Devuelve el histórico de scores de los últimos N minutos.

        Args:
            minutes: Minutos atrás (por defecto 5 min)

        Returns:
            Lista de scores de los últimos N*6 muestras (a 10s interval)
        """
        samples = minutes * 6  # 6 muestras/minuto a 10s interval
        if len(self.scores_deque) == 0:
            return []

        return list(self.scores_deque)[-samples:] if len(self.scores_deque) >= samples else list(self.scores_deque)

    def get_stats(self) -> dict:
        """
        Calcula estadísticas del histórico.

        Returns:
            Dict con min, max, mean, current score
        """
        if len(self.scores_deque) == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "current": None, "count": 0}

        scores = list(self.scores_deque)
        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "current": scores[-1] if scores else None,
            "count": len(scores),
        }
