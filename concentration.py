#!/usr/bin/env python3
"""
Cálculo de métricas de concentración basadas en landmarks faciales:
- EAR (Eye Aspect Ratio)
- Blink Rate
- Gaze Deviation
- Head Pose (Yaw)
- MAR (Mouth Aspect Ratio) + Yawn Detection
- Score Final (0-10)
"""

from typing import Optional
from collections import deque
import numpy as np
import cv2


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calcula la distancia euclidiana entre dos puntos 2D."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_eye_aspect_ratio(landmarks: list[tuple[float, float, float]]) -> tuple[float, float]:
    """
    Calcula el Eye Aspect Ratio (EAR) para ambos ojos.

    Fórmula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Donde p1,p4 son las esquinas horizontales y p2/p6, p3/p5 son pares verticales.

    Índices MediaPipe [p1, p2, p3, p4, p5, p6]:
    - Ojo derecho:   [33, 160, 158, 133, 153, 144]
    - Ojo izquierdo: [362, 385, 387, 263, 373, 380]

    Returns:
        Tupla (ear_izq, ear_dch) en rango ~0.1-0.4
    """
    def _ear(idx: list[int]) -> float:
        # p1=idx[0] y p4=idx[3]: esquinas horizontales (denominador)
        # p2=idx[1] y p6=idx[5]: par vertical 1 (numerador)
        # p3=idx[2] y p5=idx[4]: par vertical 2 (numerador)
        p1 = (landmarks[idx[0]][0], landmarks[idx[0]][1])
        p2 = (landmarks[idx[1]][0], landmarks[idx[1]][1])
        p3 = (landmarks[idx[2]][0], landmarks[idx[2]][1])
        p4 = (landmarks[idx[3]][0], landmarks[idx[3]][1])
        p5 = (landmarks[idx[4]][0], landmarks[idx[4]][1])
        p6 = (landmarks[idx[5]][0], landmarks[idx[5]][1])
        horizontal = euclidean_distance(p1, p4)
        return (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2.0 * horizontal) if horizontal > 0 else 0.0

    ear_right = _ear([33, 160, 158, 133, 153, 144])
    ear_left  = _ear([362, 385, 387, 263, 373, 380])

    return (ear_left, ear_right)


def compute_blink_rate(ear_deque: deque[float]) -> float:
    """
    Calcula el blink rate (parpadeos/minuto) desde un deque de valores EAR.

    Detecta cuántas veces EAR cruza el umbral 0.2 hacia abajo.

    Args:
        ear_deque: deque de 900 muestras (30s a 30fps)

    Returns:
        Parpadeos por minuto (0-120)
    """
    if len(ear_deque) < 30:
        return 0.0

    threshold = 0.2
    blinks = 0
    was_open = True

    for ear in ear_deque:
        is_open = ear > threshold
        if was_open and not is_open:  # Cruce hacia abajo
            blinks += 1
        was_open = is_open

    # Escalar a parpadeos/minuto
    seconds = len(ear_deque) / 30.0
    blink_rate = (blinks / seconds) * 60.0 if seconds > 0 else 0.0
    return min(blink_rate, 120.0)  # Clampear a máximo realista


def compute_gaze_deviation(landmarks: list[tuple[float, float, float]]) -> float:
    """
    Calcula la desviación de la mirada (gaze deviation) en grados.

    Usa los landmarks del iris de AMBOS ojos y promedia para mayor estabilidad.

    Índices iris: izq [468-472], dch [473-477]

    Returns:
        Ángulo en grados (-90 a +90, donde 0 = mirando al frente)
    """
    iris_left_idx = [468, 469, 470, 471, 472]
    iris_right_idx = [473, 474, 475, 476, 477]

    # Centro de cada iris
    iris_l_cx = np.mean([landmarks[i][0] for i in iris_left_idx])
    iris_r_cx = np.mean([landmarks[i][0] for i in iris_right_idx])

    # Bounding box horizontal de cada ojo
    left_eye_indices = [362, 385, 387, 263, 373, 380]
    right_eye_indices = [33, 160, 158, 133, 153, 144]

    left_eye_x = [landmarks[i][0] for i in left_eye_indices]
    right_eye_x = [landmarks[i][0] for i in right_eye_indices]

    left_bbox_min, left_bbox_max = min(left_eye_x), max(left_eye_x)
    right_bbox_min, right_bbox_max = min(right_eye_x), max(right_eye_x)

    # Posición normalizada del iris dentro del ojo (0=extremo izq, 1=extremo dch)
    iris_pos_left = (iris_l_cx - left_bbox_min) / (left_bbox_max - left_bbox_min + 1e-6)
    iris_pos_right = (iris_r_cx - right_bbox_min) / (right_bbox_max - right_bbox_min + 1e-6)

    # Promedio de ambos ojos y conversión a grados
    avg_pos = (iris_pos_left + iris_pos_right) / 2.0
    gaze_angle = (avg_pos - 0.5) * 90.0

    return np.clip(gaze_angle, -90, 90)


def compute_head_yaw(landmarks: list[tuple[float, float, float]]) -> float:
    """
    Estima el ángulo yaw de la cabeza (rotación horizontal) usando solvePnP.

    Usa puntos 3D de referencia estándar:
    - Nariz: índice 1
    - Mentón: índice 152
    - Pómulo izquierdo: índice 234
    - Pómulo derecho: índice 454

    Returns:
        Ángulo yaw en grados (-30 a +30 es rango normal)
    """
    try:
        # Puntos 3D de referencia (en mm, aproximados)
        object_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Nariz (origen)
                [0.0, -30.0, -30.0],  # Mentón
                [-30.0, -30.0, -30.0],  # Pómulo izquierdo
                [30.0, -30.0, -30.0],  # Pómulo derecho
            ],
            dtype=np.float32,
        )

        # Puntos 2D en la imagen: convertir landmarks normalizados [0,1] a píxeles (640x480)
        # MediaPipe devuelve coordenadas normalizadas, pero solvePnP necesita píxeles
        W, H = 640.0, 480.0
        image_points = np.array(
            [
                [landmarks[1][0] * W, landmarks[1][1] * H],  # Nariz
                [landmarks[152][0] * W, landmarks[152][1] * H],  # Mentón
                [landmarks[234][0] * W, landmarks[234][1] * H],  # Pómulo izquierdo
                [landmarks[454][0] * W, landmarks[454][1] * H],  # Pómulo derecho
            ],
            dtype=np.float32,
        )

        # Parámetros intrínsecos de la cámara (aproximados para 640x480)
        focal_length = 500.0
        center = (W / 2, H / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1))

        # Resolver PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs
        )

        if success:
            # Convertir vector de rotación a matriz de rotación
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            # Extraer ángulo yaw (rotación alrededor del eje Y)
            yaw = np.arctan2(rotation_mat[2, 0], rotation_mat[0, 0]) * 180 / np.pi
            return np.clip(yaw, -90, 90)
        else:
            return 0.0

    except Exception as e:
        print(f"Error computando head yaw: {e}")
        return 0.0


def compute_mouth_aspect_ratio(landmarks: list[tuple[float, float, float]]) -> float:
    """
    Calcula el Mouth Aspect Ratio (MAR) para detectar bostezos.

    Índices boca: [13, 14, 78, 308, 82, 312]

    Returns:
        MAR normalizado (0.1-0.8 es rango típico)
    """
    mouth_indices = [13, 14, 78, 308, 82, 312]
    p1 = (landmarks[mouth_indices[0]][0], landmarks[mouth_indices[0]][1])
    p2 = (landmarks[mouth_indices[1]][0], landmarks[mouth_indices[1]][1])
    p3 = (landmarks[mouth_indices[2]][0], landmarks[mouth_indices[2]][1])
    p4 = (landmarks[mouth_indices[3]][0], landmarks[mouth_indices[3]][1])
    p5 = (landmarks[mouth_indices[4]][0], landmarks[mouth_indices[4]][1])
    p6 = (landmarks[mouth_indices[5]][0], landmarks[mouth_indices[5]][1])

    dist1 = euclidean_distance(p1, p4)
    dist2 = euclidean_distance(p2, p5)
    dist3 = euclidean_distance(p3, p6)

    mar = (dist2 + dist3) / (2.0 * dist1) if dist1 > 0 else 0.0
    return np.clip(mar, 0, 1.0)


def detect_yawn(mar_deque: deque[float], threshold: float = 0.72, duration_frames: int = 90) -> bool:
    """
    Detecta si hay bostezo sostenido (MAR > threshold durante duration_frames).

    Args:
        mar_deque: deque de valores MAR (30fps)
        threshold: umbral MAR para bostezo (0.72 evita falsos positivos al mirar abajo)
        duration_frames: fotogramas para confirmar bostezo (90 = 3s a 30fps)

    Returns:
        True si se detecta bostezo sostenido
    """
    if len(mar_deque) < duration_frames:
        return False

    recent_mars = list(mar_deque)[-duration_frames:]
    yawn_count = sum(1 for mar in recent_mars if mar > threshold)
    return yawn_count > (duration_frames * 0.75)  # 75% de frames con MAR alto


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normaliza un valor al rango [0, 1].

    Args:
        value: valor a normalizar
        min_val: valor mínimo esperado
        max_val: valor máximo esperado

    Returns:
        Valor normalizado y clampado en [0, 1]
    """
    normalized = (value - min_val) / (max_val - min_val + 1e-6)
    return np.clip(normalized, 0, 1)


def compute_concentration_score(
    ear: float,
    gaze_deviation: float,
    head_yaw: float,
    blink_rate: float,
    yawn_detected: bool,
) -> float:
    """
    Calcula el score final de concentración (0-10).

    Pesos:
    - Apertura de ojos (EAR): 30%
    - Desviación de mirada: 35%
    - Pose de cabeza: 25%
    - Frecuencia de parpadeo: 10%
    - Penalización por bostezo: -1.5

    Args:
        ear: Eye Aspect Ratio (0.15-0.35 es rango normal)
        gaze_deviation: Desviación de mirada en grados (-30 a +30 es normal)
        head_yaw: Ángulo yaw de cabeza en grados (-35 a +35 es normal)
        blink_rate: Parpadeos por minuto (8-20 es rango normal)
        yawn_detected: Si hay bostezo detectado

    Returns:
        Score de concentración 0-10
    """
    # Puntuaciones por métrica (0-1)

    # EAR: mínimo bajo (0.05) porque al mirar hacia abajo el párpado baja
    # fisiológicamente — no es somnolencia, es postura de lectura normal.
    # El EAR solo penaliza ojos casi cerrados (<0.07) o adormilados.
    eye_score = normalize_value(ear, 0.05, 0.30)

    # Gaze: zona muerta de ±8° → degrada hasta ±38°
    gaze_adj = max(0.0, abs(gaze_deviation) - 8.0)
    gaze_score = 1.0 - normalize_value(gaze_adj, 0, 30)

    # Head yaw: zona muerta de ±8° → degrada hasta ±43°
    yaw_adj = max(0.0, abs(head_yaw) - 8.0)
    head_score = 1.0 - normalize_value(yaw_adj, 0, 35)

    # Blink: rango (4–25/min) para no penalizar concentración intensa
    blink_score = normalize_value(blink_rate, 4, 25)

    # Pesos: eye reducido (párpado varía con la postura), gaze y head son más fiables
    raw_score = (0.15 * eye_score + 0.45 * gaze_score + 0.30 * head_score + 0.10 * blink_score)

    # Penalización por bostezo
    yawn_penalty = 2.5 if yawn_detected else 0

    # Escalar a 0-10 y aplicar penalización
    final_score = (raw_score * 10) - yawn_penalty
    return float(np.clip(final_score, 0, 10))
