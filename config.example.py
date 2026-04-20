#!/usr/bin/env python3
"""
Archivo de configuración de ejemplo para el Monitor de Concentración.
Copia este archivo a config.py y personaliza según tus necesidades.
"""

# ============================================================================
# CONFIGURACIÓN DE CAPTURA
# ============================================================================

# Resolución de captura (ancho, alto)
RESOLUTION = (640, 480)

# FPS objetivo de captura
FPS_TARGET = 30

# ============================================================================
# CONFIGURACIÓN DE DETECCIÓN
# ============================================================================

# Usar Coral TPU (True) o solo CPU (False)
USE_TPU = True

# Ruta al modelo MediaPipe
MODEL_PATH = "models/face_landmarker.task"

# ============================================================================
# CONFIGURACIÓN DE MÉTRICAS
# ============================================================================

# Umbral de confianza mínimo para detección facial (0.0 - 1.0)
MIN_FACE_DETECTION_CONFIDENCE = 0.5

# Tamaño de las deques (historial de muestras a 30fps)
EAR_DEQUE_SIZE = 900  # 30 segundos

MAR_DEQUE_SIZE = 900  # 30 segundos

# ============================================================================
# CONFIGURACIÓN DE SERIE TEMPORAL
# ============================================================================

# Número máximo de scores guardados (a 10s interval = 30 min de histórico)
TIMESERIES_WINDOW_SIZE = 180

# Puntos para calcular media móvil (18 = 3 minutos a 10s interval)
MOVING_AVG_POINTS = 18

# Umbrales de score para recomendaciones
CRITICAL_SCORE_THRESHOLD = 4.0  # Score < esto requiere descanso inmediato
WARNING_SCORE_THRESHOLD = 6.0   # Score entre esto y OPTIMAL requiere aviso
OPTIMAL_SCORE_THRESHOLD = 7.0   # Score >= esto es concentración óptima

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

# Intervalo en segundos para loguear scores en CSV
LOG_INTERVAL_SECONDS = 10

# Directorio para guardar logs
LOG_DIRECTORY = "logs"

# ============================================================================
# CONFIGURACIÓN DE DASHBOARD
# ============================================================================

# Dimensiones del canvas del dashboard
DASHBOARD_WIDTH = 1280
DASHBOARD_HEIGHT = 720

# Mostrar todos los landmarks (True) o solo los relevantes (False)
SHOW_ALL_LANDMARKS = False

# ============================================================================
# CONFIGURACIÓN AVANZADA
# ============================================================================

# Segundos sin detectar rostro antes de resetear deques
FACE_MISSING_TIMEOUT = 5.0

# Tamaño máximo del acumulador de FPS (para promediar)
FPS_ACCUMULATOR_SIZE = 30

# ============================================================================
# NOTAS DE USO
# ============================================================================

"""
Para usar este archivo de configuración en main.py:

1. Copia este archivo a config.py
2. En main.py, añade al inicio:
   from config import *

3. Luego usa las variables como:
   monitor = ConcentrationMonitor(
       resolution=RESOLUTION,
       fps_target=FPS_TARGET,
       interval_seconds=LOG_INTERVAL_SECONDS,
       use_tpu=USE_TPU
   )

Valores recomendados por caso de uso:

PARA MÁXIMO RENDIMIENTO (RPi 5 con TPU):
  RESOLUTION = (320, 240)
  FPS_TARGET = 30
  USE_TPU = True
  LOG_INTERVAL_SECONDS = 10

PARA BALANCE (RPi 5 estándar):
  RESOLUTION = (640, 480)
  FPS_TARGET = 30
  USE_TPU = True
  LOG_INTERVAL_SECONDS = 10

PARA MÍNIMO CONSUMO (RPi 4 o bajo consumo):
  RESOLUTION = (320, 240)
  FPS_TARGET = 15
  USE_TPU = False
  LOG_INTERVAL_SECONDS = 20

PARA MÁXIMA PRECISIÓN (procesamiento offline):
  RESOLUTION = (1280, 720)
  FPS_TARGET = 30
  LOG_INTERVAL_SECONDS = 5
"""
