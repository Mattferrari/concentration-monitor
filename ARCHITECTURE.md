# Arquitectura del Sistema - Concentration Monitor

Guía técnica de la arquitectura de componentes.

## 🏗️ Diagrama General

```
┌─────────────────┐
│  Hardware I/O   │
│  Cámara RPi/USB │
└────────┬────────┘
         │ frames (640x480, 30fps)
         ▼
┌─────────────────────────────┐
│  capture.py                 │
│  FrameCapture               │
│  - picamera2 (RPi)          │
│  - OpenCV fallback (USB)    │
└────────┬────────────────────┘
         │ np.ndarray BGR
         ▼
┌──────────────────────────────┐
│  landmark_detector.py        │
│  FaceLandmarkDetector        │
│  - MediaPipe Face Landmarker │
│  - Coral TPU (if available)  │
│  - CPU fallback              │
└────────┬─────────────────────┘
         │ 478 landmarks [x,y,z] normalized [0,1]
         ▼
┌──────────────────────────────┐      ┌─────────────────┐
│  concentration.py            │◄─────►│  time_series.py │
│  Metric Calculators:         │      │ ConcentrationTS │
│  - compute_ear()             │      │ - Deque storage │
│  - compute_blink_rate()      │      │ - Moving avg    │
│  - compute_gaze_deviation()  │      │ - Recommend.    │
│  - compute_head_yaw()        │      └─────────────────┘
│  - compute_mouth_aspect()    │
│  - compute_concentration_    │      ┌──────────────────┐
│    score()                   │─────►│  dashboard.py    │
└──────────────────────────────┘      │ ConcentrationDash
                                       │ - Render frame   │
                                       │ - Draw landmarks │
                                       │ - Plot graph     │
                                       │ - Display score  │
                                       │ - Show recommend │
                                       └────────┬─────────┘
                                                 │ canvas
                                                 ▼
                                          ┌──────────────────┐
                                          │ OpenCV imshow()  │
                                          │ Display to user  │
                                          └──────────────────┘

┌──────────────────────────────┐
│  main.py                     │
│ ConcentrationMonitor         │
│ - Orchestrates all modules   │
│ - CSV logging                │
│ - Loop control               │
│ - Exception handling         │
└──────────────────────────────┘
         │
         ▼
    logs/session_*.csv
```

## 📦 Módulos Detallados

### 1. `capture.py` - Captura de Video

**Responsabilidad:** Obtener frames de la cámara en tiempo real

**Clase:** `FrameCapture`

```python
FrameCapture(resolution=(640, 480), fps=30)
  ├── _init_camera()          # Intenta picamera2, fallback OpenCV
  ├── get_frame()             # Obtiene frame individual
  ├── get_frame_generator()   # Generador infinito
  └── release()               # Limpia recursos
```

**Entrada:** Hardware (cámara)
**Salida:** `np.ndarray` BGR (h, w, 3)
**Fallback:** OpenCV USB si picamera2 no disponible

---

### 2. `landmark_detector.py` - Detección Facial

**Responsabilidad:** Detectar 478 landmarks faciales

**Clase:** `FaceLandmarkDetector`

```python
FaceLandmarkDetector(model_path, use_tpu=True)
  ├── _init_detector()        # Intenta TPU, fallback CPU
  ├── _init_detector_cpu()    # Inicializa en CPU
  ├── detect(frame)           # Devuelve landmarks o None
  └── release()               # Limpia recursos
```

**Entrada:** Frame BGR `np.ndarray`
**Salida:** `list[tuple[x, y, z]]` o `None`
  - 478 landmarks normalizados [0, 1]
  - Z: profundidad relativa

**Arquitectura MediaPipe:**
```
Face Detection (pequeño)
       ↓
Face Landmarks Detection (478 puntos)
       ↓
Salida normalizada [0, 1]
```

---

### 3. `concentration.py` - Cálculo de Métricas

**Responsabilidad:** Convertir landmarks a métricas de concentración

**Funciones:**

#### a) `compute_eye_aspect_ratio()`
```
Índices: [33, 160, 158, 133, 153, 144] (dch)
         [362, 385, 387, 263, 373, 380] (izq)
         
Fórmula: ||p2-p6|| + ||p3-p5||
         ─────────────────────────
         2 * ||p1-p4||

Rango: 0.2 (cerrado) a 0.35+ (abierto)
```

#### b) `compute_blink_rate()`
```
Entrada: deque de 900 EAR valores (30s a 30fps)
Lógica: Contar cruces de EAR < 0.2 (parpadeos)
Salida: Parpadeos/minuto (escalado)
Rango: 0-120 bpm
```

#### c) `compute_gaze_deviation()`
```
Índices iris: [468-472] (izq), [473-477] (dch)
Lógica: Centro del iris relativo al bbox del ojo
Salida: Ángulo en grados (-90 a +90)
Normal: -30 a +30 (mirando al frente)
```

#### d) `compute_head_yaw()`
```
Puntos 3D: Nariz(1), Mentón(152), Pómulos(234,454)
Método: OpenCV solvePnP
Salida: Ángulo yaw en grados
Normal: -30 a +30
```

#### e) `compute_mouth_aspect_ratio()`
```
Índices: [13, 14, 78, 308, 82, 312]
Fórmula: Igual a EAR pero para boca
Rango: 0.1-0.8
Bostezo: MAR > 0.6 durante 2s
```

#### f) `compute_concentration_score()`
```
Pesos ponderados:
  30% Eye Score (basado en EAR)
  35% Gaze Score (basado en gaze deviation)
  25% Head Score (basado en yaw)
  10% Blink Score (basado en blink rate)
  -1.5 Penalización por bostezo

Rango: 0-10
```

---

### 4. `time_series.py` - Series Temporales

**Responsabilidad:** Mantener histórico y generar recomendaciones

**Clase:** `ConcentrationTimeSeries`

```python
ConcentrationTimeSeries(
    window_size=180,           # 30 min a 10s
    moving_avg_points=18,      # 3 min media móvil
    critical_threshold=4.0,
    warning_threshold=6.0,
    optimal_threshold=7.0
)
  ├── add_score(score)
  ├── get_moving_average()
  ├── get_recommendation()
  ├── get_score_history_last_n_minutes(5)
  └── get_stats()
```

**Lógica de Recomendación:**

```
┌─────────────────────────────┐
│   Media móvil < 4.0         │
│   Por > 2 min (12 muestras) │
└──────────┬──────────────────┘
           │ SÍ
           ▼
  🚨 ¡DESCANSA AHORA!
  
┌─────────────────────────────┐
│   Media móvil 4.0-6.0       │
│   Por > 5 min (30 muestras) │
└──────────┬──────────────────┘
           │ SÍ
           ▼
  ⚠️ Descansa en 10 min
  
┌─────────────────────────────┐
│   Media móvil > 7.0         │
└──────────┬──────────────────┘
           │ SÍ
           ▼
  ✓ Concentración óptima
```

---

### 5. `dashboard.py` - Visualización

**Responsabilidad:** Renderizar interface visual

**Clase:** `ConcentrationDashboard`

```python
ConcentrationDashboard(canvas_width=1280, canvas_height=720)
  ├── _draw_landmarks()       # Dibuja puntos faciales
  ├── _score_to_color()       # Mapea score a color
  ├── _draw_score_box()       # Score grande
  ├── _draw_graph()           # Gráfica temporal
  ├── _draw_fps()             # FPS en esquina
  ├── _draw_recommendation()  # Recomendación footer
  └── render()                # Renderiza todo
```

**Layout:**
```
┌────────────────────────────────────────────────────────┐
│ FPS: 30.5                                              │
│                                                        │
│ ┌──────────────┐                    ┌─────────────┐  │
│ │              │                    │   Score     │  │
│ │   Frame      │                    │    7.5      │  │
│ │   original   │                    │  (verde)    │  │
│ │ + landmarks  │                    └─────────────┘  │
│ │              │                                     │
│ └──────────────┘                                     │
│                                                        │
│ ┌──────────────────┐                                 │
│ │  Gráfica Temporal│                                 │
│ │  (5 últimos min) │                                 │
│ │  ▁▂▃▄▅▄▃▂▁       │                                 │
│ └──────────────────┘                                 │
│                                                        │
│        ✓ Concentración óptima                         │
└────────────────────────────────────────────────────────┘
```

---

### 6. `main.py` - Orquestador

**Responsabilidad:** Integrar todos los módulos

**Clase:** `ConcentrationMonitor`

```
Loop Principal:
  while True:
    1. Capturar frame (capture.py)
    2. Detectar landmarks (landmark_detector.py)
    3. Calcular métricas (concentration.py)
    4. Actualizar serie temporal (time_series.py)
    5. Cada 10s: Loguear CSV
    6. Renderizar dashboard (dashboard.py)
    7. Mostrar en pantalla (OpenCV imshow)
    8. Procesar teclas (q para salir)
```

**Contadores y Estados:**

```python
frame_count              # Total de frames procesados
last_score_log          # Último timestamp de logging
face_missing_since      # Cuándo se perdió detección
fps_accumulator         # Deque para promediar FPS
ear_deque              # 30s de EAR (900 muestras)
mar_deque              # 30s de MAR (900 muestras)
```

**Excepciones:**

```
┌──────────────┐
│  Exceptions  │
├──────────────┤
│ face missing │─────► Resetear deques después 5s
│ (> 5s)       │
│              │
│ no capture   │─────► Cambiar a fallback
│              │
│ frame error  │─────► Log y continuar
│              │
│ CSV error    │─────► Log y continuar
│              │
│ KeyboardInt  │─────► Cleanup limpio
│ (Ctrl+C)     │
└──────────────┘
```

---

## 🔄 Flujo de Datos

```
Frame (BGR, 640x480, 30fps)
    ↓
Landmarks (478 puntos, [0,1])
    ├─► EAR ─────────────────┐
    ├─► Gaze Deviation ──────┤
    ├─► Head Yaw ───────────┤
    ├─► Blink Rate ────────┤ Score (0-10)
    └─► Yawn ──────────────┤
                            ↓
                    Series Temporal
                            ↓
                    Recomendación
                            ↓
                    Dashboard
                            ↓
                    OpenCV Display
                            +
                            CSV Log
```

---

## ⚡ Flujo de Control de FPS

```
Frame rate: 30fps → 33ms por frame máximo

Layout típico de timing:
  0ms  ┬─ Captura frame
       │
  5ms  ├─ Detección MediaPipe (TPU: 15-30ms, CPU: 50-100ms)
       │
  25ms ├─ Cálculo de métricas (< 1ms)
       │
  26ms ├─ Actualización serie temporal (< 1ms)
       │
  27ms ├─ Renderizado dashboard (1-3ms)
       │
  30ms └─ Mostrar en pantalla

Margen: ~3ms para overhead
```

---

## 💾 Formato CSV

```csv
timestamp,score,ear,gaze_dev,head_yaw,blink_rate,recommendation
2024-01-15 14:30:00,7.5,0.28,5.2,-8.3,16.0,
2024-01-15 14:30:10,7.2,0.27,6.1,-7.8,15.3,
2024-01-15 14:30:20,6.8,0.26,10.5,-6.5,17.2,⚠️ Descansa en 10 min
```

---

## 🐛 Manejo de Errores

```
Nivel 1: Try/Except en componentes individuales
  ├─ capture: silencioso, return None
  ├─ detector: silencioso, return None
  ├─ metrics: return 0, manejo de división por cero
  └─ dashboard: silencioso, skip render si error

Nivel 2: Try/Except en main loop
  ├─ Continuar si frame error
  ├─ Reset si face perdida > 5s
  └─ Log de excepciones

Nivel 3: Try/Finally en cleanup
  └─ Siempre liberar recursos
```

---

## 🔐 Recursos

```
Memory:
  Modelo MediaPipe: ~150-200 MB
  Deques (EAR+MAR): ~50 KB
  Canvas OpenCV: ~3 MB (1280x720x3)
  Total: ~200 MB típico

CPU:
  MediaPipe CPU: 40-60% en una core
  Metrics: < 1%
  Dashboard: 5-10%
  Total: 50-80% en una core

Tiempo de vida:
  Modelo: Se carga 1 vez al inicio
  Deques: Se resetean si face_missing > 5s
  Canvas: Se crea cada frame
```

---

## 📊 Decisiones de Diseño

### 1. ¿Por qué MediaPipe en lugar de dlib/OpenFace?
- **Mejor rendimiento** en RPi con TPU
- **478 landmarks vs 68** = más información
- **Optimizado para mobile**
- **Fácil integración con Coral TPU**

### 2. ¿Por qué deques en lugar de listas?
- **O(1) append y pop** vs O(n) si crecen
- **Tamaño fijo** = memoria predecible
- **FIFO automático** sin lógica adicional

### 3. ¿Por qué OpenCV en lugar de Flask?
- **Sin overhead de HTTP**
- **Rendimiento crítico en RPi**
- **Directa al frame buffer**

### 4. ¿Por qué CSV en lugar de base de datos?
- **Cero dependencias** (no SQLite, Postgres, etc.)
- **Análisis offline fácil** (Excel, pandas)
- **Portabilidad** entre sistemas

### 5. ¿Por qué ponderación 30-35-25-10?
- **Gaze (35%):** La mirada demuestra atención → más peso
- **Ojos (30%):** Pero ojos abiertos es fundamental
- **Cabeza (25%):** Postura es importante pero menos crítica
- **Parpadeo (10%):** Métrica secundaria/confirmadora

---

## 🚀 Optimizaciones Posibles

```
Mejora de rendimiento:
  - Procesar frames alternados (15fps internamente, 30fps display)
  - Downsampling de resolución intermedia
  - Usar threading para I/O (captura async)
  - Caché de modelos en memoria

Mejora de precisión:
  - Kalman filter para suavizar landmarks
  - Histograma de score en lugar de muestras individuales
  - Calibración per-usuario (baseline individual)
  - Detección de múltiples rostros

Mejora de UX:
  - Notificaciones de sonido (además de visual)
  - Guardado automático de screenshots en momentos críticos
  - API REST para integración
  - Base de datos histórica
```

---

**Última actualización:** 2024-01-15
**Versión:** 1.0
**Autor:** OpenAI/Claude
