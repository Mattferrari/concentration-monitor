# Guía de Testing - Concentration Monitor

Instrucciones para probar el sistema sin RPi o cámara Raspberry Pi.

## 🖥️ Testing en PC/Mac con Webcam

### Requisitos
- Python 3.9+
- Webcam o cámara USB
- Entorno virtual (venv)

### Instalación rápida

```bash
# Crear entorno
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo
python download_model.py
```

### Ejecución

```bash
# Con CPU (recomendado para testing)
python main.py --no-tpu

# Alternativas
python main.py --no-tpu --resolution 320x240 --fps 15
```

## 🧪 Testing sin Cámara

Para testing sin cámara disponible, podemos usar un mock:

### Crear un test con vídeo pre-grabado

1. Prepara un video MP4 con rostros:
```bash
# Guardar como test_video.mp4
```

2. Modifica `capture.py` temporalmente:
```python
# En lugar de picamera2, usa OpenCV:
def get_frame(self) -> Optional[np.ndarray]:
    if self.camera is None:
        return None
    ret, frame = cv2.VideoCapture("test_video.mp4").read()
    return frame if ret else None
```

3. Ejecuta:
```bash
python main.py --no-tpu
```

## ✅ Checklist de Testing

### Fase 1: Componentes Individuales

- [ ] `download_model.py` descarga sin errores
- [ ] Modelo existe en `models/face_landmarker.task`
- [ ] `capture.py` obtiene frames correctamente
- [ ] Rostro detectado en la pantalla

### Fase 2: Métricas

- [ ] EAR se calcula correctamente (0.2-0.35 rango normal)
- [ ] Blink rate se actualiza (12-20 bpm es normal)
- [ ] Gaze deviation cambia al mirar en diferentes direcciones
- [ ] Head yaw cambia al girar la cabeza

### Fase 3: Series Temporales

- [ ] Score oscila entre 0-10
- [ ] Media móvil se actualiza cada 10s
- [ ] Recomendaciones aparecen según umbrales

### Fase 4: Dashboard

- [ ] Ventana OpenCV se abre
- [ ] Frame se visualiza correctamente
- [ ] Score se muestra con color (rojo/amarillo/verde)
- [ ] Gráfica temporal se dibuja
- [ ] FPS aparece en esquina superior izquierda

### Fase 5: Logging

- [ ] Se crea archivo en `logs/session_*.csv`
- [ ] CSV contiene encabezados correctos
- [ ] Datos se guardan cada 10 segundos
- [ ] Análisis funciona: `python analyze_logs.py`

## 🐛 Tests Específicos

### Test 1: Detección de Ojos Cerrados
**Propósito:** Verificar que EAR < 0.2 reduce score

**Procedimiento:**
1. Abre ojos normalmente → score debe ser ~8+
2. Cierra ojos → score debe bajar a <5
3. Abre nuevamente → score debe recuperarse

**Resultado esperado:** Score rojo al cerrar ojos

### Test 2: Detección de Desviación de Mirada
**Propósito:** Verificar gaze deviation impacta score

**Procedimiento:**
1. Mira al frente → score máximo
2. Mira hacia la izquierda → score disminuye
3. Mira hacia la derecha → score disminuye

**Resultado esperado:** Score oscila 7-10 (frente) a 4-6 (desviado)

### Test 3: Detección de Rotación de Cabeza
**Propósito:** Verificar head yaw impacta score

**Procedimiento:**
1. Cabeza recta → score máximo
2. Gira cabeza 30° izquierda → score baja
3. Gira cabeza 30° derecha → score baja

**Resultado esperado:** Score 8+ (frente) a 5-6 (girada)

### Test 4: Detección de Bostezos
**Propósito:** Verificar MAR > 0.6 penaliza score

**Procedimiento:**
1. Bostezo simulado (abrir mucho la boca)
2. Mantén por 2+ segundos
3. Observa si score disminuye extra

**Resultado esperado:** Score baja -1.5 puntos durante bostezo

### Test 5: Parpadeo Normal
**Propósito:** Verificar blink rate se mantiene 12-20 bpm

**Procedimiento:**
1. Parpadea naturalmente durante 30 segundos
2. Observa blink rate en logs

**Resultado esperado:** Blink rate entre 12-20 bpm

### Test 6: Series Temporales y Recomendaciones
**Propósito:** Verificar recomendaciones basadas en histórico

**Procedimiento:**
1. Mantén los ojos muy cerrados por 2 minutos
2. Observa recomendaciones en footer

**Resultado esperado:**
- Después de 2 min: "¡DESCANSA AHORA!"
- Después de normalizarse: "Concentración óptima"

### Test 7: CSV Logging
**Propósito:** Verificar datos se guardan correctamente

**Procedimiento:**
1. Ejecuta `python main.py --no-tpu` por 2 minutos
2. Presiona 'q' para salir
3. Lee logs: `tail -n 10 logs/session_*.csv`

**Resultado esperado:**
```
timestamp,score,ear,gaze_dev,head_yaw,blink_rate,recommendation
2024-01-15 10:30:00,7.50,0.28,5.2,-8.3,16.0,
```

## 📊 Análisis de Logs

```bash
# Ver sesiones disponibles
python analyze_logs.py

# Analizar sesión específica
python analyze_logs.py logs/session_20240115_1030.csv

# Resultado esperado:
# SCORE DE CONCENTRACIÓN (0-10)
#   Promedio: 6.45
#   Rango: 3.2 - 9.8
#   Distribución:
#     🔴 Bajo (0-4):    5 (8.3%)
#     🟡 Medio (4-7):  35 (58.3%)
#     🟢 Alto (7-10):  20 (33.3%)
```

## 🔧 Troubleshooting de Testing

### Error: "No module named 'mediapipe'"
```bash
pip install mediapipe
```

### Error: "Cannot connect to display"
**En sistemas sin GUI (headless):**
```bash
# Desactivar visualización en dashboard.py temporalmente
# O usar X11 forwarding si estás en SSH
```

### Error: "Webcam no detectada"
```bash
# Verificar acceso a cámara
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### FPS muy bajo (< 10)
- Reduce resolución: `--resolution 320x240`
- Usa `--no-tpu`
- Verifica CPU usage: `top` o Task Manager

## 📈 Casos de Test Avanzados

### Test: Rendimiento bajo carga
```bash
# Abrir múltiples ventanas u aplicaciones pesadas
# Ejecutar monitor
python main.py --no-tpu

# Observar FPS (debería ser > 10 incluso bajo carga)
```

### Test: Memoria
```bash
# En otra terminal, monitorear:
watch 'free -h'  # Linux/Mac
```

### Test: Larga duración (1 hora)
```bash
# Ejecutar sin interrupción durante 1 hora
# Verificar:
# 1. No hay memory leaks (memoria crece?)
# 2. FPS se mantiene estable
# 3. CSV sigue escribiéndose correctamente
```

## 🎓 Tests Educativos

### Demostraciones en aula

**Test 1: Efecto del sueño**
- Estudiante normal: score ~7-8
- Después de noche sin dormir: score ~4-5

**Test 2: Efecto del estrés**
- Antes de examen importante: score baja a 4-5
- Después de descanso: score sube a 8-9

**Test 3: Efecto de distracciones**
- Estudiante enfocado: score ~8
- Sonido de notificación: score ~5 (mirada se desvía)
- Recuperación: score ~8 después 30 segundos

---

## ✅ Validación Final

Antes de desplegar en RPi:

- [ ] Todos los tests pasan en PC
- [ ] Logs se generan correctamente
- [ ] Análisis de logs funciona
- [ ] Dashboard se ve bien
- [ ] FPS > 15 en tu máquina

## 📝 Reporte de Testing

Plantilla para documentar testing:

```
Fecha: 2024-01-15
Sistema: [Windows/Mac/Linux]
Python: 3.9+
Cámara: [Webcam USB modelo]

Componentes probados:
- [x] Modelo descargado
- [x] Captura de frames
- [x] Detección facial
- [x] Cálculo de EAR
- [x] Dashboard visual
- [x] Logging a CSV

Problemas encontrados:
1. FPS bajo en resolución 1280x720
   Solución: Reducir a 640x480

Notas:
- Sistema funciona correctamente
- Listo para desplegar en RPi
```

---

Última actualización: 2024-01-15
