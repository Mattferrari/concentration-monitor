# Sistema de Detección de Concentración Estudiantil para Raspberry Pi 5

Sistema de visión por computador en tiempo real para monitorear concentración basado en análisis facial con MediaPipe y aceleración Coral TPU.

## 🎯 Características

- **5 Métricas de Concentración:**
  - Eye Aspect Ratio (EAR) - apertura de ojos
  - Blink Rate - frecuencia de parpadeo
  - Gaze Deviation - dirección de mirada
  - Head Pose (Yaw) - rotación horizontal de cabeza
  - Mouth Aspect Ratio (MAR) - detección de bostezos

- **Score Integrado (0-10):** Combinación ponderada de todas las métricas

- **Recomendaciones Automáticas:** Alertas de descanso basadas en series temporales

- **Dashboard en Vivo:** Visualización OpenCV con gráficas

- **Logging a CSV:** Histórico completo para análisis posterior

- **Aceleración Coral TPU:** Soporte automático con fallback a CPU

## 📋 Requisitos Hardware

- **Raspberry Pi 5** (recomendado) o RPi 4 (más lento)
- **Raspberry Pi Camera Module** (picamera2) o cámara USB
- **Coral TPU** (opcional, vía USB o M.2):
  - USB: Coral TPU USB Accelerator
  - M.2: Coral TPU PCIe
- **Fuente de poder:** 5V / 5A mínimo

## 🚀 Instalación en Raspberry Pi 5

### Paso 1: Actualizar el sistema
```bash
sudo apt update && sudo apt upgrade -y
```

### Paso 2: Instalar dependencias del sistema
```bash
# MediaPipe necesita TensorFlow Lite y librerías de desarrollo
sudo apt install -y \
  python3.11 \
  python3.11-dev \
  python3-pip \
  libatlas-base-dev \
  libjasper-dev \
  libtiff5 \
  libharfbuzz0b \
  libwebp6 \
  libjasper1 \
  libopenjp2-7 \
  libopenjp2-7-dev

# Para picamera2 (solo Raspberry Pi)
sudo apt install -y \
  libcamera0 \
  libcamera-apps \
  libcamera-tools \
  python3-libcamera \
  python3-kms++

# Para OpenCV (si es necesario compilar)
sudo apt install -y \
  build-essential \
  cmake \
  gfortran \
  git \
  graphicsmagick \
  libblas-dev \
  libbluray-dev \
  libharfbuzz0b \
  libjasper-dev \
  libjpeg-dev \
  liblapack-dev \
  libopenexr-dev \
  libpng-dev \
  libssl-dev \
  libtiff-dev \
  libwebp-dev \
  pkg-config
```

### Paso 3: Descargar el proyecto
```bash
cd ~/
git clone <repo-url> concentration_monitor
cd concentration_monitor
```

### Paso 4: Crear entorno virtual (recomendado)
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Paso 5: Instalar dependencias Python
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Alternativamente, para RPi con pre-built wheels:
pip install --index-url https://google-coral.github.io/py-repo/ \
  tflite-runtime==2.13.0
```

### Paso 6: Descargar el modelo MediaPipe
```bash
python download_model.py
```

Debería ver:
```
📥 Descargando modelo desde https://storage.googleapis.com/mediapipe-models/...
✓ Modelo descargado exitosamente en models/face_landmarker.task
```

### Paso 7 (Opcional): Configurar Coral TPU

#### Si usas USB TPU:
```bash
# Agregar usuario al grupo usb
sudo usermod -a -G plugdev $USER
newgrp plugdev

# Verificar que sea detectado
lsusb | grep Google
# Debería mostrar: "Google Inc. Coral TPU"
```

#### Si usas M.2 TPU:
```bash
# Verificar instalación
lspci | grep Coral
# Debería mostrar el dispositivo
```

#### Instalar soporte Coral (opcional pero recomendado):
```bash
# Nota: Los wheels pre-compilados están en la URL de arriba
```

## 🎬 Ejecución

### Uso básico (con TPU si está disponible):
```bash
python main.py
```

### Opciones de línea de comandos:
```bash
# Ver ayuda
python main.py --help

# Forzar CPU (sin TPU)
python main.py --no-tpu

# Cambiar resolución
python main.py --resolution 1280x720

# Cambiar FPS
python main.py --fps 15

# Cambiar intervalo de logging (segundos)
python main.py --interval 5

# Combinado
python main.py --no-tpu --resolution 320x240 --interval 20
```

### Controles en la ventana:
- **q:** Salir y cerrar aplicación
- **Ctrl+C:** Salir de emergencia

## 📊 Interpretación del Dashboard

### Score de Concentración (esquina inferior derecha):
- **🔴 Rojo (0-4):** Concentración baja - DESCANSA AHORA
- **🟡 Amarillo (4-7):** Concentración media - vigilar
- **🟢 Verde (7-10):** Concentración óptima - mantener

### Gráfica Temporal (esquina inferior izquierda):
- **Línea verde:** Scores individuales de los últimos 5 minutos
- **Línea amarilla:** Media móvil (3 minutos)

### Landmarks Visualizados (en el frame):
- **Círculos blancos:** Ojos y puntos de referencia facial
- **Círculos verdes:** Puntos de la boca (para detección de bostezos)
- **Círculos amarillos:** Iris (para gaze deviation)

### Recomendaciones (footer):
- ✓ Concentración óptima
- ⚠️ Descansa en 10 minutos
- 🚨 ¡DESCANSA AHORA!

## 📝 Archivos de Log

Los logs se guardan en `logs/` con formato:
```
logs/session_YYYYMMDD_HHMM.csv
```

### Columnas del CSV:
| Campo | Descripción |
|-------|-------------|
| timestamp | Hora del evento (YYYY-MM-DD HH:MM:SS) |
| score | Score de concentración (0-10) |
| ear | Eye Aspect Ratio (0-0.5) |
| gaze_dev | Desviación de mirada (grados) |
| head_yaw | Rotación de cabeza (grados) |
| blink_rate | Parpadeos por minuto |
| recommendation | Recomendación actual |

### Análisis posterior:
```bash
# Ver últimas 10 muestras
tail -10 logs/session_*.csv

# Calcular promedio de score
tail -n +2 logs/session_*.csv | \
  awk -F',' '{sum+=$2; count++} END {print "Promedio:", sum/count}'
```

## 🔧 Estructura del Código

```
concentration_monitor/
├── main.py                 # Orquestador principal
├── capture.py              # Captura de video
├── landmark_detector.py    # MediaPipe Face Landmarker
├── concentration.py        # Cálculo de métricas
├── time_series.py          # Series temporales
├── dashboard.py            # Visualización
├── download_model.py       # Descarga del modelo
├── requirements.txt        # Dependencias Python
├── README.md              # Este archivo
├── models/                # Directorio de modelos
│   └── face_landmarker.task
└── logs/                  # Directorio de logs
    └── session_*.csv
```

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'mediapipe'"
```bash
pip install mediapipe --upgrade
```

### "No se detecta la cámara"
```bash
# Verificar si la cámara está conectada
vcgencmd get_camera

# Habilitar cámara (si es RPi)
sudo raspi-config
# Interface Options → Camera → Enable
```

### "Face Landmarker: modelo no encontrado"
```bash
python download_model.py
```

### "Coral TPU no detectada"
- Verificar conexión USB/M.2
- Para USB: `lsusb | grep Google`
- El sistema funcionará en CPU automáticamente

### "FPS muy bajo (< 10)"
- Usar `--resolution 320x240` para reducir carga
- Usar `--no-tpu` para descartar problemas de aceleración
- Verificar temperatura de RPi: `vcgencmd measure_temp`

### "Memoria insuficiente"
- Reducir resolución
- Usar `--interval 20` para reducir logging
- Monitorear: `free -h`

## 📈 Optimización

### Para máximo rendimiento:
```bash
# CPU Governor a performance
echo performance | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Ejecutar con max performance
python main.py --resolution 320x240 --fps 20
```

### Para mínimo consumo de potencia:
```bash
python main.py --no-tpu --resolution 320x240 --fps 15 --interval 30
```

## 📚 Referencias

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [Coral TPU Documentation](https://coral.ai/)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [OpenCV Python API](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

## 📄 Licencia

MIT License - Ver LICENSE file

## 🤝 Soporte

Para problemas o sugerencias, abrir un issue en el repositorio.

---

**Nota:** Este sistema está diseñado para uso educativo y de investigación. Para aplicaciones en entornos de aula reales, considerar privacidad y aspectos éticos del monitoreo biométrico.
