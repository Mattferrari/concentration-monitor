# ⚡ Inicio Rápido - Concentration Monitor

Empieza en 5 minutos.

## En Raspberry Pi 5

### 1️⃣ Instalación (una sola vez)
```bash
cd ~/concentration_monitor
bash install_rpi.sh
```

### 2️⃣ Activar entorno
```bash
source venv/bin/activate
```

### 3️⃣ Ejecutar
```bash
python main.py
```

**¡Listo!** Deberías ver:
- ✅ Una ventana con tu rostro
- ✅ Un score 0-10 en rojo/amarillo/verde
- ✅ Una gráfica en vivo
- ✅ Recomendaciones

### 🛑 Salir
Presiona **'q'** o **Ctrl+C**

---

## En PC/Mac para Testing

### 1️⃣ Entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python download_model.py
```

### 2️⃣ Ejecutar
```bash
python main.py --no-tpu
```

---

## 📊 Opciones Útiles

```bash
# Solo CPU (sin Coral TPU)
python main.py --no-tpu

# Resolución más baja (más rápido)
python main.py --resolution 320x240

# Menos FPS (menos carga)
python main.py --fps 15

# Logging más rápido
python main.py --interval 5

# Combinar
python main.py --no-tpu --resolution 320x240 --interval 20
```

---

## 📁 Archivos Importantes

| Archivo | Qué hace |
|---------|----------|
| `main.py` | Ejecutable principal |
| `logs/` | Archivos CSV con datos |
| `models/` | Modelo MediaPipe (descargado automáticamente) |
| `analyze_logs.py` | Analizar datos guardados |

---

## 🔴 Color del Score

- **Rojo (0-4):** ¡Descansa AHORA!
- **Amarillo (4-7):** Concentración moderada
- **Verde (7-10):** Excelente concentración ✓

---

## 💾 Ver Datos Guardados

```bash
# Listar sesiones
python analyze_logs.py

# Analizar una sesión
python analyze_logs.py logs/session_20240115_1030.csv
```

---

## ⚙️ Primeras Ejecuciones

**Primera ejecución:**
- Puede ser lenta mientras carga el modelo
- Después será rápido

**Si dice "Face not detected":**
- Acerca más la cara a la cámara
- Mejora la iluminación
- Espera a que encuentre tu rostro

**Si FPS es muy bajo:**
- Usa `--no-tpu --resolution 320x240`
- O desconecta otros programas

---

## 📚 Documentación Completa

- **README.md** - Información completa
- **TESTING.md** - Cómo probar sin RPi
- **config.example.py** - Personalización avanzada

---

## 🆘 Problemas Comunes

**"Modelo no encontrado"**
```bash
python download_model.py
```

**"Cámara no encontrada"**
```bash
# En RPi
vcgencmd get_camera

# Verificar USB
lsusb | grep -i camera
```

**"ModuleNotFoundError: mediapipe"**
```bash
pip install mediapipe --upgrade
```

---

¡Ya está! Empieza con `python main.py` 🚀
