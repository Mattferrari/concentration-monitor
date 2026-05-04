#!/usr/bin/env python3
"""
Calibración Personal — Concentration Monitor.

Objetivo: aprender los rangos de métricas ESPECÍFICOS de esta persona
          para que el modelo general se adapte a su fisonomía.

Por qué es necesario:
  - Cada persona tiene diferente forma de ojos (EAR baseline distinto).
  - Cada persona inclina la cabeza de forma natural diferente.
  - Normalizar respecto al propio baseline mejora drásticamente la predicción.

Duración: ~5 minutos (3 fases)
  Fase 1 — 2 min: Trabajando concentrado/a → baseline HIGH
  Fase 2 — 1 min: Mirando lejos, relajado  → baseline LOW
  Fase 3 — 2 min: Vuelta normal            → más datos HIGH/MEDIUM

Salida:
  - calibration_profile.json    (usado por neural_monitor.py)
  - logs/calibration_<fecha>.csv

Uso:
    python calibrate.py [--name "Raquel"]
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from capture import FrameCapture
from landmark_detector import FaceLandmarkDetector
from concentration import (
    compute_eye_aspect_ratio, compute_blink_rate, compute_gaze_deviation,
    compute_head_yaw, compute_mouth_aspect_ratio, detect_yawn,
    compute_concentration_score,
)
from collections import deque

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

PROFILE_PATH = MODELS_DIR / "calibration_profile.json"

# Duración de cada fase en segundos
PHASE_DURATIONS = [120, 60, 120]   # 2min, 1min, 2min = 5 min total
PHASE_NAMES     = [
    "CONCENTRADO",    # High — mirando a la pantalla, trabajando
    "DISTRAIDO",      # Low  — mirando lejos, relajado
    "NORMAL",         # Mix  — trabajando normalmente
]
PHASE_COLORS = [
    (0, 200, 100),   # verde: concentrado
    (0, 80, 220),    # rojo:  distraído
    (200, 160, 0),   # azul:  normal
]
PHASE_INSTRUCTIONS = [
    "Mira la pantalla y trabaja con normalidad.\nNo te muevas demasiado.",
    "Mira hacia un lado. Relájate.\nComo si te hubieran distraído.",
    "Trabaja con normalidad.\nPuedes moverte, parpadear, todo normal.",
]

# ─── ARGPARSE ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Calibración personal 5 min")
parser.add_argument("--name", default="Usuario", help="Tu nombre (para el perfil)")
parser.add_argument("--no-tpu", action="store_true")
args = parser.parse_args()

# ─── TEXTO EN OPENCV ──────────────────────────────────────────────────────────
def put_text(img, text, x, y, scale=0.65, color=(255,255,255),
             thickness=1, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line_idx, line in enumerate(text.split("\n")):
        ly = y + line_idx * int(scale * 35)
        if bg:
            (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
            cv2.rectangle(img, (x-3, ly-th-3), (x+tw+3, ly+5),
                          (0,0,0), -1)
        cv2.putText(img, line, (x, ly), font, scale, color, thickness,
                    cv2.LINE_AA)

def draw_progress_bar(img, progress, x, y, w, h, color):
    cv2.rectangle(img, (x, y), (x+w, y+h), (50,50,50), -1)
    filled = int(w * progress)
    if filled > 0:
        cv2.rectangle(img, (x, y), (x+filled, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (200,200,200), 1)

# ─── INICIALIZAR COMPONENTES ──────────────────────────────────────────────────
print(f"\n[CALIBRACIÓN] Usuario: {args.name}")
print("Inicializando cámara y detector...")

capture  = FrameCapture(resolution=(640, 480), fps=30)
detector = FaceLandmarkDetector(use_tpu=not args.no_tpu)

ear_deque  = deque(maxlen=900)
mar_deque  = deque(maxlen=900)
gaze_deque = deque(maxlen=15)
yaw_deque  = deque(maxlen=15)

# Datos recogidos por fase
phase_data = {0: [], 1: [], 2: []}   # fase → lista de dicts

# CSV de calibración
ts_str  = datetime.now().strftime("%Y%m%d_%H%M")
csv_path = LOGS_DIR / f"calibration_{ts_str}.csv"
csv_fields = ["timestamp", "phase", "phase_name",
              "score", "ear", "gaze_dev", "head_yaw", "blink_rate"]
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
csv_writer.writeheader()

# ─── LOOP PRINCIPAL ───────────────────────────────────────────────────────────
print("\nAbre la ventana de calibración y sigue las instrucciones en pantalla.")
print("Presiona 'q' para cancelar.\n")

frame_gen    = capture.get_frame_generator()
current_phase = -1      # -1 = pantalla de bienvenida
phase_start   = None
total_start   = time.time()
last_log_t    = time.time()
LOG_INTERVAL  = 2.0     # loguear cada 2 segundos durante calibración

try:
    while True:
        frame = next(frame_gen)
        landmarks = detector.detect(frame)
        now = time.time()

        # ── Calcular métricas si hay cara ──────────────────────────────────
        score = None
        ear = gaze_dev = head_yaw = blink_rate = 0.0

        if landmarks:
            el, er    = compute_eye_aspect_ratio(landmarks)
            ear       = (el + er) / 2.0
            ear_deque.append(ear)

            gaze_dev  = compute_gaze_deviation(landmarks)
            head_yaw  = compute_head_yaw(landmarks)
            gaze_deque.append(gaze_dev)
            yaw_deque.append(head_yaw)
            gaze_dev  = float(np.mean(gaze_deque))
            head_yaw  = float(np.mean(yaw_deque))

            mar = compute_mouth_aspect_ratio(landmarks)
            mar_deque.append(mar)

            blink_rate = compute_blink_rate(ear_deque)
            yawn       = detect_yawn(mar_deque)
            score      = compute_concentration_score(
                ear, gaze_dev, head_yaw, blink_rate, yawn)

        # ── Dibujar overlay ────────────────────────────────────────────────
        canvas = frame.copy()
        h, w   = canvas.shape[:2]

        if current_phase == -1:
            # PANTALLA DE BIENVENIDA
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (10,20,40), -1)
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

            put_text(canvas, "CALIBRACIÓN PERSONAL", 60, 80,
                     scale=1.1, color=(0,220,140), thickness=2, bg=False)
            put_text(canvas, f"Usuario: {args.name}", 60, 145,
                     scale=0.75, color=(200,220,255), bg=False)
            put_text(canvas, "Esta calibración dura ~5 minutos.", 60, 195,
                     scale=0.65, color=(180,180,180), bg=False)
            put_text(canvas, "Sigue las instrucciones en pantalla.", 60, 230,
                     scale=0.65, color=(180,180,180), bg=False)
            put_text(canvas, "Presiona  ESPACIO  para comenzar", 60, 310,
                     scale=0.8, color=(0,220,140), thickness=2, bg=False)
            put_text(canvas, "Presiona  Q  para cancelar", 60, 360,
                     scale=0.65, color=(120,120,120), bg=False)

            face_status = "CARA DETECTADA  OK" if landmarks else "Buscando cara..."
            fc = (0,200,80) if landmarks else (50,50,220)
            put_text(canvas, face_status, 60, h-50, color=fc)

        else:
            # FASE ACTIVA
            phase_color = PHASE_COLORS[current_phase]
            elapsed     = now - phase_start
            duration    = PHASE_DURATIONS[current_phase]
            progress    = min(elapsed / duration, 1.0)
            remaining   = max(0, duration - elapsed)

            # Banner superior
            cv2.rectangle(canvas, (0,0), (w, 80), phase_color, -1)
            phase_label = PHASE_NAMES[current_phase]
            put_text(canvas, f"Fase {current_phase+1}/3 — {phase_label}",
                     15, 50, scale=1.0, color=(255,255,255),
                     thickness=2, bg=False)

            # Instrucciones
            instr = PHASE_INSTRUCTIONS[current_phase]
            put_text(canvas, instr, 15, 120, scale=0.7,
                     color=(220,220,255))

            # Barra de progreso
            draw_progress_bar(canvas, progress, 15, h-55, w-30, 22,
                               phase_color)
            put_text(canvas, f"{int(remaining)}s restantes",
                     15, h-65, scale=0.6, color=(200,200,200))

            # Métricas en tiempo real
            if landmarks:
                info = (f"Score={score:.1f}  EAR={ear:.3f}  "
                        f"Gaze={gaze_dev:.1f}  Yaw={head_yaw:.1f}  "
                        f"Blink={blink_rate:.0f}/min")
                put_text(canvas, info, 15, h-80, scale=0.55,
                         color=(180,230,180))

            # Progreso total
            total_elapsed = now - total_start
            total_progress = min(total_elapsed / sum(PHASE_DURATIONS), 1.0)
            draw_progress_bar(canvas, total_progress, 15, h-25, w-30, 12,
                               (100,160,255))
            put_text(canvas, "Progreso total", 15, h-30,
                     scale=0.45, color=(160,160,160))

            # Contador de muestras
            n_samples = len(phase_data[current_phase])
            put_text(canvas, f"Muestras fase {current_phase+1}: {n_samples}",
                     w-230, 20, scale=0.55, color=(200,255,200))

            # Loguear métrica
            if (landmarks and score is not None
                    and now - last_log_t >= LOG_INTERVAL):
                entry = {
                    "ear": ear, "gaze_dev": gaze_dev, "head_yaw": head_yaw,
                    "blink_rate": blink_rate, "score": score
                }
                phase_data[current_phase].append(entry)
                csv_writer.writerow({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "phase": current_phase,
                    "phase_name": PHASE_NAMES[current_phase],
                    "score": round(score, 2),
                    "ear": round(ear, 3),
                    "gaze_dev": round(gaze_dev, 2),
                    "head_yaw": round(head_yaw, 2),
                    "blink_rate": round(blink_rate, 2),
                })
                last_log_t = now

            # Avanzar fase
            if elapsed >= duration:
                current_phase += 1
                if current_phase >= len(PHASE_DURATIONS):
                    # Calibración completada
                    break
                phase_start  = time.time()
                ear_deque.clear()
                mar_deque.clear()
                gaze_deque.clear()
                yaw_deque.clear()
                print(f"  → Fase {current_phase+1}: {PHASE_NAMES[current_phase]}")

        cv2.imshow("Calibración — Concentration Monitor", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[CANCELADO] Calibración cancelada por el usuario.")
            raise SystemExit(0)
        if key == ord(' ') and current_phase == -1:
            if not landmarks:
                print("  [AVISO] No se detecta cara. Asegúrate de estar frente a la cámara.")
                continue
            current_phase = 0
            phase_start   = time.time()
            total_start   = time.time()
            print(f"  → Iniciando Fase 1: {PHASE_NAMES[0]}")

except (StopIteration, SystemExit):
    pass
except KeyboardInterrupt:
    print("\n[CANCELADO] Ctrl+C")
finally:
    csv_file.close()
    capture.release()
    detector.release()
    cv2.destroyAllWindows()

# ─── CALCULAR PERFIL DE CALIBRACIÓN ──────────────────────────────────────────
def stats(data_list, key):
    vals = [d[key] for d in data_list if key in d]
    if not vals:
        return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "n": 0}
    arr = np.array(vals)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()) if len(arr) > 1 else 1.0,
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "n":    len(arr),
    }

print("\n[INFO] Calculando perfil de calibración...")

all_data     = phase_data[0] + phase_data[1] + phase_data[2]
focused_data = phase_data[0] + phase_data[2]   # fases concentradas
relax_data   = phase_data[1]                    # fase relajada/distraída

if len(all_data) < 5:
    print("[WARN] Muy pocos datos capturados. Usa el perfil genérico.")
    profile = {"user": args.name, "calibrated": False}
else:
    # Baseline global (para normalización de features)
    profile = {
        "user":       args.name,
        "date":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "calibrated": True,
        "n_samples":  len(all_data),

        # Estadísticas por métrica (todas las fases)
        "ear_stats":        stats(all_data, "ear"),
        "gaze_stats":       stats(all_data, "gaze_dev"),
        "yaw_stats":        stats(all_data, "head_yaw"),
        "blink_stats":      stats(all_data, "blink_rate"),

        # Baseline concentrado vs distraído
        "focused_ear_mean":   stats(focused_data, "ear")["mean"],
        "focused_gaze_mean":  stats(focused_data, "gaze_dev")["mean"],
        "focused_yaw_mean":   stats(focused_data, "head_yaw")["mean"],
        "focused_score_mean": stats(focused_data, "score")["mean"],

        "relax_gaze_mean":  stats(relax_data, "gaze_dev")["mean"] if relax_data else 0,
        "relax_yaw_mean":   stats(relax_data, "head_yaw")["mean"]  if relax_data else 0,
        "relax_score_mean": stats(relax_data, "score")["mean"]      if relax_data else 0,

        # Umbrales personalizados (basados en el rango propio)
        "personal_ear_threshold":  stats(focused_data, "ear")["mean"] * 0.60,
        "personal_gaze_threshold": max(15.0, stats(focused_data, "gaze_dev")["std"] * 2.5),
        "personal_yaw_threshold":  max(10.0, stats(focused_data, "head_yaw")["std"] * 2.5),
    }

MODELS_DIR.mkdir(exist_ok=True)
with open(PROFILE_PATH, "w") as f:
    json.dump(profile, f, indent=2)

# ─── RESUMEN ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"CALIBRACIÓN COMPLETADA — {args.name}")
print("="*60)
if profile.get("calibrated"):
    print(f"  Muestras totales:       {profile['n_samples']}")
    print(f"  EAR baseline:           {profile['ear_stats']['mean']:.3f} "
          f"± {profile['ear_stats']['std']:.3f}")
    print(f"  Gaze baseline:          {profile['gaze_stats']['mean']:.1f}°")
    print(f"  Yaw baseline:           {profile['yaw_stats']['mean']:.1f}°")
    print(f"  Score concentrado:      {profile['focused_score_mean']:.2f}/10")
    print(f"  Score distraído:        {profile['relax_score_mean']:.2f}/10")
    print(f"\n  Umbral EAR personal:    {profile['personal_ear_threshold']:.3f}")
    print(f"  Umbral Gaze personal:   {profile['personal_gaze_threshold']:.1f}°")
    print(f"  Umbral Yaw personal:    {profile['personal_yaw_threshold']:.1f}°")
    print(f"\n  Perfil guardado en:     {PROFILE_PATH}")
    print(f"  CSV de calibración:     {csv_path}")
print("\n" + "="*60)
print("SIGUIENTE PASO:  python neural_monitor.py   ← monitor final")
print("="*60)
