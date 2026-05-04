#!/usr/bin/env python3
"""
Monitor de Concentración con Red Neuronal — versión final.

Diferencias respecto a main.py:
  - Usa el modelo MLP entrenado para clasificar concentración.
  - Carga el perfil de calibración personal para normalizar features.
  - Alarma visual (overlay rojo) + pitido de sistema cuando concentración baja.
  - Genera reporte automático al salir.

Uso:
    python neural_monitor.py [--name "Raquel"] [--no-tpu] [--no-alarm]
    python neural_monitor.py --help
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import deque
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
from time_series import ConcentrationTimeSeries

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

ALARM_HOLD_SECONDS = 30    # Mantener score < 4 durante este tiempo → alarma
ALARM_COOLDOWN     = 120   # Segundos antes de volver a alarmar
LOG_INTERVAL       = 10    # Segundos entre entradas CSV

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def put_text(img, text, x, y, scale=0.6, color=(255,255,255),
             thickness=1, bg=True, bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(text.split("\n")):
        ly = y + i * int(scale * 36)
        if bg:
            (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
            cv2.rectangle(img, (x-3, ly-th-3), (x+tw+4, ly+5),
                          bg_color, -1)
        cv2.putText(img, line, (x, ly), font, scale, color,
                    thickness, cv2.LINE_AA)

def score_color(s):
    if s is None:   return (128, 128, 128)
    if s < 4.0:     return (30,  30,  220)   # rojo (BGR)
    if s < 7.0:     return (20,  165, 225)   # naranja
    return              (80,  200, 60)        # verde

def build_features(ear, gaze_dev, head_yaw, blink_rate, yawn_detected):
    """Misma ingeniería de features que train_model.py."""
    return np.array([[
        ear,
        abs(gaze_dev),
        abs(head_yaw),
        blink_rate,
        float(yawn_detected),
        abs(gaze_dev) * abs(head_yaw) / 100.0,
        1.0 if blink_rate < 4 else 0.0,
        1.0 if ear < 0.08 else 0.0,
    ]], dtype=np.float64)

# ─── CARGAR MODELOS ───────────────────────────────────────────────────────────
def load_models():
    paths = {
        "scaler": MODELS_DIR / "feature_scaler.pkl",
        "clf":    MODELS_DIR / "concentration_classifier.pkl",
        "reg":    MODELS_DIR / "concentration_regressor.pkl",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"[ERROR] Modelos no encontrados: {missing}")
        print("  Ejecuta primero:  python train_model.py")
        sys.exit(1)
    models = {}
    for k, p in paths.items():
        with open(p, "rb") as f:
            models[k] = pickle.load(f)
    return models["scaler"], models["clf"], models["reg"]

# ─── CARGAR PERFIL DE CALIBRACIÓN ─────────────────────────────────────────────
def load_profile():
    profile_path = MODELS_DIR / "calibration_profile.json"
    if not profile_path.exists():
        print("[WARN] Sin perfil de calibración. Usando umbrales genéricos.")
        return {"calibrated": False}
    with open(profile_path) as f:
        p = json.load(f)
    if p.get("calibrated"):
        print(f"[OK] Perfil cargado: {p.get('user','?')}  "
              f"(EAR baseline={p['ear_stats']['mean']:.3f})")
    return p

# ─── REPORTE FINAL ────────────────────────────────────────────────────────────
def generate_report(csv_path: Path, session_start: float, user: str) -> Path:
    """Genera un reporte de texto plano al final de la sesión."""
    rows = []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        "score":      float(row["score"]),
                        "ear":        float(row["ear"]),
                        "gaze_dev":   float(row["gaze_dev"]),
                        "head_yaw":   float(row["head_yaw"]),
                        "blink_rate": float(row["blink_rate"]),
                        "nn_label":   row.get("nn_label", ""),
                        "timestamp":  row["timestamp"],
                    })
                except (ValueError, KeyError):
                    continue
    except Exception:
        return None

    if not rows:
        return None

    scores     = [r["score"]      for r in rows]
    ears       = [r["ear"]        for r in rows]
    gazes      = [abs(r["gaze_dev"]) for r in rows]
    yaws       = [abs(r["head_yaw"])  for r in rows]
    blinks     = [r["blink_rate"] for r in rows]

    avg_score  = np.mean(scores)
    duration_m = len(rows) * LOG_INTERVAL / 60.0
    low_pct    = 100 * sum(1 for s in scores if s < 4) / len(scores)
    med_pct    = 100 * sum(1 for s in scores if 4 <= s < 7) / len(scores)
    high_pct   = 100 * sum(1 for s in scores if s >= 7) / len(scores)
    n_alarms   = sum(1 for r in rows if r.get("alarm_triggered") == "1")

    report_path = LOGS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE SESIÓN — CONCENTRATION MONITOR\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Usuario:         {user}\n")
        f.write(f"Fecha:           {rows[0]['timestamp'][:10]}\n")
        f.write(f"Hora inicio:     {rows[0]['timestamp'][11:]}\n")
        f.write(f"Hora fin:        {rows[-1]['timestamp'][11:]}\n")
        f.write(f"Duración total:  {duration_m:.1f} minutos\n")
        f.write(f"Muestras:        {len(rows)} (cada {LOG_INTERVAL}s)\n\n")

        f.write("─" * 60 + "\n")
        f.write("SCORE DE CONCENTRACIÓN (0-10)\n")
        f.write("─" * 60 + "\n")
        f.write(f"  Promedio:        {avg_score:.2f}\n")
        f.write(f"  Máximo:          {max(scores):.2f}\n")
        f.write(f"  Mínimo:          {min(scores):.2f}\n")
        f.write(f"  Desv. estándar:  {np.std(scores):.2f}\n\n")
        f.write(f"  Distribución:\n")
        f.write(f"    Alta  (7-10):   {high_pct:.1f}%\n")
        f.write(f"    Media (4-7):    {med_pct:.1f}%\n")
        f.write(f"    Baja  (0-4):    {low_pct:.1f}%\n\n")

        f.write("─" * 60 + "\n")
        f.write("MÉTRICAS FACIALES\n")
        f.write("─" * 60 + "\n")
        f.write(f"  EAR medio:       {np.mean(ears):.3f}  (normal: 0.15-0.35)\n")
        f.write(f"  Gaze desv. media:{np.mean(gazes):.1f}°  (normal: <15°)\n")
        f.write(f"  Yaw cabeza medio:{np.mean(yaws):.1f}°  (normal: <20°)\n")
        f.write(f"  Parpadeos/min:   {np.mean(blinks):.1f}   (normal: 8-25)\n\n")

        f.write("─" * 60 + "\n")
        f.write("RECOMENDACIONES PERSONALIZADAS\n")
        f.write("─" * 60 + "\n")

        if avg_score >= 7.5:
            f.write("  ✓ Sesión excelente. Tu concentración fue muy alta.\n")
            f.write("    Mantén esta rutina de trabajo.\n")
        elif avg_score >= 5.5:
            f.write("  ! Concentración moderada. Hay margen de mejora.\n")
            f.write("    Considera descansos cada 45-50 min (técnica Pomodoro).\n")
        else:
            f.write("  ⚠ Concentración baja. La sesión fue difícil.\n")
            f.write("    Descansa antes de seguir. Revisa postura y entorno.\n")

        if low_pct > 25:
            f.write(f"\n  ⚠ El {low_pct:.0f}% del tiempo tuviste concentración baja.\n")
            f.write("    Factores posibles: fatiga, distracciones externas,\n")
            f.write("    mala iluminación o temperatura incómoda.\n")

        if np.mean(blinks) < 8:
            f.write("\n  ! Parpadeo bajo detectado → posible fatiga ocular.\n")
            f.write("    Aplica la regla 20-20-20: cada 20 min, mira a 20 pies\n")
            f.write("    durante 20 segundos.\n")
        elif np.mean(blinks) > 30:
            f.write("\n  ! Parpadeo alto → posible estrés o irritación ocular.\n")

        if np.mean(gazes) > 20:
            f.write("\n  ! Mirada frecuentemente desviada.\n")
            f.write("    Reduce notificaciones y fuentes de distracción visual.\n")

        if np.mean(yaws) > 15:
            f.write("\n  ! Cabeza girada frecuentemente → postura o monitor lateral.\n")
            f.write("    Centra el monitor frente a ti para mejorar postura.\n")

        # Mejor momento de la sesión
        if len(scores) >= 3:
            best_idx  = int(np.argmax(scores))
            best_ts   = rows[best_idx]["timestamp"][11:]
            worst_idx = int(np.argmin(scores))
            worst_ts  = rows[worst_idx]["timestamp"][11:]
            f.write(f"\n  ★ Mejor momento:  {best_ts} (score={scores[best_idx]:.1f})\n")
            f.write(f"  ↓ Peor momento:   {worst_ts} (score={scores[worst_idx]:.1f})\n")

        # Recomendación de frecuencia de descanso
        f.write("\n─" * 30 + "\n")
        if avg_score >= 7:
            f.write("  Descanso recomendado: cada 50-60 minutos\n")
        elif avg_score >= 5:
            f.write("  Descanso recomendado: cada 30-40 minutos\n")
        else:
            f.write("  Descanso recomendado: AHORA y cada 20-25 minutos\n")

        f.write("\n" + "=" * 60 + "\n")

    return report_path

# ─── ARGPARSE ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Monitor de concentración con red neuronal")
parser.add_argument("--name",     default="Usuario")
parser.add_argument("--no-tpu",   action="store_true")
parser.add_argument("--no-alarm", action="store_true",
                    help="Desactivar alarma visual/sonora")
parser.add_argument("--resolution", default="640x480")
parser.add_argument("--fps",      type=int, default=30)
parser.add_argument("--interval", type=int, default=10,
                    help="Intervalo de logging en segundos")
args = parser.parse_args()

try:
    W_CAP, H_CAP = map(int, args.resolution.split("x"))
except ValueError:
    W_CAP, H_CAP = 640, 480

# ─── INICIALIZACIÓN ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CONCENTRATION MONITOR — RED NEURONAL")
print("="*60)

scaler, clf, reg = load_models()
profile          = load_profile()
user_name        = profile.get("user", args.name)

capture  = FrameCapture(resolution=(W_CAP, H_CAP), fps=args.fps)
detector = FaceLandmarkDetector(use_tpu=not args.no_tpu)
ts_obj   = ConcentrationTimeSeries()

ear_deque  = deque(maxlen=900)
mar_deque  = deque(maxlen=900)
gaze_deque = deque(maxlen=15)
yaw_deque  = deque(maxlen=15)
score_hist = deque(maxlen=180)   # 30 min a 10s

# CSV logging
ts_str   = datetime.now().strftime("%Y%m%d_%H%M")
csv_path = LOGS_DIR / f"session_{ts_str}.csv"
CSV_FIELDS = ["timestamp", "score", "ear", "gaze_dev", "head_yaw",
              "blink_rate", "nn_label", "nn_score", "alarm_triggered",
              "recommendation"]
csv_file   = open(csv_path, "w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
csv_writer.writeheader()

# Estado alarma
alarm_active      = False
alarm_start       = None
last_alarm_time   = 0
alarm_frame_count = 0
ALARM_FLASH_RATE  = 15   # frames

# Contadores
frame_count   = 0
session_start = time.time()
last_log_t    = time.time()
fps_buf       = deque(maxlen=30)

print(f"[OK] Monitor iniciado | Usuario: {user_name}")
print("Presiona 'q' para salir y generar reporte.\n")

try:
    frame_gen = capture.get_frame_generator()

    while True:
        t_loop = time.time()
        frame  = next(frame_gen)

        landmarks = detector.detect(frame)
        now       = time.time()

        # ── Métricas ───────────────────────────────────────────────────────
        score = nn_score = None
        nn_label  = "–"
        ear = gaze_dev = head_yaw = blink_rate = 0.0
        yawn = False

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

            # Score clásico (rule-based)
            score = compute_concentration_score(
                ear, gaze_dev, head_yaw, blink_rate, yawn)

            # ── Predicción de la Red Neuronal ──────────────────────────
            feat_raw = build_features(ear, gaze_dev, head_yaw, blink_rate, yawn)

            # Aplicar normalización personal si hay calibración
            if profile.get("calibrated"):
                # Ajustar EAR respecto al baseline personal
                ear_base = profile["ear_stats"]["mean"]
                if ear_base > 0:
                    feat_raw[0, 0] = feat_raw[0, 0] / ear_base * 0.25

            feat_sc   = scaler.transform(feat_raw)
            nn_cls    = clf.predict(feat_sc)[0]
            nn_proba  = clf.predict_proba(feat_sc)[0]
            nn_score  = float(reg.predict(feat_sc)[0]) * 10.0
            nn_score  = float(np.clip(nn_score, 0, 10))

            classes   = ["low", "medium", "high"]
            nn_label  = classes[nn_cls]

            # Fusionar score clásico y NN (media ponderada)
            score = float(0.5 * score + 0.5 * nn_score)

            ts_obj.add_score(score)
            score_hist.append(score)

        # ── Alarma ─────────────────────────────────────────────────────────
        alarm_triggered = False
        if score is not None and not args.no_alarm:
            low_streak = sum(1 for s in list(score_hist)[-3:] if s < 4.0)
            if low_streak >= 3 and (now - last_alarm_time) > ALARM_COOLDOWN:
                if not alarm_active:
                    alarm_active  = True
                    alarm_start   = now
                    alarm_frame_count = 0
                    print(f"  [!] ALARMA: Concentración baja durante "
                          f">{ALARM_HOLD_SECONDS}s")
                    # Pitido del sistema (si disponible)
                    os.system("echo -e '\\a'")
                alarm_triggered = True
            else:
                if alarm_active and (low_streak < 2):
                    alarm_active   = False
                    last_alarm_time = now

        # ── Logging CSV ────────────────────────────────────────────────────
        if score is not None and (now - last_log_t) >= args.interval:
            rec = ts_obj.get_recommendation()
            csv_writer.writerow({
                "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "score":           f"{score:.2f}",
                "ear":             f"{ear:.3f}",
                "gaze_dev":        f"{gaze_dev:.2f}",
                "head_yaw":        f"{head_yaw:.2f}",
                "blink_rate":      f"{blink_rate:.2f}",
                "nn_label":        nn_label,
                "nn_score":        f"{nn_score:.2f}" if nn_score else "",
                "alarm_triggered": "1" if alarm_triggered else "0",
                "recommendation":  rec,
            })
            last_log_t = now

        # ── Dashboard ──────────────────────────────────────────────────────
        canvas = frame.copy()
        h, w   = canvas.shape[:2]

        # Overlay de alarma (parpadeo rojo)
        if alarm_active:
            alarm_frame_count += 1
            if (alarm_frame_count // ALARM_FLASH_RATE) % 2 == 0:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,0,180), -1)
                cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
                put_text(canvas, "! CONCENTRACION BAJA !", w//2-200, h//2,
                         scale=1.2, color=(255,255,255), thickness=2,
                         bg=True, bg_color=(0,0,160))
                put_text(canvas, "Toma un descanso", w//2-120, h//2+60,
                         scale=0.9, color=(200,220,255))

        # Panel superior (HUD)
        sc = score if score is not None else 0
        sc_color = score_color(sc)
        cv2.rectangle(canvas, (0, 0), (w, 55), (15,15,25), -1)

        # Score grande
        score_str = f"{sc:.1f}" if score is not None else "–"
        put_text(canvas, score_str, 15, 42, scale=1.4,
                 color=sc_color, thickness=2, bg=False)
        put_text(canvas, "/10", 95, 42, scale=0.7,
                 color=(160,160,160), bg=False)

        # Label NN
        lbl_colors = {"low": (50,50,220), "medium": (30,160,220),
                      "high": (60,200,80)}
        lbl_color = lbl_colors.get(nn_label, (160,160,160))
        put_text(canvas, f"NN: {nn_label.upper()}", 155, 38,
                 scale=0.75, color=lbl_color, bg=False)

        # Métricas
        if landmarks:
            meta = (f"EAR={ear:.3f}  Gaze={gaze_dev:.1f}°  "
                    f"Yaw={head_yaw:.1f}°  Blink={blink_rate:.0f}/m")
            put_text(canvas, meta, 15, h-45, scale=0.52,
                     color=(180,220,180))

        # Recomendación
        rec = ts_obj.get_recommendation()
        if rec:
            rec_color = (50,50,220) if "DESCANSA" in rec else (30,160,220)
            put_text(canvas, rec, 15, h-22, scale=0.55, color=rec_color)

        # Mini gráfica de score (últimos 60 puntos)
        hist = list(score_hist)[-60:]
        if len(hist) > 1:
            gx, gy, gw, gh = w-185, 5, 180, 48
            cv2.rectangle(canvas, (gx,gy), (gx+gw,gy+gh), (20,20,30), -1)
            for j in range(1, len(hist)):
                x1 = gx + int((j-1) * gw / 60)
                x2 = gx + int(j     * gw / 60)
                y1 = gy + gh - int(hist[j-1] / 10 * gh)
                y2 = gy + gh - int(hist[j]   / 10 * gh)
                cv2.line(canvas, (x1,y1), (x2,y2),
                         score_color(hist[j]), 1)

        # FPS
        loop_t = time.time() - t_loop
        if loop_t > 0:
            fps_buf.append(1.0 / loop_t)
        avg_fps = sum(fps_buf) / len(fps_buf) if fps_buf else 0
        put_text(canvas, f"{avg_fps:.0f}fps", w-50, h-5,
                 scale=0.45, color=(80,80,80))

        # Tiempo de sesión
        elapsed_m = (now - session_start) / 60
        put_text(canvas, f"{elapsed_m:.1f}min", 15, 55,
                 scale=0.5, color=(100,100,100), bg=False)

        cv2.imshow(f"Concentration Monitor — {user_name}", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_count += 1

except (StopIteration, KeyboardInterrupt):
    pass
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback; traceback.print_exc()
finally:
    csv_file.close()
    capture.release()
    detector.release()
    cv2.destroyAllWindows()

# ─── REPORTE FINAL ────────────────────────────────────────────────────────────
print(f"\n[INFO] Sesión terminada | {frame_count} frames procesados")
print(f"[LOG]  CSV guardado: {csv_path}")

report_path = generate_report(csv_path, session_start, user_name)
if report_path:
    print(f"[REPORT] Reporte generado: {report_path}")
    print("\n" + "─"*60)
    # Mostrar reporte en terminal
    with open(report_path) as f:
        print(f.read())
else:
    print("[WARN] No hay suficientes datos para generar reporte.")

print("[OK] ¡Hasta la próxima sesión!")
