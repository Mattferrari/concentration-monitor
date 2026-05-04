#!/usr/bin/env python3
"""
Generador de datos sintéticos para Concentration Monitor.

Estrategia:
1. Lee todos los CSVs reales y descarta sesiones con artefactos (gaze_dev=-90, EAR>1).
2. Calcula estadísticas por banda de concentración (low/medium/high).
3. Genera datos sintéticos usando la MISMA fórmula exacta del sistema
   (concentration.py) para que scores y métricas sean siempre consistentes.
4. Modela 5 perfiles de comportamiento con correlación temporal intra-sesión.
5. Exporta:
   - synthetic_data.csv  (solo sintético, con columna 'source')
   - combined_dataset.csv (real válido + sintético, listo para ML)
"""

import csv
import glob
import random
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
LOGS_DIR   = Path("/sessions/tender-dazzling-galileo/mnt/concentration-monitor/logs")
OUT_DIR    = Path("/sessions/tender-dazzling-galileo/mnt/concentration-monitor")
N_SYNTHETIC = 5_000   # total de muestras sintéticas a generar
SEED        = 42

rng = np.random.default_rng(SEED)

# ─── FÓRMULA EXACTA DE concentration.py ──────────────────────────────────────
def _norm(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo + 1e-9), 0.0, 1.0))

def compute_score(ear, gaze_dev, head_yaw, blink_rate, yawn=False):
    eye_score   = _norm(ear, 0.05, 0.30)
    gaze_adj    = max(0.0, abs(gaze_dev) - 8.0)
    gaze_score  = 1.0 - _norm(gaze_adj, 0, 30)
    yaw_adj     = max(0.0, abs(head_yaw) - 8.0)
    head_score  = 1.0 - _norm(yaw_adj, 0, 35)
    blink_score = _norm(blink_rate, 4, 25)
    raw  = 0.15*eye_score + 0.45*gaze_score + 0.30*head_score + 0.10*blink_score
    penalty = 2.5 if yawn else 0.0
    return float(np.clip(raw * 10 - penalty, 0, 10))

def score_to_label(s):
    if s < 4.0:  return "low"
    if s < 7.0:  return "medium"
    return "high"

def recommendation(s):
    if s < 4.0:
        return "[CRITICO] ¡DESCANSA AHORA! Concentración muy baja (< 4.0)"
    if s < 6.0:
        return "[WARN] Descansa en 10 minutos (concentración 4.0-6.0)"
    if s < 7.0:
        return ""
    return "[OK] Concentración óptima"

# ─── LECTURA DE DATOS REALES ──────────────────────────────────────────────────
def load_real_data():
    rows = []
    for path in sorted(LOGS_DIR.glob("session_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    r = {
                        "timestamp":  row["timestamp"],
                        "score":      float(row["score"]),
                        "ear":        float(row["ear"]),
                        "gaze_dev":   float(row["gaze_dev"]),
                        "head_yaw":   float(row["head_yaw"]),
                        "blink_rate": float(row["blink_rate"]),
                        "recommendation": row.get("recommendation", ""),
                        "source": "real",
                    }
                    # Filtrar sesiones con artefactos claros
                    if r["ear"] > 1.0:           continue  # EAR>1 es un bug de calibración
                    if abs(r["gaze_dev"]) >= 90:  continue  # gaze saturado, no fiable
                    rows.append(r)
                except (ValueError, KeyError):
                    continue
    return rows

real_data = load_real_data()
print(f"[INFO] Datos reales válidos: {len(real_data)} muestras")

# Estadísticas por banda
def band_stats(data, col):
    vals = [r[col] for r in data]
    if not vals:
        return 0, 1
    return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.1

for band in ["low","medium","high"]:
    subset = [r for r in real_data if score_to_label(r["score"]) == band]
    print(f"  {band:8s}: n={len(subset):3d} | "
          f"EAR={band_stats(subset,'ear')[0]:.3f}±{band_stats(subset,'ear')[1]:.3f} | "
          f"gaze={band_stats(subset,'gaze_dev')[0]:.1f}±{band_stats(subset,'gaze_dev')[1]:.1f} | "
          f"yaw={band_stats(subset,'head_yaw')[0]:.1f}±{band_stats(subset,'head_yaw')[1]:.1f} | "
          f"blink={band_stats(subset,'blink_rate')[0]:.1f}±{band_stats(subset,'blink_rate')[1]:.1f}")

# ─── PERFILES DE COMPORTAMIENTO ───────────────────────────────────────────────
# Cada perfil define distribuciones de entrada para (EAR, gaze_dev, head_yaw, blink_rate, p_yawn).
# Las muestras dentro de una sesión tienen correlación temporal (suavizado Ornstein-Uhlenbeck).

PROFILES = {
    # ── Alta concentración: leyendo/escribiendo frente a pantalla ──
    # Score esperado: 7-10
    "deep_focus": {
        "weight": 0.20,
        "ear":        {"mu": 0.26,  "sigma": 0.06,  "lo": 0.10, "hi": 0.50},
        "gaze_dev":   {"mu": 0.0,   "sigma": 4.0,   "lo":-12,   "hi": 12},
        "head_yaw":   {"mu": 0.0,   "sigma": 5.0,   "lo":-15,   "hi": 15},
        "blink_rate": {"mu": 12.0,  "sigma": 5.0,   "lo":  4,   "hi": 28},
        "p_yawn": 0.0,
    },
    # ── Hiperfoco: blink muy bajo, gaze muy estable ──
    # Score esperado: 8-10
    "hyperfocus": {
        "weight": 0.10,
        "ear":        {"mu": 0.30,  "sigma": 0.05,  "lo": 0.18, "hi": 0.55},
        "gaze_dev":   {"mu": 0.0,   "sigma": 3.0,   "lo": -8,   "hi":  8},
        "head_yaw":   {"mu": 0.0,   "sigma": 3.0,   "lo": -8,   "hi":  8},
        "blink_rate": {"mu": 5.5,   "sigma": 2.5,   "lo":  1,   "hi": 14},
        "p_yawn": 0.0,
    },
    # ── Concentración moderada: trabajo con distracciones menores ──
    # Score esperado: 5-7.5
    "moderate_focus": {
        "weight": 0.18,
        "ear":        {"mu": 0.24,  "sigma": 0.07,  "lo": 0.08, "hi": 0.55},
        "gaze_dev":   {"mu": 5.0,   "sigma": 14.0,  "lo":-30,   "hi": 35},
        "head_yaw":   {"mu": 3.0,   "sigma": 10.0,  "lo":-22,   "hi": 25},
        "blink_rate": {"mu": 18.0,  "sigma": 7.0,   "lo":  4,   "hi": 40},
        "p_yawn": 0.0,
    },
    # ── Distracción moderada: gaze/yaw fuera de zona muerta ──
    # Score esperado: 3.5-6
    "mildly_distracted": {
        "weight": 0.15,
        "ear":        {"mu": 0.26,  "sigma": 0.08,  "lo": 0.08, "hi": 0.60},
        "gaze_dev":   {"mu": 18.0,  "sigma": 12.0,  "lo":-50,   "hi": 55},
        "head_yaw":   {"mu": 12.0,  "sigma": 10.0,  "lo":-30,   "hi": 40},
        "blink_rate": {"mu": 22.0,  "sigma": 10.0,  "lo":  2,   "hi": 55},
        "p_yawn": 0.01,
    },
    # ── Distracción severa: mirando lejos, cabeza girada (simétrico izq/dch) ──
    # Score esperado: 1-4
    "heavily_distracted": {
        "weight": 0.17,
        "ear":        {"mu": 0.29,  "sigma": 0.10,  "lo": 0.08, "hi": 0.75},
        "gaze_dev":   {"mu": 0.0,   "sigma": 28.0,  "lo":-85,   "hi": 85},
        "head_yaw":   {"mu": 0.0,   "sigma": 22.0,  "lo":-65,   "hi": 65},
        "blink_rate": {"mu": 24.0,  "sigma": 12.0,  "lo":  2,   "hi": 65},
        "p_yawn": 0.03,
    },
    # ── Fatiga / somnolencia: EAR bajo, blink irregular, bostezos ──
    # Score esperado: 0.5-4.5
    "fatigue": {
        "weight": 0.20,
        "ear":        {"mu": 0.10,  "sigma": 0.05,  "lo": 0.01, "hi": 0.25},
        "gaze_dev":   {"mu": 3.0,   "sigma": 10.0,  "lo":-30,   "hi": 35},
        "head_yaw":   {"mu": 8.0,   "sigma": 10.0,  "lo":-20,   "hi": 35},
        "blink_rate": {"mu": 5.0,   "sigma": 4.0,   "lo":  0,   "hi": 18},
        "p_yawn": 0.18,
    },
}

PROFILE_NAMES   = list(PROFILES.keys())
PROFILE_WEIGHTS = [PROFILES[p]["weight"] for p in PROFILE_NAMES]

# ─── GENERADOR ORNSTEIN-UHLENBECK (correlación temporal) ─────────────────────
def ou_process(n, mu, sigma, theta=0.35, x0=None):
    """Proceso OU: mean-reverting random walk. theta controla velocidad de retorno."""
    x = x0 if x0 is not None else rng.normal(mu, sigma)
    path = [x]
    for _ in range(n - 1):
        dx = theta * (mu - x) + sigma * math.sqrt(2*theta) * rng.normal()
        x = x + dx
        path.append(x)
    return path

# ─── GENERACIÓN ───────────────────────────────────────────────────────────────
# Targets de distribución para el dataset sintético (aprox. realistas)
TARGET_DIST = {"low": 0.25, "medium": 0.40, "high": 0.35}

print(f"\n[INFO] Generando {N_SYNTHETIC} muestras sintéticas...")
print(f"       Distribución objetivo: low={TARGET_DIST['low']:.0%}  medium={TARGET_DIST['medium']:.0%}  high={TARGET_DIST['high']:.0%}")

synthetic_rows = []
start_dt = datetime(2026, 5, 1, 9, 0, 0)
current_dt = start_dt

sessions_generated = 0

while len(synthetic_rows) < N_SYNTHETIC:
    # Elegir perfil para esta sesión
    profile_name = rng.choice(PROFILE_NAMES, p=PROFILE_WEIGHTS)
    prof = PROFILES[profile_name]

    # Duración de la sesión: 6 a 25 muestras (= 1–4 min a 10 s/muestra)
    n_samples = int(rng.integers(6, 26))

    # Generar trayectorias temporalmente correladas para cada métrica
    ear_path   = ou_process(n_samples,
                            prof["ear"]["mu"],  prof["ear"]["sigma"],  theta=0.30)
    gaze_path  = ou_process(n_samples,
                            prof["gaze_dev"]["mu"], prof["gaze_dev"]["sigma"], theta=0.25)
    yaw_path   = ou_process(n_samples,
                            prof["head_yaw"]["mu"], prof["head_yaw"]["sigma"], theta=0.20)
    blink_path = ou_process(n_samples,
                            prof["blink_rate"]["mu"], prof["blink_rate"]["sigma"], theta=0.40)

    for i in range(n_samples):
        if len(synthetic_rows) >= N_SYNTHETIC:
            break

        # Clamp al rango del perfil
        ear        = float(np.clip(ear_path[i],
                                   prof["ear"]["lo"],  prof["ear"]["hi"]))
        gaze_dev   = float(np.clip(gaze_path[i],
                                   prof["gaze_dev"]["lo"], prof["gaze_dev"]["hi"]))
        head_yaw   = float(np.clip(yaw_path[i],
                                   prof["head_yaw"]["lo"], prof["head_yaw"]["hi"]))
        blink_rate = float(np.clip(blink_path[i],
                                   prof["blink_rate"]["lo"], prof["blink_rate"]["hi"]))
        yawn       = rng.random() < prof["p_yawn"]

        score = compute_score(ear, gaze_dev, head_yaw, blink_rate, yawn)

        # Añadir ruido de medición pequeño al score (simula jitter real)
        score = float(np.clip(score + rng.normal(0, 0.05), 0, 10))

        ts = current_dt.strftime("%Y-%m-%d %H:%M:%S")
        current_dt += timedelta(seconds=10)

        synthetic_rows.append({
            "timestamp":      ts,
            "score":          round(score, 2),
            "ear":            round(ear, 3),
            "gaze_dev":       round(gaze_dev, 2),
            "head_yaw":       round(head_yaw, 2),
            "blink_rate":     round(blink_rate, 2),
            "yawn_detected":  int(yawn),
            "profile":        profile_name,
            "label":          score_to_label(score),
            "recommendation": recommendation(score),
            "source":         "synthetic",
        })

    # Gap entre sesiones (2–15 min)
    gap_min = int(rng.integers(2, 16))
    current_dt += timedelta(minutes=gap_min)
    sessions_generated += 1

print(f"[INFO] Sesiones sintéticas generadas (fase 1): {sessions_generated}")

# ─── BALANCEO EXPLÍCITO ────────────────────────────────────────────────────────
# Si alguna clase está por debajo del target, generar muestras extra
# usando los perfiles que producen esa banda de score.
PROFILE_FOR_BAND = {
    "low":    "fatigue",             # fatigue + heavily_distracted → low
    "medium": "mildly_distracted",   # mildly_distracted → medium
    "high":   "deep_focus",          # deep_focus → high
}

current_counts = {lbl: sum(1 for r in synthetic_rows if r["label"] == lbl)
                  for lbl in ["low","medium","high"]}
total_now = len(synthetic_rows)

extra_rows = []
for lbl, target_frac in TARGET_DIST.items():
    target_n = int(N_SYNTHETIC * target_frac)
    deficit  = target_n - current_counts[lbl]
    if deficit <= 0:
        continue
    prof_name = PROFILE_FOR_BAND[lbl]
    # Para "low" alternar entre fatigue y heavily_distracted
    alt_prof  = "heavily_distracted" if lbl == "low" else None
    generated = 0
    attempts  = 0
    while generated < deficit and attempts < deficit * 10:
        attempts += 1
        pname = prof_name if (alt_prof is None or attempts % 2 == 0) else alt_prof
        prof  = PROFILES[pname]
        ear        = float(np.clip(rng.normal(prof["ear"]["mu"],   prof["ear"]["sigma"]),
                                   prof["ear"]["lo"],  prof["ear"]["hi"]))
        gaze_dev   = float(np.clip(rng.normal(prof["gaze_dev"]["mu"], prof["gaze_dev"]["sigma"]),
                                   prof["gaze_dev"]["lo"], prof["gaze_dev"]["hi"]))
        head_yaw   = float(np.clip(rng.normal(prof["head_yaw"]["mu"], prof["head_yaw"]["sigma"]),
                                   prof["head_yaw"]["lo"], prof["head_yaw"]["hi"]))
        blink_rate = float(np.clip(rng.normal(prof["blink_rate"]["mu"], prof["blink_rate"]["sigma"]),
                                   prof["blink_rate"]["lo"], prof["blink_rate"]["hi"]))
        yawn       = rng.random() < prof["p_yawn"]
        score      = float(np.clip(compute_score(ear, gaze_dev, head_yaw, blink_rate, yawn)
                                   + rng.normal(0, 0.05), 0, 10))
        if score_to_label(score) != lbl:
            continue  # rechazar si no cae en la banda correcta
        current_dt += timedelta(seconds=10)
        extra_rows.append({
            "timestamp":      current_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "score":          round(score, 2),
            "ear":            round(ear, 3),
            "gaze_dev":       round(gaze_dev, 2),
            "head_yaw":       round(head_yaw, 2),
            "blink_rate":     round(blink_rate, 2),
            "yawn_detected":  int(yawn),
            "profile":        pname,
            "label":          lbl,
            "recommendation": recommendation(score),
            "source":         "synthetic",
        })
        generated += 1
    print(f"  Balanceo {lbl}: +{generated} muestras extra (objetivo +{deficit})")

synthetic_rows.extend(extra_rows)

# Recortar clases que superan el target para que la distribución sea limpia
target_counts = {lbl: int(N_SYNTHETIC * f) for lbl, f in TARGET_DIST.items()}
balanced = []
counters = {"low": 0, "medium": 0, "high": 0}
rng.shuffle(synthetic_rows)
for row in synthetic_rows:
    lbl = row["label"]
    if counters[lbl] < target_counts[lbl]:
        balanced.append(row)
        counters[lbl] += 1
synthetic_rows = balanced

# Distribución final
labels_syn = [r["label"] for r in synthetic_rows]
for lbl in ["low", "medium", "high"]:
    n = labels_syn.count(lbl)
    pct = 100*n/len(labels_syn)
    mean_s = np.mean([r["score"] for r in synthetic_rows if r["label"] == lbl]) if n else 0
    print(f"  {lbl:8s}: {n:5d} ({pct:5.1f}%) | score_avg={mean_s:.2f}")

# ─── GUARDAR synthetic_data.csv ───────────────────────────────────────────────
syn_path = OUT_DIR / "synthetic_data.csv"
FIELDS_SYN = ["timestamp","score","ear","gaze_dev","head_yaw","blink_rate",
               "yawn_detected","profile","label","recommendation","source"]

with open(syn_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS_SYN)
    writer.writeheader()
    writer.writerows(synthetic_rows)

print(f"\n[OK] synthetic_data.csv → {syn_path}  ({len(synthetic_rows)} filas)")

# ─── GUARDAR combined_dataset.csv (real válido + sintético) ──────────────────
# Normalizar campos del dataset real para que tengan las mismas columnas
real_normalized = []
for r in real_data:
    # Inferir yawn: si la diferencia entre score esperado y real > 1.5, probablemente es yawn
    expected_no_yawn = compute_score(r["ear"], r["gaze_dev"], r["head_yaw"], r["blink_rate"], False)
    inferred_yawn = 1 if (expected_no_yawn - r["score"]) > 1.8 else 0
    real_normalized.append({
        "timestamp":      r["timestamp"],
        "score":          round(r["score"], 2),
        "ear":            round(r["ear"], 3),
        "gaze_dev":       round(r["gaze_dev"], 2),
        "head_yaw":       round(r["head_yaw"], 2),
        "blink_rate":     round(r["blink_rate"], 2),
        "yawn_detected":  inferred_yawn,
        "profile":        "real",
        "label":          score_to_label(r["score"]),
        "recommendation": r["recommendation"],
        "source":         "real",
    })

combined = real_normalized + synthetic_rows
# Mezclar para que real y sintético estén intercalados (mejor para cross-val)
rng.shuffle(combined)

combined_path = OUT_DIR / "combined_dataset.csv"
with open(combined_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS_SYN)
    writer.writeheader()
    writer.writerows(combined)

print(f"[OK] combined_dataset.csv → {combined_path}  ({len(combined)} filas total)")
print(f"     → {len(real_normalized)} reales válidos + {len(synthetic_rows)} sintéticos")

# ─── RESUMEN FINAL ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN DEL DATASET COMBINADO")
print("="*60)
all_labels = [r["label"] for r in combined]
for lbl in ["low", "medium", "high"]:
    n = all_labels.count(lbl)
    pct = 100*n/len(all_labels)
    sub = [r for r in combined if r["label"] == lbl]
    mean_s = np.mean([r["score"] for r in sub]) if sub else 0
    print(f"  {lbl:8s}: {n:5d} ({pct:5.1f}%) | score_avg={mean_s:.2f}")

print(f"\nTotal filas para ML: {len(combined)}")
print("Features disponibles: ear, gaze_dev, head_yaw, blink_rate, yawn_detected")
print("Targets: score (regresión), label (clasificación 3 clases)")
print("="*60)
