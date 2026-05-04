#!/usr/bin/env python3
"""
Entrenamiento de la Red Neuronal de Concentración.
Implementación desde cero con NumPy (sin dependencias externas).

Arquitectura: MLP con backpropagation
  Input (8 features) → Hidden(64, ReLU) → Hidden(32, ReLU) → Output

Guarda en models/:
  - concentration_classifier.pkl   (clasifica: low / medium / high)
  - concentration_regressor.pkl    (predice score continuo 0-10)
  - feature_scaler.pkl
  - training_stats.json

Uso:
    python train_model.py
"""

import csv
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np

# ─── RUTAS ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "combined_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── SCALER (StandardScaler manual) ──────────────────────────────────────────
class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ─── RED NEURONAL MLP ─────────────────────────────────────────────────────────
class MLP:
    """
    Perceptrón Multicapa con backpropagation.
    Soporta clasificación (softmax + cross-entropy) y regresión (lineal + MSE).
    """

    def __init__(self, layer_sizes, task="classification",
                 lr=1e-3, alpha=1e-4, batch_size=64,
                 max_epochs=300, patience=20, seed=42):
        """
        Args:
            layer_sizes : lista con tamaños [input, hidden..., output]
            task        : 'classification' o 'regression'
            lr          : learning rate
            alpha       : regularización L2
            batch_size  : tamaño del minibatch
            max_epochs  : máximo de épocas
            patience    : early stopping — épocas sin mejora
            seed        : semilla aleatoria
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.task        = task
        self.lr          = lr
        self.alpha       = alpha
        self.batch_size  = batch_size
        self.max_epochs  = max_epochs
        self.patience    = patience

        # Inicialización He para capas con ReLU
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale  = math.sqrt(2.0 / fan_in)
            W = np.random.randn(fan_in, layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)

    # ── Activaciones ──────────────────────────────────────────────────────────
    @staticmethod
    def _relu(x):      return np.maximum(0, x)
    @staticmethod
    def _relu_grad(x): return (x > 0).astype(np.float32)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def _forward(self, X):
        """Devuelve lista de activaciones para cada capa."""
        activations = [X]
        current = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            if i < len(self.weights) - 1:       # capas ocultas: ReLU
                current = self._relu(z)
            else:                                # capa de salida
                if self.task == "classification":
                    current = self._softmax(z)
                else:
                    current = z                  # lineal para regresión
            activations.append(current)
        return activations

    # ── Pérdida ───────────────────────────────────────────────────────────────
    def _loss(self, y_pred, y_true):
        n = len(y_true)
        if self.task == "classification":
            # Cross-entropy
            eps = 1e-12
            ce  = -np.log(y_pred[np.arange(n), y_true.astype(int)] + eps).mean()
            # Regularización L2
            l2 = sum((W**2).sum() for W in self.weights) * self.alpha / (2 * n)
            return ce + l2
        else:
            # MSE
            mse = ((y_pred.squeeze() - y_true)**2).mean()
            l2  = sum((W**2).sum() for W in self.weights) * self.alpha / (2 * n)
            return mse + l2

    # ── Backward pass ─────────────────────────────────────────────────────────
    def _backward(self, activations, y_true):
        n    = len(y_true)
        dW   = [None] * len(self.weights)
        db   = [None] * len(self.biases)

        # Gradiente en la capa de salida
        if self.task == "classification":
            delta = activations[-1].copy()
            delta[np.arange(n), y_true.astype(int)] -= 1
            delta /= n
        else:
            delta = (activations[-1].squeeze() - y_true).reshape(-1, 1) * 2 / n

        # Backprop
        for i in reversed(range(len(self.weights))):
            dW[i] = activations[i].T @ delta + self.alpha * self.weights[i]
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_grad(
                    activations[i])  # grad ReLU en activación pre-función
        return dW, db

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n      = len(X_train)
        best_loss  = np.inf
        best_W     = None
        best_b     = None
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            X_sh, y_sh = X_train[idx], y_train[idx]

            # Mini-batches
            for start in range(0, n, self.batch_size):
                Xb = X_sh[start:start + self.batch_size]
                yb = y_sh[start:start + self.batch_size]
                acts = self._forward(Xb)
                dW, db = self._backward(acts, yb)
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * dW[i]
                    self.biases[i]  -= self.lr * db[i]

            # Validación para early stopping
            if X_val is not None:
                val_acts = self._forward(X_val)
                val_loss = self._loss(val_acts[-1], y_val)
                if val_loss < best_loss - 1e-5:
                    best_loss  = val_loss
                    best_W = [W.copy() for W in self.weights]
                    best_b = [b.copy() for b in self.biases]
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if epoch % 50 == 0 or no_improve == self.patience:
                            print(f"       Early stopping en época {epoch} "
                                  f"(val_loss={best_loss:.4f})")
                        break

            if epoch % 100 == 0:
                tr_acts = self._forward(X_train)
                tr_loss = self._loss(tr_acts[-1], y_train)
                vl_str  = f"  val={val_loss:.4f}" if X_val is not None else ""
                print(f"       Época {epoch:4d} | train_loss={tr_loss:.4f}{vl_str}")

        # Restaurar mejores pesos
        if best_W is not None:
            self.weights = best_W
            self.biases  = best_b
        return self

    # ── Predicción ────────────────────────────────────────────────────────────
    def predict(self, X):
        acts = self._forward(X)
        out  = acts[-1]
        if self.task == "classification":
            return out.argmax(axis=1)
        return out.squeeze()

    def predict_proba(self, X):
        acts = self._forward(X)
        return acts[-1]   # softmax ya aplicado

# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────
print("="*60)
print("CONCENTRATION MONITOR — ENTRENAMIENTO RED NEURONAL")
print("="*60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró {DATA_PATH}\n"
        "Ejecuta primero: python generate_synthetic.py"
    )

rows = []
with open(DATA_PATH) as f:
    for row in csv.DictReader(f):
        try:
            rows.append({
                "ear":           float(row["ear"]),
                "gaze_dev":      float(row["gaze_dev"]),
                "head_yaw":      float(row["head_yaw"]),
                "blink_rate":    float(row["blink_rate"]),
                "yawn_detected": float(row["yawn_detected"]),
                "score":         float(row["score"]),
                "label":         row["label"],
            })
        except (ValueError, KeyError):
            continue

print(f"\n[DATA] {len(rows)} muestras cargadas")
label_map = {"low": 0, "medium": 1, "high": 2}
for lbl, idx in label_map.items():
    n = sum(1 for r in rows if r["label"] == lbl)
    print(f"       {lbl:8s}: {n} ({100*n/len(rows):.1f}%)")

# ─── FEATURES ─────────────────────────────────────────────────────────────────
def build_features(rows_list):
    X = []
    for r in rows_list:
        ear   = r["ear"]
        gaze  = r["gaze_dev"]
        yaw   = r["head_yaw"]
        blink = r["blink_rate"]
        yawn  = r["yawn_detected"]
        X.append([
            ear,
            abs(gaze),
            abs(yaw),
            blink,
            yawn,
            abs(gaze) * abs(yaw) / 100.0,
            1.0 if blink < 4 else 0.0,
            1.0 if ear < 0.08 else 0.0,
        ])
    return np.array(X, dtype=np.float64)

X_all    = build_features(rows)
y_score  = np.array([r["score"] for r in rows], dtype=np.float64)
y_label  = np.array([label_map[r["label"]] for r in rows], dtype=np.int32)
FEATURE_NAMES = ["ear", "abs_gaze", "abs_yaw", "blink_rate",
                 "yawn_detected", "gaze_x_yaw", "low_blink_flag", "low_ear_flag"]

# ─── SPLIT ────────────────────────────────────────────────────────────────────
np.random.seed(SEED)
n = len(rows)
idx = np.random.permutation(n)
split = int(n * 0.80)
train_idx, test_idx = idx[:split], idx[split:]

X_train, X_test   = X_all[train_idx], X_all[test_idx]
y_cls_train, y_cls_test = y_label[train_idx], y_label[test_idx]
y_reg_train, y_reg_test = y_score[train_idx], y_score[test_idx]

# Separar validación del train (15% del train total)
val_n = int(len(X_train) * 0.15)
X_val, y_cls_val, y_reg_val = X_train[:val_n], y_cls_train[:val_n], y_reg_train[:val_n]
X_train2, y_cls_train2, y_reg_train2 = X_train[val_n:], y_cls_train[val_n:], y_reg_train[val_n:]

# Escalar
scaler = StandardScaler()
X_train2_sc = scaler.fit_transform(X_train2)
X_val_sc    = scaler.transform(X_val)
X_test_sc   = scaler.transform(X_test)

print(f"\n[SPLIT] Train: {len(X_train2)} | Val: {len(X_val)} | Test: {len(X_test)}")
n_feats = X_all.shape[1]

# ─── CLASIFICADOR ─────────────────────────────────────────────────────────────
print("\n[TRAIN] Clasificador MLP (low / medium / high)...")
clf = MLP(
    layer_sizes=[n_feats, 64, 32, 3],
    task="classification",
    lr=5e-3, alpha=1e-4, batch_size=64,
    max_epochs=400, patience=25, seed=SEED
)
clf.fit(X_train2_sc, y_cls_train2, X_val_sc, y_cls_val)

y_pred_cls = clf.predict(X_test_sc)
acc = (y_pred_cls == y_cls_test).mean()
print(f"\n       Accuracy en test: {acc:.3f}  ({acc*100:.1f}%)")

# Informe por clase
classes = ["low", "medium", "high"]
print("       Precisión por clase:")
for i, cls in enumerate(classes):
    mask      = y_cls_test == i
    n_cls     = mask.sum()
    if n_cls == 0: continue
    tp        = ((y_pred_cls == i) & mask).sum()
    fp        = ((y_pred_cls == i) & ~mask).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / n_cls
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
    print(f"         {cls:8s}  precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}  (n={n_cls})")

# ─── REGRESOR ─────────────────────────────────────────────────────────────────
print("\n[TRAIN] Regresor MLP (score 0-10)...")
reg = MLP(
    layer_sizes=[n_feats, 64, 32, 1],
    task="regression",
    lr=5e-3, alpha=1e-4, batch_size=64,
    max_epochs=400, patience=25, seed=SEED
)
# Normalizar targets a [0,1] para mejor convergencia
y_reg_norm  = y_reg_train2 / 10.0
y_reg_val_n = y_reg_val    / 10.0
reg.fit(X_train2_sc, y_reg_norm, X_val_sc, y_reg_val_n)

y_pred_reg = reg.predict(X_test_sc) * 10.0   # volver a escala original
mae = float(np.abs(y_pred_reg - y_reg_test).mean())
ss_res = ((y_pred_reg - y_reg_test)**2).sum()
ss_tot = ((y_reg_test - y_reg_test.mean())**2).sum()
r2  = 1 - ss_res / ss_tot if ss_tot > 0 else 0

print(f"\n       MAE:  {mae:.3f} puntos")
print(f"       R²:   {r2:.3f}")

# ─── GUARDAR ──────────────────────────────────────────────────────────────────
with open(MODELS_DIR / "feature_scaler.pkl",           "wb") as f: pickle.dump(scaler, f)
with open(MODELS_DIR / "concentration_classifier.pkl", "wb") as f: pickle.dump(clf, f)
with open(MODELS_DIR / "concentration_regressor.pkl",  "wb") as f: pickle.dump(reg, f)

stats = {
    "n_samples":      len(rows),
    "n_train":        len(X_train2),
    "n_val":          len(X_val),
    "n_test":         len(X_test),
    "accuracy_cls":   float(acc),
    "mae_reg":        mae,
    "r2_reg":         float(r2),
    "feature_names":  FEATURE_NAMES,
    "label_map":      label_map,
    "classes":        classes,
    "scaler_mean":    scaler.mean_.tolist(),
    "scaler_std":     scaler.std_.tolist(),
}
with open(MODELS_DIR / "training_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n[OK] Modelos guardados en {MODELS_DIR}/")
print(f"     feature_scaler.pkl")
print(f"     concentration_classifier.pkl  (acc={acc:.1%})")
print(f"     concentration_regressor.pkl   (MAE={mae:.2f}  R²={r2:.3f})")
print(f"     training_stats.json")
print("\n" + "="*60)
print("SIGUIENTE PASO:  python calibrate.py   ← calibración personal")
print("="*60)
