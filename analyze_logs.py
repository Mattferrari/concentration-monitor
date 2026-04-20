#!/usr/bin/env python3
"""
Herramienta de análisis para los archivos CSV de logs.
Extrae estadísticas y genera reportes de sesiones de concentración.
Uso: python analyze_logs.py <archivo.csv>
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def analyze_csv(csv_path: str) -> None:
    """
    Analiza un archivo CSV de logs y genera un reporte.

    Args:
        csv_path: Ruta al archivo CSV
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ Archivo no encontrado: {csv_path}")
        sys.exit(1)

    print(f"\n📊 Analizando: {csv_file.name}\n")

    # Leer datos
    scores = []
    ears = []
    gazes = []
    yaws = []
    blinks = []
    timestamps = []
    low_concentration_periods = []

    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    score = float(row["score"])
                    ear = float(row["ear"])
                    gaze = float(row["gaze_dev"])
                    yaw = float(row["head_yaw"])
                    blink = float(row["blink_rate"])
                    timestamp = row["timestamp"]

                    scores.append(score)
                    ears.append(ear)
                    gazes.append(gaze)
                    yaws.append(yaw)
                    blinks.append(blink)
                    timestamps.append(timestamp)

                    # Detectar períodos de baja concentración
                    if score < 4.0:
                        low_concentration_periods.append((timestamp, score))

                except ValueError:
                    continue

    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        sys.exit(1)

    if not scores:
        print("❌ No hay datos válidos en el archivo")
        sys.exit(1)

    # Calcular estadísticas
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    low_score_count = sum(1 for s in scores if s < 4.0)
    medium_score_count = sum(1 for s in scores if 4.0 <= s < 7.0)
    high_score_count = sum(1 for s in scores if s >= 7.0)

    avg_ear = sum(ears) / len(ears)
    avg_gaze = sum(abs(g) for g in gazes) / len(gazes)
    avg_yaw = sum(abs(y) for y in yaws) / len(yaws)
    avg_blink = sum(blinks) / len(blinks)

    # Imprimir reporte
    print("=" * 60)
    print("REPORTE DE CONCENTRACIÓN")
    print("=" * 60)

    print(f"\n📋 INFORMACIÓN GENERAL")
    print(f"  Muestras analizadas: {len(scores)}")
    print(f"  Duración: {timestamps[0]} → {timestamps[-1]}")
    if len(scores) >= 2:
        duration_min = (len(scores) - 1) * 10 / 60  # asumiendo 10s entre muestras
        print(f"  Tiempo total: {duration_min:.1f} minutos")

    print(f"\n🎯 SCORE DE CONCENTRACIÓN (0-10)")
    print(f"  Promedio: {avg_score:.2f}")
    print(f"  Rango: {min_score:.1f} - {max_score:.1f}")
    print(f"  Distribución:")
    print(f"    🔴 Bajo (0-4):    {low_score_count:3d} ({100*low_score_count/len(scores):.1f}%)")
    print(f"    🟡 Medio (4-7):   {medium_score_count:3d} ({100*medium_score_count/len(scores):.1f}%)")
    print(f"    🟢 Alto (7-10):   {high_score_count:3d} ({100*high_score_count/len(scores):.1f}%)")

    print(f"\n👁️  MÉTRICAS INDIVIDUALES")
    print(f"  Eye Aspect Ratio (EAR):     {avg_ear:.3f}  (normal: 0.15-0.35)")
    print(f"  Desviación de Mirada:       {avg_gaze:.1f}°  (normal: <30°)")
    print(f"  Rotación de Cabeza (Yaw):   {avg_yaw:.1f}°  (normal: <35°)")
    print(f"  Parpadeos por Minuto:       {avg_blink:.1f}   (normal: 12-20)")

    print(f"\n⚠️  PERÍODOS DE BAJA CONCENTRACIÓN")
    if low_concentration_periods:
        print(f"  Total detectados: {len(low_concentration_periods)}")
        if len(low_concentration_periods) <= 10:
            for ts, score in low_concentration_periods:
                print(f"    {ts}: score={score:.1f}")
        else:
            for ts, score in low_concentration_periods[:5]:
                print(f"    {ts}: score={score:.1f}")
            print(f"    ... y {len(low_concentration_periods) - 5} más")
    else:
        print(f"  Ninguno detectado ✓")

    print(f"\n💡 RECOMENDACIONES")
    if avg_score < 5.0:
        print(f"  • La concentración es baja. Buscar factores de distracción.")
        print(f"  • Considerar descansos más frecuentes.")
    elif avg_score < 7.0:
        print(f"  • La concentración es moderada. Hay lugar para mejora.")
        print(f"  • Mantener vigilancia en períodos críticos.")
    else:
        print(f"  • Excelente concentración general. Mantener la rutina.")

    if avg_blink < 10:
        print(f"  • Parpadeo bajo: Fatiga ocular posible. Descansos más frecuentes.")
    elif avg_blink > 25:
        print(f"  • Parpadeo alto: Posible estrés o ansiedad. Técnicas de relajación.")

    if avg_gaze > 15:
        print(f"  • Mirada muy desviada: Dificultad manteniendo foco frontal.")

    if avg_yaw > 15:
        print(f"  • Cabeza rotada frecuentemente: Postura deficiente.")

    print(f"\n" + "=" * 60 + "\n")


def list_sessions() -> None:
    """Lista todas las sesiones disponibles en logs/."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("❌ Directorio logs/ no encontrado")
        return

    csv_files = sorted(logs_dir.glob("session_*.csv"))
    if not csv_files:
        print("No hay sesiones registradas en logs/")
        return

    print(f"📁 Sesiones disponibles en {logs_dir}:\n")
    for i, csv_file in enumerate(csv_files, 1):
        size = csv_file.stat().st_size
        print(f"  {i}. {csv_file.name}  ({size:,} bytes)")

    print(f"\nPara analizar una sesión:")
    print(f"  python analyze_logs.py logs/session_YYYYMMDD_HHMM.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Analizador de Logs de Concentración\n")
        list_sessions()
        sys.exit(0)

    csv_path = sys.argv[1]
    analyze_csv(csv_path)
