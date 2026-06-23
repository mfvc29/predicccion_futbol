"""
actualizador.py
================
Script todo-en-uno para actualizar los datos de la API y hacer push a GitHub.

Uso:
    python actualizador.py

Pasos que ejecuta:
    1. Descarga partidos frescos de la API (football-data.org)
    2. Recalcula estadísticas de equipos
    3. git add de los CSVs actualizados
    4. git commit con fecha de hoy
    5. git push origin main
"""

import subprocess
import sys
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(cmd, cwd=BASE_DIR):
    """Ejecuta un comando y muestra la salida en tiempo real."""
    print(f"\n▶  {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print("─" * 60)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        text=True,
    )
    if result.returncode != 0:
        print(f"\n❌ El comando falló con código {result.returncode}")
        sys.exit(result.returncode)
    print("─" * 60)


def main():
    hoy = datetime.now().strftime("%d-%b-%Y %H:%M")

    print("=" * 60)
    print("  ACTUALIZADOR — Predicción Fútbol")
    print(f"  {hoy}")
    print("=" * 60)

    # ── 1. Actualizar datos desde la API ─────────────────────────
    print("\n📡 PASO 1: Descargando datos actualizados de la API...")
    run([sys.executable, "fetch_api_data.py"])

    # ── 2. Git add de los archivos de datos ──────────────────────
    print("\n📦 PASO 2: Preparando archivos para el commit...")
    run([
        "git", "add",
        "api_matches_all.csv",
        "api_team_stats.csv",
        "api_wc2026_teams.csv",
        "api_wc2026_players.csv",
    ])

    # ── 3. Commit con fecha de hoy ────────────────────────────────
    print(f"\n💾 PASO 3: Haciendo commit ({hoy})...")
    commit_msg = f"Data: Actualización automática {hoy}"
    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
        print("ℹ️  No hay cambios nuevos en los datos desde la última actualización.")
    elif result.returncode != 0:
        print(result.stderr)
        print("⚠️  Hubo un problema con el commit (puede que no haya cambios).")
    
    # ── 4. Push a GitHub ──────────────────────────────────────────
    print("\n🚀 PASO 4: Subiendo cambios a GitHub...")
    run(["git", "push", "origin", "main"])

    # ── Resumen final ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ ¡Actualización completa!")
    print("   Los datos más recientes ya están en GitHub.")
    print("   Streamlit Cloud actualizará la app en ~1 minuto.")
    print("=" * 60)


if __name__ == "__main__":
    main()
