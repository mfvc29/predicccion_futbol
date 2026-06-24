"""
actualizador.py
================
Script todo-en-uno para actualizar los datos de la API, regenerar los
standings del bracket WC2026 en app.py, y hacer push a GitHub.

Uso:
    python actualizador.py

Pasos que ejecuta:
    1. Descarga partidos frescos de la API (football-data.org)
    2. Recalcula estadísticas de equipos
    3. Recalcula standings reales del WC2026 y actualiza GRUPOS en app.py
    4. git add de los CSVs y app.py actualizados
    5. git commit con fecha de hoy
    6. git push origin main
"""

import subprocess
import sys
import os
import re
import pandas as pd
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mapa de nombres API → nombres en español para el bracket
NOMBRE_ES = {
    "Mexico": "México",
    "South Korea": "Corea del Sur",
    "Czechia": "Chequia",
    "South Africa": "Sudáfrica",
    "Canada": "Canadá",
    "Bosnia-Herzegovina": "Bosnia-Herzegovina",
    "Qatar": "Catar",
    "Switzerland": "Suiza",
    "Brazil": "Brasil",
    "Morocco": "Marruecos",
    "Scotland": "Escocia",
    "Haiti": "Haití",
    "United States": "EE.UU.",
    "Australia": "Australia",
    "Paraguay": "Paraguay",
    "Turkey": "Turquía",
    "Germany": "Alemania",
    "Ivory Coast": "Costa de Marfil",
    "Ecuador": "Ecuador",
    "Curaçao": "Curazao",
    "Netherlands": "Países Bajos",
    "Japan": "Japón",
    "Sweden": "Suecia",
    "Tunisia": "Túnez",
    "Egypt": "Egipto",
    "Iran": "Irán",
    "Belgium": "Bélgica",
    "New Zealand": "Nueva Zelanda",
    "Spain": "España",
    "Uruguay": "Uruguay",
    "Cape Verde Islands": "Cabo Verde",
    "Saudi Arabia": "Arabia Saudí",
    "France": "Francia",
    "Norway": "Noruega",
    "Senegal": "Senegal",
    "Iraq": "Irak",
    "Argentina": "Argentina",
    "Austria": "Austria",
    "Algeria": "Argelia",
    "Jordan": "Jordania",
    "Portugal": "Portugal",
    "Colombia": "Colombia",
    "DR Congo": "RD Congo",
    "Uzbekistan": "Uzbekistán",
    "England": "Inglaterra",
    "Ghana": "Ghana",
    "Panama": "Panamá",
    "Croatia": "Croacia",
}


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


def calcular_standings_wc():
    """
    Lee api_matches_all.csv, calcula los standings reales del WC2026
    y devuelve un dict GRUPOS compatible con el formato de app.py.
    """
    csv_path = os.path.join(BASE_DIR, "api_matches_all.csv")
    df = pd.read_csv(csv_path)
    wc = df[(df["competition"] == "WC") & (df["group"].notna())].copy()
    wc = wc[wc["group"].str.startswith("GROUP_")]

    # Stats por equipo dentro de su grupo
    stats = defaultdict(lambda: {
        "grp": "", "pts": 0, "gf": 0, "gc": 0, "pj": 0
    })

    for _, m in wc.iterrows():
        grp_key = m["group"].replace("GROUP_", "")   # "A", "B", ...
        ht, at  = m["home_team"], m["away_team"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        w       = m["winner"]

        for team, gf, gc, winner_str in [
            (ht, hs, as_, "HOME_TEAM"),
            (at, as_, hs, "AWAY_TEAM"),
        ]:
            stats[team]["grp"] = grp_key
            stats[team]["gf"]  += gf
            stats[team]["gc"]  += gc
            stats[team]["pj"]  += 1
            if w == winner_str:
                stats[team]["pts"] += 3
            elif w == "DRAW":
                stats[team]["pts"] += 1

    # Agrupar por grupo
    grupos = defaultdict(list)
    for team, s in stats.items():
        grupos[s["grp"]].append({
            "team": team,
            "pts":  s["pts"],
            "gf":   s["gf"],
            "gc":   s["gc"],
            "pj":   s["pj"],
        })

    # Ordenar cada grupo por pts → dif de goles → goles a favor
    for grp in grupos:
        grupos[grp].sort(key=lambda x: (-x["pts"], -(x["gf"]-x["gc"]), -x["gf"]))

    return dict(sorted(grupos.items()))


def grupos_a_codigo(grupos):
    """Convierte el dict GRUPOS a código Python listo para insertar en app.py."""
    lines = ["    GRUPOS = {\n"]
    for grp_key, teams in sorted(grupos.items()):
        lines.append(f'        "{grp_key}": [\n')
        for t in teams:
            lines.append(
                f'            {{"team": "{t["team"]}", "pts": {t["pts"]}, '
                f'"gf": {t["gf"]}, "gc": {t["gc"]}, "pj": {t["pj"]}}},\n'
            )
        lines.append("        ],\n")
    lines.append("    }\n")
    return "".join(lines)


def actualizar_grupos_en_app(grupos):
    """
    Reemplaza el bloque GRUPOS = { ... } en app.py con los datos frescos.
    """
    app_path = os.path.join(BASE_DIR, "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()

    nuevo_codigo = grupos_a_codigo(grupos)

    # Reemplazar el bloque GRUPOS = { ... } usando regex
    patron = r"(    GRUPOS = \{.*?    \}\n)"
    nuevo_content, n = re.subn(patron, nuevo_codigo, content, count=1, flags=re.DOTALL)

    if n == 0:
        print("  ⚠️  No se encontró el bloque GRUPOS en app.py — saltando.")
        return False

    with open(app_path, "w", encoding="utf-8") as f:
        f.write(nuevo_content)

    print(f"  ✅ GRUPOS actualizado en app.py ({len(grupos)} grupos, "
          f"{sum(len(v) for v in grupos.values())} equipos)")
    return True


def main():
    hoy = datetime.now().strftime("%d-%b-%Y %H:%M")

    print("=" * 60)
    print("  ACTUALIZADOR — Predicción Fútbol")
    print(f"  {hoy}")
    print("=" * 60)

    # ── 1. Actualizar datos desde la API ─────────────────────────
    print("\n📡 PASO 1: Descargando datos actualizados de la API...")
    run([sys.executable, "fetch_api_data.py"])

    # ── 2. Recalcular standings y actualizar app.py ───────────────
    print("\n🏆 PASO 2: Recalculando standings WC2026 y actualizando app.py...")
    try:
        grupos = calcular_standings_wc()
        for grp_key, teams in sorted(grupos.items()):
            lider = teams[0]["team"] if teams else "?"
            print(f"  Grupo {grp_key}: líder → {lider} ({teams[0]['pts']}pts)")
        actualizar_grupos_en_app(grupos)
    except Exception as e:
        print(f"  ⚠️  Error actualizando standings: {e}")

    # ── 3. Reentrenar el modelo con los datos frescos ─────────────
    print("\n🤖 PASO 3: Reentrenando el modelo con los datos actualizados...")
    run([sys.executable, "train_model_api.py"])
    
    # ── 3.5. Leer métricas generadas y reportar ──────────────────
    try:
        import json
        with open("modelo_metricas.json", "r") as f:
            metricas = json.load(f)
            acc = metricas.get("accuracy", 0) * 100
            f1 = metricas.get("f1_score_weighted", 0) * 100
            print(f"  ✅ Reentrenamiento completado con éxito.")
            print(f"  📊 Nueva precisión (Accuracy): {acc:.1f}%")
            print(f"  🎯 F1-Score: {f1:.1f}%")
    except Exception as e:
        print(f"  ⚠️ No se pudieron leer las métricas del modelo: {e}")

    # ── 4. Git add de todos los archivos actualizados ─────────────
    print("\n📦 PASO 4: Preparando archivos para el commit...")
    run([
        "git", "add",
        "api_matches_all.csv",
        "api_team_stats.csv",
        "api_wc2026_teams.csv",
        "api_wc2026_players.csv",
        "modelo_futbol.pkl",
        "modelo_metricas.json",
        "equipos_scores.csv",
        "app.py",
    ])

    # ── 5. Commit con fecha de hoy ────────────────────────────────
    print(f"\n💾 PASO 5: Haciendo commit ({hoy})...")
    commit_msg = f"Data+Model: Actualización automática {hoy} — datos y modelo reentrenado"
    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
        print("ℹ️  No hay cambios nuevos desde la última actualización.")
    elif result.returncode != 0:
        print(result.stderr)
        print("⚠️  Hubo un problema con el commit.")

    # ── 6. Push a GitHub ──────────────────────────────────────────
    print("\n🚀 PASO 6: Subiendo cambios a GitHub...")
    run(["git", "push", "origin", "main"])

    # ── Resumen final ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ ¡Actualización completa!")
    print("   · Datos frescos de la API descargados")
    print("   · Standings WC2026 actualizados en app.py")
    print("   · Modelo reentrenado con los datos más recientes")
    print("   · Cambios subidos a GitHub")
    print("   · Streamlit Cloud actualizará la app en ~1 minuto")
    print("=" * 60)


if __name__ == "__main__":
    main()

