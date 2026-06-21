"""
fetch_api_data.py
==================
Descarga TODOS los datos disponibles de la API football-data.org gratuita:
  1. Partidos de todas las competiciones disponibles (WC, EC, PL, PD, SA, BL1, FL1, CL)
  2. Equipos del Mundial 2026 con plantillas completas
  3. Calcula métricas reales por equipo desde resultados reales (no CSVs con ceros)

Genera:
    api_matches_all.csv       → todos los partidos de la API
    api_team_stats.csv        → estadísticas reales por equipo (wins, goles, etc.)
    api_wc2026_players.csv    → jugadores del Mundial 2026 con datos reales
    api_wc2026_teams.csv      → equipos del WC2026 con coach, colores, etc.

Uso:
    python fetch_api_data.py
"""

import time
import json
import csv
import os
import requests
from datetime import datetime
from collections import defaultdict

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "8888a51b18884aebad14216052644c2b")
BASE_URL = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": API_KEY}
DELAY    = 7  # segundos entre llamadas (plan free: ~10/min)

# ── Competencias disponibles en plan gratuito ─────────────────────────────────
COMPETITIONS = {
    "WC":  "FIFA World Cup 2026",
    "EC":  "UEFA Euro 2024",
    "PL":  "Premier League",
    "PD":  "La Liga",
    "SA":  "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "CL":  "Champions League",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "ELC": "Championship",
    "BSA": "Brasileiro Série A",
}

def get(endpoint, params=None):
    """GET con rate limiting y reintentos respetando Retry-After header."""
    url = f"{BASE_URL}{endpoint}"
    for intento in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            # Respetar Retry-After si lo hay
            if "X-RequestCounter-Reset" in r.headers:
                pass  # solo informativo
            if r.status_code == 200:
                return r.json(), r.headers
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"  ⏳ Rate limit. Esperando {wait}s...")
                time.sleep(wait + 1)
            elif r.status_code == 403:
                print(f"  ⛔ Sin acceso: {url}")
                return {}, {}
            else:
                print(f"  HTTP {r.status_code}: {url}")
                return {}, {}
        except Exception as e:
            print(f"  Error red (intento {intento+1}): {e}")
            time.sleep(5)
    return {}, {}


def calcular_edad(dob_str):
    if not dob_str:
        return None
    try:
        dob = datetime.strptime(dob_str[:10], "%Y-%m-%d")
        hoy = datetime.now()
        return int((hoy - dob).days / 365.25)
    except:
        return None


def guardar_csv(filename, rows, fieldnames=None):
    if not rows:
        print(f"  ⚠️  Sin datos para {filename}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ {len(rows):,} filas → {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. DESCARGAR PARTIDOS DE TODAS LAS COMPETENCIAS
# ═════════════════════════════════════════════════════════════════════════════
def fetch_all_matches():
    print("\n📡 Descargando partidos de todas las competencias...")
    all_matches = []
    
    for code, name in COMPETITIONS.items():
        print(f"  → {name} ({code})...")
        data, headers = get(f"/competitions/{code}/matches")
        if not data:
            time.sleep(DELAY)
            continue
        
        matches = data.get("matches", [])
        finished = [m for m in matches if m.get("status") == "FINISHED"]
        
        for m in finished:
            score = m.get("score") or {}
            ft    = score.get("fullTime") or {}
            ht    = score.get("halfTime") or {}
            home_goals = ft.get("home", 0) or 0
            away_goals = ft.get("away", 0) or 0
            winner     = score.get("winner")
            
            all_matches.append({
                "competition":    code,
                "competition_name": name,
                "match_id":       m.get("id"),
                "date":           (m.get("utcDate") or "")[:10],
                "stage":          m.get("stage"),
                "group":          m.get("group"),
                "matchday":       m.get("matchday"),
                "home_team_id":   (m.get("homeTeam") or {}).get("id"),
                "home_team":      (m.get("homeTeam") or {}).get("name"),
                "away_team_id":   (m.get("awayTeam") or {}).get("id"),
                "away_team":      (m.get("awayTeam") or {}).get("name"),
                "home_score":     home_goals,
                "away_score":     away_goals,
                "home_score_ht":  ht.get("home", 0) or 0,
                "away_score_ht":  ht.get("away", 0) or 0,
                "winner":         winner,
                "home_win":       int(winner == "HOME_TEAM"),
                "draw":           int(winner == "DRAW"),
                "away_win":       int(winner == "AWAY_TEAM"),
                "total_goals":    home_goals + away_goals,
                "btts":           int(home_goals > 0 and away_goals > 0),
                "duration":       score.get("duration"),
            })
        
        print(f"       {len(finished):4} terminados / {len(matches)} total")
        time.sleep(DELAY)
    
    print(f"\n  🏁 Total partidos reales: {len(all_matches):,}")
    return all_matches


# ═════════════════════════════════════════════════════════════════════════════
# 2. CALCULAR ESTADÍSTICAS REALES POR EQUIPO
# ═════════════════════════════════════════════════════════════════════════════
def calcular_team_stats(matches):
    print("\n📊 Calculando estadísticas reales por equipo...")
    stats = defaultdict(lambda: {
        "partidos": 0, "wins": 0, "draws": 0, "losses": 0,
        "goles_favor": 0, "goles_contra": 0,
        "clean_sheets": 0, "failed_to_score": 0,
        "comps": set(), "match_ids": []
    })
    
    for m in matches:
        ht = m["home_team"]
        at = m["away_team"]
        hs = m["home_score"]
        as_ = m["away_score"]
        
        if not ht or not at:
            continue
        
        # Local
        stats[ht]["partidos"] += 1
        stats[ht]["goles_favor"] += hs
        stats[ht]["goles_contra"] += as_
        stats[ht]["comps"].add(m["competition"])
        if m["home_win"]: stats[ht]["wins"] += 1
        elif m["draw"]:   stats[ht]["draws"] += 1
        else:             stats[ht]["losses"] += 1
        if as_ == 0:      stats[ht]["clean_sheets"] += 1
        if hs == 0:       stats[ht]["failed_to_score"] += 1
        
        # Visitante
        stats[at]["partidos"] += 1
        stats[at]["goles_favor"] += as_
        stats[at]["goles_contra"] += hs
        stats[at]["comps"].add(m["competition"])
        if m["away_win"]: stats[at]["wins"] += 1
        elif m["draw"]:   stats[at]["draws"] += 1
        else:             stats[at]["losses"] += 1
        if hs == 0:       stats[at]["clean_sheets"] += 1
        if as_ == 0:      stats[at]["failed_to_score"] += 1
    
    rows = []
    for team, s in stats.items():
        p = s["partidos"]
        if p == 0:
            continue
        win_rate   = s["wins"] / p
        pts_p_game = (s["wins"]*3 + s["draws"]) / p
        gd         = s["goles_favor"] - s["goles_contra"]
        clean_rate = s["clean_sheets"] / p
        score_rate = 1 - (s["failed_to_score"] / p)
        
        # Fuerza Score: media ponderada de múltiples métricas
        fuerza = (
            win_rate * 40 +
            pts_p_game * 10 +
            min(gd / max(p, 1), 2) * 10 +  # capped at 2 GD per game
            clean_rate * 10 +
            score_rate * 10 +
            min(s["goles_favor"] / max(p, 1), 3) * 10  # capped at 3 goals per game
        )
        
        rows.append({
            "team":              team,
            "partidos":          p,
            "wins":              s["wins"],
            "draws":             s["draws"],
            "losses":            s["losses"],
            "goles_favor":       s["goles_favor"],
            "goles_contra":      s["goles_contra"],
            "diferencia_goles":  gd,
            "win_rate":          round(win_rate, 4),
            "pts_per_game":      round(pts_p_game, 4),
            "goles_por_partido": round(s["goles_favor"] / p, 3),
            "goles_recibidos_partido": round(s["goles_contra"] / p, 3),
            "clean_sheet_rate":  round(clean_rate, 4),
            "score_rate":        round(score_rate, 4),
            "Fuerza_Score":      round(fuerza, 4),
            "competiciones":     "|".join(sorted(s["comps"])),
        })
    
    rows.sort(key=lambda x: x["Fuerza_Score"], reverse=True)
    print(f"  🏆 {len(rows)} equipos con datos reales")
    if rows[:3]:
        print("  Top 3:")
        for r in rows[:3]:
            print(f"    {r['team']:25} Fuerza={r['Fuerza_Score']:.2f} WinRate={r['win_rate']:.2%} Partidos={r['partidos']}")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# 3. DESCARGAR EQUIPOS Y PLANTILLAS DEL MUNDIAL 2026
# ═════════════════════════════════════════════════════════════════════════════
def fetch_wc2026_teams_and_players():
    print("\n🌍 Descargando plantillas del Mundial 2026...")
    data, _ = get("/competitions/WC/teams")
    if not data:
        return [], []
    
    teams = data.get("teams", [])
    print(f"  48 equipos confirmados: {len(teams)}")
    
    all_players = []
    all_teams   = []
    
    for t in teams:
        squad = t.get("squad") or []
        coach = t.get("coach") or {}
        
        team_row = {
            "team_id":      t.get("id"),
            "team_name":    t.get("name"),
            "short_name":   t.get("shortName"),
            "tla":          t.get("tla"),
            "founded":      t.get("founded"),
            "club_colors":  t.get("clubColors"),
            "venue":        t.get("venue"),
            "website":      t.get("website"),
            "coach_name":   coach.get("name"),
            "coach_nationality": coach.get("nationality"),
            "coach_dob":    coach.get("dateOfBirth"),
            "squad_size":   len(squad),
        }
        all_teams.append(team_row)
        
        for p in squad:
            dob = p.get("dateOfBirth")
            edad = calcular_edad(dob)
            all_players.append({
                "team_id":      t.get("id"),
                "team_name":    t.get("name"),
                "tla":          t.get("tla"),
                "player_id":    p.get("id"),
                "player_name":  p.get("name"),
                "position":     p.get("position"),
                "shirt_number": p.get("shirtNumber"),
                "date_of_birth": dob,
                "edad":         edad,
                "nationality":  p.get("nationality"),
            })
    
    print(f"  👥 {len(all_players)} jugadores descargados")
    return all_teams, all_players


# ═════════════════════════════════════════════════════════════════════════════
# 4. ENRIQUECER JUGADORES CON DATOS DE SU CLUB (si posible)
# ═════════════════════════════════════════════════════════════════════════════
def enrich_players_with_clubs(players):
    """
    Intenta obtener el club actual de cada jugador vía /persons/{id}.
    Respeta el rate limit: 10 req/min = 1 req cada 6.5s.
    Como hay ~700 jugadores, esto tomaría ~1.5 horas.
    Aquí solo enrichimos los primeros N o una muestra representativa.
    """
    print("\n⚽ Enriqueciendo datos de jugadores con club actual...")
    print("   (Respetando rate limit de la API gratuita: 10 req/min)")
    
    enriched = 0
    for i, p in enumerate(players):
        pid = p.get("player_id")
        if not pid:
            continue
        
        data, _ = get(f"/persons/{pid}")
        if data:
            current_team = data.get("currentTeam") or {}
            p["club_name"]      = current_team.get("name")
            p["club_id"]        = current_team.get("id")
            p["club_short"]     = current_team.get("shortName")
            p["club_tla"]       = current_team.get("tla")
            # También contratos si los hay
            contract = current_team.get("contract") or {}
            p["contract_start"] = contract.get("start")
            p["contract_until"] = contract.get("until")
            enriched += 1
        
        if (i + 1) % 10 == 0:
            print(f"   Progreso: {i+1}/{len(players)} jugadores ({enriched} enriquecidos)...")
        
        time.sleep(DELAY)
    
    print(f"   ✅ {enriched} jugadores con datos de club")
    return players


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  FETCH API DATA — football-data.org v4")
    print("  Descargando datos reales para el modelo de predicción")
    print("=" * 65)
    
    # 1. Partidos de todas las competencias
    matches = fetch_all_matches()
    guardar_csv("api_matches_all.csv", matches)
    
    # 2. Estadísticas reales por equipo
    team_stats = calcular_team_stats(matches)
    guardar_csv("api_team_stats.csv", team_stats)
    
    # 3. Equipos y plantillas WC2026
    wc_teams, wc_players = fetch_wc2026_teams_and_players()
    time.sleep(DELAY)
    guardar_csv("api_wc2026_teams.csv", wc_teams)
    guardar_csv("api_wc2026_players.csv", wc_players)
    
    # 4. Enriquecer jugadores (respetando rate limit)
    print("\n💡 Los jugadores básicos ya están descargados.")
    print("   Para enriquecer con club actual ejecuta:")
    print("   python fetch_api_data.py --enrich-players")
    print("   (tarda ~1.5h por el rate limit del plan gratuito)")
    
    import sys
    if "--enrich-players" in sys.argv:
        wc_players = enrich_players_with_clubs(wc_players)
        guardar_csv("api_wc2026_players.csv", wc_players)
    
    print("\n" + "=" * 65)
    print("✅ ¡Datos API descargados correctamente!")
    print("   Siguiente paso: python train_model_api.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
