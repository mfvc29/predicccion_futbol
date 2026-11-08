"""Descarga partidos, tablas y equipos desde football-data.org (plan gratuito).

Las temporadas ya terminadas quedan guardadas en ligas/data y no se vuelven a pedir;
en cada corrida solo se baja la temporada en curso y la tabla de posiciones.
"""
import json
import os
import time

import pandas as pd
import requests

from config import LIGAS, RAW, TEMPORADAS_ATRAS

API = "https://api.football-data.org/v4"
PAUSA = 6.5  # el plan gratuito permite 10 consultas por minuto
EN_CURSO = {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"}

_sesion = requests.Session()
_ultima = 0.0


def get(ruta, params=None):
    global _ultima
    clave = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not clave:
        raise SystemExit("Falta la variable de entorno FOOTBALL_DATA_API_KEY")
    for _ in range(4):
        espera = PAUSA - (time.monotonic() - _ultima)
        if espera > 0:
            time.sleep(espera)
        _ultima = time.monotonic()
        try:
            r = _sesion.get(API + ruta, headers={"X-Auth-Token": clave}, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  error de red en {ruta}: {e}")
            time.sleep(10)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(int(r.headers.get("X-RequestCounter-Reset", 60)) + 1)
            continue
        if r.status_code in (400, 403, 404):
            print(f"  sin acceso a {ruta} {params or ''} (HTTP {r.status_code})")
            return None
        time.sleep(10)
    raise RuntimeError(f"No se pudo descargar {ruta}")


def _equipo(t, equipos):
    if not t or t.get("id") is None:
        return None
    equipos[str(t["id"])] = {
        "nombre": t.get("shortName") or t.get("name"),
        "completo": t.get("name"),
        "tla": t.get("tla"),
        "escudo": t.get("crest"),
    }
    return t["id"]


def _temporada(code, temporada, equipos):
    data = get(f"/competitions/{code}/matches", {"season": temporada})
    if not data:
        return None
    filas = []
    for m in data.get("matches", []):
        local = _equipo(m.get("homeTeam"), equipos)
        visita = _equipo(m.get("awayTeam"), equipos)
        if local is None or visita is None:
            continue
        ft = (m.get("score") or {}).get("fullTime") or {}
        filas.append({
            "id": m["id"],
            "liga": code,
            "temporada": temporada,
            "fecha": m["utcDate"],
            "jornada": m.get("matchday"),
            "fase": m.get("stage"),
            "estado": m["status"],
            "local_id": local,
            "visita_id": visita,
            "gl": ft.get("home"),
            "gv": ft.get("away"),
        })
    return pd.DataFrame(filas)


def _tabla(code, equipos):
    data = get(f"/competitions/{code}/standings")
    if not data:
        return None
    total = next((s for s in data.get("standings", []) if s.get("type") == "TOTAL"), None)
    if not total:
        return None
    filas = []
    for r in total["table"]:
        filas.append({
            "pos": r["position"], "id": _equipo(r["team"], equipos),
            "pj": r["playedGames"], "g": r["won"], "e": r["draw"], "p": r["lost"],
            "gf": r["goalsFor"], "gc": r["goalsAgainst"], "dg": r["goalDifference"], "pts": r["points"],
        })
    return filas


def descargar():
    RAW.mkdir(parents=True, exist_ok=True)
    ruta_eq, ruta_ligas = RAW / "equipos.json", RAW / "ligas.json"
    equipos = json.loads(ruta_eq.read_text()) if ruta_eq.exists() else {}
    ligas = json.loads(ruta_ligas.read_text()) if ruta_ligas.exists() else {}

    for code, cfg in LIGAS.items():
        print(f"→ {cfg['nombre']} ({code})")
        comp = get(f"/competitions/{code}")
        if not comp:
            continue
        cs = comp["currentSeason"]
        actual = int(cs["startDate"][:4])
        ligas[code] = {
            "temporada": actual, "inicio": cs["startDate"], "fin": cs["endDate"],
            "jornada": cs.get("currentMatchday"), "emblema": comp.get("emblem"),
        }
        for temp in range(actual - TEMPORADAS_ATRAS, actual + 1):
            archivo = RAW / f"{code}_{temp}.csv"
            if temp < actual and archivo.exists():
                if not pd.read_csv(archivo)["estado"].isin(EN_CURSO).any():
                    continue
            df = _temporada(code, temp, equipos)
            if df is not None and len(df):
                df.to_csv(archivo, index=False)
                print(f"  {temp}: {int((df['estado'] == 'FINISHED').sum())} jugados de {len(df)}")
        tabla = _tabla(code, equipos)
        if tabla:
            (RAW / f"{code}_tabla.json").write_text(json.dumps(tabla))

    ruta_eq.write_text(json.dumps(equipos, ensure_ascii=False, indent=0, sort_keys=True))
    ruta_ligas.write_text(json.dumps(ligas, indent=1))


if __name__ == "__main__":
    descargar()
