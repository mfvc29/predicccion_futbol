"""Genera los JSON que lee la web (web/data)."""
import json

import numpy as np
import pandas as pd

from config import LIGAS, RAW, WEB_DATA

PROGRAMADO = {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"}
SIN_JUGAR = PROGRAMADO | {"POSTPONED", "SUSPENDED"}
SIMULACIONES = 10000


def _limpio(o):
    if isinstance(o, dict):
        return {str(k): _limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpio(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _guardar(nombre, obj):
    texto = json.dumps(_limpio(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (WEB_DATA / nombre).write_text(texto, encoding="utf-8")


def _pct(fila):
    return [round(float(x) * 100, 1) for x in fila]


def _entero(x):
    return int(x) if pd.notna(x) else None


def _forma(jugados, eq):
    letras = []
    for r in jugados[(jugados["local_id"] == eq) | (jugados["visita_id"] == eq)].tail(5).itertuples():
        if r.gl == r.gv:
            letras.append("E")
        else:
            letras.append("G" if (r.gl > r.gv) == (r.local_id == eq) else "P")
    return "".join(letras)


def _simular(ids, tabla, restantes, prob, top, descenso):
    """Monte Carlo del resto de la temporada a partir de los puntos actuales."""
    n_eq = len(ids)
    idx = {e: i for i, e in enumerate(ids)}
    por_id = {r["id"]: r for r in tabla}
    pts0 = np.array([por_id.get(e, {}).get("pts", 0) for e in ids], dtype=float)
    dg0 = np.array([por_id.get(e, {}).get("dg", 0) for e in ids], dtype=float)
    rng = np.random.default_rng(0)
    pts = np.tile(pts0, (SIMULACIONES, 1))
    if restantes:
        acum = np.cumsum(np.array([prob[p] for p in restantes]), axis=1)
        u = rng.random((SIMULACIONES, len(restantes)))
        res = (u[:, :, None] > acum[None, :, :2]).sum(axis=2)
        for k, (l, v) in enumerate(restantes):
            r = res[:, k]
            pts[:, idx[l]] += np.where(r == 0, 3, np.where(r == 1, 1, 0))
            pts[:, idx[v]] += np.where(r == 2, 3, np.where(r == 1, 1, 0))
    clave = pts + dg0 * 1e-3 + rng.random(pts.shape) * 1e-6
    orden = np.argsort(-clave, axis=1)
    pos = np.empty_like(orden)
    pos[np.arange(SIMULACIONES)[:, None], orden] = np.arange(n_eq)
    filas = []
    for e in ids:
        i = idx[e]
        filas.append({
            "id": e,
            "pts": int(pts0[i]),
            "pts_esp": round(float(pts[:, i].mean()), 1),
            "campeon": round(float((pos[:, i] == 0).mean()) * 100, 1),
            "top": round(float((pos[:, i] < top).mean()) * 100, 1),
            "descenso": round(float((pos[:, i] >= n_eq - descenso).mean()) * 100, 1),
        })
    return sorted(filas, key=lambda f: (-f["pts_esp"], -f["campeon"]))


def _liga(code, cfg, info, df, X, estado, res, equipos, ahora):
    temp = info["temporada"]
    modelo = res["modelo"]
    partidos = df[(df["liga"] == code) & (df["temporada"] == temp)]
    jugados = partidos[partidos["terminado"]]
    ids = sorted(set(partidos["local_id"]) | set(partidos["visita_id"]),
                 key=lambda e: (equipos.get(str(e), {}).get("nombre") or "").lower())
    ruta_tabla = RAW / f"{code}_tabla.json"
    tabla = json.loads(ruta_tabla.read_text()) if ruta_tabla.exists() else []

    lista = []
    for e in ids:
        d = equipos.get(str(e), {})
        lista.append({
            "id": e, "nombre": d.get("nombre") or str(e), "tla": d.get("tla"), "escudo": d.get("escudo"),
            "elo": round(estado.elo.rating(e, code, temp, ahora)), "forma": _forma(jugados, e),
        })

    # Todos los cruces posibles con el estado actual (simulador y proyección).
    pares = [(a, b) for a in ids for b in ids if a != b]
    prob = {}
    if pares:
        F = estado.partidos([{"liga": code, "temporada": temp, "fecha": ahora,
                              "local_id": a, "visita_id": b, "descanso": 7.0} for a, b in pares])
        prob = dict(zip(pares, modelo.predict_proba(F)))
    matriz = [[_pct(prob[(a, b)]) if a != b else None for b in ids] for a in ids]

    pend = partidos[partidos["estado"].isin(PROGRAMADO)
                    & (partidos["fecha"] >= ahora - pd.Timedelta(hours=3))].sort_values("fecha")
    ventana = pend[pend["fecha"] <= ahora + pd.Timedelta(days=21)]
    if len(ventana) < 5:
        ventana = pend.head(10)
    ventana = ventana.head(60)
    proximos = []
    if len(ventana):
        F = estado.partidos([{"liga": code, "temporada": temp, "fecha": r.fecha,
                              "local_id": r.local_id, "visita_id": r.visita_id} for r in ventana.itertuples()])
        for r, p in zip(ventana.itertuples(), modelo.predict_proba(F)):
            proximos.append({"id": r.id, "fecha": r.fecha.isoformat(), "jornada": _entero(r.jornada),
                             "estado": r.estado, "l": r.local_id, "v": r.visita_id, "p": _pct(p)})

    pa = res["pred_actual"]
    resultados = []
    for r in jugados.sort_values("fecha", ascending=False).head(40).itertuples():
        if r.id not in pa.index:
            continue
        p = pa.loc[r.id].to_numpy()
        resultados.append({"fecha": r.fecha.isoformat(), "jornada": _entero(r.jornada),
                           "l": r.local_id, "v": r.visita_id, "gl": int(r.gl), "gv": int(r.gv),
                           "p": _pct(p), "real": int(r.y), "pred": int(p.argmax())})

    restantes = [(r.local_id, r.visita_id) for r in partidos[partidos["estado"].isin(SIN_JUGAR)].itertuples()
                 if (r.local_id, r.visita_id) in prob]
    proyeccion = _simular(ids, tabla, restantes, prob, cfg["top"], cfg["descenso"]) if ids else []

    _guardar(f"{code}.json", {
        "codigo": code,
        "zonas": {"top": cfg["top"], "descenso": cfg["descenso"]},
        "equipos": lista,
        "tabla": tabla,
        "proximos": proximos,
        "resultados": resultados,
        "temporada": res["actual_liga"].get(code),
        "validacion": res["validacion_liga"].get(code),
        "matriz": {"ids": ids, "p": matriz},
        "proyeccion": {"restantes": len(restantes), "simulaciones": SIMULACIONES, "equipos": proyeccion},
    })


def _etiqueta(info):
    a, b = info["inicio"][:4], info["fin"][:4]
    return a if a == b else f"{a}/{b[2:]}"


def exportar(df, X, estado, res, info_ligas):
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    equipos = json.loads((RAW / "equipos.json").read_text())
    ahora = pd.Timestamp.now(tz="UTC")
    ligas = []
    for code, cfg in LIGAS.items():
        info = info_ligas.get(code)
        if not info:
            continue
        _liga(code, cfg, info, df, X, estado, res, equipos, ahora)
        ligas.append({"codigo": code, "nombre": cfg["nombre"], "pais": cfg["pais"], "region": cfg["region"],
                      "emblema": info.get("emblema"), "temporada": _etiqueta(info), "jornada": info.get("jornada")})
        print(f"  web/data/{code}.json")
    _guardar("meta.json", {
        "actualizado": ahora.isoformat(),
        "modelo": res["elegido"],
        "partidos": res["n"],
        "ligas": ligas,
        "validacion": {"modelos": res["validacion"], "total": res["validacion_total"]},
        "temporada": res["actual_total"],
    })
    print("  web/data/meta.json")
