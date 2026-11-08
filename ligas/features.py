"""Variables de cada partido calculadas solo con información previa al partido.

- Elo por liga (con ventaja de local, peso por diferencia de goles y
  regresión a la media al cambiar de temporada).
- Forma reciente: puntos y goles de los últimos 5/10 partidos.
- Rendimiento como local (para el local) y como visita (para la visita).
- Acumulado de la temporada y días de descanso.
"""
import numpy as np
import pandas as pd

from config import LIGAS, RAW

TERMINADO = {"FINISHED", "AWARDED"}
K, HFA, REGRESION = 20.0, 60.0, 0.8
VIGENCIA = pd.Timedelta(days=400)

GENERALES = ["f5_pts", "f10_pts", "f10_gf", "f10_gc"]
SEDE = ["sede5_pts", "sede5_dg"]
TEMPORADA = ["temp_ppg", "temp_dg"]
LADO = GENERALES + SEDE + TEMPORADA + ["temp_pj", "descanso"]

FEATURES = (
    ["elo_l", "elo_v", "elo_dif", "elo_p"]
    + [f"{c}_{s}" for s in ("l", "v") for c in LADO]
    + ["dif_f10_pts", "dif_f10_dg", "dif_temp_ppg", "dif_sede", "liga_local", "liga_empate"]
    + [f"liga_{c}" for c in LIGAS]
)
FEATURES_ELO = ["elo_dif", "liga_local", "liga_empate"]


def cargar_partidos():
    archivos = sorted(RAW.glob("*_[0-9][0-9][0-9][0-9].csv"))
    df = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
    df = df[df["liga"].isin(LIGAS)].dropna(subset=["local_id", "visita_id"])
    df["local_id"] = df["local_id"].astype(int)
    df["visita_id"] = df["visita_id"].astype(int)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    df = df.drop_duplicates("id").sort_values(["fecha", "id"]).reset_index(drop=True)
    df["terminado"] = df["estado"].isin(TERMINADO) & df["gl"].notna() & df["gv"].notna()
    # 0 = gana local, 1 = empate, 2 = gana visita
    df["y"] = np.where(df["gl"] > df["gv"], 0, np.where(df["gl"] == df["gv"], 1, 2))
    return df


def _factor_goles(dg):
    dg = abs(dg)
    if dg <= 1:
        return 1.0
    if dg == 2:
        return 1.5
    return (11 + dg) / 8


class Elo:
    def __init__(self):
        self.r, self.temp, self.liga, self.ultima = {}, {}, {}, {}

    def _activos(self, liga, fecha):
        return [v for e, v in self.r.items()
                if self.liga[e] == liga and fecha - self.ultima[e] <= VIGENCIA]

    def _media(self, liga, fecha):
        activos = self._activos(liga, fecha)
        return float(np.mean(activos)) if activos else 1500.0

    def _nuevo(self, liga, fecha):
        # Un ascendido arranca con el nivel de los 3 peores de la liga.
        activos = sorted(self._activos(liga, fecha))
        return float(np.mean(activos[:3])) if len(activos) >= 10 else 1500.0

    def rating(self, eq, liga, temporada, fecha):
        """Rating vigente antes de jugar, sin modificar el estado."""
        if eq not in self.r:
            return self._nuevo(liga, fecha)
        if self.temp[eq] != temporada:
            media = self._media(liga, fecha)
            return media + REGRESION * (self.r[eq] - media)
        return self.r[eq]

    def jugar(self, p):
        rl = self.rating(p.local_id, p.liga, p.temporada, p.fecha)
        rv = self.rating(p.visita_id, p.liga, p.temporada, p.fecha)
        esperado = 1 / (1 + 10 ** ((rv - rl - HFA) / 400))
        real = 1.0 if p.gl > p.gv else 0.5 if p.gl == p.gv else 0.0
        delta = K * _factor_goles(p.gl - p.gv) * (real - esperado)
        for eq, valor in ((p.local_id, rl + delta), (p.visita_id, rv - delta)):
            self.r[eq], self.temp[eq], self.liga[eq], self.ultima[eq] = valor, p.temporada, p.liga, p.fecha
        return rl, rv


def _larga(fin):
    """Una fila por equipo y partido, ordenadas por fecha."""
    base = {"pid": fin["id"], "fecha": fin["fecha"], "liga": fin["liga"], "temporada": fin["temporada"]}
    loc = pd.DataFrame({**base, "eq": fin["local_id"], "es_local": 1, "gf": fin["gl"], "gc": fin["gv"]})
    vis = pd.DataFrame({**base, "eq": fin["visita_id"], "es_local": 0, "gf": fin["gv"], "gc": fin["gl"]})
    lg = pd.concat([loc, vis], ignore_index=True)
    lg = lg.sort_values(["fecha", "pid", "es_local"], kind="stable").reset_index(drop=True)
    lg["pts"] = np.select([lg["gf"] > lg["gc"], lg["gf"] == lg["gc"]], [3, 1], 0)
    lg["dg"] = lg["gf"] - lg["gc"]
    return lg


def _rodante(grupos, col, n, minimo):
    return grupos[col].transform(lambda s: s.rolling(n, min_periods=minimo).mean())


def _despues(lg):
    """Valores de cada equipo *después* de cada partido."""
    g = lg.groupby("eq", sort=False)
    lg["f5_pts"] = _rodante(g, "pts", 5, 3)
    lg["f10_pts"] = _rodante(g, "pts", 10, 5)
    lg["f10_gf"] = _rodante(g, "gf", 10, 5)
    lg["f10_gc"] = _rodante(g, "gc", 10, 5)
    gs = lg.groupby(["eq", "es_local"], sort=False)
    lg["sede5_pts"] = _rodante(gs, "pts", 5, 2)
    lg["sede5_dg"] = _rodante(gs, "dg", 5, 2)
    gt = lg.groupby(["eq", "temporada"], sort=False)
    lg["temp_pj"] = gt.cumcount() + 1
    lg["temp_ppg"] = gt["pts"].transform(lambda s: s.expanding().mean())
    lg["temp_dg"] = gt["dg"].transform(lambda s: s.expanding().mean())
    return lg


def _antes(lg):
    """Valores de cada equipo *antes* de cada partido (el 'después' del partido previo)."""
    pre = pd.DataFrame(index=lg.index)
    pre[GENERALES] = lg.groupby("eq", sort=False)[GENERALES].shift(1).to_numpy()
    pre[SEDE] = lg.groupby(["eq", "es_local"], sort=False)[SEDE].shift(1).to_numpy()
    pre[TEMPORADA] = lg.groupby(["eq", "temporada"], sort=False)[TEMPORADA].shift(1).to_numpy()
    pre["temp_pj"] = lg.groupby(["eq", "temporada"], sort=False).cumcount()
    previo = lg.groupby("eq", sort=False)["fecha"].shift(1)
    pre["descanso"] = ((lg["fecha"] - previo).dt.total_seconds() / 86400).clip(upper=30)
    pre["pid"] = lg["pid"]
    pre["es_local"] = lg["es_local"]
    return pre


def derivadas(X):
    X["elo_dif"] = X["elo_l"] + HFA - X["elo_v"]
    X["elo_p"] = 1 / (1 + 10 ** (-X["elo_dif"] / 400))
    X["dif_f10_pts"] = X["f10_pts_l"] - X["f10_pts_v"]
    X["dif_f10_dg"] = (X["f10_gf_l"] - X["f10_gc_l"]) - (X["f10_gf_v"] - X["f10_gc_v"])
    X["dif_temp_ppg"] = X["temp_ppg_l"] - X["temp_ppg_v"]
    X["dif_sede"] = X["sede5_pts_l"] - X["sede5_pts_v"]
    for c in LIGAS:
        X[f"liga_{c}"] = (X["liga"] == c).astype(int)
    return X


class Estado:
    """Situación actual de cada equipo, para predecir partidos futuros o hipotéticos."""

    def __init__(self, elo, lg, X):
        self.elo = elo
        ult = lg.groupby("eq").tail(1).set_index("eq")
        self.general = ult[GENERALES]
        self.ultima_fecha = ult["fecha"]
        self.sede = lg.groupby(["eq", "es_local"]).tail(1).set_index(["eq", "es_local"])[SEDE]
        self.temp = lg.groupby(["eq", "temporada"]).tail(1).set_index(["eq", "temporada"])[TEMPORADA + ["temp_pj"]]
        self.tasas = {
            liga: {"liga_local": float((g["y"] == 0).mean()), "liga_empate": float((g["y"] == 1).mean())}
            for liga, g in X.groupby("liga").tail(380).groupby("liga")
        }

    def _lado(self, eq, es_local, temporada, fecha, descanso):
        d = {c: np.nan for c in LADO}
        if eq in self.general.index:
            d.update(self.general.loc[eq].to_dict())
        if (eq, es_local) in self.sede.index:
            d.update(self.sede.loc[(eq, es_local)].to_dict())
        if (eq, temporada) in self.temp.index:
            d.update(self.temp.loc[(eq, temporada)].to_dict())
        else:
            d["temp_pj"] = 0
        if descanso is None and eq in self.ultima_fecha.index:
            descanso = min((fecha - self.ultima_fecha[eq]).total_seconds() / 86400, 30)
        d["descanso"] = np.nan if descanso is None else descanso
        return d

    def partidos(self, filas):
        """filas: dicts con liga, temporada, fecha, local_id, visita_id y opcionalmente descanso."""
        out = []
        for f in filas:
            liga, temp, fecha = f["liga"], f["temporada"], f["fecha"]
            row = {
                "liga": liga,
                "elo_l": self.elo.rating(f["local_id"], liga, temp, fecha),
                "elo_v": self.elo.rating(f["visita_id"], liga, temp, fecha),
                **self.tasas.get(liga, {"liga_local": np.nan, "liga_empate": np.nan}),
            }
            for sufijo, eq, es_local in (("l", f["local_id"], 1), ("v", f["visita_id"], 0)):
                for c, v in self._lado(eq, es_local, temp, fecha, f.get("descanso")).items():
                    row[f"{c}_{sufijo}"] = v
            out.append(row)
        return derivadas(pd.DataFrame(out))


def construir(df):
    """Devuelve (X, estado): una fila con variables por partido terminado, y el estado actual."""
    fin = df[df["terminado"]].reset_index(drop=True).copy()
    elo = Elo()
    fin[["elo_l", "elo_v"]] = np.array([elo.jugar(p) for p in fin.itertuples(index=False)])

    lg = _despues(_larga(fin))
    pre = _antes(lg)
    pl = pre[pre["es_local"] == 1].drop(columns="es_local").set_index("pid").add_suffix("_l")
    pv = pre[pre["es_local"] == 0].drop(columns="es_local").set_index("pid").add_suffix("_v")
    X = fin.join(pl, on="id").join(pv, on="id")

    for nombre, clase in (("liga_local", 0), ("liga_empate", 1)):
        serie = (X["y"] == clase).astype(float)
        X[nombre] = serie.groupby(X["liga"]).transform(lambda s: s.shift(1).rolling(380, min_periods=50).mean())

    X = derivadas(X)
    return X, Estado(elo, lg, X)
