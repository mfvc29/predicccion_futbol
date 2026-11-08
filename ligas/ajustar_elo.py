"""Busca los parámetros del Elo que mejor predicen la temporada de validación.

Prueba todas las combinaciones de la grilla con el modelo "Elo" (diferencia de Elo +
tasas de la liga → regresión logística), entrenando con las temporadas anteriores a la
pasada y midiendo el log-loss en la pasada. La temporada actual no se usa, así que su
acierto sigue siendo una medición honesta.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from entrenar import CALENTAMIENTO
from features import ELO_DEFECTO, Elo, tasas_liga

GRILLA = {
    "k": [10, 15, 20, 25, 30, 40],
    "hfa": [30, 45, 60, 75, 90, 110],
    "regresion": [0.6, 0.7, 0.8, 0.9, 1.0],
}
CALENTAMIENTOS = [120, 240, 365]


def buscar(df, temporada_actual):
    fin = df[df["terminado"] & df["liga"].isin(temporada_actual)].reset_index(drop=True).copy()
    fin = tasas_liga(fin)
    filas = list(fin[["local_id", "visita_id", "liga", "temporada", "fecha", "gl", "gv"]].itertuples(index=False))
    actual = fin["liga"].map(temporada_actual).to_numpy()
    temporada = fin["temporada"].to_numpy()
    dias = (fin["fecha"] - fin.groupby("liga")["fecha"].transform("min")).dt.days.to_numpy()
    va = temporada == actual - 1
    y = fin["y"].to_numpy()
    tasas = fin[["liga_local", "liga_empate"]].to_numpy()

    resultados = []
    combinaciones = list(itertools.product(GRILLA["k"], GRILLA["hfa"], GRILLA["regresion"]))
    for k, hfa, regresion in combinaciones:
        elo = Elo(k, hfa, regresion)
        previos = np.array([elo.jugar(p) for p in filas])
        Xm = np.column_stack([previos[:, 0] + hfa - previos[:, 1], tasas])
        for cal in CALENTAMIENTOS:
            tr = (dias >= cal) & (temporada <= actual - 2)
            modelo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(max_iter=2000))
            modelo.fit(Xm[tr], y[tr])
            ll = log_loss(y[va], modelo.predict_proba(Xm[va]), labels=[0, 1, 2])
            resultados.append({"k": k, "hfa": hfa, "regresion": regresion, "calentamiento": cal, "logloss": ll})

    tabla = pd.DataFrame(resultados).sort_values("logloss").reset_index(drop=True)
    mejor = tabla.iloc[0]
    defecto = tabla[(tabla["k"] == ELO_DEFECTO["k"]) & (tabla["hfa"] == ELO_DEFECTO["hfa"])
                    & (tabla["regresion"] == ELO_DEFECTO["regresion"]) & (tabla["calentamiento"] == CALENTAMIENTO)]
    print(f"  {len(tabla)} combinaciones probadas. Mejores 5:")
    print(tabla.head(5).to_string(index=False))
    return {
        "k": float(mejor["k"]),
        "hfa": float(mejor["hfa"]),
        "regresion": float(mejor["regresion"]),
        "calentamiento": int(mejor["calentamiento"]),
        "logloss": round(float(mejor["logloss"]), 4),
        "logloss_defecto": round(float(defecto["logloss"].iloc[0]), 4) if len(defecto) else None,
        "probadas": int(len(tabla)),
    }
