"""Entrena y evalúa el modelo 1X2 con validación temporal (nunca se usa el futuro).

- Entrenamiento: temporadas anteriores a la pasada.
- Validación: temporada pasada completa → se elige el modelo con menor log-loss.
- Temporada actual: se predice con un modelo que no la vio (predicciones honestas).
- Modelo final: se reentrena con todo para predecir los próximos partidos.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, FEATURES_ELO

CALENTAMIENTO = 120  # días iniciales de cada liga que solo sirven para que el Elo arranque


def _base(nombre):
    if nombre == "Elo":
        return FEATURES_ELO, make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                           LogisticRegression(max_iter=2000))
    if nombre == "Regresión logística":
        return FEATURES, make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       LogisticRegression(C=0.1, max_iter=2000))
    if nombre == "Gradient boosting":
        return FEATURES, HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=200, max_depth=3, min_samples_leaf=50,
            l2_regularization=1.0, random_state=0)
    raise KeyError(nombre)


OPCIONES = {
    "Elo": ["Elo"],
    "Regresión logística": ["Regresión logística"],
    "Gradient boosting": ["Gradient boosting"],
    "Mezcla logística + boosting": ["Regresión logística", "Gradient boosting"],
}


class Modelo:
    def __init__(self, nombre):
        self.nombre = nombre
        self.partes = []

    def fit(self, X, y):
        self.partes = []
        for base in OPCIONES[self.nombre]:
            cols, m = _base(base)
            m.fit(X[cols], y)
            self.partes.append((cols, m))
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X[cols]) for cols, m in self.partes], axis=0)


def evaluar(y, P):
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return {"n": 0}
    P = np.clip(np.asarray(P, dtype=float), 1e-6, 1)
    P = P / P.sum(axis=1, keepdims=True)
    return {
        "n": int(len(y)),
        "acierto": round(float((P.argmax(axis=1) == y).mean()), 4),
        "logloss": round(float(log_loss(y, P, labels=[0, 1, 2])), 4),
        "brier": round(float(((P - np.eye(3)[y]) ** 2).sum(axis=1).mean()), 4),
    }


def _frecuencias(train, objetivo):
    freq = (train.groupby("liga")["y"].value_counts(normalize=True)
            .unstack().reindex(columns=[0, 1, 2]).fillna(0))
    return freq.reindex(objetivo["liga"]).fillna(1 / 3).to_numpy()


def _con_bases(X, mask, P, train_mask):
    """Métricas del modelo y de las dos referencias simples, en total y por liga."""
    sub = X[mask]
    y = sub["y"].to_numpy()
    P_freq = _frecuencias(X[train_mask], sub)

    def bloque(m):
        return {
            "modelo": evaluar(y[m], P[m]),
            "frecuencia": evaluar(y[m], P_freq[m]),
            "siempre_local": round(float((y[m] == 0).mean()), 4) if m.any() else None,
        }

    total = bloque(np.ones(len(sub), dtype=bool))
    por_liga = {c: bloque((sub["liga"] == c).to_numpy()) for c in sorted(sub["liga"].unique())}
    return total, por_liga


def entrenar(X, temporada_actual):
    X = X[X["liga"].isin(temporada_actual)].reset_index(drop=True)
    actual = X["liga"].map(temporada_actual)
    inicio = X.groupby("liga")["fecha"].transform("min")
    util = (X["fecha"] >= inicio + pd.Timedelta(days=CALENTAMIENTO)).to_numpy()
    tr = util & (X["temporada"] <= actual - 2).to_numpy()
    va = util & (X["temporada"] == actual - 1).to_numpy()
    te = (X["temporada"] == actual).to_numpy()
    y = X["y"].to_numpy()
    print(f"Partidos — entrenamiento: {tr.sum()}, validación: {va.sum()}, temporada actual: {te.sum()}")

    validacion = {}
    for nombre in OPCIONES:
        P = Modelo(nombre).fit(X[tr], y[tr]).predict_proba(X[va])
        validacion[nombre] = evaluar(y[va], P)
        print(f"  {nombre:30} {validacion[nombre]}")
    validacion["Frecuencia histórica"] = evaluar(y[va], _frecuencias(X[tr], X[va]))
    validacion["Siempre local"] = {"n": int(va.sum()), "acierto": round(float((y[va] == 0).mean()), 4)}
    elegido = min(OPCIONES, key=lambda n: validacion[n]["logloss"])
    print(f"Modelo elegido: {elegido}")

    P_va = Modelo(elegido).fit(X[tr], y[tr]).predict_proba(X[va])
    val_total, val_liga = _con_bases(X, va, P_va, tr)

    trva = tr | va
    if te.any():
        P_te = Modelo(elegido).fit(X[trva], y[trva]).predict_proba(X[te])
        act_total, act_liga = _con_bases(X, te, P_te, trva)
    else:
        P_te = np.zeros((0, 3))
        act_total, act_liga = {}, {}
    pred_actual = pd.DataFrame(P_te, columns=[0, 1, 2], index=X.loc[te, "id"].to_numpy())

    final = Modelo(elegido).fit(X[util], y[util])
    return {
        "elegido": elegido,
        "modelo": final,
        "validacion": validacion,
        "validacion_total": val_total,
        "validacion_liga": val_liga,
        "actual_total": act_total,
        "actual_liga": act_liga,
        "pred_actual": pred_actual,
        "n": {"entrenamiento": int(tr.sum()), "validacion": int(va.sum()),
              "actual": int(te.sum()), "final": int(util.sum())},
    }
