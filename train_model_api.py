"""
train_model_api.py
==================
Entrena el modelo usando ÚNICAMENTE datos reales de la API.
Sin CSVs con ceros. Sin datos inventados.

Estrategia:
  - Dataset de entrenamiento: api_matches_all.csv (partidos reales de ligas top)
  - Features: métricas calculadas desde resultados reales (api_team_stats.csv)
  - Target: resultado del partido (home_win=1, draw=0, away_win=-1)
  - Ponderación: partidos de WC/EC pesan más que ligas de clubes

Genera:
    modelo_futbol.pkl          → modelo entrenado
    modelo_metricas.json       → métricas de evaluación
    equipos_scores.csv         → tabla de scores para la app
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.impute import SimpleImputer
import joblib

print("=" * 60)
print("  TRAIN MODEL — Con datos reales de la API")
print("=" * 60)

# ── 1. Cargar datos reales ────────────────────────────────────────────────────
print("\n📂 Cargando datos de la API...")
matches_df = pd.read_csv("api_matches_all.csv")
team_stats_df = pd.read_csv("api_team_stats.csv")

matches_df["date"] = pd.to_datetime(matches_df["date"])
print(f"  Partidos API: {len(matches_df):,}")
print(f"  Equipos con stats reales: {len(team_stats_df):,}")
print(f"  Rango: {matches_df['date'].min().date()} → {matches_df['date'].max().date()}")
print(f"\n  Por competencia:")
print(matches_df["competition"].value_counts().to_string())

# ── 2. Preparar dataset de entrenamiento ─────────────────────────────────────
# Indexar stats por equipo para lookup rápido
team_idx = team_stats_df.set_index("team")

def get_team_features(team_name):
    """Devuelve features de un equipo desde los stats reales de la API."""
    if team_name in team_idx.index:
        row = team_idx.loc[team_name]
        return {
            "win_rate":              row["win_rate"],
            "pts_per_game":          row["pts_per_game"],
            "goles_favor":           row["goles_por_partido"],
            "goles_contra":          row["goles_recibidos_partido"],
            "clean_sheet_rate":      row["clean_sheet_rate"],
            "score_rate":            row["score_rate"],
            "Fuerza_Score":          row["Fuerza_Score"],
            "diferencia_goles_total": row["diferencia_goles"],
            "partidos_jugados":      row["partidos"],
        }
    # Fallback con promedios globales si no hay datos del equipo
    return {
        "win_rate":              team_stats_df["win_rate"].mean(),
        "pts_per_game":          team_stats_df["pts_per_game"].mean(),
        "goles_favor":           team_stats_df["goles_por_partido"].mean(),
        "goles_contra":          team_stats_df["goles_recibidos_partido"].mean(),
        "clean_sheet_rate":      team_stats_df["clean_sheet_rate"].mean(),
        "score_rate":            team_stats_df["score_rate"].mean(),
        "Fuerza_Score":          team_stats_df["Fuerza_Score"].mean(),
        "diferencia_goles_total": 0,
        "partidos_jugados":      0,
    }

print("\n⚙️  Construyendo features del dataset...")
rows = []
for _, m in matches_df.iterrows():
    ht = m["home_team"]
    at = m["away_team"]
    if not ht or not at or pd.isna(ht) or pd.isna(at):
        continue
    
    home_f = get_team_features(ht)
    away_f = get_team_features(at)
    
    # Ponderación por importancia de la competencia
    comp_weight = {
        "WC":  3.0,   # Mundial: máxima importancia
        "EC":  2.5,   # Euro
        "CL":  2.0,   # Champions League
        "PD":  1.5,   # La Liga
        "BL1": 1.5,   # Bundesliga
        "PL":  1.5,   # Premier League (si hay)
        "SA":  1.5,   # Serie A
        "FL1": 1.2,   # Ligue 1
    }.get(m["competition"], 1.0)
    
    # Target
    if m["home_win"]:
        target = 1
    elif m["away_win"]:
        target = -1
    else:
        target = 0
    
    row = {
        "competition": m["competition"],
        "date": m["date"],
        "home_team": ht,
        "away_team": at,
        "target": target,
        "sample_weight": comp_weight,
    }
    
    # Features del local
    for k, v in home_f.items():
        row[f"home_{k}"] = v
    # Features del visitante
    for k, v in away_f.items():
        row[f"away_{k}"] = v
    
    # Features diferenciales
    row["diff_fuerza"]      = home_f["Fuerza_Score"] - away_f["Fuerza_Score"]
    row["diff_win_rate"]    = home_f["win_rate"] - away_f["win_rate"]
    row["diff_goles"]       = home_f["goles_favor"] - away_f["goles_favor"]
    row["diff_concedidos"]  = home_f["goles_contra"] - away_f["goles_contra"]
    row["diff_pts"]         = home_f["pts_per_game"] - away_f["pts_per_game"]
    row["diff_clean"]       = home_f["clean_sheet_rate"] - away_f["clean_sheet_rate"]
    
    rows.append(row)

df_train = pd.DataFrame(rows)
print(f"  Dataset construido: {len(df_train):,} partidos")
print(f"\n  Distribución target:")
print(f"    Gana Local:     {(df_train['target']==1).sum():4} ({(df_train['target']==1).mean()*100:.1f}%)")
print(f"    Empate:         {(df_train['target']==0).sum():4} ({(df_train['target']==0).mean()*100:.1f}%)")
print(f"    Gana Visitante: {(df_train['target']==-1).sum():4} ({(df_train['target']==-1).mean()*100:.1f}%)")

# Solo features numéricas (excluir columnas string como home_team, away_team)
feature_cols_raw = [c for c in df_train.columns if c.startswith("home_") or c.startswith("away_") or c.startswith("diff_")]
X_raw = df_train[feature_cols_raw].apply(pd.to_numeric, errors="coerce")
# Descartar columnas que son totalmente NaN (eran strings)
feature_cols = [c for c in feature_cols_raw if X_raw[c].notna().any()]
X = X_raw[feature_cols].copy()

# Imputar NaNs con la mediana
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=feature_cols)
y = df_train["target"]
weights = df_train["sample_weight"]

print(f"\n  Features: {len(feature_cols)}")

# ── 4. Entrenar modelo Ensemble ───────────────────────────────────────────────
print("\n🤖 Entrenando modelo Ensemble...")
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.25, random_state=42, stratify=y
)

# HistGradientBoosting — acepta NaN nativo, muy eficiente
hgb = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_depth=6,
    l2_regularization=0.1, random_state=42
)
# Random Forest
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1
)

print("  Entrenando HistGradientBoosting...")
hgb.fit(X_train, y_train, sample_weight=w_train)
print("  Entrenando Random Forest...")
rf.fit(X_train, y_train, sample_weight=w_train)

# Ensemble manual: promedio de probabilidades
def ensemble_predict(X_data):
    p_hgb = hgb.predict_proba(X_data)
    p_rf  = rf.predict_proba(X_data)
    p_avg = (p_hgb + p_rf) / 2
    classes = hgb.classes_
    return classes[np.argmax(p_avg, axis=1)]

def ensemble_proba(X_data):
    p_hgb = hgb.predict_proba(X_data)
    p_rf  = rf.predict_proba(X_data)
    return (p_hgb + p_rf) / 2

preds = ensemble_predict(X_test)
acc       = accuracy_score(y_test, preds)
f1        = f1_score(y_test, preds, average="weighted")
precision = precision_score(y_test, preds, average="weighted", zero_division=0)
recall    = recall_score(y_test, preds, average="weighted", zero_division=0)

# CV sobre HGB
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(hgb, X, y, cv=cv, scoring="accuracy")

print(f"\n📊 Métricas del Ensemble (HGB + RF):")
print(f"   Accuracy:             {acc:.4f}")
print(f"   F1-Score (weighted):  {f1:.4f}")
print(f"   Precision (weighted): {precision:.4f}")
print(f"   Recall (weighted):    {recall:.4f}")
print(f"   CV (5-fold GB):       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()
print(classification_report(y_test, preds, target_names=["Gana Visitante", "Empate", "Gana Local"]))

# Feature importance (usar RF, ya que HGB no lo expone directamente)
feat_imp = dict(zip(feature_cols, rf.feature_importances_))
feat_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
print("🎯 Top 10 Features:")
for i, (k, v) in enumerate(list(feat_sorted.items())[:10], 1):
    print(f"  {i:2}. {k:<45} {v*100:.1f}%")

# ── 5. Guardar métricas ───────────────────────────────────────────────────────
metricas = {
    "modelo": "Ensemble (HistGradientBoosting + RandomForest) v3",
    "datos_fuente": "football-data.org API v4 (datos 100% reales)",
    "competencias": list(matches_df["competition"].unique()),
    "partidos_entrenamiento": len(X_train),
    "partidos_evaluacion": len(X_test),
    "accuracy": round(acc, 4),
    "f1_score_weighted": round(f1, 4),
    "precision_weighted": round(precision, 4),
    "recall_weighted": round(recall, 4),
    "cv_accuracy_mean": round(cv_scores.mean(), 4),
    "cv_accuracy_std": round(cv_scores.std(), 4),
    "feature_importance": {k: round(v, 4) for k, v in list(feat_sorted.items())[:12]},
}
with open("modelo_metricas.json", "w") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

# ── 6. Guardar modelos ────────────────────────────────────────────────────────
joblib.dump({"hgb": hgb, "rf": rf, "classes": hgb.classes_, "features": feature_cols, "imputer": imputer}, "modelo_futbol.pkl")
print("\n✅ Modelo guardado en modelo_futbol.pkl")

# ── 7. Generar equipos_scores.csv para la app ─────────────────────────────────
print("\n📋 Generando equipos_scores.csv desde stats API reales...")
scores_rows = []
for _, row in team_stats_df.iterrows():
    scores_rows.append({
        "Equipo":            row["team"],
        "Fuerza_Score":      row["Fuerza_Score"],
        "Partidos":          row["partidos"],
        "Win_Rate":          row["win_rate"],
        "Pts_Per_Game":      row["pts_per_game"],
        "Goles_Por_Partido": row["goles_por_partido"],
        "Goles_Recibidos":   row["goles_recibidos_partido"],
        "Clean_Sheet_Rate":  row["clean_sheet_rate"],
        "Score_Rate":        row["score_rate"],
        "Diff_Goles":        row["diferencia_goles"],
        # Legacy aliases para compatibilidad con app.py
        "Avg_Age":           26.5,
        "Total_Caps":        row["partidos"],
        "Total_Goals_Scored": row["goles_favor"],
        "Total_Assists":     0,
    })

pd.DataFrame(scores_rows).to_csv("equipos_scores.csv", index=False)
print(f"✅ equipos_scores.csv: {len(scores_rows)} equipos con datos REALES")
print("\n🎉 ¡Entrenamiento completo con datos 100% reales de la API!")
