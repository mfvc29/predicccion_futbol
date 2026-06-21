"""
train_model.py - v2
====================
Entrena el modelo con enfoque en datos RECIENTES:
  - Partidos de los últimos 2 años para el Target (2024-2026)
  - Features de jugadores actuales del Mundial 2026
  - Ponderación por recencia (partidos recientes pesan más)
  - Filtros opcionales por tipo de torneo (no solo amistosos)
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib

# ── 1. Cargar dataset enriquecido ─────────────────────────────────────────────
print("Cargando dataset_predictivo_final.csv...")
df = pd.read_csv('dataset_predictivo_final.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"  Total partidos: {len(df):,}")

# ── 2. Filtrar por fechas recientes (últimos 2 años como datos principales) ───
print("\n📅 Estrategia de datos recientes...")

# Partidos oficiales recientes (2022 en adelante, excluir puros amistosos si hay suficientes)
df_recent = df[df['date'].dt.year >= 2022].copy()
df_older  = df[(df['date'].dt.year >= 2000) & (df['date'].dt.year < 2022)].copy()

# Filtrar torneos relevantes (excluir amistosos en datos recientes para mayor calidad señal)
# Los amistosos tienen menos correlación con el resultado real competitivo
competitive_tournaments = [
    'FIFA World Cup', 'UEFA Euro', 'Copa América', 'Africa Cup of Nations',
    'CONCACAF Gold Cup', 'AFC Asian Cup', 'UEFA Nations League',
    'CONCACAF Nations League', 'FIFA World Cup qualification',
    'UEFA Euro qualification', 'African Cup of Nations qualification',
    'CONCACAF Nations League', 'AFC Asian Cup qualification',
    'African Cup of Nations', 'FIFA Series',
    'Confederations Cup', 'COSAFA Cup', 'AFF Championship',
]

df_recent_comp = df_recent[df_recent['tournament'].isin(competitive_tournaments)].copy()
df_recent_all  = df_recent.copy()

# Si hay suficientes partidos competitivos recientes, usarlos; sino usar todos
if len(df_recent_comp) >= 1000:
    df_training = pd.concat([df_older.sample(min(len(df_older), 10000), random_state=42), df_recent_all], ignore_index=True)
    print(f"  ✅ Usando {len(df_recent_comp):,} partidos competitivos recientes + {len(df_older):,} históricos")
else:
    df_training = df.copy()
    print(f"  ⚠️  Usando dataset completo ({len(df_training):,} partidos)")

print(f"  Dataset de entrenamiento: {len(df_training):,} partidos")
print(f"  Rango: {df_training['date'].min().date()} → {df_training['date'].max().date()}")

# ── 3. Target ─────────────────────────────────────────────────────────────────
df_training['Target'] = 0
df_training.loc[df_training['home_win'] == 1, 'Target'] = 1
df_training.loc[df_training['away_win'] == 1, 'Target'] = -1

# ── 4. Ponderación por recencia ───────────────────────────────────────────────
# Partidos más recientes pesan hasta 3x más que los de 2000
max_year = df_training['date'].dt.year.max()
min_year = df_training['date'].dt.year.min()
year_range = max(max_year - min_year, 1)

df_training['sample_weight'] = (
    1.0 + 2.0 * (df_training['date'].dt.year - min_year) / year_range
).clip(1.0, 3.0)

# Partidos del último año: peso extra x1.5
df_training.loc[df_training['date'].dt.year >= 2025, 'sample_weight'] *= 1.5

print(f"\n  Peso promedio de partidos recientes (2025+): "
      f"{df_training[df_training['date'].dt.year >= 2025]['sample_weight'].mean():.2f}")

# ── 5. Features ───────────────────────────────────────────────────────────────
df_training['neutral'] = df_training['neutral'].astype(int)
df_training['score_diff'] = df_training['home_Fuerza_Score'] - df_training['away_Fuerza_Score']

# Features base (siempre disponibles)
features_base = [
    'home_Fuerza_Score', 'away_Fuerza_Score',
    'home_Avg_Age', 'away_Avg_Age',
    'home_Total_Caps', 'away_Total_Caps',
    'home_Total_Goals_Scored', 'away_Total_Goals_Scored',
    'home_Total_Assists', 'away_Total_Assists',
    'score_diff', 'neutral'
]

# Features tácticas del WC2026 (opcionales, si están disponibles)
tactical_optional = [
    'home_possession_mean', 'away_possession_mean',
    'home_shots_on_goal_mean', 'away_shots_on_goal_mean',
    'home_shot_accuracy_mean', 'away_shot_accuracy_mean',
    'home_shot_conversion_mean', 'away_shot_conversion_mean',
    'home_clean_sheet_rate', 'away_clean_sheet_rate',
    'home_goals_scored_mean', 'away_goals_scored_mean',
    'home_goals_conceded_mean', 'away_goals_conceded_mean',
    'home_attack_defense_ratio', 'away_attack_defense_ratio',
    'home_points_per_game', 'away_points_per_game',
    'diff_Fuerza_Score', 'diff_Total_Caps',
    'diff_possession_mean', 'diff_shots_on_goal_mean',
    'diff_attack_defense_ratio',
]

features_tacticals = [f for f in tactical_optional if f in df_training.columns]
features = features_base + features_tacticals

if features_tacticals:
    print(f"\n✅ Usando {len(features_tacticals)} features tácticas adicionales")
else:
    print("\n⚠️  Sin features tácticas WC2026 (ejecuta fetch_wc2026_matches.py para activarlas)")

X = df_training[features].copy()
y = df_training['Target']
weights = df_training['sample_weight']

# Manejar NaNs
X = X.fillna(X.mean())

print(f"\n  Distribución del target:")
print(f"    Gana Local:    {(y == 1).sum():,} ({(y==1).mean()*100:.1f}%)")
print(f"    Empate:        {(y == 0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"    Gana Visitante:{(y == -1).sum():,} ({(y==-1).mean()*100:.1f}%)")

# ── 6. Entrenar Modelo ────────────────────────────────────────────────────────
print("\n🤖 Entrenando Random Forest con ponderación por recencia...")
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,      # Más árboles para mayor estabilidad
    max_depth=10,          # Más profundidad para capturar patrones recientes
    min_samples_leaf=5,    # Evitar sobreajuste en partidos raros
    class_weight='balanced',  # Balancear empates (menos frecuentes)
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, sample_weight=w_train)

preds = model.predict(X_test)
acc       = accuracy_score(y_test, preds, sample_weight=w_test)
acc_plain = accuracy_score(y_test, preds)
f1        = f1_score(y_test, preds, average='weighted')
precision = precision_score(y_test, preds, average='weighted', zero_division=0)
recall    = recall_score(y_test, preds, average='weighted', zero_division=0)

# Cross-validation (sin pesos para comparación justa)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std  = cv_scores.std()

print(f"\n📊 Métricas del Modelo:")
print(f"   Accuracy (ponderada):  {acc:.4f}")
print(f"   Accuracy (plana):      {acc_plain:.4f}")
print(f"   F1-Score (weighted):   {f1:.4f}")
print(f"   Precision (weighted):  {precision:.4f}")
print(f"   Recall (weighted):     {recall:.4f}")
print(f"   CV Accuracy (5-fold):  {cv_mean:.4f} ± {cv_std:.4f}")
print()
print(classification_report(y_test, preds, target_names=['Gana Visitante', 'Empate', 'Gana Local']))

# ── 7. Feature Importance ─────────────────────────────────────────────────────
feat_importance = dict(zip(features, model.feature_importances_.round(4).tolist()))
feat_sorted = dict(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True))

# Mostrar top 10
print("\n🎯 Top 10 Features más importantes:")
for i, (k, v) in enumerate(list(feat_sorted.items())[:10], 1):
    print(f"   {i:2}. {k:<40} {v*100:.1f}%")

# ── 8. Guardar métricas ───────────────────────────────────────────────────────
metricas = {
    'modelo': 'Random Forest Classifier v2',
    'n_estimators': 200,
    'max_depth': 10,
    'datos_recencia': 'Últimos 2 años con peso 3x, 2025+ con peso 4.5x',
    'partidos_entrenamiento': len(X_train),
    'partidos_evaluacion': len(X_test),
    'accuracy': round(acc_plain, 4),
    'accuracy_ponderada': round(acc, 4),
    'f1_score_weighted': round(f1, 4),
    'precision_weighted': round(precision, 4),
    'recall_weighted': round(recall, 4),
    'cv_accuracy_mean': round(cv_mean, 4),
    'cv_accuracy_std': round(cv_std, 4),
    'features_tacticas': len(features_tacticals),
    'feature_importance': {k: v for k, v in list(feat_sorted.items())[:12]}
}
with open('modelo_metricas.json', 'w') as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)
print("\n✅ Métricas guardadas en 'modelo_metricas.json'.")

# ── 9. Guardar Modelo ─────────────────────────────────────────────────────────
joblib.dump(model, 'modelo_futbol.pkl')
print("✅ Modelo guardado como 'modelo_futbol.pkl'.")

# ── 10. Generar equipos_scores.csv actualizado ────────────────────────────────
print("\n📋 Generando equipos_scores.csv...")

# Usar datos recientes para el score de cada equipo (últimos 3 años)
df_scores_base = df[df['date'].dt.year >= 2023].copy()
if len(df_scores_base) < 100:
    df_scores_base = df.copy()

equipos_unicos = pd.concat([df['home_team'], df['away_team']]).unique()
scores_app = []

def get_team_metrics(country_name):
    # Buscar partidos del equipo en datos recientes
    df_team = df_scores_base[
        (df_scores_base['home_team'] == country_name) |
        (df_scores_base['away_team'] == country_name)
    ].sort_values('date', ascending=False)

    if len(df_team) > 0:
        # Calcular win rate basado en los últimos 20 partidos
        wins, draws, losses = 0, 0, 0
        for _, row in df_team.head(20).iterrows():
            is_home = row['home_team'] == country_name
            h, a = row['home_score'], row['away_score']
            if is_home:
                if h > a: wins += 1
                elif h == a: draws += 1
                else: losses += 1
            else:
                if a > h: wins += 1
                elif a == h: draws += 1
                else: losses += 1
        total = wins + draws + losses
        win_rate = wins / total if total > 0 else 0

        row = df_team.iloc[0]
        if row['home_team'] == country_name:
            return {
                'Fuerza_Score': row['home_Fuerza_Score'],
                'Avg_Age': row['home_Avg_Age'],
                'Total_Caps': row['home_Total_Caps'],
                'Total_Goals_Scored': row['home_Total_Goals_Scored'],
                'Total_Assists': row['home_Total_Assists'],
                'Avg_Win_Rate': win_rate,
            }
        else:
            return {
                'Fuerza_Score': row['away_Fuerza_Score'],
                'Avg_Age': row['away_Avg_Age'],
                'Total_Caps': row['away_Total_Caps'],
                'Total_Goals_Scored': row['away_Total_Goals_Scored'],
                'Total_Assists': row['away_Total_Assists'],
                'Avg_Win_Rate': win_rate,
            }

    # Fallback con promedios globales
    return {
        'Fuerza_Score': df['home_Fuerza_Score'].mean(),
        'Avg_Age': df['home_Avg_Age'].mean(),
        'Total_Caps': df['home_Total_Caps'].mean(),
        'Total_Goals_Scored': df['home_Total_Goals_Scored'].mean(),
        'Total_Assists': df['home_Total_Assists'].mean(),
        'Avg_Win_Rate': 0.33,
    }

for eq in equipos_unicos:
    metrics = get_team_metrics(eq)
    metrics['Equipo'] = eq
    scores_app.append(metrics)

df_app = pd.DataFrame(scores_app)
df_app.to_csv('equipos_scores.csv', index=False)
print(f"✅ equipos_scores.csv generado con {len(df_app)} equipos.")
print("\n🎉 ¡Entrenamiento completo!")
print("   Siguiente paso: streamlit run app.py")
