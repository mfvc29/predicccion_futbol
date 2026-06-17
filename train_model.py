import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib

# 1. Cargar Datos
print("Cargando dataset con nuevas métricas avanzadas de jugadores...")
df = pd.read_csv('dataset_predictivo_final.csv')

# Preparar Target (0: Empate, 1: Gana Local, -1: Gana Visitante)
df['Target'] = 0
df.loc[df['home_win'] == 1, 'Target'] = 1
df.loc[df['away_win'] == 1, 'Target'] = -1

# Preparar Features
df['neutral'] = df['neutral'].astype(int) # True/False a 1/0
df['score_diff'] = df['home_Fuerza_Score'] - df['away_Fuerza_Score']

features = [
    'home_Fuerza_Score', 'away_Fuerza_Score', 
    'home_Avg_Age', 'away_Avg_Age',
    'home_Total_Caps', 'away_Total_Caps',
    'home_Total_Goals_Scored', 'away_Total_Goals_Scored',
    'home_Total_Assists', 'away_Total_Assists',
    'score_diff', 'neutral'
]
X = df[features]
y = df['Target']

# Manejar NaNs si quedaron
X = X.fillna(X.mean())

# 2. Entrenar Modelo
print("Entrenando el modelo RandomForest con características humanas...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')
precision = precision_score(y_test, preds, average='weighted', zero_division=0)
recall = recall_score(y_test, preds, average='weighted', zero_division=0)

# Cross-validation accuracy (más robusta)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"Accuracy en test:       {acc:.4f}")
print(f"F1-Score (weighted):    {f1:.4f}")
print(f"Precision (weighted):   {precision:.4f}")
print(f"Recall (weighted):      {recall:.4f}")
print(f"CV Accuracy (5-fold):   {cv_mean:.4f} ± {cv_std:.4f}")
print()
print(classification_report(y_test, preds, target_names=['Gana Visitante', 'Empate', 'Gana Local']))

# Importancia de features
feat_importance = dict(zip(features, model.feature_importances_.round(4).tolist()))
feat_sorted = dict(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True))

# Guardar métricas en JSON
metricas = {
    'modelo': 'Random Forest Classifier',
    'n_estimators': 100,
    'max_depth': 7,
    'partidos_entrenamiento': len(X_train),
    'partidos_evaluacion': len(X_test),
    'accuracy': round(acc, 4),
    'f1_score_weighted': round(f1, 4),
    'precision_weighted': round(precision, 4),
    'recall_weighted': round(recall, 4),
    'cv_accuracy_mean': round(cv_mean, 4),
    'cv_accuracy_std': round(cv_std, 4),
    'feature_importance': feat_sorted
}
with open('modelo_metricas.json', 'w') as f:
    json.dump(metricas, f, indent=2)
print("Métricas guardadas en 'modelo_metricas.json'.")

# Guardar Modelo
joblib.dump(model, 'modelo_futbol.pkl')
print("Modelo guardado exitosamente como 'modelo_futbol.pkl'.")

# 3. Guardar los scores en un CSV para la App de Streamlit
equipos_unicos = pd.concat([df['home_team'], df['away_team']]).unique()
scores_app = []

def get_team_metrics(country_name):
    # Buscar el registro más reciente
    df_team = df[(df['home_team'] == country_name) | (df['away_team'] == country_name)]
    if len(df_team) > 0:
        row = df_team.iloc[-1]
        if row['home_team'] == country_name:
            return {
                'Fuerza_Score': row['home_Fuerza_Score'],
                'Avg_Age': row['home_Avg_Age'],
                'Total_Caps': row['home_Total_Caps'],
                'Total_Goals_Scored': row['home_Total_Goals_Scored'],
                'Total_Assists': row['home_Total_Assists']
            }
        else:
            return {
                'Fuerza_Score': row['away_Fuerza_Score'],
                'Avg_Age': row['away_Avg_Age'],
                'Total_Caps': row['away_Total_Caps'],
                'Total_Goals_Scored': row['away_Total_Goals_Scored'],
                'Total_Assists': row['away_Total_Assists']
            }
    # Fallback
    return {
        'Fuerza_Score': df['home_Fuerza_Score'].mean(),
        'Avg_Age': df['home_Avg_Age'].mean(),
        'Total_Caps': df['home_Total_Caps'].mean(),
        'Total_Goals_Scored': df['home_Total_Goals_Scored'].mean(),
        'Total_Assists': df['home_Total_Assists'].mean()
    }

for eq in equipos_unicos:
    metrics = get_team_metrics(eq)
    metrics['Equipo'] = eq
    scores_app.append(metrics)

df_app = pd.DataFrame(scores_app)
df_app.to_csv('equipos_scores.csv', index=False)
print("Archivo 'equipos_scores.csv' actualizado con todas las métricas.")
