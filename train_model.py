import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
print(f"Precisión del modelo en prueba (Accuracy): {acc:.2f}")

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
