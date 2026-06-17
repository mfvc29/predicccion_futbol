import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Partidos de Fútbol", page_icon="⚽", layout="centered")

st.title("⚽ Simulador de Partidos de Fútbol")
st.write("Predice el resultado basándote en la **experiencia real de los jugadores** (edad, selecciones, goles) y el historial de encuentros desde el año 2000.")

# Cargar Modelo y Datos
@st.cache_resource
def load_model_and_data():
    model = joblib.load('modelo_futbol.pkl')
    scores_df = pd.read_csv('equipos_scores.csv')
    return model, scores_df

try:
    model, scores_df = load_model_and_data()
except Exception as e:
    st.error(f"Error al cargar el modelo o los datos: {e}")
    st.stop()

equipos = sorted(scores_df['Equipo'].dropna().unique().tolist())

# Interfaz de Selección
col1, col2 = st.columns(2)

with col1:
    st.header("🏠 Equipo Local")
    equipo_local = st.selectbox("Selecciona el equipo local", equipos, index=equipos.index('France') if 'France' in equipos else 0)

with col2:
    st.header("✈️ Equipo Visitante")
    equipo_visitante = st.selectbox("Selecciona el equipo visitante", equipos, index=equipos.index('Senegal') if 'Senegal' in equipos else 1)

is_neutral = st.checkbox("¿Es partido en cancha neutral? (Ej. Mundial, Copa América)", value=True)

# Mostrar comparativa de plantillas si son distintos
if equipo_local != equipo_visitante:
    st.divider()
    st.subheader("⚖️ Comparativa de Plantillas (Métricas Reales)")
    
    stats_local = scores_df[scores_df['Equipo'] == equipo_local].iloc[0]
    stats_visit = scores_df[scores_df['Equipo'] == equipo_visitante].iloc[0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Edad Promedio", f"{stats_local['Avg_Age']:.1f} años", f"{stats_local['Avg_Age'] - stats_visit['Avg_Age']:.1f} vs Visita", delta_color="inverse")
    c2.metric("Experiencia (Partidos Int.)", f"{int(stats_local['Total_Caps'])}", f"{int(stats_local['Total_Caps'] - stats_visit['Total_Caps'])} vs Visita")
    c3.metric("Goles Totales de Plantilla", f"{int(stats_local['Total_Goals_Scored'])}", f"{int(stats_local['Total_Goals_Scored'] - stats_visit['Total_Goals_Scored'])} vs Visita")
    
if st.button("🔮 Predecir Resultado", use_container_width=True):
    if equipo_local == equipo_visitante:
        st.warning("⚠️ Debes seleccionar dos equipos diferentes.")
    else:
        # Preparar data para el modelo respetando el orden de features
        # ['home_Fuerza_Score', 'away_Fuerza_Score', 'home_Avg_Age', 'away_Avg_Age', 
        #  'home_Total_Caps', 'away_Total_Caps', 'home_Total_Goals_Scored', 'away_Total_Goals_Scored', 
        #  'home_Total_Assists', 'away_Total_Assists', 'score_diff', 'neutral']
        
        score_diff = stats_local['Fuerza_Score'] - stats_visit['Fuerza_Score']
        
        input_data = pd.DataFrame({
            'home_Fuerza_Score': [stats_local['Fuerza_Score']],
            'away_Fuerza_Score': [stats_visit['Fuerza_Score']],
            'home_Avg_Age': [stats_local['Avg_Age']],
            'away_Avg_Age': [stats_visit['Avg_Age']],
            'home_Total_Caps': [stats_local['Total_Caps']],
            'away_Total_Caps': [stats_visit['Total_Caps']],
            'home_Total_Goals_Scored': [stats_local['Total_Goals_Scored']],
            'away_Total_Goals_Scored': [stats_visit['Total_Goals_Scored']],
            'home_Total_Assists': [stats_local['Total_Assists']],
            'away_Total_Assists': [stats_visit['Total_Assists']],
            'score_diff': [score_diff],
            'neutral': [1 if is_neutral else 0]
        })
        
        # Predecir
        probs = model.predict_proba(input_data)[0]
        classes = model.classes_ # [-1, 0, 1]
        
        idx_away = np.where(classes == -1)[0][0]
        idx_draw = np.where(classes == 0)[0][0]
        idx_home = np.where(classes == 1)[0][0]
        
        prob_home = probs[idx_home] * 100
        prob_draw = probs[idx_draw] * 100
        prob_away = probs[idx_away] * 100
        
        st.divider()
        st.subheader("📊 Probabilidades del Encuentro")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric(f"Victoria de {equipo_local}", f"{prob_home:.1f}%")
        col_res2.metric("Empate", f"{prob_draw:.1f}%")
        col_res3.metric(f"Victoria de {equipo_visitante}", f"{prob_away:.1f}%")
        
        st.bar_chart(pd.DataFrame({
            'Probabilidad (%)': [prob_home, prob_draw, prob_away]
        }, index=[equipo_local, 'Empate', equipo_visitante]))
