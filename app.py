import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor ML Mundial",
    page_icon="⚽",
    layout="wide"
)

# ─── CSS PREMIUM ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── FONDO BLANCO LIMPIO ── */
    .stApp {
        background: #f8f9fc;
    }
    .stApp > header { background: transparent !important; }

    /* Sidebar blanco */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8ecf0;
    }

    /* Inputs y selects */
    .stSelectbox > div > div,
    .stTextInput > div > div {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 10px !important;
        color: #1a202c !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    /* Checkbox */
    .stCheckbox > label { color: #4a5568 !important; font-weight: 500; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #e8ecf0;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #718096;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: #6366f1 !important;
        color: white !important;
    }

    /* Tarjetas */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e8ecf0;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Hero */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 4px;
    }
    .hero-sub {
        text-align: center;
        color: #718096;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 32px;
    }

    /* Labels */
    .team-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #a0aec0;
        margin-bottom: 6px;
    }

    /* Resultados predicción */
    .result-home { color: #6366f1; font-size: 2.6rem; font-weight: 900; text-align: center; }
    .result-draw { color: #d97706; font-size: 2rem; font-weight: 700; text-align: center; }
    .result-away { color: #ef4444; font-size: 2.6rem; font-weight: 900; text-align: center; }
    .result-label { color: #a0aec0; font-size: 0.8rem; text-align: center; margin-top: 4px; }
    .winner-badge {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; padding: 6px 16px;
        border-radius: 999px; display: inline-block; margin-top: 8px;
    }
    .badge-home { background: rgba(99,102,241,0.1); color: #6366f1; border: 1px solid rgba(99,102,241,0.3); }
    .badge-away { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge-draw { background: rgba(217,119,6,0.1); color: #d97706; border: 1px solid rgba(217,119,6,0.3); }

    /* Players */
    .danger-player {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 0; border-bottom: 1px solid #f1f5f9;
    }
    .danger-rank { font-size: 1.1rem; font-weight: 900; color: #cbd5e0; min-width: 28px; }
    .danger-name { font-weight: 600; color: #1a202c; flex: 1; }
    .danger-pos {
        font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;
        background: rgba(99,102,241,0.1); color: #6366f1; border: 1px solid rgba(99,102,241,0.25);
    }
    .danger-score { font-weight: 700; color: #6366f1; min-width: 60px; text-align: right; }

    /* Section title */
    .section-title {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.15em;
        text-transform: uppercase; color: #a0aec0;
        margin-bottom: 16px; padding-bottom: 10px;
        border-bottom: 1px solid #e8ecf0;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: #ffffff !important;
        border: 1px solid #e8ecf0 !important;
        border-radius: 12px !important;
        padding: 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1a202c !important;
    }
    [data-testid="stMetricLabel"] { color: #718096 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important; border: none !important;
        font-weight: 700 !important; font-size: 1rem !important;
        padding: 14px 32px !important; border-radius: 12px !important;
        letter-spacing: 0.04em; transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(99,102,241,0.4) !important;
    }

    /* Divider */
    hr { border-color: #e8ecf0 !important; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #e8ecf0 !important; border-radius: 12px !important; }

    /* Expander */
    .streamlit-expanderHeader { color: #4a5568 !important; font-weight: 600 !important; }
    .streamlit-expanderContent { background: #ffffff !important; border: 1px solid #e8ecf0 !important; border-radius: 0 0 12px 12px !important; }

    /* General text */
    p, li, span, div { color: #2d3748; }
    h1, h2, h3, h4 { color: #1a202c !important; }
    label { color: #4a5568 !important; }

    /* Hide streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    model = joblib.load('modelo_futbol.pkl')
    scores_df = pd.read_csv('equipos_scores.csv')
    try:
        top_players_df = pd.read_csv('top_players.csv')
    except:
        top_players_df = pd.DataFrame()
    try:
        with open('modelo_metricas.json', 'r') as f:
            metricas = json.load(f)
    except:
        metricas = {}
    try:
        jugadores_df = pd.read_csv('jugadores_mundial_2026_con_stats_final.csv')
    except:
        jugadores_df = pd.DataFrame()
    try:
        partidos_df = pd.read_csv('partidos_internacionales_historico.csv')
        partidos_df['date'] = pd.to_datetime(partidos_df['date'])
        partidos_df = partidos_df.sort_values('date', ascending=False)
    except:
        partidos_df = pd.DataFrame()
    return model, scores_df, top_players_df, metricas, jugadores_df, partidos_df

try:
    model, scores_df, top_players_df, metricas, jugadores_df, partidos_df = load_all()
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.stop()

equipos = sorted(scores_df['Equipo'].dropna().unique().tolist())

# ─── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predicción de Partidos", "🌍 Análisis de Selecciones", "🏃 Buscador de Jugadores"])

with tab1:

    # ─── HERO HEADER ───────────────────────────────────────────────────────────────
    st.markdown('<div class="hero-title">⚽ Predictor ML Mundial</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Machine Learning entrenado con 25,000+ partidos históricos y estadísticas reales de 1,247 jugadores</div>', unsafe_allow_html=True)

    # ─── MODEL METRICS EXPANDER ────────────────────────────────────────────────────
    if metricas:
        with st.expander("🤖 Ver métricas del modelo de Machine Learning", expanded=False):
            st.markdown('<div class="section-title">Información del Modelo</div>', unsafe_allow_html=True)

            em1, em2, em3, em4, em5 = st.columns(5)
            em1.metric("Algoritmo", metricas.get('modelo', 'RF').replace('Random Forest Classifier', 'Random Forest'))
            em2.metric("✅ Accuracy", f"{metricas.get('accuracy', 0)*100:.1f}%")
            em3.metric("📐 F1-Score", f"{metricas.get('f1_score_weighted', 0)*100:.1f}%")
            em4.metric("🎯 Precision", f"{metricas.get('precision_weighted', 0)*100:.1f}%")
            em5.metric("🔁 CV Accuracy", f"{metricas.get('cv_accuracy_mean', 0)*100:.1f}% ± {metricas.get('cv_accuracy_std', 0)*100:.1f}%")

            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px;margin-top:8px;color:rgba(255,255,255,0.55);font-size:0.82rem;line-height:1.7'>
            📦 <b style='color:white'>Configuración:</b> {metricas.get('n_estimators',100)} árboles · profundidad máxima {metricas.get('max_depth',7)}<br>
            📊 <b style='color:white'>Datos:</b> {metricas.get('partidos_entrenamiento',0):,} partidos entrenamiento · {metricas.get('partidos_evaluacion',0):,} evaluación<br>
            📋 <b style='color:white'>Nota:</b> Para clasificación de victorias/empates/derrotas no se usa R², sino <b>Accuracy</b> (% de partidos correctamente predichos)
            y <b>F1-Score ponderado</b> (equilibrio entre precisión y sensibilidad por clase).
            </div>
            """, unsafe_allow_html=True)

            # Feature importance chart
            if 'feature_importance' in metricas:
                feat_imp = metricas['feature_importance']
                labels = list(feat_imp.keys())
                values = list(feat_imp.values())

                # Limpiar nombres
                label_map = {
                    'home_Fuerza_Score': 'Fuerza Local', 'away_Fuerza_Score': 'Fuerza Visitante',
                    'score_diff': 'Diferencia Fuerza', 'home_Total_Caps': 'Partidos Int. Local',
                    'away_Total_Caps': 'Partidos Int. Visita', 'home_Avg_Age': 'Edad Local',
                    'away_Avg_Age': 'Edad Visita', 'home_Total_Assists': 'Asist. Local',
                    'away_Total_Assists': 'Asist. Visita', 'home_Total_Goals_Scored': 'Goles Local',
                    'away_Total_Goals_Scored': 'Goles Visita', 'neutral': 'Sede Neutral'
                }
                labels_clean = [label_map.get(l, l) for l in labels]

                colors = [f'rgba(99, 102, 241, {0.35 + 0.65 * (v / max(values))})' for v in values]
                fig_imp = go.Figure(go.Bar(
                    x=values, y=labels_clean, orientation='h',
                    marker=dict(color=colors),
                    text=[f"{v*100:.1f}%" for v in values],
                    textposition='outside', textfont=dict(color='#2d3748', size=11)
                ))
                fig_imp.update_layout(
                    title=dict(text='Importancia de Variables (Feature Importance)', font=dict(color='#718096', size=12)),
                    paper_bgcolor='rgba(255,255,255,0)', plot_bgcolor='rgba(255,255,255,0)',
                    xaxis=dict(visible=False, range=[0, max(values) * 1.3]),
                    yaxis=dict(tickfont=dict(color='#4a5568', size=11)),
                    margin=dict(l=10, r=60, t=40, b=10), height=320, showlegend=False
                )
                st.plotly_chart(fig_imp, use_container_width=True)

    # ─── TEAM SELECTION ────────────────────────────────────────────────────────────
    col1, spacer, col2 = st.columns([5, 1, 5])

    with col1:
        st.markdown('<div class="team-label">🏠 Equipo Local</div>', unsafe_allow_html=True)
        equipo_local = st.selectbox("", equipos, index=equipos.index('France') if 'France' in equipos else 0, key="local", label_visibility="collapsed")

    with spacer:
        st.markdown("<br><br><h2 style='text-align:center;color:#ffffff33;margin-top:8px'>VS</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="team-label">✈️ Equipo Visitante</div>', unsafe_allow_html=True)
        equipo_visitante = st.selectbox("", equipos, index=equipos.index('Senegal') if 'Senegal' in equipos else 1, key="visit", label_visibility="collapsed")

    is_neutral = st.checkbox("⚖️ Partido en cancha neutral (Mundial, Copa América, etc.)", value=True)

    # ─── COMPARATIVA + PLAYERS ─────────────────────────────────────────────────────
    if equipo_local != equipo_visitante:
        stats_local = scores_df[scores_df['Equipo'] == equipo_local].iloc[0]
        stats_visit = scores_df[scores_df['Equipo'] == equipo_visitante].iloc[0]

        st.markdown("---")
        st.markdown('<div class="section-title">⚖️ Comparativa de Plantillas</div>', unsafe_allow_html=True)

        # Cabecera con nombre de equipos
        mc_header_l, mc_header_r = st.columns(2)
        mc_header_l.markdown(f"<div style='font-weight:700;font-size:1.05rem;color:#6366f1;margin-bottom:10px'>🏠 {equipo_local}</div>", unsafe_allow_html=True)
        mc_header_r.markdown(f"<div style='font-weight:700;font-size:1.05rem;color:#ef4444;margin-bottom:10px'>✈️ {equipo_visitante}</div>", unsafe_allow_html=True)

        # Fila 1: Fuerza y Edad
        m1a, m1b, m1c, m1d = st.columns(4)
        m1a.metric("💪 Fuerza Histórica", f"{int(stats_local['Fuerza_Score'])}")
        m1b.metric("🎂 Edad Promedio", f"{stats_local['Avg_Age']:.1f} años")
        m1c.metric("💪 Fuerza Histórica", f"{int(stats_visit['Fuerza_Score'])}")
        m1d.metric("🎂 Edad Promedio", f"{stats_visit['Avg_Age']:.1f} años")

        # Fila 2: Caps y Win Rate
        m2a, m2b, m2c, m2d = st.columns(4)
        m2a.metric("🌍 Partidos Int. Plantilla", f"{int(stats_local['Total_Caps'])}")
        win_local = stats_local.get('Avg_Win_Rate', None)
        win_visit = stats_visit.get('Avg_Win_Rate', None)
        m2b.metric("🏆 % Victorias Promedio", f"{win_local*100:.1f}%" if pd.notna(win_local) else "N/D")
        m2c.metric("🌍 Partidos Int. Plantilla", f"{int(stats_visit['Total_Caps'])}")
        m2d.metric("🏆 % Victorias Promedio", f"{win_visit*100:.1f}%" if pd.notna(win_visit) else "N/D")

        # ─── RADAR CHART ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Análisis Radar de Equipos</div>', unsafe_allow_html=True)

        categorias = ['Fuerza<br>Histórica', 'Experiencia<br>Internacional', 'Edad<br>Óptima', 'Asistencias<br>Plantilla', 'Victorias<br>Totales']

        # Normalizar 0-100 relativo a ambos equipos para el radar
        def norm(v, ref_min, ref_max):
            if ref_max == ref_min: return 50
            return round(100 * (v - ref_min) / (ref_max - ref_min), 1)

        # Edad óptima: penalizar los extremos (mejor edad ~27)
        age_score_local = max(0, 100 - abs(stats_local['Avg_Age'] - 27) * 8)
        age_score_visit = max(0, 100 - abs(stats_visit['Avg_Age'] - 27) * 8)

        values_local_raw = [stats_local['Fuerza_Score'], stats_local['Total_Caps'], age_score_local, stats_local['Total_Assists'], stats_local['Fuerza_Score'] * 0.6]
        values_visit_raw = [stats_visit['Fuerza_Score'], stats_visit['Total_Caps'], age_score_visit, stats_visit['Total_Assists'], stats_visit['Fuerza_Score'] * 0.6]

        # Normalizar las métricas de escala diferente (Fuerza, Caps, Assists)
        all_fuerza = [r['Fuerza_Score'] for _, r in scores_df.iterrows() if not pd.isna(r['Fuerza_Score'])]
        all_caps = [r['Total_Caps'] for _, r in scores_df.iterrows() if not pd.isna(r['Total_Caps'])]
        all_assists = [r['Total_Assists'] for _, r in scores_df.iterrows() if not pd.isna(r['Total_Assists'])]

        values_local = [
            norm(stats_local['Fuerza_Score'], min(all_fuerza), max(all_fuerza)),
            norm(stats_local['Total_Caps'], min(all_caps), max(all_caps)),
            age_score_local,
            norm(stats_local['Total_Assists'], min(all_assists), max(all_assists)),
            norm(stats_local['Fuerza_Score'], min(all_fuerza), max(all_fuerza))
        ]
        values_visit = [
            norm(stats_visit['Fuerza_Score'], min(all_fuerza), max(all_fuerza)),
            norm(stats_visit['Total_Caps'], min(all_caps), max(all_caps)),
            age_score_visit,
            norm(stats_visit['Total_Assists'], min(all_assists), max(all_assists)),
            norm(stats_visit['Fuerza_Score'], min(all_fuerza), max(all_fuerza))
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_local + [values_local[0]],
            theta=categorias + [categorias[0]],
            fill='toself',
            name=equipo_local,
            line=dict(color='#6366f1', width=2.5),
            fillcolor='rgba(99, 102, 241, 0.12)',
            marker=dict(size=6, color='#6366f1')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=values_visit + [values_visit[0]],
            theta=categorias + [categorias[0]],
            fill='toself',
            name=equipo_visitante,
            line=dict(color='#ef4444', width=2.5),
            fillcolor='rgba(239, 68, 68, 0.10)',
            marker=dict(size=6, color='#ef4444')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=True, tickfont=dict(color='#a0aec0', size=9), gridcolor='#e8ecf0'),
                angularaxis=dict(tickfont=dict(color='#4a5568', size=12), gridcolor='#e8ecf0'),
                bgcolor='rgba(255,255,255,0)'
            ),
            paper_bgcolor='rgba(255,255,255,0)',
            plot_bgcolor='rgba(255,255,255,0)',
            legend=dict(font=dict(color='#2d3748', size=12), bgcolor='rgba(255,255,255,0)'),
            margin=dict(l=60, r=60, t=40, b=40),
            height=420
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ─── JUGADORES PELIGROSOS ───────────────────────────────────────────────────
        if not top_players_df.empty:
            st.markdown("---")
            st.markdown('<div class="section-title">🚨 Jugadores Más Peligrosos (Índice de Peligrosidad)</div>', unsafe_allow_html=True)

            pcol1, pcol2 = st.columns(2)
            for col_ui, equipo_nombre, color in [(pcol1, equipo_local, '6366f1'), (pcol2, equipo_visitante, 'ef4444')]:
                with col_ui:
                    st.markdown(f"<h4 style='color:#{color};margin-bottom:12px'>{equipo_nombre}</h4>", unsafe_allow_html=True)
                    players = top_players_df[top_players_df['Equipo'] == equipo_nombre].head(5)
                    if not players.empty:
                        max_score = players['Peligrosidad'].max() if players['Peligrosidad'].max() > 0 else 1
                        for i, (_, p) in enumerate(players.iterrows(), 1):
                            pct = p['Peligrosidad'] / max_score * 100 if max_score > 0 else 0
                            sels  = int(p.get('Partidos_Int', p.get('Sel_n', 0)) or 0)
                            vics  = int(p.get('Victoria_Total', p.get('Vic_n', 0)) or 0)
                            goles = int(p.get('Goles_Int', 0) or 0)
                            goles_str = f" · ⚽ {goles} goles int." if goles > 0 else ""
                            st.markdown(f"""
                            <div class='danger-player'>
                                <span class='danger-rank'>#{i}</span>
                                <div style='flex:1;padding: 0 12px'>
                                    <div class='danger-name'>{p['Nombre']}</div>
                                    <div style='font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:2px'>
                                        {sels} partidos int. · {vics} victorias{goles_str}
                                    </div>
                                    <div style='height:3px;background:rgba(255,255,255,0.06);border-radius:3px;margin-top:6px'>
                                        <div style='height:100%;width:{pct:.0f}%;background:#{color};border-radius:3px;opacity:0.7'></div>
                                    </div>
                                </div>
                                <span class='danger-pos'>{p['Pos.']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='color:rgba(255,255,255,0.3)'>Sin datos disponibles</p>", unsafe_allow_html=True)

    # ─── PREDICT BUTTON ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮  Predecir Resultado del Partido", use_container_width=True)

    if predict_clicked:
        if equipo_local == equipo_visitante:
            st.warning("⚠️ Selecciona dos equipos diferentes.")
        else:
            stats_local = scores_df[scores_df['Equipo'] == equipo_local].iloc[0]
            stats_visit = scores_df[scores_df['Equipo'] == equipo_visitante].iloc[0]
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

            probs = model.predict_proba(input_data)[0]
            classes = model.classes_

            idx_away = np.where(classes == -1)[0][0]
            idx_draw = np.where(classes == 0)[0][0]
            idx_home = np.where(classes == 1)[0][0]

            prob_home = probs[idx_home] * 100
            prob_draw = probs[idx_draw] * 100
            prob_away = probs[idx_away] * 100

            winner = equipo_local if prob_home > prob_away and prob_home > prob_draw else (equipo_visitante if prob_away > prob_home and prob_away > prob_draw else "Empate")
            winner_badge = "badge-home" if winner == equipo_local else ("badge-away" if winner == equipo_visitante else "badge-draw")

            st.markdown("---")
            st.markdown('<div class="section-title">🔮 Predicción del Encuentro</div>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div class='result-home'>{prob_home:.1f}%</div>
                <div class='result-label'>Victoria de {equipo_local}</div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class='result-draw'>{prob_draw:.1f}%</div>
                <div class='result-label'>Empate</div>
                """, unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div class='result-away'>{prob_away:.1f}%</div>
                <div class='result-label'>Victoria de {equipo_visitante}</div>
                """, unsafe_allow_html=True)

            # Resultado más probable
            st.markdown(f"""
            <div style='text-align:center;margin-top:24px'>
                <div style='color:rgba(255,255,255,0.4);font-size:0.8rem;margin-bottom:8px'>RESULTADO MÁS PROBABLE</div>
                <span class='winner-badge {winner_badge}'>🏆 {winner}</span>
            </div>
            """, unsafe_allow_html=True)

            # Gráfico de barras horizontales
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[prob_home, prob_draw, prob_away],
                y=[equipo_local, 'Empate', equipo_visitante],
                orientation='h',
                marker=dict(color=['#6366f1', '#d97706', '#ef4444'], opacity=0.85),
                text=[f"{prob_home:.1f}%", f"{prob_draw:.1f}%", f"{prob_away:.1f}%"],
                textposition='outside',
                textfont=dict(color='#2d3748', size=14, family='Inter')
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(255,255,255,0)',
                plot_bgcolor='rgba(255,255,255,0)',
                xaxis=dict(visible=False, range=[0, 115]),
                yaxis=dict(tickfont=dict(color='#2d3748', size=13)),
                margin=dict(l=10, r=70, t=10, b=10),
                height=180,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ─── BLOQUE DE CONFIABILIDAD ───────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔬 Confiabilidad del Modelo</div>', unsafe_allow_html=True)

            max_prob = max(prob_home, prob_draw, prob_away) / 100
            model_acc = metricas.get('accuracy', 0)
            model_f1  = metricas.get('f1_score_weighted', 0)
            cv_mean   = metricas.get('cv_accuracy_mean', 0)

            # El nivel de confianza combina dos señales:
            # 1) Accuracy global del modelo (qué tan bueno es en general)
            # 2) Probabilidad máxima de ESTE partido (qué tan diferenciado está el resultado)
            # Un partido muy desigual (Alemania vs San Marino) da max_prob ~85%,
            # lo que indica alta certeza aunque el modelo tenga solo 53% global.
            if max_prob >= 0.60 and model_acc >= 0.60:
                conf_label = "✅ Alta confianza — resultado casi seguro"
                conf_desc  = f"El modelo asigna {max_prob*100:.1f}% al favorito y en general acierta el {model_acc*100:.1f}% de partidos históricos. Ambas señales apuntan a un resultado bastante definido."
                conf_color = "#00c48c"
                conf_bg    = "rgba(0, 196, 140, 0.08)"
                conf_border= "#00c48c44"
            elif max_prob >= 0.60 and model_acc < 0.60:
                conf_label = "🟡 Partido muy desigual — pero modelo con precisión limitada"
                conf_desc  = f"Este partido específico tiene un favorito muy claro ({max_prob*100:.1f}%). Sin embargo, el modelo acierta globalmente solo el {model_acc*100:.1f}%, así que toma la predicción como orientativa."
                conf_color = "#ffd700"
                conf_bg    = "rgba(255, 215, 0, 0.08)"
                conf_border= "#ffd70044"
            elif max_prob >= 0.45:
                conf_label = "⚠️ Incertidumbre moderada — tomar con precaución"
                conf_desc  = f"El favorito tiene {max_prob*100:.1f}% de probabilidad pero los equipos están relativamente parejos. El modelo acerta el {model_acc*100:.1f}% en general — hay margen de sorpresa."
                conf_color = "#ffd700"
                conf_bg    = "rgba(255, 215, 0, 0.08)"
                conf_border= "#ffd70044"
            else:
                conf_label = "❌ Partido muy abierto — resultado impredecible"
                conf_desc  = f"Ningún resultado supera el {max_prob*100:.1f}% de probabilidad. El modelo acerta solo el {model_acc*100:.1f}% en general. Este partido es una moneda al aire."
                conf_color = "#ff6b6b"
                conf_bg    = "rgba(255, 107, 107, 0.08)"
                conf_border= "#ff6b6b44"

            st.markdown(f"""
            <div style='background:{conf_bg};border:1px solid {conf_border};border-radius:16px;padding:24px;margin-bottom:16px'>
                <div style='font-size:1.1rem;font-weight:700;color:{conf_color};margin-bottom:8px'>{conf_label}</div>
                <div style='color:rgba(255,255,255,0.55);font-size:0.85rem;line-height:1.6'>{conf_desc}</div>
                <div style='margin-top:14px;display:flex;gap:24px;flex-wrap:wrap'>
                    <div>
                        <div style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:rgba(255,255,255,0.3);margin-bottom:4px'>Accuracy del modelo</div>
                        <div style='font-size:1.6rem;font-weight:800;color:{conf_color}'>{model_acc*100:.1f}%</div>
                    </div>
                    <div>
                        <div style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:rgba(255,255,255,0.3);margin-bottom:4px'>F1-Score ponderado</div>
                        <div style='font-size:1.6rem;font-weight:800;color:rgba(255,255,255,0.7)'>{model_f1*100:.1f}%</div>
                    </div>
                    <div>
                        <div style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:rgba(255,255,255,0.3);margin-bottom:4px'>CV Accuracy (5-fold)</div>
                        <div style='font-size:1.6rem;font-weight:800;color:rgba(255,255,255,0.7)'>{cv_mean*100:.1f}%</div>
                    </div>
                </div>
            </div>
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:14px;color:rgba(255,255,255,0.35);font-size:0.76rem;line-height:1.7'>
            📋 <b style='color:rgba(255,255,255,0.5)'>¿Cómo leer esto?</b><br>
            El <b style='color:rgba(255,255,255,0.5)'>Accuracy</b> indica con qué frecuencia el modelo acertó el resultado en partidos históricos reales.<br>
            Si el modelo tiene un Accuracy de <b style='color:rgba(255,255,255,0.5)'>{model_acc*100:.1f}%</b>, significa que de cada 100 partidos del pasado,
            acertó el resultado en aproximadamente {int(model_acc*100)}. Eso determina cuánto puedes confiar en ese
            <b style='color:rgba(255,255,255,0.5)'>{max_prob*100:.1f}%</b> que asigna al favorito.<br>
            Nota: El fútbol tiene alta aleatoriedad — ningún modelo supera ~60-65% de accuracy de forma consistente.
            </div>
            """, unsafe_allow_html=True)



with tab2:
    st.markdown('<div class="hero-title" style="font-size:2rem; margin-top:20px;">🌍 Análisis de Selecciones</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Historial, rendimiento reciente y estadísticas de cada selección nacional</div>', unsafe_allow_html=True)
    
    sel_equipo = st.selectbox("Selecciona un País", equipos, key="sel_pais_tab2")
    
    if not partidos_df.empty:
        # Get historical matches for the selected team
        team_matches = partidos_df[(partidos_df['home_team'] == sel_equipo) | (partidos_df['away_team'] == sel_equipo)]
        if not team_matches.empty:
            team_stats = scores_df[scores_df['Equipo'] == sel_equipo].iloc[0] if not scores_df.empty and sel_equipo in scores_df['Equipo'].values else None
            
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            if team_stats is not None:
                c1.metric("Fuerza Histórica", f"{int(team_stats.get('Fuerza_Score', 0))}")
                c2.metric("Tasa de Victorias", f"{team_stats.get('Avg_Win_Rate', 0)*100:.1f}%")
                c3.metric("Partidos Int. de Plantilla", f"{int(team_stats.get('Total_Caps', 0))}")
                c4.metric("Edad Promedio", f"{team_stats.get('Avg_Age', 0):.1f}")
            
            st.markdown('<br><div class="section-title">📅 Últimos 15 Partidos</div>', unsafe_allow_html=True)
            ultimos = team_matches.head(15)[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']]
            
            # Format the table
            def color_result(row):
                if row['home_score'] > row['away_score']:
                    return ['background-color: rgba(99, 102, 241, 0.08)'] * len(row) if row['home_team'] == sel_equipo else ['background-color: rgba(239, 68, 68, 0.08)'] * len(row)
                elif row['home_score'] < row['away_score']:
                    return ['background-color: rgba(239, 68, 68, 0.08)'] * len(row) if row['home_team'] == sel_equipo else ['background-color: rgba(99, 102, 241, 0.08)'] * len(row)
                else:
                    return ['background-color: rgba(217, 119, 6, 0.08)'] * len(row)
                    
            st.dataframe(ultimos.style.apply(color_result, axis=1), use_container_width=True, hide_index=True)
            
        else:
            st.warning("No hay partidos históricos registrados para esta selección.")
    else:
        st.warning("No se pudo cargar el dataset de partidos históricos.")

with tab3:
    st.markdown('<div class="hero-title" style="font-size:2rem; margin-top:20px;">🏃 Buscador de Jugadores</div>', unsafe_allow_html=True)
    
    if not jugadores_df.empty:
        paises_disp = sorted(jugadores_df['Pais'].dropna().unique().tolist())
        col_p, col_j = st.columns(2)
        with col_p:
            sel_pais_jug = st.selectbox("Filtro por País", paises_disp, key="sel_pais_jug")
        
        jugadores_pais = jugadores_df[jugadores_df['Pais'] == sel_pais_jug]
        jugadores_nombres = sorted(jugadores_pais['Nombre'].dropna().unique().tolist())
        
        with col_j:
            sel_jugador = st.selectbox("Selecciona un Jugador", jugadores_nombres, key="sel_jugador")
            
        if sel_jugador:
            jug = jugadores_pais[jugadores_pais['Nombre'] == sel_jugador].iloc[0]
            
            st.markdown("---")
            
            def get_val(val, default="N/A"):
                if pd.isna(val) or val == "":
                    return default
                return val
                
            st.markdown(f"""
            <div class="glass-card" style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h2 style="margin:0; color:#6366f1;">{get_val(jug.get('Nombre'))}</h2>
                    <p style="color:#718096; font-size:1.1rem; margin:5px 0;">{get_val(jug.get('Pos.'))} · {get_val(jug.get('Clubes'))}</p>
                </div>
                <div style="text-align:right; margin-left: 20px;">
                    <div style="color:#a0aec0; font-size:0.8rem; text-transform:uppercase;">Edad</div>
                    <div style="font-size:1.5rem; font-weight:bold; color:#1a202c;">{get_val(jug.get('Edad'))}</div>
                </div>
                <div style="text-align:right; margin-left: 20px;">
                    <div style="color:#a0aec0; font-size:0.8rem; text-transform:uppercase;">Físico</div>
                    <div style="font-size:1.5rem; font-weight:bold; color:#1a202c;">{get_val(jug.get('Tamaño'), '-')} / {get_val(jug.get('Peso'), '-')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">📊 Estadísticas Globales</div>', unsafe_allow_html=True)
            
            def parse_num(val):
                try:
                    if pd.isna(val): return 0
                    return int(float(str(val).replace(',', '')))
                except:
                    return 0
            
            j1, j2, j3, j4 = st.columns(4)
            j1.metric("Partidos Int.", parse_num(jug.get('Selecciones')))
            j2.metric("Goles Int.", parse_num(jug.get('Goles_Int')))
            j3.metric("Victorias Totales", parse_num(jug.get('Victoria_Total')))
            j4.metric("Partidos Clubes", parse_num(jug.get('PartidosPartidos jugados')))
            
            j5, j6, j7, j8 = st.columns(4)
            j5.metric("Goles Totales", parse_num(jug.get('Goles paraGoles')))
            j6.metric("Efectividad", str(get_val(jug.get('EfectividadEfectividad'), '0%')))
            j7.metric("Minutos Jugados", parse_num(jug.get('Minutas jugadasMinutes')))
            j8.metric("Asistencias", parse_num(jug.get('Pases dec.Pases dec.')))
            
            j9, j10, j11, j12 = st.columns(4)
            j9.metric("Empates Int.", parse_num(jug.get('Empate_Total')))
            j10.metric("Derrotas Int.", parse_num(jug.get('Derrota_Total')))
            j11.metric("Tarjetas Amarillas", parse_num(jug.get('Tarjetas amarillasyc')))
            j12.metric("Tarjetas Rojas", parse_num(jug.get('Tarjetas rojasrc')))
    else:
        st.warning("No se pudo cargar el dataset de jugadores.")
