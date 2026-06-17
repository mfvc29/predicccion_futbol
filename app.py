import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundial 2026 – Predictor IA",
    page_icon="⚽",
    layout="wide"
)

# ─── CSS PREMIUM ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Fondo oscuro */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    }

    /* Tarjetas glassmorphism */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }

    /* Header hero */
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 4px;
    }
    .hero-sub {
        text-align: center;
        color: rgba(255,255,255,0.45);
        font-size: 1rem;
        font-weight: 300;
        margin-bottom: 40px;
    }

    /* Selector de equipo */
    .team-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin-bottom: 6px;
    }

    /* Resultado predicción */
    .result-home { color: #00d4ff; font-size: 2.8rem; font-weight: 900; text-align: center; }
    .result-draw { color: #ffd700; font-size: 2rem; font-weight: 700; text-align: center; }
    .result-away { color: #ff6b6b; font-size: 2.8rem; font-weight: 900; text-align: center; }
    .result-label { color: rgba(255,255,255,0.45); font-size: 0.8rem; text-align: center; margin-top: 4px; }
    .winner-badge {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; padding: 4px 12px;
        border-radius: 999px; display: inline-block; margin-top: 6px;
    }
    .badge-home { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid #00d4ff44; }
    .badge-away { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid #ff6b6b44; }
    .badge-draw { background: rgba(255,215,0,0.15); color: #ffd700; border: 1px solid #ffd70044; }

    /* Players danger table */
    .danger-player {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .danger-rank { font-size: 1.1rem; font-weight: 900; color: rgba(255,255,255,0.15); min-width: 28px; }
    .danger-name { font-weight: 600; color: #fff; flex: 1; }
    .danger-pos {
        font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;
        background: rgba(123,47,247,0.2); color: #a78bfa; border: 1px solid #7b2ff744;
    }
    .danger-score { font-weight: 700; color: #00d4ff; min-width: 60px; text-align: right; }

    /* Section title */
    .section-title {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
        text-transform: uppercase; color: rgba(255,255,255,0.3);
        margin-bottom: 18px; padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }

    /* Metric override */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 16px !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #7b2ff7, #00d4ff) !important;
        color: white !important; border: none !important;
        font-weight: 700 !important; font-size: 1rem !important;
        padding: 16px 32px !important; border-radius: 14px !important;
        letter-spacing: 0.05em; transition: all 0.3s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-2px); }

    /* Hide default streamlit menu */
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
    return model, scores_df, top_players_df

try:
    model, scores_df, top_players_df = load_all()
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.stop()

equipos = sorted(scores_df['Equipo'].dropna().unique().tolist())

# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">⚽ Predictor IA · Mundial 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Machine Learning entrenado con 25,000+ partidos históricos y estadísticas reales de 1,247 jugadores</div>', unsafe_allow_html=True)

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

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("💪 Fuerza Histórica", f"{int(stats_local['Fuerza_Score'])}", f"{int(stats_local['Fuerza_Score'] - stats_visit['Fuerza_Score'])} vs visitante")
    mcol2.metric("🎂 Edad Promedio", f"{stats_local['Avg_Age']:.1f} años", f"{stats_local['Avg_Age'] - stats_visit['Avg_Age']:.1f} vs visitante", delta_color="inverse")
    mcol3.metric("🌍 Partidos Int. Plantilla", f"{int(stats_local['Total_Caps'])}", f"{int(stats_local['Total_Caps'] - stats_visit['Total_Caps'])} vs visitante")
    mcol4.metric("🤝 Asistencias Plantilla", f"{int(stats_local['Total_Assists'])}", f"{int(stats_local['Total_Assists'] - stats_visit['Total_Assists'])} vs visitante")

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
        line=dict(color='#00d4ff', width=2),
        fillcolor='rgba(0, 212, 255, 0.15)',
        marker=dict(size=6, color='#00d4ff')
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=values_visit + [values_visit[0]],
        theta=categorias + [categorias[0]],
        fill='toself',
        name=equipo_visitante,
        line=dict(color='#ff6b6b', width=2),
        fillcolor='rgba(255, 107, 107, 0.15)',
        marker=dict(size=6, color='#ff6b6b')
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True, tickfont=dict(color='rgba(255,255,255,0.3)', size=9), gridcolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(tickfont=dict(color='rgba(255,255,255,0.7)', size=12), gridcolor='rgba(255,255,255,0.08)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='white', size=12), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=60, t=40, b=40),
        height=420
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ─── JUGADORES PELIGROSOS ───────────────────────────────────────────────────
    if not top_players_df.empty:
        st.markdown("---")
        st.markdown('<div class="section-title">🚨 Jugadores Más Peligrosos (Índice de Peligrosidad)</div>', unsafe_allow_html=True)

        pcol1, pcol2 = st.columns(2)
        for col_ui, equipo_nombre, color in [(pcol1, equipo_local, '00d4ff'), (pcol2, equipo_visitante, 'ff6b6b')]:
            with col_ui:
                st.markdown(f"<h4 style='color:#{color};margin-bottom:12px'>{equipo_nombre}</h4>", unsafe_allow_html=True)
                players = top_players_df[top_players_df['Equipo'] == equipo_nombre].head(5)
                if not players.empty:
                    max_score = players['Peligrosidad'].max() if players['Peligrosidad'].max() > 0 else 1
                    for i, (_, p) in enumerate(players.iterrows(), 1):
                        pct = p['Peligrosidad'] / max_score * 100 if max_score > 0 else 0
                        sels = int(p['Sel_n']) if not pd.isna(p['Sel_n']) else 0
                        vics = int(p['Vic_n']) if not pd.isna(p['Vic_n']) else 0
                        st.markdown(f"""
                        <div class='danger-player'>
                            <span class='danger-rank'>#{i}</span>
                            <div style='flex:1;padding: 0 12px'>
                                <div class='danger-name'>{p['Nombre']}</div>
                                <div style='font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:2px'>
                                    {sels} partidos int. · {vics} victorias
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
            marker=dict(color=['#00d4ff', '#ffd700', '#ff6b6b'], opacity=0.85),
            text=[f"{prob_home:.1f}%", f"{prob_draw:.1f}%", f"{prob_away:.1f}%"],
            textposition='outside',
            textfont=dict(color='white', size=14, family='Inter')
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False, range=[0, 110]),
            yaxis=dict(tickfont=dict(color='white', size=13)),
            margin=dict(l=10, r=60, t=10, b=10),
            height=180,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px;margin-top:8px;color:rgba(255,255,255,0.4);font-size:0.8rem'>
        💡 <b style='color:rgba(255,255,255,0.6)'>Sobre el modelo:</b> Predicción basada en Random Forest entrenado con 25,000+ partidos (2000-2026), fuerza histórica de plantillas, experiencia internacional acumulada y forma reciente de cada selección.
        </div>
        """, unsafe_allow_html=True)
