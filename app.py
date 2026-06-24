"""
app.py - Predictor Mundial 2026 ML
====================================
Aplicación Streamlit de predicción de partidos de fútbol.
Usa datos 100% reales de la API football-data.org.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Predictor Mundial 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS PREMIUM BLANCO ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── FONDO PREMIUM ── */
    .stApp {
        background: linear-gradient(160deg, #f0f4ff 0%, #fafbff 50%, #f5f0ff 100%);
        min-height: 100vh;
    }
    .stApp > header { background: transparent !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 20px rgba(99,102,241,0.06);
    }

    /* ── HERO BANNER ── */
    .hero-wrap {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0ea5e9 100%);
        border-radius: 24px;
        padding: 40px 48px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(79,70,229,0.3);
    }
    .hero-wrap::before {
        content: '';
        position: absolute; top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: rgba(255,255,255,0.07);
        border-radius: 50%;
    }
    .hero-wrap::after {
        content: '';
        position: absolute; bottom: -40%; left: -10%;
        width: 300px; height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.1;
    }
    .hero-title span { color: #93c5fd; }
    .hero-sub {
        color: rgba(255,255,255,0.75);
        font-size: 1.05rem;
        margin-top: 10px;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        color: white;
        letter-spacing: 0.06em;
        margin-right: 8px; margin-top: 14px;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 5px;
        border: 1px solid #e2e8f0;
        gap: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 10px 18px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(79,70,229,0.3) !important;
    }

    /* ── CARDS ── */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .card:hover {
        box-shadow: 0 8px 24px rgba(99,102,241,0.12);
        transform: translateY(-2px);
    }
    .card-header {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
        display: flex; align-items: center; gap: 8px;
    }
    .card-header span { color: #4f46e5; font-size: 1rem; }

    /* ── TEAM SELECTOR ── */
    .team-pill {
        display: flex; align-items: center; justify-content: center;
        gap: 12px;
        background: linear-gradient(135deg, #f0f4ff, #faf5ff);
        border: 2px solid #e0e7ff;
        border-radius: 16px;
        padding: 16px 20px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e1b4b;
        margin: 8px 0;
    }
    .vs-badge {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 900;
        color: #94a3b8;
        padding: 12px;
    }

    /* ── RESULTADO PREDICCIÓN ── */
    .result-container {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
        border: 2px solid #e0e7ff;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        margin: 18px 0;
    }
    .result-winner {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #4f46e5;
        margin-bottom: 6px;
    }
    .result-score-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 24px;
        margin: 18px 0;
    }
    .result-prob {
        text-align: center;
        flex: 1;
    }
    .result-prob-value {
        font-size: 2.2rem;
        font-weight: 900;
        line-height: 1;
    }
    .result-prob-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-top: 4px;
    }
    .prob-home   { color: #4f46e5; }
    .prob-draw   { color: #f59e0b; }
    .prob-away   { color: #ef4444; }
    .result-divider { color: #cbd5e1; font-size: 1.2rem; }

    /* Confianza badge */
    .confidence-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 16px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 700; margin-top: 12px;
    }
    .conf-high   { background: #dcfce7; color: #16a34a; border: 1px solid #86efac; }
    .conf-medium { background: #fef9c3; color: #ca8a04; border: 1px solid #fde047; }
    .conf-low    { background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }

    /* ── PLAYERS TABLE ── */
    .player-row {
        display: flex; align-items: center; gap: 12px;
        padding: 12px 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .player-rank {
        font-size: 1.05rem; font-weight: 900;
        color: #e2e8f0; min-width: 30px; text-align: center;
    }
    .player-info { flex: 1; }
    .player-name { font-weight: 700; color: #1e293b; font-size: 0.95rem; }
    .player-meta { font-size: 0.78rem; color: #94a3b8; margin-top: 2px; }
    .player-pos {
        font-size: 0.7rem; font-weight: 700; padding: 3px 10px;
        border-radius: 999px; white-space: nowrap;
    }
    .pos-gk  { background: #fef9c3; color: #854d0e; border: 1px solid #fde68a; }
    .pos-def { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .pos-mid { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
    .pos-fwd { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

    /* ── STAT BOX ── */
    .stat-box {
        background: #f8faff;
        border: 1px solid #e0e7ff;
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-box-value {
        font-size: 1.6rem; font-weight: 800; color: #1e1b4b;
        line-height: 1;
    }
    .stat-box-label {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #94a3b8; margin-top: 4px;
    }
    .stat-box-delta {
        font-size: 0.8rem; font-weight: 600; margin-top: 3px;
    }
    .delta-pos { color: #16a34a; }
    .delta-neg { color: #dc2626; }

    /* ── RANKING TABLE ── */
    .rank-row {
        display: flex; align-items: center; gap: 14px;
        padding: 10px 14px; border-radius: 12px; margin-bottom: 6px;
        background: #f8faff; border: 1px solid #e0e7ff;
        transition: all 0.15s;
    }
    .rank-row:hover { background: #ede9fe; border-color: #c4b5fd; }
    .rank-num { font-weight: 900; color: #c7d2fe; min-width: 24px; font-size: 0.95rem; }
    .rank-name { flex: 1; font-weight: 700; color: #1e1b4b; font-size: 0.92rem; }
    .rank-bar-wrap { width: 100px; height: 6px; background: #e0e7ff; border-radius: 999px; overflow: hidden; }
    .rank-bar { height: 100%; background: linear-gradient(90deg, #4f46e5, #7c3aed); border-radius: 999px; }
    .rank-score { font-weight: 800; color: #4f46e5; font-size: 0.9rem; min-width: 40px; text-align: right; }

    /* ── METRICAS GLOBALES ── */
    [data-testid="metric-container"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important; font-weight: 800 !important;
        color: #1e1b4b !important;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricDelta"] { font-weight: 600 !important; }

    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important; border: none !important;
        font-weight: 700 !important; font-size: 1rem !important;
        padding: 14px 32px !important; border-radius: 14px !important;
        letter-spacing: 0.03em;
        box-shadow: 0 6px 20px rgba(79,70,229,0.3) !important;
        transition: all 0.25s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(79,70,229,0.4) !important;
    }

    /* ── SELECTBOX ── */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1e1b4b !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
    }

    /* Checkbox */
    .stCheckbox > label { color: #64748b !important; font-weight: 500; }

    /* Divider */
    hr { border-color: #e2e8f0 !important; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #e2e8f0 !important; border-radius: 14px !important; }

    /* Texto general */
    p, li { color: #374151; }
    h1, h2, h3, h4 { color: #1e1b4b !important; }
    label { color: #64748b !important; }

    /* Expander */
    .streamlit-expanderHeader { color: #4f46e5 !important; font-weight: 700 !important; }

    /* LIVE badge */
    .live-dot {
        display: inline-block; width: 8px; height: 8px;
        background: #22c55e; border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(34,197,94,0.25);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 3px rgba(34,197,94,0.25); }
        50%       { box-shadow: 0 0 0 6px rgba(34,197,94,0.1); }
    }

    /* Hide streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_datos():
    """Carga todos los recursos necesarios para la app."""
    import os

    # Modelo
    model_data = joblib.load("modelo_futbol.pkl")
    is_ensemble = isinstance(model_data, dict) and ("gb" in model_data or "hgb" in model_data)

    # Métricas
    with open("modelo_metricas.json") as f:
        metricas = json.load(f)

    # Scores de equipos
    equipos_df = pd.read_csv("equipos_scores.csv")

    # Partidos API reales (si existen)
    matches_api = None
    if os.path.exists("api_matches_all.csv"):
        matches_api = pd.read_csv("api_matches_all.csv")
        matches_api["date"] = pd.to_datetime(matches_api["date"])

    # Partidos históricos (fallback)
    partidos_hist = pd.read_csv("partidos_internacionales_historico.csv") if os.path.exists("partidos_internacionales_historico.csv") else None
    if partidos_hist is not None:
        partidos_hist["date"] = pd.to_datetime(partidos_hist["date"])

    # Jugadores WC2026 — preferir de la API
    players_api = None
    if os.path.exists("api_wc2026_players.csv"):
        players_api = pd.read_csv("api_wc2026_players.csv")
    elif os.path.exists("jugadores_mundial_2026_con_stats_final.csv"):
        players_api = pd.read_csv("jugadores_mundial_2026_con_stats_final.csv")

    # Team stats API
    team_stats = None
    if os.path.exists("api_team_stats.csv"):
        team_stats = pd.read_csv("api_team_stats.csv")

    # Partidos WC2026 en vivo
    wc_matches = None
    if matches_api is not None:
        wc_matches = matches_api[matches_api["competition"] == "WC"].copy()

    # Predicciones honestas (OOS)
    predicciones_honestas = {}
    if os.path.exists("predicciones_honestas.json"):
        with open("predicciones_honestas.json", "r") as f:
            predicciones_honestas = json.load(f)

    return {
        "model": model_data,
        "is_ensemble": is_ensemble,
        "metricas": metricas,
        "equipos": equipos_df,
        "matches_api": matches_api,
        "partidos_hist": partidos_hist,
        "players": players_api,
        "team_stats": team_stats,
        "wc_matches": wc_matches,
        "predicciones_honestas": predicciones_honestas,
    }

datos = cargar_datos()
model_data   = datos["model"]
is_ensemble  = datos["is_ensemble"]
metricas     = datos["metricas"]
equipos_df   = datos["equipos"]
matches_api  = datos["matches_api"]
partidos_hist = datos["partidos_hist"]
players_df   = datos["players"]
team_stats   = datos["team_stats"]
wc_matches   = datos["wc_matches"]
predicciones_honestas = datos["predicciones_honestas"]

equipos_list = sorted(equipos_df["Equipo"].dropna().unique().tolist())

# Rankings FIFA aproximados (Top 50)
RANKINGS_FIFA = {
    "Argentina": 1, "France": 2, "Belgium": 3, "England": 4, "Brazil": 5,
    "Portugal": 6, "Netherlands": 7, "Spain": 8, "Croatia": 10, "United States": 11,
    "Colombia": 12, "Morocco": 13, "Mexico": 15, "Uruguay": 15, "Germany": 16,
    "Senegal": 17, "Japan": 18, "Switzerland": 19, "Iran": 20, "South Korea": 22,
    "Australia": 23, "Austria": 25, "Sweden": 26, "Turkey": 35, "Ecuador": 31,
    "Egypt": 36, "Tunisia": 41, "Canada": 48, "Saudi Arabia": 53, "Ghana": 67,
    "South Africa": 58, "Algeria": 43, "New Zealand": 104, "Czechia": 36,
    "Paraguay": 56, "Bosnia-Herzegovina": 71, "Panama": 43, "Cape Verde Islands": 65,
    "Congo DR": 63, "Ivory Coast": 38, "Qatar": 34, "Jordan": 71, "Iraq": 58,
    "Uzbekistan": 64, "Norway": 47, "Scotland": 39, "Curaçao": 91, "Haiti": 90
}

def get_rank(team):
    return RANKINGS_FIFA.get(team, 80)

# ═════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE PREDICCIÓN
# ═════════════════════════════════════════════════════════════════════════════
def predecir(home, away):
    """Hace la predicción usando el modelo (ensemble o simple)."""
    eq_idx = equipos_df.set_index("Equipo")

    def feat(team):
        if team in eq_idx.index:
            r = eq_idx.loc[team]
            return {
                "win_rate":       r.get("Win_Rate", 0.33),
                "pts_per_game":   r.get("Pts_Per_Game", 1.0),
                "goles_favor":    r.get("Goles_Por_Partido", 1.0),
                "goles_contra":   r.get("Goles_Recibidos", 1.2),
                "clean_sheet_rate": r.get("Clean_Sheet_Rate", 0.25),
                "score_rate":     r.get("Score_Rate", 0.7),
                "Fuerza_Score":   r.get("Fuerza_Score", 50),
                "diferencia_goles_total": r.get("Diff_Goles", 0),
                "partidos_jugados": r.get("Total_Caps", r.get("Partidos", 10)),
                "rank":           get_rank(team),
            }
        return {k: 0 for k in ["win_rate","pts_per_game","goles_favor","goles_contra",
                                 "clean_sheet_rate","score_rate","Fuerza_Score",
                                 "diferencia_goles_total","partidos_jugados", "rank"]}


    h = feat(home)
    a = feat(away)

    row = {}
    for k, v in h.items():
        row[f"home_{k}"] = v
    for k, v in a.items():
        row[f"away_{k}"] = v
    row["diff_fuerza"]   = h["Fuerza_Score"] - a["Fuerza_Score"]
    row["diff_win_rate"] = h["win_rate"] - a["win_rate"]
    row["diff_goles"]    = h["goles_favor"] - a["goles_favor"]
    row["diff_concedidos"] = h["goles_contra"] - a["goles_contra"]
    row["diff_pts"]      = h["pts_per_game"] - a["pts_per_game"]
    row["diff_clean"]    = h["clean_sheet_rate"] - a["clean_sheet_rate"]
    row["diff_rank"]     = h["rank"] - a["rank"]

    if is_ensemble:
        features = model_data["features"]
        X = pd.DataFrame([{f: row.get(f, 0) for f in features}])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Soportar tanto hgb (nuevo) como gb (viejo)
        if "hgb" in model_data:
            p1 = model_data["hgb"].predict_proba(X)
        elif "gb" in model_data:
            p1 = model_data["gb"].predict_proba(X)
        else:
            p1 = np.array([[0.33, 0.33, 0.34]])

        p2 = model_data["rf"].predict_proba(X)
        proba = (p1 + p2) / 2
        classes = model_data["classes"]
    else:
        # Modelo simple (legacy)
        fuerza_h = h["Fuerza_Score"]
        fuerza_a = a["Fuerza_Score"]
        legacy_row = {
            "home_Fuerza_Score": fuerza_h,
            "away_Fuerza_Score": fuerza_a,
            "home_Avg_Age": 27, "away_Avg_Age": 27,
            "home_Total_Caps": h["partidos_jugados"],
            "away_Total_Caps": a["partidos_jugados"],
            "home_Total_Goals_Scored": h["goles_favor"],
            "away_Total_Goals_Scored": a["goles_favor"],
            "home_Total_Assists": 0, "away_Total_Assists": 0,
            "score_diff": fuerza_h - fuerza_a, "neutral": 0
        }
        X = pd.DataFrame([legacy_row])
        proba = model_data.predict_proba(X)
        classes = model_data.classes_

    # Mapear probabilidades
    prob_map = dict(zip(classes.tolist(), proba[0].tolist()))
    prob_home = round(prob_map.get(1, 0) * 100, 1)
    prob_draw = round(prob_map.get(0, 0) * 100, 1)
    prob_away = round(prob_map.get(-1, 0) * 100, 1)
    pred_class = int(classes[np.argmax(proba[0])])

    return prob_home, prob_draw, prob_away, pred_class


def get_pos_class(pos):
    if not pos: return "pos-mid"
    pos = str(pos).lower()
    if "goal" in pos or "portero" in pos: return "pos-gk"
    if "defend" in pos or "defensa" in pos: return "pos-def"
    if "forward" in pos or "delantero" in pos or "atacante" in pos: return "pos-fwd"
    return "pos-mid"

def pos_label(pos):
    if not pos: return "MF"
    pos = str(pos)
    m = {"Goalkeeper":"PO","Defender":"DF","Midfielder":"MC","Forward":"DE"}
    return m.get(pos, pos[:2].upper())

def get_team_logo(team_name):
    logos = {
        "Argentina": "🇦🇷", "France": "🇫🇷", "Brazil": "🇧🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
        "Spain": "🇪🇸", "Germany": "🇩🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱",
        "Italy": "🇮🇹", "Belgium": "🇧🇪", "Uruguay": "🇺🇾", "Croatia": "🇭🇷",
        "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Japan": "🇯🇵", "South Korea": "🇰🇷",
        "Mexico": "🇲🇽", "United States": "🇺🇸", "Canada": "🇨🇦", "Australia": "🇦🇺",
        "Colombia": "🇨🇴", "Chile": "🇨🇱", "Ecuador": "🇪🇨", "Peru": "🇵🇪",
        "Nigeria": "🇳🇬", "Cameroon": "🇨🇲", "Ghana": "🇬🇭", "Algeria": "🇩🇿",
        "Poland": "🇵🇱", "Switzerland": "🇨🇭", "Denmark": "🇩🇰", "Serbia": "🇷🇸",
        "Austria": "🇦🇹", "Turkey": "🇹🇷", "Ukraine": "🇺🇦", "Hungary": "🇭🇺",
        "Iran": "🇮🇷", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Costa Rica": "🇨🇷",
        "Paraguay": "🇵🇾", "Bolivia": "🇧🇴", "Venezuela": "🇻🇪",
    }
    return logos.get(team_name, "🏳️")


# ═════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Predictor <span>Mundial 2026</span> ⚽</div>
    <div class="hero-sub">
        Machine Learning con datos 100% reales · Ensemble GradientBoosting + RandomForest
    </div>
    <div style="margin-top: 16px;">
        <span class="hero-badge">🤖 ML Avanzado</span>
        <span class="hero-badge"><span class="live-dot"></span>&nbsp; Datos Live API</span>
        <span class="hero-badge">🏆 48 Selecciones</span>
        <span class="hero-badge">📊 Datos Reales</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ Predictor",
    "🏆 Resultados WC2026",
    "🌍 Selecciones",
    "👤 Jugadores",
    "🥇 Ganadores",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1.3, 1], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>⚙️</span> Configurar Partido</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            equipo_local = st.selectbox("🏠 Equipo Local", equipos_list, index=equipos_list.index("Argentina") if "Argentina" in equipos_list else 0, key="sel_home")
        with c2:
            opciones_visit = [e for e in equipos_list if e != equipo_local]
            equipo_visit = st.selectbox("✈️ Equipo Visitante", opciones_visit, index=opciones_visit.index("France") if "France" in opciones_visit else 0, key="sel_away")

        # Visual equipo selector
        e1f = get_team_logo(equipo_local)
        e2f = get_team_logo(equipo_visit)
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin: 18px 0;">
            <div style="flex:1; text-align:center;">
                <div style="font-size:2.5rem;">{e1f}</div>
                <div style="font-weight:700; color:#1e1b4b; font-size:0.95rem; margin-top:6px;">{equipo_local}</div>
                <div style="font-size:0.75rem; color:#94a3b8;">Local</div>
            </div>
            <div style="color:#cbd5e1; font-size:1.8rem; font-weight:900;">VS</div>
            <div style="flex:1; text-align:center;">
                <div style="font-size:2.5rem;">{e2f}</div>
                <div style="font-weight:700; color:#1e1b4b; font-size:0.95rem; margin-top:6px;">{equipo_visit}</div>
                <div style="font-size:0.75rem; color:#94a3b8;">Visitante</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        neutral = st.checkbox("⚖️ Sede neutral (torneo / cancha única)", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Comparación de Stats
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>📊</span> Comparación de Métricas Reales</div>', unsafe_allow_html=True)
        if team_stats is not None:
            ts_idx = team_stats.set_index("team")
            ht_row = ts_idx.loc[equipo_local] if equipo_local in ts_idx.index else None
            at_row = ts_idx.loc[equipo_visit] if equipo_visit in ts_idx.index else None

            if ht_row is not None and at_row is not None:
                # Radar Chart
                cats = ["Win Rate", "Goles/Partido", "Limpieza", "Efectividad", "Pts/Partido"]
                max_vals = [1.0, 4.0, 1.0, 1.0, 3.0]
                h_vals = [
                    ht_row["win_rate"],
                    min(ht_row["goles_por_partido"], 4.0),
                    ht_row["clean_sheet_rate"],
                    ht_row["score_rate"],
                    min(ht_row["pts_per_game"], 3.0),
                ]
                a_vals = [
                    at_row["win_rate"],
                    min(at_row["goles_por_partido"], 4.0),
                    at_row["clean_sheet_rate"],
                    at_row["score_rate"],
                    min(at_row["pts_per_game"], 3.0),
                ]
                h_norm = [v/m*100 for v, m in zip(h_vals, max_vals)]
                a_norm = [v/m*100 for v, m in zip(a_vals, max_vals)]

                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(
                    r=h_norm + [h_norm[0]], theta=cats + [cats[0]],
                    fill="toself", name=equipo_local,
                    line=dict(color="#4f46e5", width=2.5),
                    fillcolor="rgba(79,70,229,0.10)",
                    marker=dict(size=6, color="#4f46e5")
                ))
                fig_r.add_trace(go.Scatterpolar(
                    r=a_norm + [a_norm[0]], theta=cats + [cats[0]],
                    fill="toself", name=equipo_visit,
                    line=dict(color="#ef4444", width=2.5),
                    fillcolor="rgba(239,68,68,0.08)",
                    marker=dict(size=6, color="#ef4444")
                ))
                fig_r.update_layout(
                    polar=dict(
                        bgcolor="rgba(255,255,255,0)",
                        radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#e0e7ff"),
                        angularaxis=dict(tickfont=dict(color="#475569", size=12), gridcolor="#e0e7ff"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#1e1b4b", size=12), bgcolor="rgba(255,255,255,0)", orientation="h", y=-0.15),
                    margin=dict(l=50, r=50, t=20, b=40), height=320,
                )
                st.plotly_chart(fig_r, use_container_width=True)

                # Barras de stats lado a lado
                stats_labels = ["Victorias", "Empates", "Derrotas", "Goles a Favor", "Goles en Contra", "Partidos"]
                stats_h = [int(ht_row["wins"]), int(ht_row["draws"]), int(ht_row["losses"]),
                           int(ht_row["goles_favor"]), int(ht_row["goles_contra"]), int(ht_row["partidos"])]
                stats_a = [int(at_row["wins"]), int(at_row["draws"]), int(at_row["losses"]),
                           int(at_row["goles_favor"]), int(at_row["goles_contra"]), int(at_row["partidos"])]

                mc1, mc2, mc3 = st.columns(3)
                for col_s, lab, hv, av in zip(
                    [mc1, mc2, mc3, mc1, mc2, mc3],
                    stats_labels, stats_h, stats_a
                ):
                    diff = hv - av
                    delta_html = f"<div class='stat-box-delta {'delta-pos' if diff >= 0 else 'delta-neg'}'>{'▲' if diff > 0 else ('▼' if diff < 0 else '─')} {abs(diff)}</div>" if diff != 0 else ""
                    col_s.markdown(f"""
                    <div class="stat-box">
                        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#94a3b8;">{lab}</div>
                        <div style="display:flex;justify-content:center;gap:12px;margin-top:8px;align-items:center;">
                            <div style="text-align:center;">
                                <div style="font-size:1.2rem;font-weight:800;color:#4f46e5;">{hv}</div>
                                <div style="font-size:0.65rem;color:#94a3b8;">{equipo_local[:8]}</div>
                            </div>
                            <div style="color:#e2e8f0;font-size:1rem;">|</div>
                            <div style="text-align:center;">
                                <div style="font-size:1.2rem;font-weight:800;color:#ef4444;">{av}</div>
                                <div style="font-size:0.65rem;color:#94a3b8;">{equipo_visit[:8]}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")
            else:
                st.info("Sin estadísticas suficientes para este enfrentamiento.")
        else:
            st.info("Ejecuta fetch_api_data.py para cargar stats reales de equipos.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # BOTÓN PREDICTOR
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>🤖</span> Resultado Previsto</div>', unsafe_allow_html=True)

        if st.button("⚡ PREDECIR RESULTADO", key="btn_predict"):
            with st.spinner("Analizando con IA..."):
                ph, pd_p, pa, pred = predecir(equipo_local, equipo_visit)

            # Determinar ganador
            if pred == 1:
                ganador = equipo_local
                badge_class = "conf-high" if ph > 50 else "conf-medium"
                resultado_texto = f"Gana {equipo_local}"
                resultado_emoji = get_team_logo(equipo_local)
            elif pred == -1:
                ganador = equipo_visit
                badge_class = "conf-high" if pa > 50 else "conf-medium"
                resultado_texto = f"Gana {equipo_visit}"
                resultado_emoji = get_team_logo(equipo_visit)
            else:
                ganador = "Empate"
                badge_class = "conf-medium"
                resultado_texto = "🤝 Empate"
                resultado_emoji = "⚖️"

            confianza_val = max(ph, pd_p, pa)
            conf_class = "conf-high" if confianza_val > 55 else ("conf-medium" if confianza_val > 40 else "conf-low")
            conf_label = "Alta" if confianza_val > 55 else ("Media" if confianza_val > 40 else "Baja")

            st.markdown(f"""
            <div class="result-container">
                <div style="font-size:2.5rem;">{resultado_emoji}</div>
                <div class="result-winner">{resultado_texto}</div>
                <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:18px;">Predicción del modelo ML</div>
                <div class="result-score-row">
                    <div class="result-prob">
                        <div class="result-prob-value prob-home">{ph}%</div>
                        <div class="result-prob-label">🏠 {equipo_local}</div>
                    </div>
                    <div class="result-divider">|</div>
                    <div class="result-prob">
                        <div class="result-prob-value prob-draw">{pd_p}%</div>
                        <div class="result-prob-label">🤝 Empate</div>
                    </div>
                    <div class="result-divider">|</div>
                    <div class="result-prob">
                        <div class="result-prob-value prob-away">{pa}%</div>
                        <div class="result-prob-label">✈️ {equipo_visit}</div>
                    </div>
                </div>
                <span class="confidence-badge {conf_class}">
                    {'✅' if conf_class=='conf-high' else ('⚠️' if conf_class=='conf-medium' else '❓')}
                    Confianza {conf_label} ({confianza_val:.0f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Gráfico barras probabilidades
            fig_b = go.Figure(go.Bar(
                x=[ph, pd_p, pa],
                y=[equipo_local, "Empate", equipo_visit],
                orientation="h",
                marker=dict(
                    color=["#4f46e5", "#f59e0b", "#ef4444"],
                    line=dict(color=["#3730a3","#d97706","#dc2626"], width=1.5)
                ),
                text=[f"{ph:.1f}%", f"{pd_p:.1f}%", f"{pa:.1f}%"],
                textposition="outside",
                textfont=dict(color="#1e1b4b", size=14, family="Inter")
            ))
            fig_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False, range=[0, 120]),
                yaxis=dict(tickfont=dict(color="#1e1b4b", size=13)),
                margin=dict(l=10, r=70, t=10, b=10), height=180, showlegend=False,
            )
            st.plotly_chart(fig_b, use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:40px 20px; color:#94a3b8;">
                <div style="font-size:3rem; margin-bottom:12px;">⚡</div>
                <div style="font-weight:600; font-size:1rem;">Configura los equipos y presiona</div>
                <div style="font-size:0.85rem; margin-top:4px;">PREDECIR RESULTADO</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # MODELO INFO
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>🔬</span> Métricas del Modelo</div>', unsafe_allow_html=True)
        acc = metricas.get("accuracy", 0)
        f1  = metricas.get("f1_score_weighted", 0)
        cv  = metricas.get("cv_accuracy_mean", 0)
        fuente = metricas.get("datos_fuente", "CSV histórico")
        comps = metricas.get("competencias", [])

        ma, mb, mc = st.columns(3)
        ma.metric("Accuracy", f"{acc*100:.1f}%")
        mb.metric("F1-Score", f"{f1*100:.1f}%")
        mc.metric("CV Score", f"{cv*100:.1f}%")


        st.markdown("</div>", unsafe_allow_html=True)

        # TOP RANKING
        if team_stats is not None and len(team_stats) > 0:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span>🏆</span> Ranking de Fuerza (Datos Reales)</div>', unsafe_allow_html=True)
            top10 = team_stats.nlargest(10, "Fuerza_Score")
            max_f = top10["Fuerza_Score"].max()
            rows_html = ""
            for i, (_, r) in enumerate(top10.iterrows(), 1):
                logo = get_team_logo(r["team"])
                bar_pct = int(r["Fuerza_Score"] / max_f * 100)
                medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
                rows_html += f"""
                <div class="rank-row">
                    <div class="rank-num">{medal}</div>
                    <div style="font-size:1.3rem;">{logo}</div>
                    <div class="rank-name">{r['team']}</div>
                    <div class="rank-bar-wrap"><div class="rank-bar" style="width:{bar_pct}%"></div></div>
                    <div class="rank-score">{r['Fuerza_Score']:.1f}</div>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.75rem; color:#94a3b8; margin-top:12px;'><i>*Nota: La <b>Fuerza Score</b> es un índice inteligente de 0 a 100+ calculado desde datos reales de la API (tasa de victorias, goles, vallas invictas) de los últimos torneos (Euro, Mundial, Ligas).</i></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: RESULTADOS WC 2026
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    if wc_matches is not None and len(wc_matches) > 0:
        finished = wc_matches[wc_matches["status"] == "FINISHED"] if "status" in wc_matches.columns else wc_matches
        scheduled = wc_matches[wc_matches["status"].isin(["TIMED","SCHEDULED"])] if "status" in wc_matches.columns else pd.DataFrame()

        # Métricas rápidas
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("⚽ Partidos Jugados", len(finished))
        mc2.metric("📅 Programados", len(scheduled))
        mc3.metric("🎯 Goles Totales", int(finished["total_goals"].sum()) if "total_goals" in finished.columns else "-")
        mc4.metric("📊 Promedio Goles", f"{finished['total_goals'].mean():.2f}" if len(finished) > 0 and "total_goals" in finished.columns else "-")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if True:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span>✅</span> Historial (Predicciones Honestas "Out-Of-Sample")</div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.8rem;color:#64748b;margin-bottom:15px;line-height:1.3;">Para asegurar transparencia matemática, el modelo muestra qué habría predicho para estos partidos <b>sin conocer su resultado final</b>.</p>', unsafe_allow_html=True)

            # Contador de aciertos
            total_pred, aciertos_pred = 0, 0

            if len(finished) > 0:
                for _, m in finished.sort_values("date", ascending=False).iterrows():
                    ht  = m.get("home_team", "?")
                    at  = m.get("away_team", "?")
                    hs  = m.get("home_score", "-")
                    as_ = m.get("away_score", "-")
                    w   = m.get("winner", "")
                    st_date = str(m.get("date", ""))[:10]

                    home_bold = "font-weight:900;" if w == "HOME_TEAM" else ""
                    away_bold = "font-weight:900;" if w == "AWAY_TEAM" else ""
                    home_col  = "color:#4f46e5;" if w == "HOME_TEAM" else "color:#64748b;"
                    away_col  = "color:#ef4444;" if w == "AWAY_TEAM" else "color:#64748b;"
                    ht_logo   = get_team_logo(ht)
                    at_logo   = get_team_logo(at)

                    # ── Predicción del modelo ──────────────────────────────
                    # resultado real: "Local", "Empate", "Visita"
                    real_str = "Local" if w == "HOME_TEAM" else ("Visita" if w == "AWAY_TEAM" else "Empate")
                    
                    key = f"{ht}_{at}"
                    if key in predicciones_honestas:
                        pred_str = predicciones_honestas[key]
                        total_pred += 1
                        if pred_str == real_str:
                            aciertos_pred += 1
                            pred_badge = "<span style='background:#d1fae5;color:#065f46;font-size:0.65rem;font-weight:800;padding:3px 8px;border-radius:999px;white-space:nowrap;'>✅ Acertó</span>"
                        else:
                            pred_badge = f"<span style='background:#fee2e2;color:#991b1b;font-size:0.65rem;font-weight:800;padding:3px 8px;border-radius:999px;white-space:nowrap;'>❌ Predijo: {pred_str}</span>"
                    else:
                        # Fallback por si no hay historial honesto para este partido
                        try:
                            _, _, _, pred_class = predecir(ht, at)
                            total_pred += 1
                            real_result = 1 if w == "HOME_TEAM" else (-1 if w == "AWAY_TEAM" else 0)
                            if pred_class == real_result:
                                aciertos_pred += 1
                                pred_badge = "<span style='background:#d1fae5;color:#065f46;font-size:0.65rem;font-weight:800;padding:3px 8px;border-radius:999px;white-space:nowrap;'>✅ Acertó</span>"
                            else:
                                pred_nombre = "Local" if pred_class == 1 else ("Empate" if pred_class == 0 else "Visita")
                                pred_badge = f"<span style='background:#fee2e2;color:#991b1b;font-size:0.65rem;font-weight:800;padding:3px 8px;border-radius:999px;white-space:nowrap;'>❌ Predijo: {pred_nombre}</span>"
                        except Exception:
                            pred_badge = "<span style='background:#f1f5f9;color:#94a3b8;font-size:0.65rem;padding:3px 8px;border-radius:999px;'>— sin datos</span>"

                    st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;padding:10px 12px;margin-bottom:6px;
            background:#f8faff;border-radius:12px;border:1px solid #e0e7ff;">
    <div style="flex:1;text-align:right;{home_bold}{home_col}font-size:0.88rem;">{ht_logo} {ht}</div>
    <div style="text-align:center;min-width:70px;background:#fff;border-radius:8px;padding:6px 10px;border:1px solid #e0e7ff;">
        <div style="font-size:1.1rem;font-weight:900;color:#1e1b4b;">{hs} – {as_}</div>
        <div style="font-size:0.65rem;color:#94a3b8;">{st_date}</div>
    </div>
    <div style="flex:1;text-align:left;{away_bold}{away_col}font-size:0.88rem;">{at_logo} {at}</div>
    <div style="min-width:110px;text-align:right;">{pred_badge}</div>
</div>
""", unsafe_allow_html=True)

            # Resumen de aciertos
            if total_pred > 0:
                pct = aciertos_pred / total_pred * 100
                color_bg = "#d1fae5" if pct >= 50 else "#fef3c7"
                color_tx = "#065f46" if pct >= 50 else "#92400e"
                st.markdown(f"""
<div style="background:{color_bg};border-radius:10px;padding:10px 16px;margin-top:10px;display:flex;align-items:center;gap:10px;">
    <span style="font-size:1.3rem;">🎯</span>
    <div>
        <div style="font-size:0.82rem;font-weight:800;color:{color_tx};">Precisión del modelo en WC2026: {aciertos_pred}/{total_pred} partidos acertados ({pct:.0f}%)</div>
    </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)



        # Gráfico goles por partido
        if len(finished) > 0 and "total_goals" in finished.columns and "group" in finished.columns:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span>📈</span> Goles por Partido — Distribución</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(
                finished, x="total_goals", nbins=8,
                color_discrete_sequence=["#4f46e5"],
                labels={"total_goals": "Goles Totales en el Partido"},
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,255,1)",
                yaxis=dict(gridcolor="#e0e7ff"),
                xaxis=dict(gridcolor="#f1f5f9"),
                margin=dict(l=20,r=20,t=20,b=20), height=250,
                bargap=0.1,
            )
            fig_hist.update_traces(marker_line_color="#3730a3", marker_line_width=1.5)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("📡 Ejecuta `python fetch_api_data.py` desde la carpeta BASE para descargar los partidos del Mundial 2026.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ANÁLISIS DE SELECCIONES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([1.2, 1], gap="large")

    with col_s1:
        sel_equipo = st.selectbox("🌍 Selecciona una Selección", equipos_list, key="sel_pais_tab3")
        logo_sel = get_team_logo(sel_equipo)

        if team_stats is not None and sel_equipo in team_stats["team"].values:
            ts = team_stats[team_stats["team"] == sel_equipo].iloc[0]

            # Header del equipo
            st.markdown(f"""
            <div class="card" style="background:linear-gradient(135deg,#f0f4ff,#faf5ff);border:2px solid #e0e7ff;">
                <div style="display:flex;align-items:center;gap:16px;">
                    <div style="font-size:4rem;">{logo_sel}</div>
                    <div>
                        <div style="font-size:1.8rem;font-weight:800;color:#1e1b4b;">{sel_equipo}</div>
                        <div style="color:#64748b;font-size:0.9rem;margin-top:4px;">
                            {int(ts['partidos'])} partidos registrados · {int(ts['wins'])}V {int(ts['draws'])}E {int(ts['losses'])}D
                        </div>
                        <div style="margin-top:8px;">
                            <span style="background:#4f46e5;color:white;padding:4px 12px;border-radius:999px;font-size:0.72rem;font-weight:700;">
                                Fuerza: {ts['Fuerza_Score']:.1f}
                            </span>
                            <span style="background:#f0fdf4;color:#16a34a;padding:4px 12px;border-radius:999px;font-size:0.72rem;font-weight:700;border:1px solid #86efac;margin-left:6px;">
                                WinRate: {ts['win_rate']*100:.1f}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Stats cards
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("⚽ Goles/Partido", f"{ts['goles_por_partido']:.2f}")
            s2.metric("🛡️ Goles Contra", f"{ts['goles_recibidos_partido']:.2f}")
            s3.metric("🏆 Pts/Partido", f"{ts['pts_per_game']:.2f}")
            s4.metric("🧤 Clean Sheets", f"{ts['clean_sheet_rate']*100:.0f}%")

            # Historial reciente desde partidos API
            if matches_api is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header"><span>📋</span> Historial Reciente (API)</div>', unsafe_allow_html=True)
                team_matches = matches_api[
                    (matches_api["home_team"] == sel_equipo) |
                    (matches_api["away_team"] == sel_equipo)
                ].sort_values("date", ascending=False).head(12)

                for _, m in team_matches.iterrows():
                    is_home = m["home_team"] == sel_equipo
                    opp = m["away_team"] if is_home else m["home_team"]
                    gs = m["home_score"] if is_home else m["away_score"]
                    gc = m["away_score"] if is_home else m["home_score"]
                    w = m.get("winner", "")
                    won = (w == "HOME_TEAM" and is_home) or (w == "AWAY_TEAM" and not is_home)
                    drew = w == "DRAW"
                    result_label = "W" if won else ("D" if drew else "L")
                    result_color = "#16a34a" if won else ("#f59e0b" if drew else "#dc2626")
                    result_bg = "#dcfce7" if won else ("#fef9c3" if drew else "#fee2e2")
                    comp_badge = m.get("competition", "")
                    opp_logo = get_team_logo(opp)

                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid #f1f5f9;">
                        <div style="background:{result_bg};color:{result_color};font-weight:800;font-size:0.82rem;
                                    padding:4px 10px;border-radius:6px;min-width:28px;text-align:center;">{result_label}</div>
                        <div style="flex:1;font-size:0.88rem;color:#1e1b4b;font-weight:600;">
                            {opp_logo} {opp}
                        </div>
                        <div style="font-size:0.9rem;font-weight:700;color:#1e1b4b;">{gs} – {gc}</div>
                        <div style="font-size:0.72rem;color:#94a3b8;">{str(m['date'])[:10]}</div>
                        <div style="font-size:0.65rem;background:#f0f4ff;color:#4f46e5;padding:2px 8px;
                                    border-radius:999px;border:1px solid #e0e7ff;">{comp_badge}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info(f"No hay estadísticas de API disponibles para **{sel_equipo}**. Puede ser un equipo solo del histórico CSV.")

    with col_s2:
        # Ranking global de todas las selecciones con datos
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>📊</span> Ranking Global por Fuerza</div>', unsafe_allow_html=True)
        if team_stats is not None:
            top_all = team_stats.nlargest(25, "Fuerza_Score")
            max_f = top_all["Fuerza_Score"].max()
            for i, (_, r) in enumerate(top_all.iterrows(), 1):
                logo = get_team_logo(r["team"])
                bar = int(r["Fuerza_Score"] / max_f * 100)
                is_sel = r["team"] == sel_equipo
                bg = "#ede9fe" if is_sel else "transparent"
                border = "border:2px solid #7c3aed;" if is_sel else ""
                medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;padding:7px 10px;margin-bottom:4px;
                            background:{bg};border-radius:10px;{border}">
                    <div style="min-width:22px;font-size:0.8rem;font-weight:800;color:#cbd5e1;">{medal}</div>
                    <div style="font-size:1.1rem;">{logo}</div>
                    <div style="flex:1;font-size:0.85rem;font-weight:{'800' if is_sel else '600'};
                                color:{'#4f46e5' if is_sel else '#1e1b4b'};">{r['team']}</div>
                    <div style="width:80px;height:5px;background:#e0e7ff;border-radius:999px;overflow:hidden;">
                        <div style="width:{bar}%;height:100%;background:linear-gradient(90deg,#4f46e5,#7c3aed);border-radius:999px;"></div>
                    </div>
                    <div style="font-size:0.82rem;font-weight:800;color:#4f46e5;min-width:38px;text-align:right;">{r['Fuerza_Score']:.1f}</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">{int(r['wins'])}W</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.75rem; color:#94a3b8; margin-top:12px;'><i>*Nota: La <b>Fuerza Score</b> es un índice inteligente calculado desde datos reales de la API (tasa de victorias, goles, vallas invictas) de los últimos torneos (Euro, Mundial, Ligas).</i></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: JUGADORES
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    if players_df is not None and len(players_df) > 0:
        # Columnas de referencia
        team_col   = next((c for c in ["team_name","Equipo","Seleccion"] if c in players_df.columns), None)
        name_col   = next((c for c in ["player_name","Nombre"] if c in players_df.columns), None)
        pos_col    = next((c for c in ["position","Pos.","Pos"] if c in players_df.columns), None)
        age_col    = next((c for c in ["edad","Edad"] if c in players_df.columns), None)
        nat_col    = next((c for c in ["nationality","Nacionalidad"] if c in players_df.columns), None)
        club_col   = next((c for c in ["club_name","Clubes"] if c in players_df.columns), None)

        col_p1, col_p2 = st.columns([1, 1.5], gap="large")

        with col_p1:
            # Búsqueda de jugador
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span>🔍</span> Buscar Jugador</div>', unsafe_allow_html=True)
            search_q = st.text_input("Nombre del jugador...", placeholder="Ej: Mbappé, Messi, Vinicius...", key="player_search")

            if search_q and len(search_q) >= 2:
                mask = players_df[name_col].str.contains(search_q, case=False, na=False) if name_col else pd.Series([False]*len(players_df))
                results = players_df[mask].head(8)
                for _, p in results.iterrows():
                    pname = p.get(name_col, "?") if name_col else "?"
                    ppos  = p.get(pos_col, "MF") if pos_col else "MF"
                    pteam = p.get(team_col, "?") if team_col else "?"
                    page  = p.get(age_col, "?") if age_col else "?"
                    pclub = p.get(club_col, "") if club_col else ""
                    pos_c = get_pos_class(ppos)
                    pos_l = pos_label(ppos)
                    team_logo = get_team_logo(pteam)
                    html_str = f"""
<div class="player-row">
    <div class="player-info">
        <div class="player-name">{pname}</div>
        <div class="player-meta">{team_logo} {pteam} &middot; {pclub or 'Club n/d'} &middot; {page} años</div>
    </div>
    <span class="player-pos {pos_c}">{pos_l}</span>
</div>
"""
                    st.markdown(html_str, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Selector de equipo para ver plantilla
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span>🏳️</span> Ver Plantilla del Equipo</div>', unsafe_allow_html=True)
            if team_col:
                teams_avail = sorted(players_df[team_col].dropna().unique().tolist())
                sel_team_p = st.selectbox("Equipo", teams_avail, key="sel_team_players")
                squad_df = players_df[players_df[team_col] == sel_team_p]

                logo_t = get_team_logo(sel_team_p)
                st.markdown(f"""
                <div style="text-align:center;padding:10px 0 16px;">
                    <div style="font-size:2.5rem;">{logo_t}</div>
                    <div style="font-weight:800;font-size:1.1rem;color:#1e1b4b;">{sel_team_p}</div>
                    <div style="color:#94a3b8;font-size:0.82rem;">{len(squad_df)} jugadores</div>
                </div>
                """, unsafe_allow_html=True)

                # Agrupar por posición
                pos_order = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
                if pos_col:
                    squad_df = squad_df.copy()
                    squad_df["pos_order"] = squad_df[pos_col].map(pos_order).fillna(99)
                    squad_df = squad_df.sort_values("pos_order")

                for _, p in squad_df.iterrows():
                    pname = p.get(name_col, "?") if name_col else "?"
                    ppos  = p.get(pos_col, "") if pos_col else ""
                    page  = p.get(age_col, "?") if age_col else "?"
                    pclub = p.get(club_col, "") if club_col else ""
                    shirt = p.get("shirt_number", "") if "shirt_number" in p.index else ""
                    pos_c = get_pos_class(ppos)
                    pos_l = pos_label(ppos)
                    club_txt = f" · {pclub}" if pclub and str(pclub) != "nan" else ""
                    shirt_txt = f"<span style='color:#94a3b8;font-size:0.75rem;'>#{shirt}</span>" if shirt and str(shirt) != "nan" else ""

                    html_str = f"""
<div class="player-row">
    {shirt_txt}
    <div class="player-info">
        <div class="player-name">{pname}</div>
        <div class="player-meta">{page} años{club_txt}</div>
    </div>
    <span class="player-pos {pos_c}">{pos_l}</span>
</div>
"""
                    st.markdown(html_str, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_p2:
            # Estadísticas de la plantilla seleccionada
            if team_col and age_col and pos_col:
                squad_df2 = players_df[players_df[team_col] == sel_team_p].copy() if 'sel_team_p' in dir() else players_df.copy()
                squad_df2[age_col] = pd.to_numeric(squad_df2[age_col], errors="coerce")

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header"><span>📊</span> Análisis de la Plantilla</div>', unsafe_allow_html=True)

                avg_age = squad_df2[age_col].mean()
                min_age = squad_df2[age_col].min()
                max_age = squad_df2[age_col].max()
                total   = len(squad_df2)

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("👥 Jugadores", total)
                a2.metric("📅 Edad Prom.", f"{avg_age:.1f}" if not pd.isna(avg_age) else "N/A")
                a3.metric("🌱 + Joven", f"{int(min_age)}" if not pd.isna(min_age) else "N/A")
                a4.metric("🧓 Mayor", f"{int(max_age)}" if not pd.isna(max_age) else "N/A")

                # Gráfico distribución por edad
                if not squad_df2[age_col].isna().all():
                    fig_age = px.histogram(
                        squad_df2.dropna(subset=[age_col]), x=age_col,
                        nbins=10, color_discrete_sequence=["#4f46e5"],
                        labels={age_col: "Edad"},
                    )
                    fig_age.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,255,1)",
                        yaxis=dict(gridcolor="#e0e7ff", title=""),
                        xaxis=dict(title="Edad"),
                        margin=dict(l=20,r=20,t=10,b=20), height=220, showlegend=False,
                        bargap=0.1,
                    )
                    fig_age.update_traces(marker_line_color="#3730a3", marker_line_width=1.5)
                    st.plotly_chart(fig_age, use_container_width=True)

                # Distribución por posición — pie chart
                if pos_col in squad_df2.columns:
                    pos_counts = squad_df2[pos_col].value_counts()
                    fig_pos = go.Figure(go.Pie(
                        labels=pos_counts.index.tolist(),
                        values=pos_counts.values.tolist(),
                        hole=0.5,
                        marker=dict(colors=["#fde68a","#86efac","#93c5fd","#fca5a5"]),
                        textinfo="label+percent",
                        textfont=dict(size=12),
                    ))
                    fig_pos.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10,r=10,t=10,b=10), height=240,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Comparativa de edades entre selecciones
            if age_col and team_col and players_df is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header"><span>🌍</span> Edad Promedio por Selección WC2026</div>', unsafe_allow_html=True)
                players_df2 = players_df.copy()
                players_df2[age_col] = pd.to_numeric(players_df2[age_col], errors="coerce")
                avg_by_team = (
                    players_df2.groupby(team_col)[age_col].mean()
                    .dropna().sort_values()
                )
                fig_avg = go.Figure(go.Bar(
                    y=avg_by_team.index.tolist(),
                    x=avg_by_team.values.tolist(),
                    orientation="h",
                    marker=dict(
                        color=avg_by_team.values.tolist(),
                        colorscale=[[0, "#4f46e5"], [0.5, "#7c3aed"], [1, "#ec4899"]],
                        showscale=False,
                    ),
                ))
                fig_avg.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,255,1)",
                    xaxis=dict(gridcolor="#e0e7ff", title="Edad Promedio"),
                    yaxis=dict(tickfont=dict(size=9), gridcolor="#f1f5f9"),
                    margin=dict(l=10,r=20,t=10,b=20),
                    height=max(350, len(avg_by_team)*18),
                )
                st.plotly_chart(fig_avg, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👤 Ejecuta `python fetch_api_data.py` para descargar los datos de los jugadores del Mundial 2026 desde la API.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: GANADORES — BRACKET WC2026
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .bracket-title { font-size:1.5rem; font-weight:900; color:#1e1b4b; margin-bottom:4px; }
    .bracket-sub   { font-size:0.85rem; color:#94a3b8; margin-bottom:20px; }
    .group-card {
        background: linear-gradient(135deg,#f8faff 0%,#f0f4ff 100%);
        border: 1px solid #e0e7ff; border-radius:14px;
        padding:14px; margin-bottom:10px;
    }
    .group-title { font-size:0.7rem; font-weight:800; color:#4f46e5; letter-spacing:1px; margin-bottom:8px; }
    .team-row {
        display:flex; align-items:center; gap:8px;
        padding:5px 6px; border-radius:8px; margin-bottom:3px;
    }
    .team-row.qualified { background:#ede9fe; }
    .team-row.maybe     { background:#fef9c3; }
    .team-rank  { font-size:0.7rem; font-weight:800; color:#94a3b8; min-width:14px; }
    .team-name  { font-size:0.82rem; font-weight:700; color:#1e1b4b; flex:1; }
    .team-pts   { font-size:0.72rem; font-weight:800; color:#4f46e5; background:#e0e7ff;
                  padding:2px 7px; border-radius:999px; }
    .q-badge    { font-size:0.6rem; color:#7c3aed; font-weight:700; }
    .bracket-box {
        background:#fff; border:2px solid #e0e7ff; border-radius:12px;
        padding:10px 14px; margin:4px 0; display:flex; align-items:center; gap:8px;
        min-height:42px;
    }
    .bracket-box.winner { border-color:#4f46e5; background:#ede9fe; }
    .bracket-box.tbd    { border-color:#e2e8f0; background:#f8faff; opacity:0.6; }
    .bname { font-size:0.84rem; font-weight:700; color:#1e1b4b; flex:1; }
    .bpts  { font-size:0.7rem; color:#4f46e5; font-weight:800; }
    .final-box {
        background: linear-gradient(135deg,#4f46e5,#7c3aed);
        color:#fff; border-radius:16px; padding:18px 24px;
        text-align:center; margin:10px 0;
    }
    .final-title { font-size:0.7rem; font-weight:700; letter-spacing:2px; opacity:0.7; }
    .final-name  { font-size:1.3rem; font-weight:900; margin-top:4px; }
    .final-sub   { font-size:0.75rem; opacity:0.75; margin-top:2px; }
    .connector   { text-align:center; color:#c4b5fd; font-size:1.2rem; padding:2px 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="bracket-title">🥇 Camino al Título — WC2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="bracket-sub">Resultados reales de fase de grupos · Se actualiza con cada ejecución del actualizador</div>', unsafe_allow_html=True)

    # ── Datos de grupos (se actualizan al correr el actualizador) ──────────────
    GRUPOS = {
        "A": [
            {"team": "Mexico", "pts": 6, "gf": 3, "gc": 0, "pj": 2},
            {"team": "South Korea", "pts": 3, "gf": 2, "gc": 2, "pj": 2},
            {"team": "Czechia", "pts": 1, "gf": 2, "gc": 3, "pj": 2},
            {"team": "South Africa", "pts": 1, "gf": 1, "gc": 3, "pj": 2},
        ],
        "B": [
            {"team": "Canada", "pts": 4, "gf": 7, "gc": 1, "pj": 2},
            {"team": "Switzerland", "pts": 4, "gf": 5, "gc": 2, "pj": 2},
            {"team": "Bosnia-Herzegovina", "pts": 1, "gf": 2, "gc": 5, "pj": 2},
            {"team": "Qatar", "pts": 1, "gf": 1, "gc": 7, "pj": 2},
        ],
        "C": [
            {"team": "Brazil", "pts": 4, "gf": 4, "gc": 1, "pj": 2},
            {"team": "Morocco", "pts": 4, "gf": 2, "gc": 1, "pj": 2},
            {"team": "Scotland", "pts": 3, "gf": 1, "gc": 1, "pj": 2},
            {"team": "Haiti", "pts": 0, "gf": 0, "gc": 4, "pj": 2},
        ],
        "D": [
            {"team": "United States", "pts": 6, "gf": 6, "gc": 1, "pj": 2},
            {"team": "Australia", "pts": 3, "gf": 2, "gc": 2, "pj": 2},
            {"team": "Paraguay", "pts": 3, "gf": 2, "gc": 4, "pj": 2},
            {"team": "Turkey", "pts": 0, "gf": 0, "gc": 3, "pj": 2},
        ],
        "E": [
            {"team": "Germany", "pts": 6, "gf": 9, "gc": 2, "pj": 2},
            {"team": "Ivory Coast", "pts": 3, "gf": 2, "gc": 2, "pj": 2},
            {"team": "Ecuador", "pts": 1, "gf": 0, "gc": 1, "pj": 2},
            {"team": "Curaçao", "pts": 1, "gf": 1, "gc": 7, "pj": 2},
        ],
        "F": [
            {"team": "Netherlands", "pts": 4, "gf": 7, "gc": 3, "pj": 2},
            {"team": "Japan", "pts": 4, "gf": 6, "gc": 2, "pj": 2},
            {"team": "Sweden", "pts": 3, "gf": 6, "gc": 6, "pj": 2},
            {"team": "Tunisia", "pts": 0, "gf": 1, "gc": 9, "pj": 2},
        ],
        "G": [
            {"team": "Egypt", "pts": 4, "gf": 4, "gc": 2, "pj": 2},
            {"team": "Iran", "pts": 2, "gf": 2, "gc": 2, "pj": 2},
            {"team": "Belgium", "pts": 2, "gf": 1, "gc": 1, "pj": 2},
            {"team": "New Zealand", "pts": 1, "gf": 3, "gc": 5, "pj": 2},
        ],
        "H": [
            {"team": "Spain", "pts": 4, "gf": 4, "gc": 0, "pj": 2},
            {"team": "Uruguay", "pts": 2, "gf": 3, "gc": 3, "pj": 2},
            {"team": "Cape Verde Islands", "pts": 2, "gf": 2, "gc": 2, "pj": 2},
            {"team": "Saudi Arabia", "pts": 1, "gf": 1, "gc": 5, "pj": 2},
        ],
        "I": [
            {"team": "France", "pts": 6, "gf": 6, "gc": 1, "pj": 2},
            {"team": "Norway", "pts": 6, "gf": 7, "gc": 3, "pj": 2},
            {"team": "Senegal", "pts": 0, "gf": 3, "gc": 6, "pj": 2},
            {"team": "Iraq", "pts": 0, "gf": 1, "gc": 7, "pj": 2},
        ],
        "J": [
            {"team": "Argentina", "pts": 6, "gf": 5, "gc": 0, "pj": 2},
            {"team": "Austria", "pts": 3, "gf": 3, "gc": 3, "pj": 2},
            {"team": "Algeria", "pts": 3, "gf": 2, "gc": 4, "pj": 2},
            {"team": "Jordan", "pts": 0, "gf": 2, "gc": 5, "pj": 2},
        ],
        "K": [
            {"team": "Portugal", "pts": 4, "gf": 6, "gc": 1, "pj": 2},
            {"team": "Colombia", "pts": 3, "gf": 3, "gc": 1, "pj": 1},
            {"team": "Congo DR", "pts": 1, "gf": 1, "gc": 1, "pj": 1},
            {"team": "Uzbekistan", "pts": 0, "gf": 1, "gc": 8, "pj": 2},
        ],
        "L": [
            {"team": "England", "pts": 4, "gf": 4, "gc": 2, "pj": 2},
            {"team": "Ghana", "pts": 4, "gf": 1, "gc": 0, "pj": 2},
            {"team": "Croatia", "pts": 3, "gf": 3, "gc": 4, "pj": 2},
            {"team": "Panama", "pts": 0, "gf": 0, "gc": 2, "pj": 2},
        ],
    }

    def get_qualifiers(grupos):
        """Devuelve los 2 primeros de cada grupo (clasificados seguros o proyectados)."""
        qualifiers = {}
        for grp, teams in grupos.items():
            sorted_teams = sorted(teams, key=lambda x: (-x["pts"], -(x["gf"]-x["gc"]), -x["gf"]))
            qualifiers[grp] = sorted_teams[:2]
        return qualifiers

    qualifiers = get_qualifiers(GRUPOS)

    # ── SECCIÓN: Fase de Grupos ───────────────────────────────────────────────
    st.markdown("### 📋 Fase de Grupos — Clasificados")
    st.markdown("<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:16px;'>🟣 Clasificado seguro &nbsp;|&nbsp; 🟡 En disputa</div>", unsafe_allow_html=True)

    # Mostrar grupos en columnas de 3
    grupo_keys = list(GRUPOS.keys())
    for row_start in range(0, len(grupo_keys), 3):
        cols = st.columns(3)
        for ci, grp_key in enumerate(grupo_keys[row_start:row_start+3]):
            teams = GRUPOS[grp_key]
            sorted_t = sorted(teams, key=lambda x: (-x["pts"], -(x["gf"]-x["gc"]), -x["gf"]))
            with cols[ci]:
                html = f'<div class="group-card"><div class="group-title">GRUPO {grp_key}</div>'
                for i, t in enumerate(sorted_t):
                    emoji = get_team_logo(t["team"])
                    all_played = t["pj"] >= 2
                    if i == 0 and all_played:
                        cls, badge = "qualified", "✓"
                    elif i == 1 and all_played:
                        cls, badge = "qualified", "✓"
                    elif i <= 1:
                        cls, badge = "maybe", "~"
                    else:
                        cls, badge = "", ""
                    dg = t['gf'] - t['gc']
                    dg_str = f"+{dg}" if dg > 0 else str(dg)
                    html += f"""
<div class="team-row {cls}">
    <span class="team-rank">{i+1}</span>
    <span style="font-size:1rem">{emoji}</span>
    <span class="team-name">{t['team']}</span>
    <span style="font-size:0.65rem;color:#94a3b8;">{dg_str}</span>
    <span class="team-pts">{t['pts']}pts</span>
</div>"""
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── SECCIÓN: Bracket árbol ────────────────────────────────────────────────
    st.markdown("### 🏆 Bracket — Árbol del Torneo (Proyectado)")
    st.markdown("<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:12px;'>Basado en clasificados actuales · Se actualiza automáticamente con el actualizador · <b>✅</b> modelo acertó · <b>❌</b> modelo falló · <b>—</b> no jugado</div>", unsafe_allow_html=True)

    def fuerza_equipo(team_name, ts):
        if ts is None:
            return 50
        for _, row in ts.iterrows():
            if row.get("team","").lower() in team_name.lower() or team_name.lower() in row.get("team","").lower():
                return float(row.get("Fuerza_Score", 50))
        return 50

    def win(t1, t2, ts):
        # Intentar predecir con el modelo ML primero
        try:
            _, _, _, pred_class = predecir(t1["team"], t2["team"])
            if pred_class == 1:
                return t1
            elif pred_class == -1:
                return t2
            else:
                # Si predice empate, desempata usando la Fuerza Score y puntos
                f1 = fuerza_equipo(t1["team"], ts) + t1["pts"] * 3
                f2 = fuerza_equipo(t2["team"], ts) + t2["pts"] * 3
                return t1 if f1 >= f2 else t2
        except Exception:
            # Fallback si no hay suficientes datos para el modelo
            f1 = fuerza_equipo(t1["team"], ts) + t1["pts"] * 3
            f2 = fuerza_equipo(t2["team"], ts) + t2["pts"] * 3
            return t1 if f1 >= f2 else t2

    # Cruces R32 — izquierda (8 partidos) y derecha (8 partidos)
    q = qualifiers
    r32_l = [
        (q["A"][0], q["B"][1]),  # 1A vs 2B
        (q["C"][0], q["D"][1]),  # 1C vs 2D
        (q["E"][0], q["F"][1]),  # 1E vs 2F
        (q["G"][0], q["H"][1]),  # 1G vs 2H
        (q["I"][0], q["J"][1]),  # 1I vs 2J
        (q["K"][0], q["L"][1]),  # 1K vs 2L
        (q["B"][0], q["A"][1]),  # 1B vs 2A
        (q["D"][0], q["C"][1]),  # 1D vs 2C
    ]
    r32_r = [
        (q["F"][0], q["E"][1]),  # 1F vs 2E
        (q["H"][0], q["G"][1]),  # 1H vs 2G
        (q["J"][0], q["I"][1]),  # 1J vs 2I
        (q["L"][0], q["K"][1]),  # 1L vs 2K
        (q["A"][0], q["C"][1]),
        (q["E"][0], q["G"][1]),
        (q["I"][0], q["K"][1]),
        (q["C"][0], q["E"][1]),
    ]

    def proj_round(pairs, ts):
        return [win(a, b, ts) for a, b in pairs]

    def pair_up(teams):
        return [(teams[i], teams[i+1]) for i in range(0, len(teams)-1, 2)]

    r16_l  = proj_round(r32_l, team_stats)
    r16_r  = proj_round(r32_r, team_stats)
    qf_l   = proj_round(pair_up(r16_l),  team_stats)
    qf_r   = proj_round(pair_up(r16_r),  team_stats)
    sf_l   = proj_round(pair_up(qf_l),   team_stats)
    sf_r   = proj_round(pair_up(qf_r),   team_stats)
    final_l = sf_l[0] if sf_l else q["A"][0]
    final_r = sf_r[0] if sf_r else q["B"][0]
    campeon = win(final_l, final_r, team_stats)

    # ── Lookup de resultados reales para badges ───────────────────────────────
    # Índice: (equipo_local_norm, equipo_visita_norm) → winner real
    wc_results_index = {}
    if wc_matches is not None:
        fin_wc = wc_matches[wc_matches["status"] == "FINISHED"] if "status" in wc_matches.columns else wc_matches
        for _, m in fin_wc.iterrows():
            ht_n = str(m.get("home_team","")).strip().lower()
            at_n = str(m.get("away_team","")).strip().lower()
            wc_results_index[(ht_n, at_n)] = m.get("winner","")

    def find_result(ht, at):
        """Busca si hay resultado real para home_team vs away_team."""
        ht_n, at_n = ht.strip().lower(), at.strip().lower()
        return wc_results_index.get((ht_n, at_n), None)

    def pred_badge(t1, t2):
        """Devuelve badge HTML ✅/❌/— comparando predicción del modelo con resultado real."""
        result = find_result(t1["team"], t2["team"])
        if result is None:
            # try reversed
            result_rev = find_result(t2["team"], t1["team"])
            if result_rev is not None:
                # result was played with teams reversed in CSV
                real_win = 1 if result_rev == "AWAY_TEAM" else (-1 if result_rev == "HOME_TEAM" else 0)
                real_winner_name = t1["team"] if real_win == 1 else (t2["team"] if real_win == -1 else "Empate")
            else:
                return '<span style="font-size:0.65rem;color:#94a3b8;font-weight:700;padding:1px 5px;background:#f1f5f9;border-radius:999px;">—</span>'
        else:
            real_win = 1 if result == "HOME_TEAM" else (-1 if result == "AWAY_TEAM" else 0)
            real_winner_name = t1["team"] if real_win == 1 else (t2["team"] if real_win == -1 else "Empate")

        try:
            _, _, _, pred_class = predecir(t1["team"], t2["team"])
            pred_winner_name = t1["team"] if pred_class == 1 else (t2["team"] if pred_class == -1 else "Empate")
            if pred_class == real_win:
                return '<span style="font-size:0.65rem;color:#065f46;font-weight:800;padding:1px 6px;background:#d1fae5;border-radius:999px;">✅</span>'
            else:
                return f'<span style="font-size:0.65rem;color:#991b1b;font-weight:800;padding:1px 6px;background:#fee2e2;border-radius:999px;" title="Predijo: {pred_winner_name}">❌</span>'
        except Exception:
            return '<span style="font-size:0.65rem;color:#94a3b8;padding:1px 5px;background:#f1f5f9;border-radius:999px;">—</span>'

    # ── Funciones para generar los team_box HTML (fondo blanco) ──────────────
    def team_box(t, highlight=False, champion=False):
        logo = get_team_logo(t["team"]) if t else "❓"
        name = t["team"] if t else "TBD"
        pts  = t["pts"]  if t else 0
        if champion:
            bg   = "linear-gradient(135deg,#fef3c7,#fde68a)"
            bord = "2px solid #f59e0b"
            col  = "#78350f"
            ptsc = "#92400e"
        elif highlight:
            bg   = "linear-gradient(135deg,#ede9fe,#ddd6fe)"
            bord = "1.5px solid #7c3aed"
            col  = "#3730a3"
            ptsc = "#4f46e5"
        else:
            bg   = "#f8faff"
            bord = "1px solid #e0e7ff"
            col  = "#475569"
            ptsc = "#94a3b8"
        return (
            f'<div style="background:{bg};border:{bord};border-radius:8px;'
            f'padding:5px 8px;margin:2px 0;display:flex;align-items:center;gap:5px;'
            f'white-space:nowrap;min-width:125px;max-width:150px;">'
            f'<span style="font-size:0.85rem">{logo}</span>'
            f'<span style="font-size:0.7rem;font-weight:700;color:{col};flex:1;overflow:hidden;text-overflow:ellipsis;">{name}</span>'
            f'<span style="font-size:0.58rem;color:{ptsc};font-weight:700;">{pts}p</span>'
            f'</div>'
        )

    # ── Construir HTML completo del bracket ───────────────────────────────────
    col_h = 760

    # Pre-calcular badges para R32 (únicos que pueden tener resultados reales)
    r32_l_badges = [pred_badge(t1, t2) for t1, t2 in r32_l]
    r32_r_badges = [pred_badge(t1, t2) for t1, t2 in r32_r]

    def r32_match_html(t1, t2, badge):
        return (
            f'<div style="margin-bottom:2px;">'
            f'{team_box(t1)}'
            f'<div style="font-size:0.5rem;color:#94a3b8;text-align:center;padding:1px 0;">vs</div>'
            f'{team_box(t2)}'
            f'<div style="text-align:center;margin-top:1px;">{badge}</div>'
            f'</div>'
        )

    bracket_html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
  body {{ margin:0; background:#ffffff; font-family: 'Segoe UI', Arial, sans-serif; }}
  .bwrap {{
      display:grid;
      grid-template-columns: 158px 152px 138px 98px 116px 98px 138px 152px 158px;
      width:100%; height:{col_h}px;
      align-items:stretch; padding:8px 4px; box-sizing:border-box;
  }}
  .col {{ display:flex; flex-direction:column; height:100%; padding:0 2px; }}
  .lbl {{ font-size:0.52rem;font-weight:800;color:#6366f1;letter-spacing:1px;
           text-align:center;padding:4px 0 5px;text-transform:uppercase;border-bottom:2px solid #e0e7ff;margin-bottom:4px; }}
  .mid {{ display:flex;flex-direction:column;justify-content:center;align-items:center;padding:0 3px; }}
  .trophy {{ font-size:2rem;text-align:center; }}
  .chbox {{
      background:linear-gradient(135deg,#fef3c7,#fde68a);
      border:2px solid #f59e0b; border-radius:12px;
      padding:8px 10px; text-align:center; margin-top:5px; min-width:95px;
  }}
  .chlbl {{ font-size:0.47rem;font-weight:800;color:#78350f;letter-spacing:1.5px; }}
  .chlogo {{ font-size:1.6rem; }}
  .chname {{ font-size:0.7rem;font-weight:900;color:#78350f;margin-top:2px; }}
  .chsub  {{ font-size:0.46rem;color:#92400e;margin-top:2px; }}
  .vs {{ font-size:0.5rem;color:#94a3b8;text-align:center;padding:1px 0; }}
  .sep {{ border-left:2px dashed #e0e7ff; height:80%; align-self:center; }}
</style>
</head>
<body>
<div class="bwrap">

  <!-- R32 IZQ -->
  <div class="col">
    <div class="lbl">R32</div>
    <div style="display:flex;flex-direction:column;justify-content:space-around;height:100%;gap:2px;">
      {chr(10).join(r32_match_html(r32_l[i][0], r32_l[i][1], r32_l_badges[i]) for i in range(8))}
    </div>
  </div>

  <!-- OCTAVOS IZQ -->
  <div class="col">
    <div class="lbl">Octavos</div>
    <div style="display:flex;flex-direction:column;justify-content:space-around;height:100%;gap:2px;">
      {chr(10).join(
        f'<div>{team_box(r16_l[i], highlight=True)}<div class="vs">vs</div>{team_box(r16_l[i+1])}</div>'
        for i in range(0, 8, 2)
      )}
    </div>
  </div>

  <!-- CUARTOS IZQ -->
  <div class="col">
    <div class="lbl">Cuartos</div>
    <div style="display:flex;flex-direction:column;justify-content:space-around;height:100%;gap:2px;">
      {chr(10).join(
        f'<div>{team_box(qf_l[i], highlight=True)}<div class="vs">vs</div>{team_box(qf_l[i+1])}</div>'
        for i in range(0, len(qf_l)-1, 2)
      )}
    </div>
  </div>

  <!-- SEMI IZQ -->
  <div class="col">
    <div class="lbl">Semi</div>
    <div style="display:flex;flex-direction:column;justify-content:center;height:100%;">
      {team_box(final_l, highlight=True)}
    </div>
  </div>

  <!-- CENTRO: Copa + Campeón -->
  <div class="mid">
    <div class="trophy">🏆</div>
    <div class="chbox">
      <div class="chlbl">CAMPEÓN</div>
      <div class="chlogo">{get_team_logo(campeon["team"])}</div>
      <div class="chname">{campeon["team"]}</div>
      <div class="chsub">ML + datos reales</div>
    </div>
    <div style="font-size:0.5rem;color:#6366f1;margin-top:6px;text-align:center;font-weight:700;">FINAL</div>
    <div style="font-size:0.46rem;color:#818cf8;text-align:center;margin-top:2px;line-height:1.4;">
      {final_l["team"]}<br>vs<br>{final_r["team"]}
    </div>
  </div>

  <!-- SEMI DER -->
  <div class="col">
    <div class="lbl">Semi</div>
    <div style="display:flex;flex-direction:column;justify-content:center;height:100%;align-items:flex-end;">
      {team_box(final_r, highlight=True)}
    </div>
  </div>

  <!-- CUARTOS DER -->
  <div class="col">
    <div class="lbl">Cuartos</div>
    <div style="display:flex;flex-direction:column;justify-content:space-around;height:100%;gap:2px;align-items:flex-end;">
      {chr(10).join(
        f'<div style="text-align:right">{team_box(qf_r[i], highlight=True)}<div class="vs">vs</div>{team_box(qf_r[i+1])}</div>'
        for i in range(0, len(qf_r)-1, 2)
      )}
    </div>
  </div>

  <!-- OCTAVOS DER -->
  <div class="col">
    <div class="lbl">Octavos</div>
    <div style="display:flex;flex-direction:column;justify-content:space-around;height:100%;gap:2px;align-items:flex-end;">
      {chr(10).join(
        f'<div style="text-align:right">{team_box(r16_r[i], highlight=True)}<div class="vs">vs</div>{team_box(r16_r[i+1])}</div>'
        for i in range(0, 8, 2)
      )}
    </div>
  </div>

  <!-- R32 DER -->
  <div class="col">
    <div class="lbl">R32</div>
    <div style="display:flex;flex-direction:column;justify-content:space-around;height:100%;gap:2px;align-items:flex-end;">
      {chr(10).join(r32_match_html(r32_r[i][0], r32_r[i][1], r32_r_badges[i]) for i in range(8))}
    </div>
  </div>

</div>
</body>
</html>
"""

    import streamlit.components.v1 as components
    components.html(bracket_html, height=col_h + 40, scrolling=False)

    st.markdown(
        "<div style='font-size:0.72rem;color:#94a3b8;text-align:center;margin-top:8px;'>"
        "⚠️ Proyección basada en resultados reales de fase de grupos + Fuerza Score del modelo ML. "
        "✅ = modelo acertó el ganador · ❌ = modelo falló · — = partido no jugado aún."
        "</div>",
        unsafe_allow_html=True
    )






