import os

with open('/home/mvargasch/Descargas/futbol/BASE/app.py', 'r') as f:
    lines = f.readlines()

new_lines = []
in_load_data = False
tabs_added = False

for i, line in enumerate(lines):
    # Update load_all function
    if "return model, scores_df, top_players_df, metricas" in line and not tabs_added:
        new_lines.append("    try:\n")
        new_lines.append("        jugadores_df = pd.read_csv('../jugadores_mundial_2026_con_stats_final.csv')\n")
        new_lines.append("    except:\n")
        new_lines.append("        jugadores_df = pd.DataFrame()\n")
        new_lines.append("    try:\n")
        new_lines.append("        partidos_df = pd.read_csv('../partidos_internacionales_historico.csv')\n")
        new_lines.append("        partidos_df['date'] = pd.to_datetime(partidos_df['date'])\n")
        new_lines.append("        partidos_df = partidos_df.sort_values('date', ascending=False)\n")
        new_lines.append("    except:\n")
        new_lines.append("        partidos_df = pd.DataFrame()\n")
        new_lines.append("    return model, scores_df, top_players_df, metricas, jugadores_df, partidos_df\n")
        continue

    if "model, scores_df, top_players_df, metricas = load_all()" in line and not tabs_added:
        new_lines.append("    model, scores_df, top_players_df, metricas, jugadores_df, partidos_df = load_all()\n")
        continue

    if "equipos = sorted(scores_df['Equipo'].dropna().unique().tolist())" in line:
        new_lines.append(line)
        new_lines.append("\n")
        new_lines.append("# ─── MAIN TABS ─────────────────────────────────────────────────────────────────\n")
        new_lines.append("tab1, tab2, tab3 = st.tabs([\"🔮 Predicción de Partidos\", \"🌍 Análisis de Selecciones\", \"🏃 Buscador de Jugadores\"])\n")
        new_lines.append("\n")
        new_lines.append("with tab1:\n")
        tabs_added = True
        continue
        
    if tabs_added:
        # Indent everything that belongs to the original app
        if line.strip() == "":
            new_lines.append("\n")
        else:
            new_lines.append("    " + line)
    else:
        new_lines.append(line)

with open('/home/mvargasch/Descargas/futbol/BASE/app.py', 'w') as f:
    f.writelines(new_lines)
