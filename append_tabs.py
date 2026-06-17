with open('/home/mvargasch/Descargas/futbol/BASE/app.py', 'a', encoding='utf-8') as f:
    f.write('''
with tab2:
    st.markdown('<div class="hero-title" style="font-size:2rem; margin-top:20px;">🌍 Análisis de Selecciones</div>', unsafe_allow_html=True)
    
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
                    return ['background-color: rgba(0, 212, 255, 0.1)'] * len(row) if row['home_team'] == sel_equipo else ['background-color: rgba(255, 107, 107, 0.1)'] * len(row)
                elif row['home_score'] < row['away_score']:
                    return ['background-color: rgba(255, 107, 107, 0.1)'] * len(row) if row['home_team'] == sel_equipo else ['background-color: rgba(0, 212, 255, 0.1)'] * len(row)
                else:
                    return ['background-color: rgba(255, 215, 0, 0.1)'] * len(row)
                    
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
                    <h2 style="margin:0; color:#00d4ff;">{get_val(jug.get('Nombre'))}</h2>
                    <p style="color:rgba(255,255,255,0.6); font-size:1.1rem; margin:5px 0;">{get_val(jug.get('Pos.'))} · {get_val(jug.get('Clubes'))}</p>
                </div>
                <div style="text-align:right; margin-left: 20px;">
                    <div style="color:rgba(255,255,255,0.4); font-size:0.8rem; text-transform:uppercase;">Edad</div>
                    <div style="font-size:1.5rem; font-weight:bold;">{get_val(jug.get('Edad'))}</div>
                </div>
                <div style="text-align:right; margin-left: 20px;">
                    <div style="color:rgba(255,255,255,0.4); font-size:0.8rem; text-transform:uppercase;">Físico</div>
                    <div style="font-size:1.5rem; font-weight:bold;">{get_val(jug.get('Tamaño'), '-')} / {get_val(jug.get('Peso'), '-')}</div>
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
''')
