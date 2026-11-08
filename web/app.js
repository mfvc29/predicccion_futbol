"use strict";

const TABS = [
  ["partidos", "Partidos"],
  ["simular", "Simular"],
  ["tabla", "Tabla"],
  ["proyeccion", "Proyección"],
  ["modelo", "Modelo"],
];
const RESULTADO = ["Local", "Empate", "Visita"];

const st = { meta: null, liga: null, tab: "partidos", cache: {} };
const $ = (s, el = document) => el.querySelector(s);

function leer(k) { try { return localStorage.getItem(k); } catch { return null; } }
function guardar(k, v) { try { localStorage.setItem(k, v); } catch { /* sin almacenamiento */ } }

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const pct = (x) => (x == null ? "—" : `${Math.round(x)}%`);
const pct1 = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

const fDia = new Intl.DateTimeFormat("es-PE", { weekday: "long", day: "numeric", month: "long" });
const fHora = new Intl.DateTimeFormat("es-PE", { hour: "2-digit", minute: "2-digit" });
const fCorta = new Intl.DateTimeFormat("es-PE", { day: "numeric", month: "short" });
const fCompleta = new Intl.DateTimeFormat("es-PE", { dateStyle: "medium", timeStyle: "short" });

async function json(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function datosLiga(code) {
  if (!st.cache[code]) st.cache[code] = json(`data/${code}.json`);
  return st.cache[code];
}

/* ---------- piezas ---------- */

function escudo(eq, chico) {
  if (!eq?.escudo) return `<span class="escudo${chico ? " chico" : ""}"></span>`;
  return `<img class="escudo${chico ? " chico" : ""}" src="${esc(eq.escudo)}" alt="" loading="lazy" width="26" height="26">`;
}

function barra(p, compacta) {
  const [l, e, v] = p;
  const t = (x) => (compacta || x < 11 ? "" : pct(x));
  return `<div class="barra${compacta ? " compacta" : ""}" role="img" aria-label="Local ${pct(l)}, empate ${pct(e)}, visita ${pct(v)}">
    <span class="seg l" style="flex:${l}">${t(l)}</span><span class="seg e" style="flex:${e}">${t(e)}</span><span class="seg v" style="flex:${v}">${t(v)}</span>
  </div>`;
}

function forma(s) {
  if (!s) return "";
  return `<span class="forma" aria-label="Forma ${esc(s)}">${[...s].map((c) => `<i class="${c}">${c}</i>`).join("")}</span>`;
}

function cruce(L, V, centro) {
  return `<div class="cruce">
    <div class="equipo">${escudo(L)}<span>${esc(L?.nombre)}</span></div>
    <div class="centro">${centro}</div>
    <div class="equipo der">${escudo(V)}<span>${esc(V?.nombre)}</span></div>
  </div>`;
}

const vacio = (t) => `<p class="vacio">${t}</p>`;

/* ---------- vistas ---------- */

function vistaPartidos(d, eq) {
  let html = "<h2>Próximos partidos</h2>";
  if (!d.proximos.length) html += vacio("No hay partidos programados por ahora.");
  let dia = "";
  for (const p of d.proximos) {
    const f = new Date(p.fecha);
    const nuevoDia = fDia.format(f);
    if (nuevoDia !== dia) {
      if (dia) html += "</div>";
      html += `<div class="dia">${nuevoDia}</div><div class="lista dos">`;
      dia = nuevoDia;
    }
    const enJuego = p.estado === "IN_PLAY" || p.estado === "PAUSED";
    const centro = `${fHora.format(f)}${enJuego ? '<span class="vivo">En juego</span>' : ""}`;
    html += `<article class="tarjeta partido">
      ${cruce(eq[p.l], eq[p.v], centro)}
      ${barra(p.p)}
      <div class="leyenda"><span>Local</span><span>Empate</span><span>Visita</span></div>
    </article>`;
  }
  if (dia) html += "</div>";

  html += `<h2>Últimos resultados</h2><p class="texto">Lo que el modelo predijo <em>antes</em> de cada partido, con un modelo que no había visto esta temporada.</p>`;
  if (!d.resultados.length) html += vacio("Todavía no hay partidos jugados esta temporada.");
  html += '<div class="lista dos">';
  for (const r of d.resultados) {
    const ok = r.real === r.pred;
    html += `<article class="tarjeta partido">
      ${cruce(eq[r.l], eq[r.v], `<span class="marcador">${r.gl} – ${r.gv}</span>`)}
      ${barra(r.p, true)}
      <div class="res-pie">
        <span>${fCorta.format(new Date(r.fecha))} · Predijo ${RESULTADO[r.pred].toLowerCase()} (${pct(r.p[r.pred])})</span>
        <span class="sello ${ok ? "bien" : "mal"}">${ok ? "Acertó" : "Falló"}</span>
      </div>
    </article>`;
  }
  return html + "</div>";
}

function vistaSimular(d, eq) {
  const ids = d.matriz.ids;
  if (ids.length < 2) return vacio("No hay equipos para simular.");
  const orden = [...ids].sort((a, b) => eq[b].elo - eq[a].elo);
  const guardado = (leer(`sim_${d.codigo}`) || "").split(",").map(Number);
  const [l0, v0] = guardado.length === 2 && guardado.every((x) => ids.includes(x)) && guardado[0] !== guardado[1]
    ? guardado : [orden[0], orden[1]];
  const opciones = (sel) => ids.map((id) => `<option value="${id}"${id === sel ? " selected" : ""}>${esc(eq[id].nombre)}</option>`).join("");
  return `<h2>Simulador de partido</h2>
  <div class="tarjeta sim">
    <div class="selectores">
      <label class="campo">Local<select id="sel-l">${opciones(l0)}</select></label>
      <button class="boton" id="invertir" type="button" aria-label="Invertir local y visita">⇄</button>
      <label class="campo">Visita<select id="sel-v">${opciones(v0)}</select></label>
    </div>
    <div class="sim-res" id="sim-res"></div>
  </div>
  <p class="texto">Usa el estado actual de cada equipo (Elo, forma y rendimiento de local/visita) y supone una semana de descanso.</p>`;
}

function pintarSimulacion(d, eq) {
  const sl = $("#sel-l"), sv = $("#sel-v");
  if (!sl) return;
  const actualizar = () => {
    const l = Number(sl.value), v = Number(sv.value);
    guardar(`sim_${d.codigo}`, `${l},${v}`);
    if (l === v) { $("#sim-res").innerHTML = vacio("Elige dos equipos distintos."); return; }
    const ids = d.matriz.ids;
    const p = d.matriz.p[ids.indexOf(l)][ids.indexOf(v)];
    const L = eq[l], V = eq[v];
    const posicion = (id) => d.tabla.find((t) => t.id === id)?.pos ?? "—";
    $("#sim-res").innerHTML = `${cruce(L, V, "vs")}
      ${barra(p)}
      <div class="leyenda"><span>Gana ${esc(L.nombre)}</span><span>Empate</span><span>Gana ${esc(V.nombre)}</span></div>
      <div class="comparar">
        <div>${L.elo}</div><div class="k">Elo</div><div class="d">${V.elo}</div>
        <div>${posicion(l)}º</div><div class="k">Posición</div><div class="d">${posicion(v)}º</div>
        <div>${forma(L.forma)}</div><div class="k">Forma</div><div class="d">${forma(V.forma)}</div>
      </div>`;
  };
  sl.onchange = sv.onchange = actualizar;
  $("#invertir").onclick = () => { [sl.value, sv.value] = [sv.value, sl.value]; actualizar(); };
  actualizar();
}

function vistaTabla(d, eq) {
  if (!d.tabla.length) return vacio("La tabla aún no está disponible.");
  const n = d.tabla.length;
  const filas = d.tabla.map((t) => {
    const e = eq[t.id] || {};
    const zona = t.pos <= d.zonas.top ? "zona-top" : t.pos > n - d.zonas.descenso ? "zona-desc" : "";
    return `<tr class="${zona}">
      <td class="pos izq">${t.pos}</td>
      <td class="izq"><div class="equipo">${escudo(e, true)}<span>${esc(e.nombre)}</span></div></td>
      <td>${t.pj}</td>
      <td class="ocultar-movil">${t.g}</td><td class="ocultar-movil">${t.e}</td><td class="ocultar-movil">${t.p}</td>
      <td>${t.dg > 0 ? "+" : ""}${t.dg}</td>
      <td><b>${t.pts}</b></td>
      <td class="ocultar-movil">${e.elo ?? "—"}</td>
      <td class="ocultar-movil">${forma(e.forma)}</td>
    </tr>`;
  }).join("");
  return `<h2>Tabla de posiciones</h2>
  <div class="tarjeta tabla-env"><table>
    <thead><tr><th class="izq">#</th><th class="izq">Equipo</th><th>PJ</th>
      <th class="ocultar-movil">G</th><th class="ocultar-movil">E</th><th class="ocultar-movil">P</th>
      <th>DG</th><th>Pts</th><th class="ocultar-movil">Elo</th><th class="ocultar-movil">Forma</th></tr></thead>
    <tbody>${filas}</tbody>
  </table></div>`;
}

function vistaProyeccion(d, eq) {
  const pr = d.proyeccion;
  if (!pr.equipos.length) return vacio("Sin datos para proyectar.");
  const celda = (v, color) => `<td class="pct" style="--v:${v};--c:${color}"><b>${v < 0.5 && v > 0 ? "<1%" : pct(v)}</b></td>`;
  const filas = pr.equipos.map((t, i) => {
    const e = eq[t.id] || {};
    return `<tr>
      <td class="pos izq">${i + 1}</td>
      <td class="izq"><div class="equipo">${escudo(e, true)}<span>${esc(e.nombre)}</span></div></td>
      <td class="ocultar-movil">${t.pts}</td>
      <td><b>${t.pts_esp.toFixed(1)}</b></td>
      ${celda(t.campeon, "var(--acento)")}
      ${celda(t.top, "var(--visita)")}
      ${celda(t.descenso, "var(--mal)")}
    </tr>`;
  }).join("");
  return `<h2>Proyección de fin de temporada</h2>
  <p class="texto">${pr.restantes} partidos por jugar, simulados ${pr.simulaciones.toLocaleString("es-PE")} veces con las probabilidades del modelo. Los desempates se aproximan con la diferencia de goles actual.</p>
  <div class="tarjeta tabla-env"><table>
    <thead><tr><th class="izq">#</th><th class="izq">Equipo</th><th class="ocultar-movil">Pts hoy</th>
      <th>Pts esperados</th><th>Campeón</th><th>Top ${d.zonas.top}</th><th>Desciende</th></tr></thead>
    <tbody>${filas}</tbody>
  </table></div>`;
}

function vistaModelo(d) {
  const m = st.meta;
  const t = d.temporada, v = d.validacion;
  let html = "<h2>¿Qué tan bien predice?</h2>";
  if (t?.modelo?.n) {
    html += `<div class="kpis">
      <div class="tarjeta kpi"><div class="v">${pct1(t.modelo.acierto)}</div><div class="t">Acierto esta temporada (${t.modelo.n} partidos)</div></div>
      <div class="tarjeta kpi"><div class="v">${pct1(t.siempre_local)}</div><div class="t">Si siempre dijeras "gana el local"</div></div>
      <div class="tarjeta kpi"><div class="v">${t.modelo.logloss.toFixed(3)}</div><div class="t">Log-loss del modelo (menor es mejor)</div></div>
      <div class="tarjeta kpi"><div class="v">${t.frecuencia.logloss.toFixed(3)}</div><div class="t">Log-loss usando solo frecuencias históricas</div></div>
    </div>`;
  } else {
    html += vacio("Todavía no hay partidos jugados esta temporada para medir.");
  }
  if (v?.modelo?.n) {
    html += `<p class="texto">Temporada anterior completa en esta liga: acierto ${pct1(v.modelo.acierto)} (${v.modelo.n} partidos), frente a ${pct1(v.siempre_local)} de "siempre local".</p>`;
  }

  const modelos = Object.entries(m.validacion.modelos);
  html += `<h2>Comparación de modelos</h2>
  <p class="texto">Todas las ligas juntas, evaluadas en la temporada anterior completa sin haberla visto. Se usa el de menor log-loss: <b>${esc(m.modelo)}</b>.</p>
  <div class="tarjeta tabla-env"><table>
    <thead><tr><th class="izq">Modelo</th><th>Acierto</th><th>Log-loss</th><th class="ocultar-movil">Brier</th></tr></thead>
    <tbody>${modelos.map(([nombre, r]) => `<tr class="${nombre === m.modelo ? "elegido" : ""}">
      <td class="izq">${esc(nombre)}</td><td>${pct1(r.acierto)}</td>
      <td>${r.logloss != null ? r.logloss.toFixed(3) : "—"}</td>
      <td class="ocultar-movil">${r.brier != null ? r.brier.toFixed(3) : "—"}</td></tr>`).join("")}</tbody>
  </table></div>

  ${m.elo ? `<h2>Parámetros del Elo</h2>
  <p class="texto">Elegidos automáticamente entre ${m.elo.probadas} combinaciones, según cuál predijo mejor la temporada anterior.</p>
  <div class="kpis">
    <div class="tarjeta kpi"><div class="v">${m.elo.k}</div><div class="t">K: cuánto se mueve el Elo por partido</div></div>
    <div class="tarjeta kpi"><div class="v">${m.elo.hfa}</div><div class="t">Ventaja de local (puntos Elo)</div></div>
    <div class="tarjeta kpi"><div class="v">${Math.round(m.elo.regresion * 100)}%</div><div class="t">Fuerza que se conserva al cambiar de temporada</div></div>
    <div class="tarjeta kpi"><div class="v">${m.elo.logloss_defecto != null ? m.elo.logloss_defecto.toFixed(3) + " → " + m.elo.logloss.toFixed(3) : m.elo.logloss.toFixed(3)}</div><div class="t">Log-loss con los valores iniciales → ajustados</div></div>
  </div>` : ""}

  <h2>Cómo funciona</h2>
  <ul class="texto">
    <li><b>Elo por liga:</b> cada resultado sube o baja la fuerza del equipo según lo esperado, con ventaja de local y más peso a goleadas.</li>
    <li><b>Forma:</b> puntos y goles de los últimos 5 y 10 partidos, y rendimiento reciente como local o visita.</li>
    <li><b>Temporada:</b> puntos y diferencia de gol por partido en la temporada actual, y días de descanso.</li>
    <li>Todo se calcula solo con partidos <em>anteriores</em> a cada predicción. Se reentrena dos veces al día con los partidos recién jugados.</li>
    <li>Partidos usados: ${m.partidos.final.toLocaleString("es-PE")} (temporada actual y las 3 anteriores).</li>
  </ul>`;
  return html;
}

/* ---------- navegación ---------- */

function pintarLigas() {
  let html = "", region = "";
  for (const l of st.meta.ligas) {
    if (l.region !== region) { html += `<span class="region">${esc(l.region)}</span>`; region = l.region; }
    html += `<button class="chip" type="button" data-liga="${l.codigo}" aria-pressed="${l.codigo === st.liga}">
      ${l.emblema ? `<img src="${esc(l.emblema)}" alt="" width="22" height="22">` : ""}${esc(l.nombre)}</button>`;
  }
  const nav = $("#ligas");
  nav.innerHTML = html;
  nav.onclick = (ev) => {
    const b = ev.target.closest("[data-liga]");
    if (b) { st.liga = b.dataset.liga; guardar("liga", st.liga); mostrar(); }
  };
}

function pintarPestanas() {
  const nav = $("#pestanas");
  nav.innerHTML = TABS.map(([id, t]) =>
    `<button class="pestana" role="tab" type="button" data-tab="${id}" aria-selected="${id === st.tab}">${t}</button>`).join("");
  nav.onclick = (ev) => {
    const b = ev.target.closest("[data-tab]");
    if (b) { st.tab = b.dataset.tab; guardar("tab", st.tab); mostrar(); }
  };
}

async function mostrar() {
  history.replaceState(null, "", `#${st.liga}/${st.tab}`);
  document.querySelectorAll("[data-liga]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.liga === st.liga));
  document.querySelectorAll("[data-tab]").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === st.tab));
  const cont = $("#contenido");
  const liga = st.meta.ligas.find((l) => l.codigo === st.liga);
  let d;
  try { d = await datosLiga(st.liga); } catch { cont.innerHTML = vacio("No se pudieron cargar los datos de esta liga."); return; }
  const eq = Object.fromEntries(d.equipos.map((e) => [e.id, e]));
  const vistas = { partidos: vistaPartidos, simular: vistaSimular, tabla: vistaTabla, proyeccion: vistaProyeccion, modelo: vistaModelo };
  cont.innerHTML = `<p class="sutil" style="margin:14px 0 0;font-size:.85rem">${esc(liga.nombre)} · ${esc(liga.pais)} · Temporada ${esc(liga.temporada)}${liga.jornada ? ` · Jornada ${liga.jornada}` : ""}</p>`
    + vistas[st.tab](d, eq);
  if (st.tab === "simular") pintarSimulacion(d, eq);
}

async function iniciar() {
  try {
    st.meta = await json("data/meta.json");
  } catch {
    $("#actualizado").textContent = "";
    $("#contenido").innerHTML = vacio("No se pudieron cargar los datos.");
    return;
  }
  const codigos = st.meta.ligas.map((l) => l.codigo);
  const [hLiga, hTab] = location.hash.slice(1).split("/");
  const ligaGuardada = leer("liga"), tabGuardada = leer("tab");
  const tabValida = (t) => TABS.some(([id]) => id === t);
  st.liga = codigos.includes(hLiga) ? hLiga : codigos.includes(ligaGuardada) ? ligaGuardada : codigos[0];
  st.tab = tabValida(hTab) ? hTab : tabValida(tabGuardada) ? tabGuardada : "partidos";
  $("#actualizado").textContent = `Actualizado: ${fCompleta.format(new Date(st.meta.actualizado))}`;
  pintarLigas();
  pintarPestanas();
  mostrar();
}

iniciar();
