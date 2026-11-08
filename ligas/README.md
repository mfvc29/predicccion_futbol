# Predictor de ligas

Probabilidades 1X2 (local / empate / visita) para Premier League, LaLiga, Serie A,
Bundesliga, Ligue 1 y Brasileirão. Independiente del predictor del Mundial (`app.py`).

```
ligas/
  config.py     ligas incluidas (agregar una liga = una línea aquí)
  fetch.py      descarga de football-data.org → ligas/data (temporadas pasadas quedan en caché)
  features.py   Elo + forma, calculados solo con partidos previos
  entrenar.py   validación temporal, elección de modelo por log-loss
  exportar.py   JSON para la web → web/data
  pipeline.py   corre todo
web/            sitio estático (Vercel, carpeta raíz: web)
```

Se ejecuta solo en GitHub Actions (`.github/workflows/ligas.yml`) dos veces al día
con el secret `FOOTBALL_DATA_API_KEY`, y commitea los datos nuevos; Vercel redespliega
la web con cada commit.

A mano: `FOOTBALL_DATA_API_KEY=... python ligas/pipeline.py`
