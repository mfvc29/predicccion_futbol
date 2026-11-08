"""Configuración de las ligas que cubre el predictor."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / "data"
WEB_DATA = BASE.parent / "web" / "data"

# Temporadas completas hacia atrás que se descargan además de la actual.
# El plan gratuito de football-data.org da acceso a la actual + las 3 anteriores.
TEMPORADAS_ATRAS = 3

# top: puestos de clasificación que se muestran en la proyección.
# descenso: puestos de descenso directo.
LIGAS = {
    "PL":  {"nombre": "Premier League", "pais": "Inglaterra", "region": "Europa",     "top": 4, "descenso": 3},
    "PD":  {"nombre": "LaLiga",         "pais": "España",     "region": "Europa",     "top": 4, "descenso": 3},
    "SA":  {"nombre": "Serie A",        "pais": "Italia",     "region": "Europa",     "top": 4, "descenso": 3},
    "BL1": {"nombre": "Bundesliga",     "pais": "Alemania",   "region": "Europa",     "top": 4, "descenso": 2},
    "FL1": {"nombre": "Ligue 1",        "pais": "Francia",    "region": "Europa",     "top": 4, "descenso": 2},
    "BSA": {"nombre": "Brasileirão",    "pais": "Brasil",     "region": "Sudamérica", "top": 4, "descenso": 4},
}
