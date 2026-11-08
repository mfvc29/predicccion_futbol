"""Descarga datos, entrena el modelo y genera los JSON de la web.

Uso:
    FOOTBALL_DATA_API_KEY=... python ligas/pipeline.py
    python ligas/pipeline.py --sin-descarga   # reusa lo que ya está en ligas/data
"""
import json
import sys

from config import RAW
from entrenar import entrenar
from exportar import exportar
from features import cargar_partidos, construir
from fetch import descargar


def main():
    if "--sin-descarga" not in sys.argv:
        print("① Descargando datos")
        descargar()
    info = json.loads((RAW / "ligas.json").read_text())
    print("② Calculando variables")
    df = cargar_partidos()
    df = df[df["liga"].isin(info)].reset_index(drop=True)
    X, estado = construir(df)
    print("③ Entrenando")
    res = entrenar(X, {c: v["temporada"] for c, v in info.items()})
    print("④ Exportando")
    exportar(df, X, estado, res, info)


if __name__ == "__main__":
    main()
