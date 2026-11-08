"""Descarga datos, entrena el modelo y genera los JSON de la web.

Uso:
    FOOTBALL_DATA_API_KEY=... python ligas/pipeline.py
    python ligas/pipeline.py --sin-descarga   # reusa lo que ya está en ligas/data
"""
import json
import sys

from ajustar_elo import buscar
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
    temporadas = {c: v["temporada"] for c, v in info.items()}
    df = cargar_partidos()
    df = df[df["liga"].isin(info)].reset_index(drop=True)
    print("② Ajustando parámetros del Elo")
    elo = buscar(df, temporadas)
    print(f"  Elegidos: {elo}")
    print("③ Calculando variables")
    X, estado = construir(df, elo)
    print("④ Entrenando")
    res = entrenar(X, temporadas, elo["calentamiento"])
    res["elo"] = elo
    print("⑤ Exportando")
    exportar(df, X, estado, res, info)


if __name__ == "__main__":
    main()
