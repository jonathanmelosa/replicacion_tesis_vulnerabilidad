"""
conteo_gsv_por_anio.py
======================
Cuenta las fotos de Google Street View obtenidas para los hogares de la
ELCA por año de captura del panorama, y guarda el resultado agregado.

De aquí salen la densidad por año de la barra de Street View en la figura
de cobertura temporal (`timeline_cobertura_geoespacial.py`) y la cifra del
texto sobre la proporción de fotos tomadas de 2023 en adelante (Sección 3.2
y Anexo de Street View).

Se cuentan las filas con `exito == True` del registro de descargas
(20,935 fotos). El registro tiene filas cuyo mensaje de error contiene
comas sin comillas, por eso se lee con `csv.reader` y se toman los campos
por posición en vez de usar `pandas.read_csv`.

INPUT

    data/processed/embeddings/descargas_final.csv

OUTPUT

    outputs/tables/gsv/gsv_fotos_por_anio.csv  (anio, n_fotos; sin datos por hogar)

COMO CORRER

    python src/conteo_gsv_por_anio.py
"""
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "embeddings" / "descargas_final.csv"
OUTPUT_PATH = REPO_ROOT / "outputs" / "tables" / "gsv" / "gsv_fotos_por_anio.csv"

COL_FECHA = 4  # fecha_pano, formato AAAA-MM
COL_EXITO = 9  # exito


def main() -> None:
    conteo = Counter()
    with open(INPUT_PATH, encoding="utf-8-sig", newline="") as f:
        lector = csv.reader(f)
        next(lector)
        for fila in lector:
            if len(fila) > COL_EXITO and fila[COL_EXITO] == "True" and fila[COL_FECHA][:4].isdigit():
                conteo[int(fila[COL_FECHA][:4])] += 1

    tabla = pd.DataFrame(sorted(conteo.items()), columns=["anio", "n_fotos"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUT_PATH, index=False)

    total = tabla["n_fotos"].sum()
    desde_2023 = tabla.loc[tabla["anio"] >= 2023, "n_fotos"].sum()
    print(tabla.to_string(index=False))
    print(f"\nTotal: {total:,} fotos; de 2023 en adelante: {100 * desde_2023 / total:.1f}%")
    print(f"Guardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
