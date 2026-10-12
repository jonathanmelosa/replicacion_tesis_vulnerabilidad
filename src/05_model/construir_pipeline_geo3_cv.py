"""
construir_pipeline_geo3_cv.py
================================

PIPELINE 2 de incorporacion de fuentes geoespaciales al benchmark: agrega
las variables de las TRES fuentes nuevas (DMSP-OLS, ALOS PALSAR, Landsat 5
TM) a las covariables de la transicion 2010->2013, para un ejercicio
EXPLORATORIO evaluado con validacion cruzada dentro de esa misma
transicion -- NO con el holdout temporal principal.

POR QUE UN ESQUEMA DE VALIDACION DISTINTO AQUI
------------------------------------------------
ALOS PALSAR y Landsat 5 TM solo tienen datos reales en la ola 2010 (0%
en 2013 -- ver Seccion 3.2 de la tesis). Si se agregaran al benchmark
principal (train=2010, test=2013, `construir_pipeline_geo_dmsp.py`), esas
columnas quedarian 100% vacias en el conjunto de prueba: el modelo
correria sin error, pero cualquier patron aprendido de ellas en
entrenamiento dejaria de aplicarse al evaluar en 2013 (missingness
universal, no missingness real) -- una prueba fuera de muestra invalida.

Por eso este ejercicio NO usa un holdout temporal: evalua las tres
fuentes juntas dentro de la MISMA transicion 2010->2013, con validacion
cruzada agrupando por hogar (cada hogar se predice con un modelo que
nunca lo vio en entrenamiento, pero todos los hogares -- train y
"prueba" -- son del mismo periodo). Es una prueba mas debil que el
holdout temporal (no confirma que el modelo generaliza a OTRO momento),
pero es la unica forma honesta de aprovechar ALOS PALSAR y Landsat 5 TM
sin inventar cobertura de 2013 que no existe. Los resultados de este
ejercicio NO son comparables cifra a cifra contra
ESPECIFICACIONES_PRINCIPAL (esquemas de validacion distintos) -- ver
`modelo_utils.evaluar_cv_semillas`.

QUE HACE
--------
    1. Carga variables_geoespaciales_unificadas.parquet, filtra a ola ==
       2010, se queda con dmsp_* + alos_* + l5_* + consecutivo.
    2. Excluye hogares divididos (es_split == 1), igual que
       construir_pipeline_geo_dmsp.py.
    3. Para cada especificacion (A, B): carga
       modelo_{espec}_2010_2013.parquet (el archivo de TRAIN del
       benchmark principal -- aqui se usa como la muestra completa del
       ejercicio, no solo como train), pega las columnas geoespaciales
       por `consecutivo`, LEFT JOIN.
    4. Exporta modelo_{espec}geo3_2010_2013.parquet -- especificacion
       nueva ("Ageo3"/"Bgeo3"). El archivo original modelo_{espec}_2010_2013.parquet
       NO se toca.

INPUTS
------
    data/processed/benchmark_train_test/modelo_{A,B}_2010_2013.parquet
    data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet

OUTPUTS
-------
    data/processed/benchmark_train_test/modelo_{A,B}geo3_2010_2013.parquet

CORRER
------
    python construir_pipeline_geo3_cv.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"
GEOESPACIAL_PATH = PROJECT_ROOT / "data" / "processed" / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"

ESPECIFICACIONES_BASE = ["A", "B"]
OLA_OBJETIVO = 2010
PREFIJOS_GEO = ("dmsp_", "alos_", "l5_")

# Variables de "cambio ENTRE olas" (2010 vs. 2013 del mismo hogar, ver
# construir_pipeline_geo_dmsp.py) -- se excluyen aqui tambien: para la
# ola 2010 quedan 100% NaN por construccion (dmsp_*) o siempre vacias
# (l5_*, Landsat 5 no tiene ningun par 2010-2013 real, ver Seccion 3.2),
# asi que no aportan nada como covariable de estado y solo aumentan
# columnas sin informacion.
COLS_CAMBIO_ENTRE_OLAS = [
    "dmsp_crecimiento_entre_olas", "dmsp_cambio_nivel_migracion",
    "dmsp_hogar_se_movio", "dmsp_distancia_movimiento_m",
    "l5_hogar_se_movio", "l5_distancia_movimiento_m",
]


def cargar_geoespacial_ola2010() -> pd.DataFrame:
    """Carga el archivo unificado, filtra a ola 2010, se queda con las
    columnas de las tres fuentes + consecutivo, y excluye hogares
    divididos para que el merge por consecutivo sea 1 a 1 seguro."""
    if not GEOESPACIAL_PATH.exists():
        print(f"ERROR: no se encontro {GEOESPACIAL_PATH}", file=sys.stderr)
        sys.exit(1)
    geo = pd.read_parquet(GEOESPACIAL_PATH)
    geo = geo[(geo["ola"] == OLA_OBJETIVO) & (geo["es_split"] == 0)].copy()

    cols_geo = [c for c in geo.columns if c.startswith(PREFIJOS_GEO) and c not in COLS_CAMBIO_ENTRE_OLAS]
    geo = geo[["consecutivo"] + cols_geo]

    n_dup = geo.duplicated(subset=["consecutivo"]).sum()
    if n_dup > 0:
        print(f"ERROR: {n_dup} filas duplicadas en consecutivo tras excluir es_split -- revisar.", file=sys.stderr)
        sys.exit(1)

    # Variable "estado" representativa por fuente para el diagnostico de
    # cobertura -- NO usar .notna().any() sobre TODAS las columnas: los
    # campos *_acum_n_anios quedan poblados con 0 (no NaN) incluso cuando
    # un hogar no tiene ningun dato real en la ventana, lo que inflaria
    # artificialmente la cobertura reportada (ver
    # utils/{dmsp,alos}_utils.py::calcular_estadisticos_ventana_acumulada).
    VARIABLE_INSIGNIA = {"dmsp": "dmsp_stable_lights", "alos": "alos_hh_db", "l5": "l5_ndvi"}
    n_por_fuente = {
        fuente: int(geo[col].notna().sum()) if col in geo.columns else None
        for fuente, col in VARIABLE_INSIGNIA.items()
    }
    print(f"Geoespacial ola {OLA_OBJETIVO} cargado: {len(geo):,} filas x {len(cols_geo)} columnas -- cobertura por fuente (variable insignia): {n_por_fuente}")
    return geo


def agregar_geo_a_benchmark(geo: pd.DataFrame) -> None:
    """Para cada especificacion, pega las columnas de las tres fuentes al
    archivo de train de la transicion 2010->2013 y exporta la version geo3."""
    for espec in ESPECIFICACIONES_BASE:
        ruta_in = DATA_DIR / f"modelo_{espec}_2010_2013.parquet"
        if not ruta_in.exists():
            print(f"ERROR: no se encontro {ruta_in}", file=sys.stderr)
            sys.exit(1)
        base = pd.read_parquet(ruta_in)

        antes = len(base)
        resultado = base.merge(geo, on="consecutivo", how="left", validate="one_to_one")
        assert len(resultado) == antes, f"El merge cambio el numero de filas en {ruta_in.name}"

        espec_geo = f"{espec}geo3"
        ruta_out = DATA_DIR / f"modelo_{espec_geo}_2010_2013.parquet"
        resultado.to_parquet(ruta_out, index=False)

        cols_geo = [c for c in geo.columns if c != "consecutivo"]
        n_con_geo = int(resultado[cols_geo].notna().any(axis=1).sum())
        print(f"  modelo_{espec_geo}_2010_2013.parquet: {resultado.shape[0]:,} filas x {resultado.shape[1]:,} columnas -- {n_con_geo:,}/{antes:,} hogares con al menos una variable geoespacial")


def main() -> None:
    print("=== construir_pipeline_geo3_cv.py ===")
    geo = cargar_geoespacial_ola2010()
    print("\nAgregando las 3 fuentes geoespaciales a la transicion 2010->2013 (sin holdout):")
    agregar_geo_a_benchmark(geo)
    print("\nListo. Especificaciones nuevas disponibles: Ageo3, Bgeo3")
    print("(agregadas a modelo_utils.ESPECIFICACIONES_CV_2010_2013 -- evaluar con evaluar_cv_semillas, NO con evaluar_multiples_semillas)")


if __name__ == "__main__":
    main()
