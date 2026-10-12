"""
construir_pipeline_geo_dmsp.py
=================================

PIPELINE 1 de incorporacion de fuentes geoespaciales al benchmark: agrega
las variables DMSP-OLS (Seccion 3.2/3.3 de la tesis) a los conjuntos de
entrenamiento/prueba YA construidos por `build_benchmark_train_test.py`,
manteniendo intacto el diseno de holdout temporal principal (train =
covariables 2010, test = covariables 2013).

POR QUE SOLO DMSP-OLS AQUI
---------------------------
De las tres fuentes geoespaciales nuevas, DMSP-OLS es la UNICA con datos
reales tanto en la ola base de entrenamiento (2010, 100% cobertura) como
en la ola base de prueba (2013, 100% cobertura) -- ver
`variables_geoespaciales_unificadas.parquet` y la Seccion 3.2 de la
tesis. ALOS PALSAR y Landsat 5 TM solo tienen datos reales en 2010: si se
agregaran aqui, sus columnas quedarian 100% vacias en el conjunto de
prueba (covariables de 2013), y un modelo entrenado con ellas no podria
evaluarse fuera de muestra de forma valida -- ver
`construir_pipeline_geo3_cv.py` para el ejercicio que SI las usa, con un
esquema de validacion distinto (CV dentro de 2010->2013, sin holdout
temporal).

QUE HACE
--------
    1. Carga variables_geoespaciales_unificadas.parquet, se queda solo
       con las columnas dmsp_* + consecutivo + ola.
    2. Excluye hogares divididos (es_split == 1): el benchmark original
       ya excluye estos hogares de train/test (`consecutivo` es unico ahi
       por construccion, ver build_benchmark_train_test.py) -- si se
       dejaran en el archivo geoespacial, el merge por consecutivo podria
       encontrar match ambiguo.
    3. Para cada especificacion (A, B) y cada archivo (train=2010_2013,
       test=2013_2016): carga modelo_{espec}_{transicion}.parquet, pega
       las columnas dmsp_* de la ola base correspondiente (2010 para
       train, 2013 para test) por `consecutivo`, LEFT JOIN (ningun hogar
       del benchmark original se pierde).
    4. Verifica que el merge no haya cambiado el numero de filas.
    5. Exporta modelo_{espec}geoDMSP_{transicion}.parquet -- especificacion
       nueva ("AgeoDMSP"/"BgeoDMSP"), archivos ORIGINALES sin tocar.

INPUTS
------
    data/processed/benchmark_train_test/modelo_{A,B}_{2010_2013,2013_2016}.parquet
    data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet

OUTPUTS
-------
    data/processed/benchmark_train_test/modelo_{A,B}geoDMSP_{2010_2013,2013_2016}.parquet

CORRER
------
    python construir_pipeline_geo_dmsp.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"
GEOESPACIAL_PATH = PROJECT_ROOT / "data" / "processed" / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"

ESPECIFICACIONES_BASE = ["A", "B"]
# (archivo del benchmark, ola base cuyas covariables geoespaciales debe llevar)
ARCHIVOS_Y_OLA_BASE = [("2010_2013", 2010), ("2013_2016", 2013)]

# Variables de "cambio ENTRE olas" (comparan el estado 2010 vs. 2013 del
# MISMO hogar, ver Seccion 3.3.7 de la tesis) -- se EXCLUYEN aqui porque no
# son covariables "medidas en la ola base" validas para este diseno: por
# construccion quedan 100% NaN en la ola 2010 (no hay ola anterior con la
# que compararse), lo que ademas rompe alinear_columnas_categoricas()
# (una columna booleana 100% NaN se tipa distinto a una con valores
# reales -- 'dtype of categories must be the same'). Solo se conservan
# las variables de ESTADO y VENTANA ACUMULADA, que si estan bien
# definidas usando unicamente datos hasta la ola base.
COLS_CAMBIO_ENTRE_OLAS = [
    "dmsp_crecimiento_entre_olas", "dmsp_cambio_nivel_migracion",
    "dmsp_hogar_se_movio", "dmsp_distancia_movimiento_m",
]


def cargar_geoespacial_dmsp() -> pd.DataFrame:
    """Carga el archivo unificado, se queda con dmsp_* + consecutivo + ola,
    y excluye hogares divididos (es_split == 1) para que el merge por
    consecutivo sea 1 a 1 seguro."""
    if not GEOESPACIAL_PATH.exists():
        print(f"ERROR: no se encontro {GEOESPACIAL_PATH}", file=sys.stderr)
        sys.exit(1)
    geo = pd.read_parquet(GEOESPACIAL_PATH)
    cols_dmsp = [c for c in geo.columns if c.startswith("dmsp_") and c not in COLS_CAMBIO_ENTRE_OLAS]
    geo = geo[geo["es_split"] == 0][["consecutivo", "ola"] + cols_dmsp].copy()

    n_dup = geo.duplicated(subset=["consecutivo", "ola"]).sum()
    if n_dup > 0:
        print(f"ERROR: {n_dup} filas duplicadas en (consecutivo, ola) tras excluir es_split -- revisar.", file=sys.stderr)
        sys.exit(1)

    print(f"Geoespacial DMSP-OLS cargado: {len(geo):,} filas x {len(cols_dmsp)} columnas dmsp_* (hogares no divididos)")
    return geo


def agregar_geo_a_benchmark(geo: pd.DataFrame) -> None:
    """Para cada especificacion x archivo, pega las columnas dmsp_* de la
    ola base correspondiente y exporta la version geoDMSP."""
    for espec in ESPECIFICACIONES_BASE:
        for transicion, ola_base in ARCHIVOS_Y_OLA_BASE:
            ruta_in = DATA_DIR / f"modelo_{espec}_{transicion}.parquet"
            if not ruta_in.exists():
                print(f"ERROR: no se encontro {ruta_in}", file=sys.stderr)
                sys.exit(1)
            base = pd.read_parquet(ruta_in)

            geo_ola = geo[geo["ola"] == ola_base].drop(columns=["ola"])
            antes = len(base)
            resultado = base.merge(geo_ola, on="consecutivo", how="left", validate="one_to_one")
            assert len(resultado) == antes, f"El merge cambio el numero de filas en {ruta_in.name}"

            cols_dmsp = [c for c in geo_ola.columns if c.startswith("dmsp_")]
            n_con_dmsp = int(resultado[cols_dmsp].notna().any(axis=1).sum())

            espec_geo = f"{espec}geoDMSP"
            ruta_out = DATA_DIR / f"modelo_{espec_geo}_{transicion}.parquet"
            resultado.to_parquet(ruta_out, index=False)
            print(
                f"  modelo_{espec_geo}_{transicion}.parquet: {resultado.shape[0]:,} filas x "
                f"{resultado.shape[1]:,} columnas -- {n_con_dmsp:,}/{antes:,} hogares con DMSP-OLS (ola base {ola_base})"
            )


def main() -> None:
    print("=== construir_pipeline_geo_dmsp.py ===")
    geo = cargar_geoespacial_dmsp()
    print("\nAgregando DMSP-OLS a train/test del benchmark:")
    agregar_geo_a_benchmark(geo)
    print("\nListo. Especificaciones nuevas disponibles: AgeoDMSP, BgeoDMSP")
    print("(agregadas a modelo_utils.ESPECIFICACIONES_PRINCIPAL)")


if __name__ == "__main__":
    main()
