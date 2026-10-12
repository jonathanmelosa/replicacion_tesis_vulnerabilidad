"""
construir_pipeline_geo_dmsp_4clases.py
=========================================

Identico a `construir_pipeline_geo_dmsp.py` (que NO se modifica), pero
sobre las especificaciones A4/B4 del benchmark de 4 clases
(`build_benchmark_train_test_4clases.py`) en vez de las binarias A/B --
 Pega
DMSP-OLS (unica fuente con cobertura real en ambas olas del holdout
principal) manteniendo intacto el holdout temporal (train=2010, test=2013).

QUE HACE / INPUTS / OUTPUTS: identico a construir_pipeline_geo_dmsp.py,
sustituyendo modelo_{A,B}_* por modelo_{A4,B4}_* -> modelo_{A4,B4}geoDMSP_*.

COMO CORRER
-----------
    cd src/05_model && python construir_pipeline_geo_dmsp_4clases.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"
GEOESPACIAL_PATH = PROJECT_ROOT / "data" / "processed" / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"

ESPECIFICACIONES_BASE = ["A4", "B4"]
ARCHIVOS_Y_OLA_BASE = [("2010_2013", 2010), ("2013_2016", 2013)]

COLS_CAMBIO_ENTRE_OLAS = [
    "dmsp_crecimiento_entre_olas", "dmsp_cambio_nivel_migracion",
    "dmsp_hogar_se_movio", "dmsp_distancia_movimiento_m",
]


def cargar_geoespacial_dmsp() -> pd.DataFrame:
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
    print("=== construir_pipeline_geo_dmsp_4clases.py ===")
    geo = cargar_geoespacial_dmsp()
    print("\nAgregando DMSP-OLS a train/test del benchmark de 4 clases:")
    agregar_geo_a_benchmark(geo)
    print("\nListo. Especificaciones nuevas disponibles: A4geoDMSP, B4geoDMSP")
    print("(agregadas a modelo_utils_multiclase.ESPECIFICACIONES_4CLASES_PRINCIPAL)")


if __name__ == "__main__":
    main()
