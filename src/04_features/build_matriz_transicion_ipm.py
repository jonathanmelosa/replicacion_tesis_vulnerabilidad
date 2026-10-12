"""
Matriz de transicion pobre/no-pobre entre olas consecutivas para el
target IPM (pobreza multidimensional) -- espejo de las matrices de
transicion de pobreza monetaria ya construidas en
`build_pobreza_desagregaciones.py::construir_matriz_transicion` (Lopez-
Calva y Ortiz-Juarez 2014, Tabla 3) (2026-08-31).

Reusa literalmente `construir_matriz_transicion` (IMPORTADA de
`build_pobreza_desagregaciones.py`, no duplicada) parametrizada con
`col_pobre="pobre_ipm"` sobre
`ipm_multidimensional_elca_longitudinal.parquet` (ver
`build_ipm_multidimensional.py`) en vez de `pobre_ingreso` sobre
`pobreza_monetaria_elca_longitudinal.parquet` -- mismo criterio de
emparejamiento 1 a 1 por `consecutivo` (excluye hogares divididos, mismo
conteo de excluidos que ya reporta la funcion original), mismas 4
categorias (Nunca pobre / Siempre pobre / Sale de la pobreza / Entra en
pobreza).

Version NO ponderada unicamente (a diferencia del pipeline monetario,
que tambien genera versiones ponderadas por factor de expansion y de
robustez -- sin_excepcional, banda LP, etc., que no aplican al IPM tal
como esta definido). Si se necesita una version ponderada, agregar
`peso_col` reusando la misma logica de `peso_transversal`/
`peso_longitudinal` ya construida para pobreza monetaria.

Tambien genera la figura de barras apiladas (espejo de
`06_transiciones_pobreza.png`, la de pobreza monetaria) reusando
`graf_transiciones_generico` -- IMPORTADA de `build_pobreza_desagregaciones.py`,
no duplicada -- con dos paneles.

INPUTS

    data/processed/ipm_multidimensional_elca_longitudinal.parquet

OUTPUTS

    outputs/tables/pobreza/transicion_conteo_ipm_ola{1_a_2,2_a_3}.csv
    outputs/tables/pobreza/transicion_pct_ipm_ola{1_a_2,2_a_3}.csv
    outputs/tables/pobreza/transicion_categorias_ipm_ola{1_a_2,2_a_3}.csv
    outputs/figures/pobreza/06_transiciones_pobreza_ipm.png

COMO CORRER

    cd src/04_features && python build_matriz_transicion_ipm.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pobreza_desagregaciones import (  # noqa: E402
    ANO_POR_OLA,
    FIGURES_DIR,
    OUTPUT_DIR,
    construir_matriz_transicion,
    graf_transiciones_generico,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IPM_PATH = PROJECT_ROOT / "data" / "processed" / "ipm_multidimensional_elca_longitudinal.parquet"


def main() -> None:
    ipm = pd.read_parquet(IPM_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    resumen_transiciones_ipm = []
    for ola_ini, ola_fin in [(1, 2), (2, 3)]:
        sufijo = f"{ola_ini}_a_{ola_fin}"
        periodo = f"{ANO_POR_OLA[ola_ini]}-{ANO_POR_OLA[ola_fin]}"

        resultado = construir_matriz_transicion(ipm, ola_ini, ola_fin, col_pobre="pobre_ipm")
        resultado["matriz_porcentaje_fila"].to_csv(OUTPUT_DIR / f"transicion_pct_ipm_ola{sufijo}.csv")
        resultado["matriz_conteo"].to_csv(OUTPUT_DIR / f"transicion_conteo_ipm_ola{sufijo}.csv")
        resultado["distribucion_categorias"].to_csv(
            OUTPUT_DIR / f"transicion_categorias_ipm_ola{sufijo}.csv", header=["porcentaje"]
        )
        resumen_transiciones_ipm.append(resultado)

        print(
            f"\nTransicion IPM {periodo} (ola {ola_ini} -> ola {ola_fin}) "
            f"(n={resultado['n_hogares_panel']}, excluidos por division={resultado['n_excluidos_por_division']}):"
        )
        print(resultado["matriz_porcentaje_fila"])
        print(resultado["distribucion_categorias"])

    graf_transiciones_generico(
        resumen_transiciones_ipm,
        titulo="Matriz de transicion de pobreza multidimensional (IPM) entre olas\n(Lopez-Calva y Ortiz-Juarez, 2014)",
        nombre_archivo="06_transiciones_pobreza_ipm.png",
    )

    print(f"\nGuardado en: {OUTPUT_DIR} y {FIGURES_DIR}")


if __name__ == "__main__":
    main()
