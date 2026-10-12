"""
Trayectorias de pobreza en las 3 olas de ELCA (2010, 2013, 2016), para
hogares NO pobres en la ola base (2010) -- complemento de la Seccion 5.2: a diferencia de la
Seccion 5.2 y de `eda_transicion_covariables.py` (que tratan cada
transicion de 2 olas como una poblacion independiente, igual que los
modelos de ML), este script seguir a los MISMOS hogares en las 3 olas
para responder si la entrada a pobreza en 2010->2013 es un evento
pasajero o el inicio de una trayectoria persistente.

Poblacion: hogares con match 1 a 1 (sin division de hogar en NINGUNA de
las 3 olas -- misma regla exacta que `_excluir_hogares_divididos` de
`eda_transicion_covariables.py`, reutilizada aqui) que ademas son NO
pobres en la ola 1 (2010). Verificado (2026-09-10): 2,530 hogares
sobreviven este filtro para pobreza monetaria.

Tipologia: con el hogar
fijo en NO pobre en 2010, las unicas 2 variables libres son su estado en
2013 y en 2016, lo que da exactamente 4 combinaciones:

    Nunca cae: No pobre en 2013 Y no pobre en 2016
    Cae tarde: No pobre en 2013 Y Si pobre en 2016
    Entra y sale (transitorio): Si pobre en 2013 Y no pobre en 2016
    Entra y se queda (persist.): Si pobre en 2013 Y Si pobre en 2016

Se corre para AMBAS definiciones de pobreza, cada una con su propia tipologia (un hogar puede tener
trayectorias distintas segun la definicion, igual que ya se documenta en
la Seccion 5.2 para las transiciones de 2 olas).

Ponderacion: `peso_longitudinal` (fexhog_2010), tomado de la fila de ola
3 de cada hogar. CASI constante por hogar entre olas 2 y 3 -- pero NO
perfectamente: verificado (2026-09-10, ver caveat agregado en
`cargar_pesos_muestrales`) que 43 de 6,911 hogares (0.6%) tienen un
`fexhog_2010` ligeramente distinto entre su fila de ola 2 y su fila de
ola 3 en la fuente misma (`HOGAR_PATH`), no un artefacto de esta funcion.
Efecto en `pct_ponderado`: inmaterial dado el tamano y magnitud de la
discrepancia, pero no es estrictamente "sin ambiguedad" como se afirmaba
antes en este docstring.

INPUTS
    data/processed/pobreza_monetaria_elca_longitudinal.parquet
    data/processed/ipm_multidimensional_elca_longitudinal.parquet
    data/processed/hogar_elca_longitudinal_clean.parquet (pesos)

OUTPUTS
    outputs/tables/eda_transicion_covariables/trayectorias_3olas_{monetaria,ipm}.csv

COMO CORRER
    cd src/02_build && python eda_trayectorias_3olas.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import (  # noqa: E402
    _llave_compuesta,
    cargar_pesos_muestrales,
)
from panel_transicion import (  # noqa: E402
    _excluir_hogares_divididos,
    cargar_peso_longitudinal_por_consecutivo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POBREZA_PATH = PROJECT_ROOT / "data" / "processed" / "pobreza_monetaria_elca_longitudinal.parquet"
IPM_PATH = PROJECT_ROOT / "data" / "processed" / "ipm_multidimensional_elca_longitudinal.parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"

CATEGORIAS_TRAYECTORIA = [
    "Nunca cae",
    "Cae tarde (solo en 2016)",
    "Entra y sale (transitorio)",
    "Entra y se queda (persistente)",
]


def _categoria(pobre_2013: bool, pobre_2016: bool) -> str:
    if not pobre_2013 and not pobre_2016:
        return "Nunca cae"
    if not pobre_2013 and pobre_2016:
        return "Cae tarde (solo en 2016)"
    if pobre_2013 and not pobre_2016:
        return "Entra y sale (transitorio)"
    return "Entra y se queda (persistente)"


def construir_trayectorias(df: pd.DataFrame, col_pobre: str, peso_col: str = "peso_longitudinal") -> dict:
    """A partir de `df` (consecutivo, ola, col_pobre, peso_col ya
    calculado), construye la tabla de trayectorias de 3 olas para hogares
    no pobres en ola 1, excluyendo hogares divididos en cualquiera de las
    3 olas (misma regla que el resto del proyecto)."""
    por_ola = {}
    for ola in (1, 2, 3):
        sub = df[df["ola"] == ola].dropna(subset=[col_pobre])
        sub = _excluir_hogares_divididos(sub, ola)
        por_ola[ola] = sub.set_index("consecutivo")

    comunes = por_ola[1].index.intersection(por_ola[2].index).intersection(por_ola[3].index)
    n_o1, n_o2, n_o3 = len(por_ola[1]), len(por_ola[2]), len(por_ola[3])

    no_pobres_2010 = por_ola[1].loc[por_ola[1][col_pobre] == False].index  # noqa: E712
    poblacion = comunes.intersection(no_pobres_2010)

    tabla = pd.DataFrame({
        "consecutivo": poblacion,
        "pobre_2013": por_ola[2].loc[poblacion, col_pobre].to_numpy(),
        "pobre_2016": por_ola[3].loc[poblacion, col_pobre].to_numpy(),
        "peso_longitudinal": por_ola[3].loc[poblacion, peso_col].to_numpy(),
    })
    tabla["trayectoria"] = [
        _categoria(p13, p16) for p13, p16 in zip(tabla["pobre_2013"], tabla["pobre_2016"])
    ]
    tabla["trayectoria"] = pd.Categorical(tabla["trayectoria"], categories=CATEGORIAS_TRAYECTORIA, ordered=True)

    resumen = tabla.groupby("trayectoria", observed=True).agg(
        n_hogares=("consecutivo", "size"),
        peso_total=("peso_longitudinal", "sum"),
    )
    resumen["pct_n"] = (resumen["n_hogares"] / resumen["n_hogares"].sum() * 100).round(1)
    resumen["pct_ponderado"] = (resumen["peso_total"] / resumen["peso_total"].sum() * 100).round(1)
    resumen = resumen.reindex(CATEGORIAS_TRAYECTORIA)

    return {
        "tabla_hogares": tabla,
        "resumen": resumen,
        "n_o1": n_o1, "n_o2": n_o2, "n_o3": n_o3,
        "n_comunes_3olas": len(comunes),
        "n_poblacion": len(poblacion),
    }


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pobreza = pd.read_parquet(POBREZA_PATH)
    llave_pobreza = _llave_compuesta(pobreza)
    cargar_pesos_muestrales(pobreza, llave_pobreza)
    resultado_monetaria = construir_trayectorias(pobreza, col_pobre="pobre_ingreso")

    ipm = pd.read_parquet(IPM_PATH)
    cargar_peso_longitudinal_por_consecutivo(ipm, ola_fin=3)
    resultado_ipm = construir_trayectorias(ipm, col_pobre="pobre_ipm")

    for nombre, resultado in [("monetaria", resultado_monetaria), ("ipm", resultado_ipm)]:
        print(f"\n=== TRAYECTORIAS 3 OLAS -- {nombre.upper()} ===")
        print(f"Hogares no-divididos con dato valido -- ola1: {resultado['n_o1']} | "
              f"ola2: {resultado['n_o2']} | ola3: {resultado['n_o3']}")
        print(f"Presentes y no-divididos en las 3 olas: {resultado['n_comunes_3olas']}")
        print(f"No pobres en 2010 (poblacion de interes): {resultado['n_poblacion']}")
        print("\nDistribucion de trayectorias:")
        print(resultado["resumen"])

        resultado["resumen"].to_csv(TABLES_DIR / f"trayectorias_3olas_{nombre}.csv")
        resultado["tabla_hogares"].to_csv(
            TABLES_DIR / f"trayectorias_3olas_{nombre}_hogares.csv", index=False
        )

    print(f"\nGuardado en: {TABLES_DIR}")


if __name__ == "__main__":
    main()
