"""
eda_split_trayectoria_nucleo.py
=====================================
Cuarto hallazgo de la Seccion 5.2 restringido al nucleo SHAP: compara, entre
los hogares que entran en pobreza en 2010->2013, a los transitorios (vuelven
a salir antes de 2016) con los persistentes (siguen pobres en 2016), solo en
las 22 variables del nucleo SHAP (`diagnostico_shap_nucleo_perfil.csv`).

Cambio 2026-09-30: se retira el perfil univariado del
paper. La comparacion anterior (`eda_perfil_split_trayectoria.py` y
`robustez_split_trayectoria.py`) usaba las 53 variables del perfil, con
pruebas de Welch y correccion de Benjamini-Hochberg. Decision:
sin correccion por comparaciones multiples; se mantiene solo lo que se
sostiene en las variables del nucleo SHAP, con descriptivos.

METODOLOGIA (descriptiva)
    Mismo panel que `panel_transicion.construir_panel_split`
    (matriz de transicion monetaria 2010->2013 con "Entra en pobreza"
    separada por trayectoria a 2016) y mismas covariables de la ola base
    (`cargar_covariables_ola_base`). Por variable:
      - media ponderada por `peso_longitudinal` en cada grupo (siempre pobre,
        sale, transitorio, persistente, nunca pobre), misma formula que
        `construir_tabla_comparativa`;
      - diferencia persistente - transitorio y diferencia estandarizada
        ponderada, (m_p - m_t) / sqrt((s2_p + s2_t) / 2), marcada como
        sustantiva si |dif. estandarizada| >= 0.20 (mismo umbral UMBRAL_SMD
        de `robustez_split_trayectoria.py`).
    Las variables categoricas (material de pisos, estado civil del jefe) se
    expanden en un indicador 0/1 por nivel, expresado en %.

INPUTS
    data/processed/benchmark_resultados/diagnostico_shap_nucleo_perfil.csv
    Fuentes de `panel_transicion.py` (panel y trayectorias a 2016)

OUTPUTS
    outputs/tables/eda_transicion_covariables/split_trayectoria_nucleo_monetaria.csv

COMO CORRER
    cd src/02_build && python eda_split_trayectoria_nucleo.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from panel_transicion import (  # noqa: E402
    CATEGORIAS_ORDEN_SPLIT, TABLES_DIR, cargar_covariables_ola_base, cargar_dmsp_por_consecutivo,
    cargar_tipos_variables, construir_panel_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUTA_NUCLEO = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_shap_nucleo_perfil.csv"
RUTA_SALIDA = TABLES_DIR / "split_trayectoria_nucleo_monetaria.csv"
UMBRAL_SMD = 0.20
TRANSITORIO, PERSISTENTE = "Entra (transitorio)", "Entra (persistente)"


def media_var_ponderada(x: pd.Series, w: pd.Series) -> tuple:
    mask = x.notna() & w.notna()
    x, w = x[mask].astype(float), w[mask]
    if not len(x):
        return float("nan"), float("nan"), 0
    m = np.average(x, weights=w)
    return m, np.average((x - m) ** 2, weights=w), int(mask.sum())


def indicadores(df: pd.DataFrame, var: str, tipo: str) -> dict:
    """{etiqueta: serie numerica}. Numerica: la variable (en % si es 0/1).
    Categorica/Booleana: un indicador 0/1 por nivel, en %."""
    valores = df[var]
    if tipo == "Numerica":
        no_nulos = valores.dropna()
        if len(no_nulos) and no_nulos.between(0, 1).all():
            return {f"{var} (%)": valores * 100}
        return {var: valores}
    salida = {}
    for nivel in sorted(valores.dropna().unique(), key=str):
        salida[f"{var} = {nivel} (%)"] = (valores == nivel).astype(float).where(valores.notna()) * 100
    return salida


def main() -> None:
    panel = construir_panel_split()
    nucleo = pd.read_csv(RUTA_NUCLEO)["variable"].tolist()
    covariables = cargar_covariables_ola_base(cargar_dmsp_por_consecutivo(2010), 1)
    tipos = cargar_tipos_variables()
    faltantes = set(nucleo) - set(covariables.columns)
    if faltantes:
        raise ValueError(f"Variables del nucleo sin covariable en la ola base: {faltantes}")

    df = panel.merge(covariables[nucleo], left_on="consecutivo", right_index=True, how="left")
    peso = df["peso_longitudinal"]
    n_t, n_p = int((df["categoria"] == TRANSITORIO).sum()), int((df["categoria"] == PERSISTENTE).sum())
    print(f"Transitorio: {n_t} hogares | Persistente: {n_p} hogares")

    filas = []
    for var in nucleo:
        for etiqueta, serie in indicadores(df, var, tipos.get(var, "Numerica")).items():
            fila = {"variable": var, "indicador": etiqueta}
            stats = {}
            for cat in CATEGORIAS_ORDEN_SPLIT:
                mask = df["categoria"] == cat
                stats[cat] = media_var_ponderada(serie[mask], peso[mask])
                fila[cat] = stats[cat][0]
            (m_t, v_t, n_t_var), (m_p, v_p, n_p_var) = stats[TRANSITORIO], stats[PERSISTENTE]
            sd = np.sqrt((v_t + v_p) / 2)
            fila.update({
                "n_transitorio": n_t_var, "n_persistente": n_p_var,
                "dif_persistente_menos_transitorio": m_p - m_t,
                "dif_estandarizada": (m_p - m_t) / sd if sd > 0 else float("nan"),
            })
            fila["sustantiva"] = abs(fila["dif_estandarizada"]) >= UMBRAL_SMD
            filas.append(fila)

    tabla = pd.DataFrame(filas)
    tabla = tabla.reindex(tabla["dif_estandarizada"].abs().sort_values(ascending=False).index)
    tabla.to_csv(RUTA_SALIDA, index=False)
    columnas = ["indicador", TRANSITORIO, PERSISTENTE, "Sale de la pobreza", "Siempre pobre", "dif_estandarizada", "sustantiva"]
    print(tabla[columnas].round(3).to_string(index=False))
    print(f"\nSustantivas (|dif. est.| >= {UMBRAL_SMD}): {int(tabla['sustantiva'].sum())} de {len(tabla)} indicadores")
    print(f"Guardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
