"""
diagnostico_shap_variables_necesarias.py
=====================================
Responde una pregunta operativa que ninguno de los diagnosticos SHAP
anteriores contesta directamente: de las ~90 variables originales del
benchmark, ?cuantas bastan para capturar la mayor parte de la senal
predictiva? Post-procesa el ranking ya calculado por
`diagnostico_shap_ab.py` (`diagnostico_shap_importancia_ab.csv`) -- NO
recalcula SHAP ni reentrena nada.

Por que hay que agregar las columnas one-hot antes de contar
---------------------------------------------------------------------
`diagnostico_shap_importancia_ab.csv` reporta SHAP por COLUMNA del
pipeline ya transformado: para XGBoost/LightGBM/HistGradientBoosting eso
son las ~165-169 variables originales (categoricas nativas, sin
expandir), pero para Random Forest y Logistica son ~328-332 columnas
porque el preprocesador expande cada variable categorica en una columna
dummy por categoria (`cat__categoria_ocupacional_jefe_Cuenta propia`,
`cat__categoria_ocupacional_jefe_Jornalero`, ...) y cada numerica en una
columna de indicador de faltante adicional
(`num__missingindicator_<variable>`). Contar "variables necesarias" sin
agregar estas columnas de vuelta a su variable original haria que Random
Forest/Logistica parecieran necesitar muchas mas "variables" que los
arboles nativos por un artefacto de codificacion, no por una diferencia
real de concentracion de la senal -- este script suma el |SHAP| medio de
todas las columnas que pertenecen a la misma variable original (dummies
de una misma categorica, o el indicador de faltante de una misma
numerica) antes de calcular la masa acumulada, usando la lista real de
columnas categoricas de `modelo_A_2010_2013.parquet` (no hardcodeada a
ciegas) para hacer el prefix-matching de forma robusta.

QUE HACE
---------------------------------------------------------------------
Para cada algoritmo x especificacion (A, B) de
`diagnostico_shap_importancia_ab.csv`:
    1. Agrega el |SHAP| medio de columnas one-hot/missingindicator de
       vuelta a su variable original.
    2. Ordena por SHAP agregado, calcula la masa acumulada.
    3. Reporta el % de la masa total en las 10 variables mas importantes,
       y cuantas variables originales hacen falta para acumular el 80%.

INPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia_ab.csv
    data/processed/benchmark_train_test/modelo_A_2010_2013.parquet
    (solo para la lista de columnas categoricas originales)

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_variables_necesarias.csv
    (columnas: algoritmo, especificacion, n_variables_originales,
    pct_shap_top10, n_variables_para_80pct)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_variables_necesarias.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu

RUTA_SHAP = mu.PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_shap_importancia_ab.csv"
RUTA_DATOS_A = mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_A_2010_2013.parquet"
RUTA_SALIDA = mu.PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_shap_variables_necesarias.csv"

UMBRAL_MASA_ACUMULADA = 0.80
TOP_N_REPORTADO = 10


def _variables_categoricas_originales() -> list:
    df = pd.read_parquet(RUTA_DATOS_A)
    return df.select_dtypes(include="object").columns.tolist()


def _construir_mapa_variable_base(categoricas: list):
    categoricas_por_largo = sorted(categoricas, key=len, reverse=True)

    def variable_base(columna: str) -> str:
        columna = re.sub(r"^(num__|cat__)", "", columna)
        columna = re.sub(r"^missingindicator_", "", columna)
        for c in categoricas_por_largo:
            if columna == c or columna.startswith(c + "_"):
                return c
        return columna

    return variable_base


def main() -> None:
    shap_raw = pd.read_csv(RUTA_SHAP)
    categoricas = _variables_categoricas_originales()
    variable_base = _construir_mapa_variable_base(categoricas)

    filas = []
    for (algoritmo, espec), g in shap_raw.groupby(["algoritmo", "especificacion"]):
        g = g.copy()
        g["variable_base"] = g["variable"].map(variable_base)
        agregado = g.groupby("variable_base")["shap_abs_medio"].sum().sort_values(ascending=False)
        total = agregado.sum()
        masa_acumulada = agregado.cumsum() / total

        pct_top10 = masa_acumulada.iloc[TOP_N_REPORTADO - 1] if len(agregado) >= TOP_N_REPORTADO else masa_acumulada.iloc[-1]
        n_para_umbral = int((masa_acumulada < UMBRAL_MASA_ACUMULADA).sum() + 1)

        filas.append({
            "algoritmo": algoritmo,
            "especificacion": espec,
            "n_variables_originales": len(agregado),
            "pct_shap_top10": round(float(pct_top10), 4),
            "n_variables_para_80pct": n_para_umbral,
        })

    resultado = pd.DataFrame(filas).sort_values(["especificacion", "algoritmo"]).reset_index(drop=True)
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(RUTA_SALIDA, index=False)
    print(resultado.to_string(index=False))
    print(f"\nGuardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
