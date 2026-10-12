"""
comparar_nucleos_shap.py
=====================================
Calcula el "nucleo" de importancia SHAP (misma definicion exacta de
`diagnostico_shap_nucleo_perfil.py`: variable dentro del 80% de masa SHAP
acumulada en al menos 5 de 10 combinaciones algoritmo/especificacion)
para las 4 combinaciones definicion-de-pobreza x ventana-temporal, y
compara que variables se mantienen en las 4 -- para saber si el nucleo de variables de la tesis (hasta ahora
solo verificado para monetaria, 2010->2013 evaluado en test 2013->2016)
es robusto a la definicion de pobreza (IPM) y a la ventana temporal
(2013->2016 como su propia transicion).

Reutiliza la logica de agregacion one-hot->variable de
`diagnostico_shap_nucleo_perfil.py` (duplicada deliberadamente, incluye
solo el calculo del nucleo -- NO la categorizacion tematica para tabla
LaTeX, que no hace falta para esta comparacion).

INPUTS (los 4 SHAP rankings ya calculados, ninguno se recalcula aqui)

    data/processed/benchmark_resultados/diagnostico_shap_importancia_ab.csv
        (Monetaria, train 2010->2013, evaluado en test 2013->2016)
    data/processed/benchmark_resultados/diagnostico_shap_importancia_2013_2016.csv
        (Monetaria, 2013->2016 como propia ventana, in-sample)
    data/processed/benchmark_resultados/diagnostico_shap_importancia_ipm.csv
        (IPM, train 2010->2013, evaluado en test 2013->2016)
    data/processed/benchmark_resultados/diagnostico_shap_importancia_ipm_2013_2016.csv
        (IPM, 2013->2016 como propia ventana, in-sample)

OUTPUT

    data/processed/benchmark_resultados/comparacion_nucleos_shap.csv
    (una fila por variable, columnas: en cuantas de las 4 combinaciones
    entra al nucleo, y su n_combinaciones dentro de cada una)

COMO CORRER

    cd src/05_model && python comparar_nucleos_shap.py
"""

import re
from pathlib import Path

import pandas as pd

import modelo_utils as mu

UMBRAL_MASA_ACUMULADA = 0.80
UMBRAL_COMBINACIONES = 5  # de 10 (5 algoritmos x 2 especificaciones)

FUENTES = {
    "Monetaria_2010-2013(test 2013-2016)": {
        "shap_csv": mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ab.csv",
        "parquet_categoricas": mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_A_2010_2013.parquet",
    },
    "Monetaria_2013-2016(propia)": {
        "shap_csv": mu.RESULTADOS_DIR / "diagnostico_shap_importancia_2013_2016.csv",
        "parquet_categoricas": mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_A_2013_2016.parquet",
    },
    "IPM_2010-2013(test 2013-2016)": {
        "shap_csv": mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ipm.csv",
        "parquet_categoricas": mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_Aipm_2010_2013.parquet",
    },
    "IPM_2013-2016(propia)": {
        "shap_csv": mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ipm_2013_2016.csv",
        "parquet_categoricas": mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_Aipm_2013_2016.parquet",
    },
}

RUTA_SALIDA = mu.RESULTADOS_DIR / "comparacion_nucleos_shap.csv"


def _variables_categoricas_originales(parquet_path: Path) -> list:
    df = pd.read_parquet(parquet_path)
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


def calcular_nucleo(shap_csv: Path, parquet_categoricas: Path) -> set:
    shap_raw = pd.read_csv(shap_csv)
    categoricas = _variables_categoricas_originales(parquet_categoricas)
    variable_base = _construir_mapa_variable_base(categoricas)
    shap_raw = shap_raw.copy()
    shap_raw["base"] = shap_raw["variable"].map(variable_base)

    en_80_por_combinacion = {}
    for (algoritmo, espec), g in shap_raw.groupby(["algoritmo", "especificacion"]):
        agregado = g.groupby("base")["shap_abs_medio"].sum().sort_values(ascending=False)
        masa_acumulada = agregado.cumsum() / agregado.sum()
        n_para_umbral = int((masa_acumulada < UMBRAL_MASA_ACUMULADA).sum() + 1)
        conjunto_80 = set(agregado.index[:n_para_umbral])
        en_80_por_combinacion[f"{algoritmo}_{espec}"] = conjunto_80

    n_combinaciones_totales = len(en_80_por_combinacion)
    todas_las_variables = set().union(*en_80_por_combinacion.values())
    conteo = pd.Series({
        v: sum(v in conjunto for conjunto in en_80_por_combinacion.values())
        for v in todas_las_variables
    })
    print(f"    ({n_combinaciones_totales} combinaciones algoritmo/especificacion detectadas)")
    nucleo = set(conteo[conteo >= UMBRAL_COMBINACIONES].index)
    return nucleo, conteo


def main() -> None:
    nucleos = {}
    conteos = {}
    for nombre, cfg in FUENTES.items():
        print(f"=== {nombre} ===")
        nucleo, conteo = calcular_nucleo(cfg["shap_csv"], cfg["parquet_categoricas"])
        nucleos[nombre] = nucleo
        conteos[nombre] = conteo
        print(f"    Nucleo ({UMBRAL_COMBINACIONES}/10): {len(nucleo)} variables")

    todas = sorted(set().union(*nucleos.values()))
    filas = []
    for var in todas:
        fila = {"variable": var}
        n_en = 0
        for nombre in FUENTES:
            en_nucleo = var in nucleos[nombre]
            fila[f"en_nucleo_{nombre}"] = en_nucleo
            fila[f"n_combinaciones_{nombre}"] = int(conteos[nombre].get(var, 0))
            n_en += int(en_nucleo)
        fila["n_fuentes_donde_esta_en_nucleo"] = n_en
        filas.append(fila)

    resultado = pd.DataFrame(filas).sort_values(
        ["n_fuentes_donde_esta_en_nucleo", "variable"], ascending=[False, True]
    )
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(RUTA_SALIDA, index=False)

    print(f"\n=== Variables en el nucleo de las 4 combinaciones a la vez ===")
    en_las_4 = resultado[resultado["n_fuentes_donde_esta_en_nucleo"] == 4]["variable"].tolist()
    print(en_las_4 if en_las_4 else "(ninguna)")

    print(f"\n=== Variables en 3 de 4 ===")
    en_3 = resultado[resultado["n_fuentes_donde_esta_en_nucleo"] == 3]["variable"].tolist()
    print(en_3 if en_3 else "(ninguna)")

    print(f"\nGuardado: {RUTA_SALIDA} ({len(resultado)} variables)")


if __name__ == "__main__":
    main()
