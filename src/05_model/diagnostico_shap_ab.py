"""
diagnostico_shap_ab.py
=====================================
Analisis SHAP (SHapley Additive exPlanations) para los Modelos A y B
PUROS (sin variables geoespaciales) -- target pobreza monetaria, los 5
algoritmos de la suite, ya bajo la metodologia final de la tesis (umbral
F-beta beta=2, CV_FOLDS=10, N_ITER_BUSQUEDA=30, `registro_modelos_
fbeta2_cv10.csv`).

Por que un script separado de `diagnostico_shap.py`
---------------------------------------------------------------------
`diagnostico_shap.py` calcula SHAP para las especificaciones CON
DMSP-OLS (`AgeoDMSP`/`AipmgeoDMSP`) -- responde la pregunta de
Seccion~5.4 ("¿aporta DMSP-OLS a nivel de prediccion individual?"). Este
script responde una pregunta distinta, la de Seccion~4.5
("comparacion de estas listas de variables mas importantes entre el
Modelo A y el Modelo B... antes de la incorporacion de las variables
geoespaciales"): SHAP sobre el benchmark puro (A/B, sin DMSP-OLS), para
tener una escala de importancia COMPARABLE entre los 5 algoritmos (a
diferencia de comparar directamente feature_importances_ de arboles
contra coeficientes estandarizados de la logistica, que no estan en la
misma unidad) -- justo lo que la Seccion~4.5 señalaba como tarea
pendiente. Reutiliza literalmente `entrenar()`/`calcular_shap()` de
`diagnostico_shap.py` (importadas, no duplicadas).

QUE HACE
---------------------------------------------------------------------
Para cada uno de los 5 algoritmos x 2 especificaciones (A, B):
    1. Reconstruye el modelo ganador (balanceo/hiperparametros ya
       decididos en registro_modelos_fbeta2_cv10.csv, UN solo fit, sin
       repetir RandomizedSearchCV).
    2. Calcula SHAP values sobre el conjunto de PRUEBA (holdout temporal
       2013->2016).
    3. Exporta el ranking completo de variables por |SHAP| medio.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_train_test/modelo_{A,B}_2010_2013.parquet
    data/processed/benchmark_train_test/modelo_{A,B}_2013_2016.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia_ab.csv
    (ranking completo de variables por algoritmo x especificacion)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_ab.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd

import modelo_utils as mu
from algoritmos_suite import algoritmos_presentes_en_registro, resolver_algoritmo
from diagnostico_shap import calcular_shap, entrenar

REGISTRO = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv"
ESPECIFICACIONES = ["A", "B"]
RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ab.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def main() -> None:
    registro = pd.read_csv(REGISTRO)
    algoritmos_crudos = algoritmos_presentes_en_registro(REGISTRO)
    log(f"Algoritmos detectados en {REGISTRO.name}: {algoritmos_crudos}")

    filas = []
    for espec in ESPECIFICACIONES:
        for algoritmo_raw in algoritmos_crudos:
            nombre_algo = resolver_algoritmo(algoritmo_raw)["nombre_bonito"]
            existe = ((registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)).any()
            if not existe:
                log(f"OMITIDO: no hay fila para {algoritmo_raw!r}/{espec!r} en {REGISTRO.name}")
                continue

            log(f"=== {nombre_algo} -- Modelo {espec} ===")
            pipe, x_train, x_test, y_test, cat_cols = entrenar(algoritmo_raw, espec, registro)
            log(f"  Calculando SHAP sobre {x_test.shape[0]} hogares de test...")
            shap_values, nombres, _ = calcular_shap(algoritmo_raw, pipe, x_train, x_test, cat_cols)

            importancia_media = pd.Series(np.abs(shap_values).mean(axis=0), index=nombres).sort_values(ascending=False)
            ranking = importancia_media.rank(ascending=False)

            for var, val in importancia_media.items():
                filas.append({
                    "algoritmo": nombre_algo, "especificacion": espec, "variable": var,
                    "shap_abs_medio": round(val, 6), "rank": int(ranking[var]),
                })

            log(f"  Top 5 variables (|SHAP| medio): {importancia_media.head(5).index.tolist()}")

    ranking_df = pd.DataFrame(filas)
    _comparar_con_corrida_previa(RUTA_SALIDA, ranking_df)
    ranking_df.to_csv(RUTA_SALIDA, index=False)
    log(f"FIN. Guardado: {RUTA_SALIDA} ({len(ranking_df)} filas)")


def _comparar_con_corrida_previa(ruta, df_nuevo: pd.DataFrame) -> None:
    """Verificacion explicita de reproducibilidad (mismo patron que
    modelo_fbeta2_cv10_importancia.py): si RUTA_SALIDA ya existe de una
    corrida anterior, compara shap_abs_medio fila a fila (misma llave
    algoritmo+especificacion+variable) y avisa si hay cualquier
    diferencia -- el calculo no tiene aleatoriedad (TreeExplainer/
    LinearExplainer sobre un modelo ya entrenado y el mismo x_test), asi
    que una corrida que reproduce otra debe dar EXACTAMENTE el mismo
    valor, no solo uno parecido."""
    if not ruta.exists():
        log(f"  (sin corrida previa de {ruta.name} contra la cual comparar -- primera vez que se genera)")
        return
    df_previo = pd.read_csv(ruta)
    clave = ["algoritmo", "especificacion", "variable"]
    merged = df_previo.merge(df_nuevo, on=clave, suffixes=("_previo", "_nuevo"), how="outer", indicator=True)
    solo_en_uno = merged[merged["_merge"] != "both"]
    if not solo_en_uno.empty:
        log(f"  ADVERTENCIA: {len(solo_en_uno)} filas presentes en solo una de las dos corridas (claves distintas).")
        return
    diffs = merged.loc[merged["shap_abs_medio_previo"] != merged["shap_abs_medio_nuevo"]]
    if diffs.empty:
        log(f"  Reproducibilidad OK: {ruta.name} identico a la corrida anterior ({len(df_nuevo)} filas comparadas).")
    else:
        max_diff = (diffs["shap_abs_medio_previo"] - diffs["shap_abs_medio_nuevo"]).abs().max()
        log(f"  ADVERTENCIA: {len(diffs)} filas de {ruta.name} difieren de la corrida anterior (diferencia maxima: {max_diff:.2e}).")


if __name__ == "__main__":
    main()
