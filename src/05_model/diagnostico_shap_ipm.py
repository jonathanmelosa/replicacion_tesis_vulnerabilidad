"""
diagnostico_shap_ipm.py
=====================================
Analogo a `diagnostico_shap_ab.py`, pero para pobreza multidimensional
(IPM) en vez de monetaria
comparar el nucleo de importancia SHAP entre las dos definiciones de
pobreza, ademas de entre ventanas temporales (ver
`diagnostico_shap_2013_2016.py`).

Reutiliza literalmente `entrenar()`/`calcular_shap()` de
`diagnostico_shap.py` (mismo patron que `diagnostico_shap_ab.py`): el
unico cambio real es apuntar a `registro_modelos_ipm.csv` (Aipm/Bipm) en
vez de `registro_modelos_fbeta2_cv10.csv` (A/B) -- misma metodologia
robusta (F-beta beta=2, CV_FOLDS=10, N_ITER_BUSQUEDA=30, RANDOM_STATE=42),
confirmada en la columna "observaciones" del registro IPM como
"comparabilidad directa entre targets".

QUE HACE
---------------------------------------------------------------------
Para cada uno de los 5 algoritmos x 2 especificaciones (Aipm, Bipm):
    1. Reconstruye el modelo ganador (balanceo/hiperparametros ya
       decididos en registro_modelos_ipm.csv, UN solo fit).
    2. Calcula SHAP values sobre el conjunto de PRUEBA (holdout temporal
       2013->2016, target = pobre_ipm).
    3. Exporta el ranking completo de variables por |SHAP| medio.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_ipm.csv
    data/processed/benchmark_train_test/modelo_{Aipm,Bipm}_2010_2013.parquet
    data/processed/benchmark_train_test/modelo_{Aipm,Bipm}_2013_2016.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia_ipm.csv
    (mismo formato que diagnostico_shap_importancia_ab.csv)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_ipm.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd

import modelo_utils as mu
from algoritmos_suite import algoritmos_presentes_en_registro, resolver_algoritmo
from diagnostico_shap import calcular_shap, entrenar

REGISTRO = mu.RESULTADOS_DIR / "registro_modelos_ipm.csv"
ESPECIFICACIONES = ["Aipm", "Bipm"]
RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ipm.csv"

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
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(RUTA_SALIDA, index=False)
    log(f"FIN. Guardado: {RUTA_SALIDA} ({len(ranking_df)} filas)")


if __name__ == "__main__":
    main()
