"""
diagnostico_shap_ipm_2013_2016.py
=====================================
Analogo a `diagnostico_shap_2013_2016.py` (chequeo de robustez temporal,
version "liviana": reutiliza hiperparametros ya ganadores, sin repetir la
busqueda), pero para pobreza multidimensional (IPM) en vez de monetaria.

Igual que la version monetaria, esto es SHAP in-sample: no hay una cuarta
ola de la ELCA para evaluar fuera de muestra un modelo entrenado en
2013->2016, asi que se reentrena UNA vez sobre
`modelo_{Aipm,Bipm}_2013_2016.parquet` reutilizando el
balanceo/hiperparametros ya ganadores de `registro_modelos_ipm.csv`
(entrenados originalmente sobre 2010->2013) y se explica sobre esa misma
poblacion -- responde "¿el ranking de importancia bajo IPM es consistente
entre ventanas?", no "¿cual es el mejor modelo posible para esta ventana?".

INPUTS

    data/processed/benchmark_resultados/registro_modelos_ipm.csv
    data/processed/benchmark_train_test/modelo_{Aipm,Bipm}_2013_2016.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia_ipm_2013_2016.csv
    (mismo formato que diagnostico_shap_importancia_2013_2016.csv,
    especificacion marcada "Aipm_2013_2016"/"Bipm_2013_2016")

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_ipm_2013_2016.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd

import modelo_utils as mu
from algoritmos_suite import (
    algoritmos_presentes_en_registro,
    filtrar_params_modelo,
    preparar_x_y,
    resolver_algoritmo,
)
from diagnostico_shap import calcular_shap

REGISTRO = mu.RESULTADOS_DIR / "registro_modelos_ipm.csv"
ESPECIFICACIONES = ["Aipm", "Bipm"]
RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ipm_2013_2016.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def entrenar_sobre_2013_2016(algoritmo_raw: str, espec: str, registro: pd.DataFrame):
    """Identico en logica a `entrenar_sobre_2013_2016` de
    diagnostico_shap_2013_2016.py -- duplicado deliberadamente para
    mantener cada script autocontenido."""
    fila = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)].iloc[0]
    balanceo = fila["balanceo_elegido"]
    params = filtrar_params_modelo(fila["hiperparametros"])

    poblacion = pd.read_parquet(mu.DATA_DIR / f"modelo_{espec}_2013_2016.parquet")
    x, y, x_de_nuevo, _, cat_cols = preparar_x_y(algoritmo_raw, poblacion, poblacion)

    pipe = resolver_algoritmo(algoritmo_raw)["construir_pipeline_fn"](x, y, balanceo, mu.RANDOM_STATE)
    if params:
        pipe.set_params(**params)
    pipe.fit(x, y)

    return pipe, x, x_de_nuevo, cat_cols


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

            log(f"=== {nombre_algo} -- Modelo {espec} (entrenado y explicado sobre 2013->2016) ===")
            pipe, x, x_shap_pobl, cat_cols = entrenar_sobre_2013_2016(algoritmo_raw, espec, registro)
            log(f"  Calculando SHAP sobre {x_shap_pobl.shape[0]} hogares (in-sample)...")
            shap_values, nombres, _ = calcular_shap(algoritmo_raw, pipe, x, x_shap_pobl, cat_cols)

            importancia_media = pd.Series(np.abs(shap_values).mean(axis=0), index=nombres).sort_values(ascending=False)
            ranking = importancia_media.rank(ascending=False)

            espec_salida = f"{espec}_2013_2016"
            for var, val in importancia_media.items():
                filas.append({
                    "algoritmo": nombre_algo, "especificacion": espec_salida, "variable": var,
                    "shap_abs_medio": round(val, 6), "rank": int(ranking[var]),
                })

            log(f"  Top 5 variables (|SHAP| medio): {importancia_media.head(5).index.tolist()}")

    ranking_df = pd.DataFrame(filas)
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(RUTA_SALIDA, index=False)
    log(f"FIN. Guardado: {RUTA_SALIDA} ({len(ranking_df)} filas)")


if __name__ == "__main__":
    main()
