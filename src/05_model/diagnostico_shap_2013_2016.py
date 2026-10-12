"""
diagnostico_shap_2013_2016.py
=====================================
Version "liviana" de repetir
el analisis de importancia SHAP (`diagnostico_shap_ab.py`) usando la
transicion 2013->2016 como su PROPIA transicion de entrenamiento, en vez
de como el holdout de prueba del benchmark 2010->2013.

Pregunta que responde: la Seccion 5.1.1 ya verifico que el Hallazgo 1 del
perfil univariado (entra se parece a sale) es robusto a la ola
(2010->2013 y 2013->2016). Este script pregunta lo mismo pero para la
importancia MULTIVARIADA de variables (Seccion 5.2.2): en particular, si
las variables de choque (`tuvo_choque_economico_hogar`, rank 7 en
2010->2013 para al menos un algoritmo) siguen siendo importantes cuando
se mide en la otra transicion, o si es una casualidad de esa ventana
especifica.

Por que es la version LIVIANA y no una replica completa del benchmark
---------------------------------------------------------------------
No hay una cuarta ola de la ELCA para evaluar un modelo entrenado en
2013->2016 fuera de muestra (a diferencia de 2010->2013, que sí tiene
2013->2016 como holdout real) -- cualquier evaluacion de desempeño
tendria que ser por CV dentro de la misma transicion (como ya hacen las
especificaciones `*geo3`). Repetir la busqueda de hiperparametros
(RandomizedSearchCV, CV_FOLDS=10, N_ITER_BUSQUEDA=30) completa para 5
algoritmos x 2 especificaciones sobre una transicion que de todos modos
no se puede validar con un holdout propio no se justifica solo para
chequear si el ranking de importancia es consistente. En su lugar, este
script REUTILIZA el balanceo/hiperparametros YA ganadores de la
transicion 2010->2013 (`registro_modelos_fbeta2_cv10.csv`, sin repetir la
busqueda) y los reentrena UNA vez sobre la poblacion de
`modelo_{A,B}_2013_2016.parquet` -- que ya existe como el holdout de
prueba del benchmark, y tiene la poblacion/covariables/target correctos
para esta pregunta (hogares no pobres en 2013, Y=entran en pobreza para
2016).

Consecuencia de esto: el SHAP se calcula EN LA MISMA muestra que se usa
para ajustar el modelo (no hay holdout propio para esta transicion) --
igual de IN-SAMPLE que las especificaciones `*geo3`, mismo matiz de
cautela ya aplicado ahi (mide que tanto se apoya el ajuste en cada
variable, no su aporte fuera de muestra). Los hiperparametros tampoco son
necesariamente los OPTIMOS para esta ventana especifica (son los que
ganaron en 2010->2013) -- esto responde "¿el ranking de importancia es
consistente entre ventanas usando una configuracion razonable?", no
"¿cual es el mejor modelo posible para 2013->2016?".

INPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_train_test/modelo_{A,B}_2013_2016.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia_2013_2016.csv
    (mismo formato que diagnostico_shap_importancia_ab.csv, especificacion
    marcada "A_2013_2016"/"B_2013_2016" para no confundirse con la
    corrida original)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_2013_2016.py
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

REGISTRO = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv"
ESPECIFICACIONES = ["A", "B"]
RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_shap_importancia_2013_2016.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def entrenar_sobre_2013_2016(algoritmo_raw: str, espec: str, registro: pd.DataFrame):
    """Analogo a `entrenar()` de diagnostico_shap.py, pero ajusta Y explica
    SHAP sobre LA MISMA poblacion (modelo_{espec}_2013_2016.parquet) en vez
    de entrenar en 2010->2013 y evaluar en 2013->2016 -- ver docstring del
    modulo para la justificacion de por que es in-sample aqui."""
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
