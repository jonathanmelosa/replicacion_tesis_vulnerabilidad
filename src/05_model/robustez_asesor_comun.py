"""
robustez_asesor_comun.py
========================
Utilidades compartidas por las cuatro pruebas de robustez que responden
los comentarios del asesor del 2026-10-07:

  - robustez_hogares_no_repetidos.py (comentario 1b)
  - robustez_historia_hogar.py (comentario 1a)
  - robustez_techo_comunidad.py (comentario 2)
  - robustez_encuesta_desactualizada.py (comentarios 1a y 2)

Decisiones comunes:
  - Los cinco algoritmos de la suite, Modelos A y B.
  - Hiperparametros y balanceo YA elegidos en el paper, sin nueva
    busqueda: `registro_modelos_fbeta2_cv10.csv` (pobreza monetaria) y
    `registro_modelos_ipm.csv` (IPM). Las diferencias entre variantes
    reflejan solo el cambio de diseño o de variables.
  - Semillas del paper (`mu.SEMILLAS` = 42, 1, 2, 3, 4): cada semilla fija
    el modelo y la particion de la validacion cruzada.
  - Validacion cruzada de 10 particiones (`N_FOLDS`).
  - Metrica principal: AUC-PR (average precision), como la Seccion 5 del
    paper. Se reporta tambien AUC-ROC.
  - Cifra puntual = metrica sobre la probabilidad PROMEDIO de las 5
    semillas; se reporta ademas la desviacion estandar entre semillas.
  - IC95% de una diferencia = bootstrap pareado por COMUNIDAD (2,000
    remuestras), con la misma funcion y la misma regla de clusters que
    `diagnostico_curva_pr.py` (comunidad no identificada, 8888888 =
    cluster de un hogar).

Computo en paralelo: cada ajuste (semilla x particion x variante) y
cada bootstrap es una tarea independiente que se reparte entre todos los
nucleos (`N_JOBS` = os.cpu_count()) con joblib/loky. Cada tarea usa UN
hilo (n_jobs=1 en el modelo e `inner_max_num_threads=1` para OpenMP/BLAS)
para no sobresuscribir los nucleos. Efecto secundario deseable: los
resultados no dependen del numero de nucleos de la maquina. Las semillas
del bootstrap se asignan por posicion en una lista fija, asi que el
resultado es el mismo en serie o en paralelo.

Nada de este modulo escribe archivos; cada script guarda sus salidas en
`OUT_DIR`.
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import diagnostico_curva_pr as dcp
import modelo_utils as mu
from algoritmos_suite import algoritmos_presentes_en_registro, filtrar_params_modelo, preparar_x_y, resolver_algoritmo

OUT_DIR = mu.RESULTADOS_DIR / "robustez_asesor_2026_10"
PANEL_PATH = mu.PROJECT_ROOT / "data" / "processed" / "benchmark_consolidado_elca_longitudinal.parquet"
IPM_PATH = mu.PROJECT_ROOT / "data" / "processed" / "ipm_multidimensional_elca_longitudinal.parquet"
NUCLEO_PATH = mu.RESULTADOS_DIR / "nucleo_por_especificacion.csv"

N_FOLDS = 10
N_JOBS = os.cpu_count()
SEMILLAS = mu.SEMILLAS
N_BOOT = dcp.N_BOOT
SENTINEL_COMUNIDAD = dcp.SENTINEL_COMUNIDAD

# definicion -> (registro de hiperparametros, {espec: (sin luz, con luz)})
REGISTROS = {
    "monetaria": ("registro_modelos_fbeta2_cv10.csv", {"A": ("A", "AgeoDMSP"), "B": ("B", "BgeoDMSP")}),
    "ipm": ("registro_modelos_ipm.csv", {"A": ("Aipm", "AipmgeoDMSP"), "B": ("Bipm", "BipmgeoDMSP")}),
}


def cargar_registro(definicion: str) -> tuple[pd.DataFrame, list]:
    """Registro de hiperparametros de la definicion y los algoritmos que
    contiene (sin listas a mano, ver `algoritmos_suite.py`)."""
    ruta = mu.RESULTADOS_DIR / REGISTROS[definicion][0]
    return pd.read_csv(ruta), algoritmos_presentes_en_registro(ruta)


def nombre_corto(algoritmo_raw: str) -> str:
    return resolver_algoritmo(algoritmo_raw)["nombre_bonito"]


def cargar(espec: str, transicion: str) -> pd.DataFrame:
    """Archivo del benchmark ya construido (`transicion` = '2010_2013' o
    '2013_2016')."""
    return pd.read_parquet(mu.DATA_DIR / f"modelo_{espec}_{transicion}.parquet")


def ajustar_y_predecir(algoritmo_raw: str, espec_registro: str, registro: pd.DataFrame,
                       df_train: pd.DataFrame, df_pred: pd.DataFrame, semilla: int) -> np.ndarray:
    """Entrena con `df_train` usando el balanceo y los hiperparametros de
    (algoritmo, espec_registro) y devuelve la probabilidad para `df_pred`.
    Misma receta que `generar_predicciones_test_dmsp.entrenar_y_predecir`,
    con la semilla como argumento."""
    fila = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec_registro)]
    assert len(fila) == 1, f"no hay una fila unica para {algoritmo_raw} / {espec_registro}"
    fila = fila.iloc[0]
    x_tr, y_tr, x_pr, _, _ = preparar_x_y(algoritmo_raw, df_train, df_pred)
    pipe = resolver_algoritmo(algoritmo_raw)["construir_pipeline_fn"](x_tr, y_tr, fila["balanceo_elegido"], semilla)
    params = filtrar_params_modelo(fila["hiperparametros"])
    if params:
        pipe.set_params(**params)
    if "modelo__n_jobs" in pipe.get_params():
        pipe.set_params(modelo__n_jobs=1)  # un hilo por tarea; el paralelismo va entre tareas
    pipe.fit(x_tr, y_tr)
    return pipe.predict_proba(x_pr)[:, 1]


def en_paralelo(tareas) -> list:
    """Ejecuta una lista de `delayed(...)` en todos los nucleos, un hilo por
    tarea; devuelve los resultados en el orden de la lista."""
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=N_JOBS)(list(tareas))


def particiones(y: np.ndarray, semilla: int, grupos: np.ndarray = None):
    """10 particiones estratificadas; agrupadas si se pasan `grupos`."""
    if grupos is None:
        return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=semilla).split(np.zeros(len(y)), y))
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=semilla)
    return list(cv.split(np.zeros(len(y)), y, grupos))


def _ajustar_particion(algoritmo_raw, espec_registro, registro, df, idx_tr, idx_va, semilla, por_fold):
    df_tr, df_va = df.iloc[idx_tr], df.iloc[idx_va]
    if por_fold is not None:
        df_tr, df_va = por_fold(df_tr, df_va)
    return ajustar_y_predecir(algoritmo_raw, espec_registro, registro, df_tr, df_va, semilla)


def oof_semillas(algoritmo_raw: str, espec_registro: str, registro: pd.DataFrame, df: pd.DataFrame,
                 grupos: np.ndarray = None, por_fold=None) -> dict:
    """{semilla: probabilidades fuera de la particion (out-of-fold)}, con las
    5 semillas x 10 particiones ajustadas en paralelo. `por_fold`, si se
    pasa, recibe (df_train, df_val) de cada particion y devuelve ambos con
    columnas que dependen de la particion (p. ej. la tasa de los vecinos,
    que solo puede usar el resultado de la parte de entrenamiento)."""
    df = df.reset_index(drop=True)
    y = df["Y"].values
    planes = [(s, tr, va) for s in SEMILLAS for tr, va in particiones(y, s, grupos)]
    resultados = en_paralelo(delayed(_ajustar_particion)(algoritmo_raw, espec_registro, registro, df, tr, va, s, por_fold)
                             for s, tr, va in planes)
    oof = {s: np.full(len(df), np.nan) for s in SEMILLAS}
    for (s, _, va), p in zip(planes, resultados):
        oof[s][va] = p
    assert not any(np.isnan(p).any() for p in oof.values())
    return oof


def cluster_id(df: pd.DataFrame) -> np.ndarray:
    """Clusters del bootstrap: la regla de `diagnostico_curva_pr.py`."""
    return dcp.asignar_cluster_id(df.reset_index(drop=True)).values


def _comparar_con_semilla(y, p_ref, p_var, clusters, semilla):
    return comparar(y, p_ref, p_var, clusters, np.random.default_rng(semilla))


def comparar_varios(casos: list, semilla_base: int) -> list:
    """`casos`: lista de (y, p_ref, p_var, clusters). Corre `comparar` en
    paralelo; el caso i usa la semilla `semilla_base + i`."""
    return en_paralelo(delayed(_comparar_con_semilla)(y, a, b, c, semilla_base + i)
                       for i, (y, a, b, c) in enumerate(casos))


def comparar(y: np.ndarray, p_ref: np.ndarray, p_var: np.ndarray, clusters: np.ndarray, rng) -> dict:
    """AUC-PR y AUC-ROC de una referencia y una variante, su diferencia y
    el IC95% de la diferencia de AUC-PR (bootstrap pareado por comunidad,
    `dcp.bootstrap_ap`)."""
    _, delta = dcp.bootstrap_ap(y, p_ref, p_var, clusters, rng)
    low, high = np.percentile(delta, [2.5, 97.5])
    ap_ref, ap_var = average_precision_score(y, p_ref), average_precision_score(y, p_var)
    return {
        "n_hogares": len(y), "prevalencia": round(y.mean(), 4),
        "ap_ref": round(ap_ref, 4), "ap_var": round(ap_var, 4), "delta_ap": round(ap_var - ap_ref, 4),
        "delta_ap_ci95_low": round(low, 4), "delta_ap_ci95_high": round(high, 4),
        "delta_ap_cruza_cero": bool(low <= 0 <= high),
        "auc_ref": round(roc_auc_score(y, p_ref), 4), "auc_var": round(roc_auc_score(y, p_var), 4),
    }


def metricas_por_semilla(y: np.ndarray, probas: dict) -> dict:
    """`probas`: {semilla: proba}. Devuelve media y desviacion del AUC-PR
    entre semillas (la cifra puntual sale del promedio de probabilidades)."""
    aps = [average_precision_score(y, p) for p in probas.values()]
    return {"ap_media_semillas": round(float(np.mean(aps)), 4), "ap_sd_semillas": round(float(np.std(aps, ddof=1)), 4)}


def promedio_semillas(probas: dict) -> np.ndarray:
    return np.mean(np.column_stack(list(probas.values())), axis=1)


def panel_ola(ola: int, columnas: list = None) -> pd.DataFrame:
    """Filas de una ola del panel consolidado, sin hogares divididos (misma
    regla 1 a 1 de `build_benchmark_train_test.py`)."""
    panel = pd.read_parquet(PANEL_PATH, columns=None if columnas is None else list(dict.fromkeys(["consecutivo", "ola"] + columnas)))
    panel = panel[panel["ola"] == ola]
    return panel[~panel["consecutivo"].duplicated(keep=False)].copy()


def guardar(df: pd.DataFrame, nombre: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta = OUT_DIR / nombre
    if nombre.endswith(".parquet"):
        df.to_parquet(ruta, index=False)
    else:
        df.to_csv(ruta, index=False)
    print(f"Guardado: {ruta} ({df.shape})", flush=True)
