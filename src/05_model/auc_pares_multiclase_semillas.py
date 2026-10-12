"""
auc_pares_multiclase_semillas.py
=====================================
AUC-ROC de los 6 pares de grupos de la matriz de transicion (nunca pobre,
entra, sale, siempre pobre) con las 5 semillas de la suite multiclase, para
las especificaciones del Modelo B con holdout temporal (B4 y B4geoDMSP):
entrenamiento en 2010->2013, prueba en 2013->2016.

Motivo: el perfil univariado se retira del
paper y el Tercer hallazgo de la Seccion 5.2 ("entra" se parece mas a
"sale" que a "nunca pobre") pasa a apoyarse solo en el modelo multiclase.
La tabla de AUC por par (`auc_pares_multiclase_predicciones.py`) usaba una
sola semilla (42) y no tenia intervalos; aqui se reentrena con las 5
semillas (`mcu.SEMILLAS`) y se reporta media, desviacion e IC95 (misma
convencion t de Student que `mcu._resumir_semillas`). El cambio al agregar
DMSP-OLS se calcula por semilla (B4geoDMSP - B4, mismo algoritmo y
semilla) y se resume igual.

NO se repite la busqueda de hiperparametros: se reconstruye el pipeline
ganador con `balanceo_elegido` e `hiperparametros` de
`registro_modelos_multiclase.csv` (mismo patron que
`modelo_multiclase_predicciones.py`, aqui con la semilla como parametro).

Definicion del AUC por par (igual que `auc_pares_multiclase_predicciones.py`):
para el par (a, b) se restringe a los hogares de a o b y se usa P(a) como
puntaje, con a como clase positiva.

INPUTS
    data/processed/benchmark_resultados/multiclase/registro_modelos_multiclase.csv
    data/processed/benchmark_train_test/modelo_{B4,B4geoDMSP}_*.parquet

OUTPUTS
    data/processed/benchmark_resultados/multiclase/auc_pares_multiclase_semillas_detalle.csv
        (una fila por algoritmo x especificacion x semilla)
    data/processed/benchmark_resultados/multiclase/auc_pares_multiclase_semillas.csv
        (media/std/IC95 por algoritmo x especificacion, y filas
        especificacion="delta_DMSP" con el cambio B4geoDMSP - B4)

Paralelismo (2026-09-30): la logistica (solver saga) y la red neuronal
ajustan en un solo nucleo, asi que sus 10 ajustes (2 especificaciones x 5
semillas) corren en paralelo con joblib, uno por nucleo y con 1 hilo
interno cada uno. Los algoritmos de arboles ya usan todos los nucleos
internamente (n_jobs=-1) y corren en secuencia. Los resultados no dependen
del orden: cada ajuste fija su propia semilla.

COMO CORRER
    cd src/05_model && python -u auc_pares_multiclase_semillas.py
"""

import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
import modelo_utils_multiclase as mcu
import modelo_multiclase_robusto_comparacion as mrc
from modelo_multiclase_descomponer_auc import ALGORITMOS_ARBOL_NATIVO, ALGORITMOS_PREPROCESADOR

# La red neuronal no esta en los mapas de `modelo_multiclase_descomponer_auc`
# (se agrego a la suite despues); usa preprocesador como RF y logistica.
ALGORITMOS_PREPROCESADOR = {**ALGORITMOS_PREPROCESADOR, "Red neuronal (MLP)": mrc.pipeline_nn}

# Algoritmos que ajustan en un solo nucleo: sus ajustes se reparten en paralelo.
ALGORITMOS_UN_NUCLEO = {"Logistica regularizada (elastic net, benchmark)", "Red neuronal (MLP)"}
N_JOBS_PARALELO = 10

ESPECIFICACIONES = ["B4", "B4geoDMSP"]
GRUPOS = list(mcu.CATEGORIAS_Y_GRUPO)
PARES = list(itertools.combinations(GRUPOS, 2))
RUTA_DETALLE = mcu.RESULTADOS_DIR / "auc_pares_multiclase_semillas_detalle.csv"
RUTA_RESUMEN = mcu.RESULTADOS_DIR / "auc_pares_multiclase_semillas.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def auc_pares(y_grupo: np.ndarray, proba: np.ndarray) -> dict:
    """proba en el orden de CATEGORIAS_Y_GRUPO; y_grupo como etiquetas."""
    fila = {}
    for a, b in PARES:
        mask = np.isin(y_grupo, [a, b])
        score = proba[mask, GRUPOS.index(a)]
        fila[f"auc_{a}_vs_{b}"] = roc_auc_score((y_grupo[mask] == a).astype(int), score)
    return fila


def ajustar_semilla(algoritmo: str, espec: str, balanceo: str, params: dict, semilla: int) -> dict:
    """Reentrena el pipeline ganador con una semilla y devuelve los 6 AUC
    por par sobre la prueba 2013->2016. Carga sus propios datos para poder
    correr en un proceso aparte."""
    train, test = mcu.cargar_datos_4clases(espec)
    if algoritmo in ALGORITMOS_ARBOL_NATIVO:
        pipeline_fn, fit_params_fn = ALGORITMOS_ARBOL_NATIVO[algoritmo]
        x_train, y_train, cat_cols = mcu.preparar_arboles_nativos_multiclase(train)
        x_test, y_test, _ = mcu.preparar_arboles_nativos_multiclase(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)
        pipe = pipeline_fn(balanceo, semilla=semilla)
        fit_params = fit_params_fn(balanceo, y_train) if fit_params_fn else {}
    else:
        pipeline_fn = ALGORITMOS_PREPROCESADOR[algoritmo]
        x_train, y_train = mcu.preparar_xy_crudo_multiclase(train)
        x_test, y_test = mcu.preparar_xy_crudo_multiclase(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)
        pipe = pipeline_fn(x_train, balanceo, semilla=semilla)
        fit_params = {}

    if params:
        pipe.set_params(**params)
    pipe.fit(x_train, y_train, **(fit_params or {}))
    clases = list(pipe.classes_)
    orden = [clases.index(i) for i in range(len(GRUPOS))]
    proba = pipe.predict_proba(x_test)[:, orden]
    y_grupo = np.array([mcu.CATEGORIAS_Y_GRUPO[k] for k in np.asarray(y_test)])
    return {"algoritmo": algoritmo, "especificacion": espec, "semilla": semilla,
            "n": len(y_grupo), **auc_pares(y_grupo, proba)}


def resumir(detalle: pd.DataFrame) -> pd.DataFrame:
    columnas = [c for c in detalle.columns if c.startswith("auc_")]
    n = detalle["semilla"].nunique()
    t_mult = float(scipy_stats.t.ppf(0.975, df=n - 1))
    filas = []
    for (algoritmo, espec), g in detalle.groupby(["algoritmo", "especificacion"], sort=False):
        fila = {"algoritmo": algoritmo, "especificacion": espec, "n_semillas": len(g)}
        for c in columnas:
            media, std = g[c].mean(), g[c].std(ddof=1)
            margen = t_mult * std / np.sqrt(len(g))
            fila.update({f"{c}_media": round(media, 4), f"{c}_std": round(std, 4),
                         f"{c}_ci95_low": round(media - margen, 4), f"{c}_ci95_high": round(media + margen, 4)})
        filas.append(fila)
    return pd.DataFrame(filas)


def main() -> None:
    registro = pd.read_csv(mcu.REGISTRO_CSV)
    registro = registro[registro["especificacion"].isin(ESPECIFICACIONES)]
    log(f"INICIO: {len(registro)} filas x {len(mcu.SEMILLAS)} semillas")

    tareas = []
    for _, fila in registro.iterrows():
        params = json.loads(fila["hiperparametros"]) if pd.notna(fila["hiperparametros"]) else {}
        if "modelo__hidden_layer_sizes" in params:  # JSON la guarda como lista
            params["modelo__hidden_layer_sizes"] = tuple(params["modelo__hidden_layer_sizes"])
        for semilla in mcu.SEMILLAS:
            tareas.append((fila["algoritmo"], fila["especificacion"], fila["balanceo_elegido"], params, semilla))

    paralelas = [t for t in tareas if t[0] in ALGORITMOS_UN_NUCLEO]
    secuenciales = [t for t in tareas if t[0] not in ALGORITMOS_UN_NUCLEO]

    log(f"=== {len(paralelas)} ajustes de un solo nucleo en paralelo (n_jobs={N_JOBS_PARALELO}) -- INICIO ===")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        filas = Parallel(n_jobs=N_JOBS_PARALELO, verbose=10)(delayed(ajustar_semilla)(*t) for t in paralelas)
    log("=== ajustes en paralelo -- FIN ===")

    for t in secuenciales:
        log(f"--- {t[0]} -- {t[1]} -- semilla {t[4]}")
        filas.append(ajustar_semilla(*t))

    detalle = pd.DataFrame(filas)

    # Cambio al agregar DMSP-OLS, pareado por algoritmo y semilla.
    columnas = [c for c in detalle.columns if c.startswith("auc_")]
    base = detalle[detalle["especificacion"] == "B4"].set_index(["algoritmo", "semilla"])[columnas]
    geo = detalle[detalle["especificacion"] == "B4geoDMSP"].set_index(["algoritmo", "semilla"])[columnas]
    delta = (geo - base).dropna().reset_index()
    delta["especificacion"] = "delta_DMSP"
    delta["n"] = np.nan
    detalle = pd.concat([detalle, delta[detalle.columns]], ignore_index=True)

    detalle.to_csv(RUTA_DETALLE, index=False)
    resumen = resumir(detalle)
    resumen.to_csv(RUTA_RESUMEN, index=False)

    medias = [c for c in resumen.columns if c.endswith("_media")]
    print(resumen[["algoritmo", "especificacion"] + medias].round(3).to_string(index=False))
    log(f"FIN. Guardado: {RUTA_RESUMEN}")


if __name__ == "__main__":
    main()
