"""
modelo_multiclase_descomponer_auc.py
=======================================

NO LANZAR TODAVIA -- preparado a (2026-09-16) para
correrlo DESPUES de que termine `modelo_multiclase_robusto_comparacion.py`
(evita competir por CPU con esa corrida, que ya esta en curso).

Motivacion
----------
`modelo_utils_multiclase.calcular_metricas_multiclase` solo reporta dos
metricas de separabilidad para "entra": `auc_entra_vs_resto` (entra vs.
{nunca, sale, siempre} combinados) y `auc_entra_vs_sale`. Al revisar los
primeros resultados (XGBoost) se encontro que `auc_entra_vs_sale` cae a
~0.49 (azar) en las especificaciones B4/B4geoDMSP -- pero
`auc_entra_vs_resto` da ~0.59 (por encima del azar) en esas MISMAS
especificaciones. Eso significa que la falta de señal es especifica de
la frontera entra/sale, NO general: el modelo si distingue algo a
"entra" del resto, solo que ese "resto" mezcla a "sale" (indistinguible)
con "nunca pobre" y "siempre pobre" (presumiblemente mas faciles). El
agregado no permite ver eso, de ahi el desglose.

Que hace este script
---------------------
Para cada fila YA REGISTRADA en `registro_modelos_multiclase.csv`
(algoritmo x especificacion, con `balanceo_elegido` y `hiperparametros`
ya elegidos por la busqueda cara de
`modelo_multiclase_robusto_comparacion.py` -- NO se repite esa busqueda
aqui, solo se reentrena con la configuracion YA ganadora):
  1. Reconstruye el pipeline (mismas funciones `pipeline_*` de
     `modelo_multiclase_robusto_comparacion.py`) con los hiperparametros
     guardados.
  2. Reentrena con las mismas 5 semillas (`modelo_utils_multiclase.SEMILLAS`)
     -- holdout temporal para las especificaciones principales,
     out-of-fold (CV) para las especificaciones *geo3.
  3. Calcula 3 AUC pairwise por semilla: entra-vs-nunca, entra-vs-sale,
     entra-vs-siempre (generalizacion de la logica ya usada para
     `auc_entra_vs_sale` en `calcular_metricas_multiclase`, aqui para
     los 3 pares en vez de 1 solo).
  4. Guarda media/std/IC95 (misma convencion que el resto de la suite)
     en un CSV NUEVO, separado del registro principal.

Por que no se repite la busqueda de hiperparametros: ya se hizo una vez
(costosa, CV_FOLDS=10/N_ITER=30) y no hay razon para pagarla de nuevo
solo para descomponer una metrica adicional sobre el MISMO modelo ya
elegido -- este script solo reentrena (barato: un fit por semilla, sin
RandomizedSearchCV) para obtener las probabilidades necesarias.

INPUTS
    data/processed/benchmark_resultados/multiclase/registro_modelos_multiclase.csv
    data/processed/benchmark_train_test/modelo_{espec}_*.parquet

OUTPUTS
    data/processed/benchmark_resultados/multiclase/descomposicion_auc_entra.csv

COMO CORRER (cuando termine modelo_multiclase_robusto_comparacion.py)
-----------------------------------------------------------------------
    cd src/05_model && python -u modelo_multiclase_descomponer_auc.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
import modelo_utils_multiclase as mcu
import modelo_multiclase_robusto_comparacion as mrc

OUTPUT_CSV = mcu.RESULTADOS_DIR / "descomposicion_auc_entra.csv"

# (nombre_par, idx_a, idx_b) -- idx_a siempre "entra" (1), idx_b el otro
# grupo. Generaliza la logica ya usada para auc_entra_vs_sale en
# modelo_utils_multiclase.calcular_metricas_multiclase.
PARES = [
    ("auc_entra_vs_nunca", mcu.GRUPO_A_ENTERO["entra"], mcu.GRUPO_A_ENTERO["nunca_pobre"]),
    ("auc_entra_vs_sale", mcu.GRUPO_A_ENTERO["entra"], mcu.GRUPO_A_ENTERO["sale"]),
    ("auc_entra_vs_siempre", mcu.GRUPO_A_ENTERO["entra"], mcu.GRUPO_A_ENTERO["siempre_pobre"]),
]

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def auc_pares(y_true: np.ndarray, proba: np.ndarray, clases_estimador: list) -> dict:
    """`proba`/`clases_estimador`: igual convencion que
    calcular_metricas_multiclase -- se reindexa a orden GRUPO_A_ENTERO."""
    orden = [clases_estimador.index(i) for i in range(len(mcu.CATEGORIAS_Y_GRUPO))]
    proba = proba[:, orden]

    resultado = {}
    for nombre, idx_a, idx_b in PARES:
        mask = np.isin(y_true, [idx_a, idx_b])
        if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
            resultado[nombre] = float("nan")
            continue
        y_par = (y_true[mask] == idx_a).astype(int)
        score_par = proba[mask, idx_a]
        resultado[nombre] = roc_auc_score(y_par, score_par)
    return resultado


# Mapa algoritmo (como aparece en el registro) -> como preparar X/y y
# como construir el pipeline con hiperparametros ya fijos.
ALGORITMOS_ARBOL_NATIVO = {
    "XGBoost": (mrc.pipeline_xgb, mrc.fit_params_xgb),
    "LightGBM": (mrc.pipeline_lgb, None),
    "HistGradientBoosting (sklearn)": (mrc.pipeline_hgb, None),
}
ALGORITMOS_PREPROCESADOR = {
    "Random Forest": mrc.pipeline_rf,
    "Logistica regularizada (elastic net, benchmark)": mrc.pipeline_log,
}


def _procesar_fila_principal(algoritmo: str, espec: str, balanceo: str, params: dict) -> dict:
    train, test = mcu.cargar_datos_4clases(espec)

    if algoritmo in ALGORITMOS_ARBOL_NATIVO:
        pipeline_fn, fit_params_fn = ALGORITMOS_ARBOL_NATIVO[algoritmo]
        x_train, y_train, cat_cols = mcu.preparar_arboles_nativos_multiclase(train)
        x_test, y_test, _ = mcu.preparar_arboles_nativos_multiclase(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)
        construir = lambda s: pipeline_fn(balanceo, semilla=s)
        fit_params = fit_params_fn(balanceo, y_train) if fit_params_fn else {}
    else:
        pipeline_fn = ALGORITMOS_PREPROCESADOR[algoritmo]
        x_train, y_train = mcu.preparar_xy_crudo_multiclase(train)
        x_test, y_test = mcu.preparar_xy_crudo_multiclase(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)
        construir = lambda s: pipeline_fn(x_train, balanceo, semilla=s)
        fit_params = {}

    filas = []
    for semilla in mcu.SEMILLAS:
        pipe = construir(semilla)
        if params:
            pipe.set_params(**params)
        pipe.fit(x_train, y_train, **fit_params)
        proba = pipe.predict_proba(x_test)
        filas.append({"semilla": semilla, **auc_pares(np.asarray(y_test), proba, list(pipe.classes_))})
    return pd.DataFrame(filas)


def _procesar_fila_cv(algoritmo: str, espec: str, balanceo: str, params: dict) -> pd.DataFrame:
    datos = mcu.cargar_datos_cv_4clases(espec)

    if algoritmo in ALGORITMOS_ARBOL_NATIVO:
        pipeline_fn, fit_params_fn = ALGORITMOS_ARBOL_NATIVO[algoritmo]
        x, y, _ = mcu.preparar_arboles_nativos_multiclase(datos)
        construir = lambda s: pipeline_fn(balanceo, semilla=s)
        fit_params_base = fit_params_fn(balanceo, y) if fit_params_fn else None
    else:
        pipeline_fn = ALGORITMOS_PREPROCESADOR[algoritmo]
        x, y = mcu.preparar_xy_crudo_multiclase(datos)
        construir = lambda s: pipeline_fn(x, balanceo, semilla=s)
        fit_params_base = None

    filas = []
    for semilla in mcu.SEMILLAS:
        pipe = construir(semilla)
        if params:
            pipe.set_params(**params)
        cv = StratifiedKFold(n_splits=mrc.CV_FOLDS, shuffle=True, random_state=semilla)
        proba_oof = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba", n_jobs=-1, params=fit_params_base)

        pipe_ref = construir(semilla)
        if params:
            pipe_ref.set_params(**params)
        pipe_ref.fit(x, y, **(fit_params_base or {}))
        filas.append({"semilla": semilla, **auc_pares(np.asarray(y), proba_oof, list(pipe_ref.classes_))})
    return pd.DataFrame(filas)


def main() -> None:
    if not mcu.REGISTRO_CSV.exists():
        print(f"ERROR: no existe {mcu.REGISTRO_CSV} -- correr primero modelo_multiclase_robusto_comparacion.py", file=sys.stderr)
        sys.exit(1)

    registro = pd.read_csv(mcu.REGISTRO_CSV)
    log(f"INICIO descomposicion de AUC entra-vs-{{nunca,sale,siempre}} sobre {len(registro)} filas ya registradas")

    resultados = []
    for _, fila in registro.iterrows():
        algoritmo, espec = fila["algoritmo"], fila["especificacion"]
        balanceo = fila["balanceo_elegido"]
        params = json.loads(fila["hiperparametros"]) if pd.notna(fila["hiperparametros"]) else {}

        log(f"=== {algoritmo} -- {espec} -- INICIO ===")
        if espec in mcu.ESPECIFICACIONES_4CLASES_CV:
            detalle = _procesar_fila_cv(algoritmo, espec, balanceo, params)
        else:
            detalle = _procesar_fila_principal(algoritmo, espec, balanceo, params)

        n = len(detalle)
        t_mult = float(scipy_stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
        fila_resumen = {"algoritmo": algoritmo, "especificacion": espec, "balanceo": balanceo}
        for nombre, _, _ in PARES:
            vals = detalle[nombre].dropna()
            media = float(vals.mean()) if len(vals) else float("nan")
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            margen = t_mult * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            fila_resumen[f"{nombre}_media"] = round(media, 4)
            fila_resumen[f"{nombre}_std"] = round(std, 4)
            fila_resumen[f"{nombre}_ci95_low"] = round(media - margen, 4)
            fila_resumen[f"{nombre}_ci95_high"] = round(media + margen, 4)
        resultados.append(fila_resumen)
        log(f"  {fila_resumen}")
        log(f"=== {algoritmo} -- {espec} -- FIN ===")

    salida = pd.DataFrame(resultados)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(OUTPUT_CSV, index=False)
    log(f"FIN. Guardado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
