"""
modelo_multiclase_robusto_comparacion.py
===========================================

Suite de comparacion de 6 algoritmos (XGBoost, LightGBM,
HistGradientBoosting, Random Forest, Red neuronal (MLPClassifier),
Logistica regularizada -- en ese orden de ejecucion, Logistica al final
por ser la mas lenta/impredecible) sobre el benchmark de 4 CLASES
(nunca_pobre/entra/sale/siempre_pobre,
`build_benchmark_train_test_4clases.py`), en "version robusta"
.

NO modifica ni sobrescribe nada del benchmark binario (`modelo_utils.py`,
`registro_modelos*.csv`, los scripts `modelo_*.py` originales) -- usa
`modelo_utils_multiclase.py` (nuevo) y escribe en un registro separado
(`registro_modelos_multiclase.csv`, bajo
`data/processed/benchmark_resultados/multiclase/`).

Alcance:
  - Pista principal (holdout temporal 2010->2013/2013->2016): A4, B4,
    A4geoDMSP, B4geoDMSP -- los 6 algoritmos.
  - Pista exploratoria (CV dentro de 2010->2013, sin holdout, las 3
    fuentes geoespaciales completas -- DMSP-OLS+ALOS PALSAR+Landsat 5 TM):
    A4geo3, B4geo3 -- los 6 algoritmos.

Balanceo de clases: identico en estructura al binario (3 estrategias,
elegidas por AUC-ROC-OVR en CV) -- ver docstring de
`modelo_utils_multiclase.py` para el detalle de como XGBoost maneja
"balanced" (sample_weight en vez de class_weight, unico caso especial).
EXCEPCION: la red neuronal (MLPClassifier) NO soporta class_weight ni
sample_weight en absoluto -- para ella solo se compite "ninguno" vs.
"oversampling" (BALANCEOS_NN, ver pipeline_nn), para no gastar una busqueda de hiperparametros
completa en una estrategia "balanced" que seria identica a "ninguno".

Salida por algoritmo (bajo
`data/processed/benchmark_resultados/multiclase/{algoritmo}/`):
  - `metricas_multiples_semillas_modelo_{espec}.csv` (accuracy,
    balanced_accuracy, f1_macro/weighted, precision/recall/f1 de "entra",
    auc_ovr_macro, auc_entra_vs_resto, auc_entra_vs_sale,
    precision_top10_entra -- por semilla)
  - `importancia_variables_modelo_{espec}.csv` (arboles y NN -- para NN,
    permutation_importance sobre el pipeline completo, ver
    `_exportar_interpretabilidad`) o `coeficientes_por_clase_modelo_{espec}.csv`
    (logistica -- un coeficiente por clase y por variable, mas la columna
    `entra_menos_sale`, el contraste que responde directamente la
    pregunta de la Seccion 5.1 sobre la frontera entra/sale)

Ademas, bajo `.../multiclase/predicciones/{algoritmo}_{espec}.parquet`:
probabilidades por hogar en las 4 clases (prob_nunca_pobre, prob_entra,
prob_sale, prob_siempre_pobre), guardadas DENTRO de esta misma corrida
 -- ver `_guardar_predicciones`
para el detalle de que estimador y que muestra se usa en cada pista.

COMO CORRER
-----------
    cd src/05_model && python -u modelo_multiclase_robusto_comparacion.py

"""

import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

from sklearn.neural_network import MLPClassifier

import modelo_utils as mu
import modelo_utils_multiclase as mcu

CV_FOLDS = 10
N_ITER_BUSQUEDA = 30

# Pista exploratoria A4geo3/B4geo3 (CV dentro de 2010->2013): el paper no la
# usa (solo A4, B4 y B4geoDMSP), asi que queda apagada por defecto y el script
# no necesita la base de construir_pipeline_geo3_cv_4clases.py. Poner True
# para reproducirla.
INCLUIR_PISTA_GEO3 = False
ESPECIFICACIONES_CV = mcu.ESPECIFICACIONES_4CLASES_CV if INCLUIR_PISTA_GEO3 else []

REGISTRO_CSV = mcu.REGISTRO_CSV
REGISTRO_XLSX = mcu.REGISTRO_XLSX

# Probabilidades por hogar (4 clases)
# (2026-09-18): guardarlas DENTRO de esta misma corrida, no como paso
# aparte despues. Se guarda 1 parquet por algoritmo/especificacion, con
# el `resultado["estimador"]` YA AJUSTADO que devuelve
# `comparar_balanceo_y_tunear_multiclase` (RandomizedSearchCV con
# refit=True lo reentrena sobre el 100% de x_train con los mejores
# hiperparametros) -- CERO ajustes adicionales, es pura inferencia sobre
# ese estimador ya en memoria. Para las especificaciones *geo3 (sin
# holdout) las probabilidades son IN-SAMPLE (el estimador se ajusto sobre
# la misma x que aqui se le pide predecir) -- no out-of-fold como las
# metricas de evaluar_cv_semillas_multiclase -- interpretar con la misma
# cautela ya aplicada a la importancia por permutacion in-sample de
# HistGB en el benchmark binario CV-only.
PREDICCIONES_DIR = mcu.RESULTADOS_DIR / "predicciones"


def _guardar_predicciones(algoritmo: str, espec: str, pipe, x_eval, y_eval, consecutivo_eval) -> None:
    proba = pipe.predict_proba(x_eval)
    orden = [list(pipe.classes_).index(i) for i in range(len(mcu.CATEGORIAS_Y_GRUPO))]
    proba = proba[:, orden]
    salida = pd.DataFrame({
        "consecutivo": pd.Series(consecutivo_eval).values,
        "algoritmo": algoritmo,
        "especificacion": espec,
        "Y_grupo": [mcu.CATEGORIAS_Y_GRUPO[k] for k in np.asarray(y_eval)],
    })
    for k, nombre in enumerate(mcu.CATEGORIAS_Y_GRUPO):
        salida[f"prob_{nombre}"] = proba[:, k]

    PREDICCIONES_DIR.mkdir(parents=True, exist_ok=True)
    slug = algoritmo.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    salida.to_parquet(PREDICCIONES_DIR / f"{slug}_{espec}.parquet", index=False)

OBSERVACIONES_PRINCIPAL = (
    "Benchmark de 4 clases (nunca_pobre/entra/sale/siempre_pobre), "
    "poblacion completa (NO filtrada a no-pobres en la ola base -- ver "
    "build_benchmark_train_test_4clases.py). Holdout temporal "
    "2010->2013 / 2013->2016, CV_FOLDS=10, N_ITER_BUSQUEDA=30, "
    "SEMILLAS=[42,1,2,3,4]. Como interpretar precision/recall de "
    "'entra': si bajan respecto al "
    "benchmark binario, NO es 'el modelo empeoro' -- es que 'sale' "
    "(el grupo mas parecido a 'entra', Seccion 5.1) ahora esta en el "
    "'resto', haciendo la tarea mas dificil y mas fiel a la definicion "
    "de vulnerabilidad. NO comparable cifra a cifra contra "
    "registro_modelos.csv/registro_modelos_fbeta2_cv10.csv (target y "
    "poblacion distintos)."
)
OBSERVACIONES_CV = (
    "PISTA EXPLORATORIA (igual que Ageo3/Bgeo3 en el binario): ELCA + "
    "DMSP-OLS + ALOS PALSAR + Landsat 5 TM, restringido a la transicion "
    "2010->2013, benchmark de 4 clases. Sin holdout temporal -- metricas "
    "sobre probabilidades OUT-OF-FOLD dentro de la misma muestra. NO "
    "comparable cifra a cifra contra A4/B4/A4geoDMSP/B4geoDMSP (holdout) "
    "ni contra Ageo3/Bgeo3 del binario (target distinto)."
)

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


# ---------------------------------------------------------------------
# XGBoost (arboles nativos -- NaN/categoricas sin imputar)
# ---------------------------------------------------------------------

PARAM_DIST_XGB = {
    "modelo__n_estimators": randint(100, 500),
    "modelo__max_depth": randint(3, 8),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__reg_lambda": loguniform(0.1, 10),
    "modelo__subsample": uniform(0.6, 0.4),
    "modelo__colsample_bytree": uniform(0.6, 0.4),
}


def pipeline_xgb(balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    modelo = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True,
        objective="multi:softprob", num_class=len(mcu.CATEGORIAS_Y_GRUPO),
        random_state=semilla, n_jobs=-1, eval_metric="mlogloss",
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


def fit_params_xgb(balanceo: str, y_train) -> dict:
    if balanceo == "balanced":
        return {"modelo__sample_weight": mcu.sample_weight_balanced(y_train)}
    return {}


# ---------------------------------------------------------------------
# LightGBM (arboles nativos)
# ---------------------------------------------------------------------

PARAM_DIST_LGB = {
    "modelo__n_estimators": randint(100, 500),
    "modelo__max_depth": randint(3, 8),
    "modelo__num_leaves": randint(15, 63),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__reg_lambda": loguniform(0.1, 10),
    "modelo__min_child_samples": randint(5, 50),
}


def pipeline_lgb(balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(mcu.CATEGORIAS_Y_GRUPO),
        class_weight=class_weight, random_state=semilla, n_jobs=-1, verbosity=-1,
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# HistGradientBoosting (arboles nativos) -- multiclase automatico
# ---------------------------------------------------------------------

PARAM_DIST_HGB = {
    "modelo__max_iter": randint(100, 400),
    "modelo__max_depth": randint(3, 8),
    "modelo__max_leaf_nodes": randint(15, 63),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__l2_regularization": loguniform(0.01, 10),
}


def pipeline_hgb(balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = HistGradientBoostingClassifier(
        categorical_features="from_dtype", class_weight=class_weight, random_state=semilla,
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# Random Forest (preprocesador 0+indicador/one-hot, sin escalar)
# ---------------------------------------------------------------------

PARAM_DIST_RF = {
    "modelo__n_estimators": randint(200, 600),
    "modelo__max_depth": randint(3, 15),
    "modelo__min_samples_leaf": randint(2, 30),
    "modelo__max_features": uniform(0.2, 0.6),
}


def pipeline_rf(x_train: pd.DataFrame, balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    preprocesador = mu.construir_preprocesador(x_train, escalar=False)
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = RandomForestClassifier(class_weight=class_weight, random_state=semilla, n_jobs=-1)
    pasos = [("prep", preprocesador)]
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# Logistica regularizada (preprocesador + escalado; multinomial nativo
# de sklearn para >2 clases)
# ---------------------------------------------------------------------

PARAM_DIST_LOG = {
    "modelo__C": loguniform(1e-3, 1e2),
    "modelo__l1_ratio": uniform(0, 1),
}


def pipeline_log(x_train: pd.DataFrame, balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    preprocesador = mu.construir_preprocesador(x_train, escalar=True)
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = LogisticRegression(
        penalty="elasticnet", solver="saga", class_weight=class_weight,
        max_iter=5000, random_state=semilla,
    )
    pasos = [("prep", preprocesador)]
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# Red neuronal (MLPClassifier) -- preprocesador + escalado, igual que
# Logistica. MLPClassifier.fit() NO acepta sample_weight (a diferencia de
# casi todos los demas estimadores de sklearn) ni tiene class_weight en el
# constructor -- no existe una via nativa para "balanced" en este
# algoritmo. BALANCEOS_NN excluye "balanced": solo compite "ninguno" vs.
# "oversampling" (para no
# gastar una busqueda de hiperparametros completa en una estrategia que
# seria identica a "ninguno").
# ---------------------------------------------------------------------

BALANCEOS_NN = ["ninguno", "oversampling"]

PARAM_DIST_NN = {
    "modelo__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32), (128, 64)],
    "modelo__alpha": loguniform(1e-5, 1e-1),
    "modelo__learning_rate_init": loguniform(1e-4, 1e-2),
}


def pipeline_nn(x_train: pd.DataFrame, balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    preprocesador = mu.construir_preprocesador(x_train, escalar=True)
    modelo = MLPClassifier(
        early_stopping=True, max_iter=2000, random_state=semilla,
    )
    pasos = [("prep", preprocesador)]
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------

def _log_resumen(r: dict) -> None:
    log(
        f"  accuracy: {r['accuracy']['media']:.3f}  f1_macro: {r['f1_macro']['media']:.3f}  "
        f"recall_entra: {r['recall_entra']['media']:.3f}  precision_entra: {r['precision_entra']['media']:.3f}  "
        f"auc_ovr_macro: {r['auc_ovr_macro']['media']:.3f}  auc_entra_vs_sale: {r['auc_entra_vs_sale']['media']:.3f}"
    )


# Checkpoint/reanudacion (, 2026-09-18, tras
# un crash con exit 137/SIGKILL durante el smoke test): cada combinacion
# algoritmo/especificacion escribe su fila en REGISTRO_CSV como ULTIMO
# paso (via registrar_resultado_multiclase, al final de cada iteracion de
# cada funcion correr_*), asi que esa fila es un marcador fiable de
# "esta combinacion ya termino, con predicciones e importancia ya
# guardadas". Si el proceso muere a mitad de camino (OOM, hibernacion,
# etc.), reiniciar `python -u modelo_multiclase_robusto_comparacion.py`
# SALTA automaticamente las combinaciones ya presentes en el registro en
# vez de repetirlas -- no se pierde el trabajo ya hecho, ni se duplica.
def _ya_completado(algoritmo: str, espec: str) -> bool:
    if not REGISTRO_CSV.exists():
        return False
    registro = pd.read_csv(REGISTRO_CSV, usecols=["algoritmo", "especificacion"])
    return bool(((registro["algoritmo"] == algoritmo) & (registro["especificacion"] == espec)).any())


def correr_arbol_principal(nombre_algoritmo: str, out_subdir: str, pipeline_fn, param_dist: dict, fit_params_fn=None) -> None:
    out_dir = mcu.RESULTADOS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL:
        if _ya_completado(nombre_algoritmo, espec):
            log(f"=== {nombre_algoritmo} -- Modelo {espec} -- YA COMPLETADO (checkpoint), se salta ===")
            continue
        train, test = mcu.cargar_datos_4clases(espec)
        x_train, y_train, cat_cols = mcu.preparar_arboles_nativos_multiclase(train)
        x_test, y_test, _ = mcu.preparar_arboles_nativos_multiclase(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        log(f"=== {nombre_algoritmo} -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        ffn = (lambda b: fit_params_fn(b, y_train)) if fit_params_fn else None
        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=lambda b: pipeline_fn(b),
            param_distributions_fn=lambda b: param_dist,
            x_train=x_train, y_train=y_train, fit_params_fn=ffn,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        modelo_final = pipe.named_steps["modelo"]
        if hasattr(modelo_final, "feature_importances_"):
            importancias = pd.DataFrame({
                "variable": x_train.columns, "importancia": modelo_final.feature_importances_,
            }).sort_values("importancia", ascending=False)
            importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)

        _guardar_predicciones(nombre_algoritmo, espec, pipe, x_test, y_test, test["consecutivo"])

        ffn_semilla = (lambda s: fit_params_fn(resultado["balanceo_elegido"], y_train)) if fit_params_fn else None
        multi = mcu.evaluar_multiples_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_fn(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            fit_params_fn=ffn_semilla,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo=nombre_algoritmo, especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_PRINCIPAL,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== {nombre_algoritmo} -- Modelo {espec} -- FIN ===")


def correr_arbol_cv(nombre_algoritmo: str, out_subdir: str, pipeline_fn, param_dist: dict, fit_params_fn=None) -> None:
    out_dir = mcu.RESULTADOS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES_CV:
        if _ya_completado(nombre_algoritmo, espec):
            log(f"=== {nombre_algoritmo} -- Modelo {espec} -- YA COMPLETADO (checkpoint), se salta ===")
            continue
        datos = mcu.cargar_datos_cv_4clases(espec)
        x, y, cat_cols = mcu.preparar_arboles_nativos_multiclase(datos)

        log(f"=== {nombre_algoritmo} -- Modelo {espec} (CV, sin holdout) -- INICIO (n={x.shape[0]}) ===")

        ffn = (lambda b: fit_params_fn(b, y)) if fit_params_fn else None
        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=lambda b: pipeline_fn(b),
            param_distributions_fn=lambda b: param_dist,
            x_train=x, y_train=y, fit_params_fn=ffn,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        modelo_final = pipe.named_steps["modelo"]
        if hasattr(modelo_final, "feature_importances_"):
            importancias = pd.DataFrame({
                "variable": x.columns, "importancia": modelo_final.feature_importances_,
            }).sort_values("importancia", ascending=False)
            importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)

        _guardar_predicciones(nombre_algoritmo, espec, pipe, x, y, datos["consecutivo"])

        ffn_semilla = (lambda s: fit_params_fn(resultado["balanceo_elegido"], y)) if fit_params_fn else None
        multi = mcu.evaluar_cv_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_fn(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y, fit_params_fn=ffn_semilla, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo=nombre_algoritmo, especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_CV,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== {nombre_algoritmo} -- Modelo {espec} (CV) -- FIN ===")


def correr_hgb() -> None:
    """HistGB no tiene feature_importances_ nativo -- permutation importance, igual que el binario."""
    out_dir = mcu.RESULTADOS_DIR / "histgradientboosting"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL:
        if _ya_completado("HistGradientBoosting (sklearn)", espec):
            log(f"=== HistGradientBoosting (sklearn) -- Modelo {espec} -- YA COMPLETADO (checkpoint), se salta ===")
            continue
        train, test = mcu.cargar_datos_4clases(espec)
        x_train, y_train, cat_cols = mcu.preparar_arboles_nativos_multiclase(train)
        x_test, y_test, _ = mcu.preparar_arboles_nativos_multiclase(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        log(f"=== HistGradientBoosting -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=pipeline_hgb, param_distributions_fn=lambda b: PARAM_DIST_HGB,
            x_train=x_train, y_train=y_train, cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        imp = permutation_importance(pipe.named_steps["modelo"], x_test, y_test, scoring="roc_auc_ovr", n_repeats=10, random_state=mcu.RANDOM_STATE, n_jobs=-1)
        importancias = pd.DataFrame({
            "variable": x_test.columns, "importancia_media": imp.importances_mean, "importancia_std": imp.importances_std,
        }).sort_values("importancia_media", ascending=False)
        importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)

        _guardar_predicciones("HistGradientBoosting (sklearn)", espec, pipe, x_test, y_test, test["consecutivo"])

        multi = mcu.evaluar_multiples_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_hgb(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo="HistGradientBoosting (sklearn)", especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_PRINCIPAL,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== HistGradientBoosting -- Modelo {espec} -- FIN ===")

    for espec in ESPECIFICACIONES_CV:
        if _ya_completado("HistGradientBoosting (sklearn)", espec):
            log(f"=== HistGradientBoosting (sklearn) -- Modelo {espec} -- YA COMPLETADO (checkpoint), se salta ===")
            continue
        datos = mcu.cargar_datos_cv_4clases(espec)
        x, y, cat_cols = mcu.preparar_arboles_nativos_multiclase(datos)

        log(f"=== HistGradientBoosting -- Modelo {espec} (CV, sin holdout) -- INICIO (n={x.shape[0]}) ===")

        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=pipeline_hgb, param_distributions_fn=lambda b: PARAM_DIST_HGB,
            x_train=x, y_train=y, cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        _guardar_predicciones("HistGradientBoosting (sklearn)", espec, resultado["estimador"], x, y, datos["consecutivo"])

        multi = mcu.evaluar_cv_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_hgb(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"], x=x, y=y, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo="HistGradientBoosting (sklearn)", especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_CV,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== HistGradientBoosting -- Modelo {espec} (CV) -- FIN ===")


def _exportar_interpretabilidad(pipe, x_test_o_x, y_test_o_y, familia: str, out_dir, espec: str) -> None:
    """familia en {"logistica", "arbol", "nn"} -- Logistica exporta
    coeficientes, arboles (Random Forest) exportan feature_importances_,
    ambos indexados por las columnas YA transformadas por "prep"
    (`get_feature_names_out()`). NN no tiene ninguno de los dos atributos
    (MLPClassifier) -- se usa permutation_importance sobre el PIPELINE
    COMPLETO (incluyendo "prep") aplicado a `x_test_o_x` EN CRUDO (sin
    transformar) -- por eso ese caso indexa por las columnas ORIGINALES de
    `x_test_o_x`, no por `get_feature_names_out()` (que describe columnas
    post-one-hot que no existen en el x crudo que se permuta)."""
    modelo_final = pipe.named_steps["modelo"]
    if familia == "logistica":
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pd.DataFrame(modelo_final.coef_.T, columns=[f"coef_{mcu.CATEGORIAS_Y_GRUPO[k]}" for k in modelo_final.classes_])
        coefs.insert(0, "variable", nombres_features)
        coefs["entra_menos_sale"] = coefs["coef_entra"] - coefs["coef_sale"]
        coefs = coefs.sort_values("entra_menos_sale", key=np.abs, ascending=False)
        coefs.to_csv(out_dir / f"coeficientes_por_clase_modelo_{espec}.csv", index=False)
    elif familia == "arbol":
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        importancias = pd.DataFrame({
            "variable": nombres_features, "importancia": modelo_final.feature_importances_,
        }).sort_values("importancia", ascending=False)
        importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)
    else:
        imp = permutation_importance(pipe, x_test_o_x, y_test_o_y, scoring="roc_auc_ovr", n_repeats=10, random_state=mcu.RANDOM_STATE, n_jobs=-1)
        importancias = pd.DataFrame({
            "variable": x_test_o_x.columns, "importancia_media": imp.importances_mean, "importancia_std": imp.importances_std,
        }).sort_values("importancia_media", ascending=False)
        importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)


def _correr_lineal(nombre_algoritmo: str, out_subdir: str, pipeline_fn_train, familia: str = "arbol", balanceos: list = None) -> None:
    """Comun a Random Forest, Logistica y NN (preprocesador dependiente de
    x_train). `familia` controla el export de interpretabilidad (ver
    `_exportar_interpretabilidad`). `balanceos` acota la lista de
    estrategias a probar (usado por NN, que no soporta "balanced" -- ver
    pipeline_nn)."""
    out_dir = mcu.RESULTADOS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    balanceos = balanceos or mcu.BALANCEOS
    if familia == "logistica":
        param_dist = PARAM_DIST_LOG
    elif familia == "nn":
        param_dist = PARAM_DIST_NN
    else:
        param_dist = PARAM_DIST_RF
    estrategia_imputacion = "0 + indicador (numericas), 'Sin dato' + one-hot (categoricas)" + (", estandarizacion" if familia in ("logistica", "nn") else "")

    for espec in mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL:
        if _ya_completado(nombre_algoritmo, espec):
            log(f"=== {nombre_algoritmo} -- Modelo {espec} -- YA COMPLETADO (checkpoint), se salta ===")
            continue
        train, test = mcu.cargar_datos_4clases(espec)
        x_train, y_train = mcu.preparar_xy_crudo_multiclase(train)
        x_test, y_test = mcu.preparar_xy_crudo_multiclase(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)

        log(f"=== {nombre_algoritmo} -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=lambda b: pipeline_fn_train(x_train, b),
            param_distributions_fn=lambda b: param_dist,
            x_train=x_train, y_train=y_train, cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
            balanceos=balanceos,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        _exportar_interpretabilidad(pipe, x_test, y_test, familia, out_dir, espec)
        _guardar_predicciones(nombre_algoritmo, espec, pipe, x_test, y_test, test["consecutivo"])

        multi = mcu.evaluar_multiples_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_fn_train(x_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo=nombre_algoritmo, especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test, multi_resultado=multi,
            estrategia_imputacion=estrategia_imputacion,
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_PRINCIPAL,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== {nombre_algoritmo} -- Modelo {espec} -- FIN ===")

    for espec in ESPECIFICACIONES_CV:
        if _ya_completado(nombre_algoritmo, espec):
            log(f"=== {nombre_algoritmo} -- Modelo {espec} -- YA COMPLETADO (checkpoint), se salta ===")
            continue
        datos = mcu.cargar_datos_cv_4clases(espec)
        x, y = mcu.preparar_xy_crudo_multiclase(datos)

        log(f"=== {nombre_algoritmo} -- Modelo {espec} (CV, sin holdout) -- INICIO (n={x.shape[0]}) ===")

        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=lambda b: pipeline_fn_train(x, b),
            param_distributions_fn=lambda b: param_dist,
            x_train=x, y_train=y, cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
            balanceos=balanceos,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        _exportar_interpretabilidad(pipe, x, y, familia, out_dir, espec)
        _guardar_predicciones(nombre_algoritmo, espec, pipe, x, y, datos["consecutivo"])

        multi = mcu.evaluar_cv_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_fn_train(x, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"], x=x, y=y, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo=nombre_algoritmo, especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion=estrategia_imputacion,
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_CV,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== {nombre_algoritmo} -- Modelo {espec} (CV) -- FIN ===")


def main() -> None:
    mcu.RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"INICIO suite multiclase robusta: CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}. "
        f"6 algoritmos x ({len(mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL)} principal + {len(ESPECIFICACIONES_CV)} CV) especificaciones. "
        f"Orden: XGBoost, LightGBM, HistGradientBoosting, Random Forest, Red neuronal, Logistica regularizada (ultima -- la mas lenta/impredecible).")

    log("\n### XGBoost ###")
    correr_arbol_principal("XGBoost", "xgboost", pipeline_xgb, PARAM_DIST_XGB, fit_params_fn=fit_params_xgb)
    correr_arbol_cv("XGBoost", "xgboost", pipeline_xgb, PARAM_DIST_XGB, fit_params_fn=fit_params_xgb)

    log("\n### LightGBM ###")
    correr_arbol_principal("LightGBM", "lightgbm", pipeline_lgb, PARAM_DIST_LGB)
    correr_arbol_cv("LightGBM", "lightgbm", pipeline_lgb, PARAM_DIST_LGB)

    log("\n### HistGradientBoosting ###")
    correr_hgb()

    log("\n### Random Forest ###")
    _correr_lineal("Random Forest", "random_forest", pipeline_rf, familia="arbol")

    log("\n### Red neuronal (MLPClassifier) ###")
    _correr_lineal("Red neuronal (MLP)", "red_neuronal", pipeline_nn, familia="nn", balanceos=BALANCEOS_NN)

    log("\n### Logistica regularizada (elastic net) -- ultima, la mas lenta/impredecible ###")
    _correr_lineal("Logistica regularizada (elastic net, benchmark)", "logistica_regularizada", pipeline_log, familia="logistica")

    log(f"\nFIN. Registro: {REGISTRO_CSV}")


if __name__ == "__main__":
    main()
