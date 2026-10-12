"""
modelo_geo3_robusto_cv.py
=====================================
Version bajo config ROBUSTA (CV_FOLDS=10, N_ITER_BUSQUEDA=30, beta=2) de
Ageo3/Bgeo3 (pobreza monetaria).

Ageo3/Bgeo3 se calcularon originalmente con `mu.comparar_balanceo_y_tunear`
SIN pasar cv_folds/n_iter_busqueda, es decir con los defaults antiguos
(CV_FOLDS=3, N_ITER_BUSQUEDA=8, beta=1/F1 -- ver
`ESPECIFICACIONES_CV_2010_2013` en `modelo_xgboost.py`/
`modelo_histgradientboosting.py`/`modelo_logistica_regularizada.py`),
config que quedo congelada desde ANTES de que la comparacion robusta
(folds=10/iter=30/beta=2) se volviera el estandar del resto de la sesion
(ver `modelo_fbeta2_comparacion.py`, linea 37: ESPECIFICACIONES_CV_2010_2013
explicitamente excluida de ese re-run). Por eso Ageo3/Bgeo3 NO son
comparables cifra a cifra contra `modelo_geo3_baseline_cv.py` (que si usa
la config robusta, igual que Aipmgeo3/Bipmgeo3/su baseline bajo IPM).

Este script llena ese hueco: recalcula Ageo3/Bgeo3 bajo la MISMA config
robusta que el resto de la comparacion geo3 (IPM y baseline monetaria),
en un registro NUEVO -- no toca ni sobreescribe el Ageo3/Bgeo3 original
en registro_modelos.csv (ese se conserva intacto).

Espejo EXACTO de `modelo_ipm_geo3_cv.py`, aplicado a "Ageo3"/"Bgeo3" en
vez de "Aipmgeo3"/"Bipmgeo3" -- usa los mismos archivos
modelo_{A,B}geo3_2010_2013.parquet que ya genera
`construir_pipeline_geo3_cv.py` (no requiere ningun paso de construccion
de datos nuevo).

Corre en PARALELO a `modelo_geo3_baseline_cv.py` y
`modelo_ipm_geo3_baseline_cv.py`.

Resultados comparables cifra a cifra contra A/B bajo
`registro_modelos_geo3_baseline.csv` (mismo esquema CV, mismo n, mismos
folds/iteraciones/beta) -- exactamente el punto de este script. NO
comparables contra el Ageo3/Bgeo3 original en registro_modelos.csv
(folds/iter/beta distintos) ni contra A/B/AgeoDMSP/BgeoDMSP (holdout).

OUTPUTS

    data/processed/benchmark_resultados/registro_modelos_geo3_robusto.csv
    data/processed/benchmark_resultados/geo3_robusto/{xgboost,histgradientboosting,logistica_regularizada}/
        metricas_multiples_semillas_modelo_{espec}.csv

COMO CORRER

    cd src/05_model && caffeinate -i python -u modelo_geo3_robusto_cv.py
"""

import time
from datetime import datetime

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb
import modelo_logistica_regularizada as m_log
import modelo_xgboost as m_xgb

BETA = 2.0
CV_FOLDS = 10
N_ITER_BUSQUEDA = 30
ESPECIFICACIONES = ["Ageo3", "Bgeo3"]

REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_geo3_robusto.csv"
REGISTRO_XLSX = mu.RESULTADOS_DIR / "registro_modelos_geo3_robusto.xlsx"
OUTPUT_ROOT = mu.RESULTADOS_DIR / "geo3_robusto"

OBSERVACIONES = (
    f"Ageo3/Bgeo3 (DMSP-OLS + ALOS PALSAR + Landsat 5 TM juntas, target "
    f"pobreza monetaria) recalculados bajo la config ROBUSTA para quedar "
    f"comparables cifra a cifra contra el baseline sin geo bajo el mismo "
    f"esquema (registro_modelos_geo3_baseline.csv). El Ageo3/Bgeo3 original "
    f"en registro_modelos.csv quedo congelado en la config antigua "
    f"(CV_FOLDS=3, N_ITER_BUSQUEDA=8, beta=1) desde antes de adoptar la "
    f"config robusta como estandar -- ver docstring del modulo. Sin "
    f"holdout temporal -- metricas sobre probabilidades OUT-OF-FOLD dentro "
    f"de la transicion 2010->2013 completa. "
    f"F-beta (beta={BETA}), CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, "
    f"balanceo/hiperparametros por AUC-ROC, RANDOM_STATE={mu.RANDOM_STATE}, "
    f"SEMILLAS={mu.SEMILLAS}."
)

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def correr_xgboost() -> None:
    out_dir = OUTPUT_ROOT / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES:
        datos = mu.cargar_datos_cv(espec)
        x, y, cat_cols = mu.preparar_arboles_nativos(datos)

        log(f"=== XGBoost -- {espec} -- INICIO (n={x.shape[0]}, columnas={x.shape[1]}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_xgb.construir_pipeline(x, y, b),
            param_distributions_fn=lambda b: m_xgb.PARAM_DIST,
            x_train=x, y_train=y,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: m_xgb.construir_pipeline(x, y, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y, beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC(OOF): {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "scale_pos_weight": "auto (balanced)" if resultado["balanceo_elegido"] == "balanced" else 1.0}
        mu.registrar_resultado(
            algoritmo="XGBoost", especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (enable_categorical)",
            balanceo_info=resultado, hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === XGBoost -- {espec} -- FIN ===")


def correr_histgradientboosting() -> None:
    out_dir = OUTPUT_ROOT / "histgradientboosting"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES:
        datos = mu.cargar_datos_cv(espec)
        x, y, cat_cols = mu.preparar_arboles_nativos(datos)

        log(f"=== HistGradientBoosting -- {espec} -- INICIO (n={x.shape[0]}, columnas={x.shape[1]}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=m_hgb.construir_pipeline,
            param_distributions_fn=lambda b: m_hgb.PARAM_DIST,
            x_train=x, y_train=y,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: m_hgb.construir_pipeline(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y, beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC(OOF): {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="HistGradientBoosting (sklearn)", especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (categorical_features='from_dtype')",
            balanceo_info=resultado, hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === HistGradientBoosting -- {espec} -- FIN ===")


def correr_logistica() -> None:
    out_dir = OUTPUT_ROOT / "logistica_regularizada"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES:
        datos = mu.cargar_datos_cv(espec)
        x, y = mu.preparar_xy_crudo(datos)

        log(f"=== Logistica regularizada -- {espec} -- INICIO (n={x.shape[0]}, columnas={x.shape[1]}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_log.construir_pipeline(x, b),
            param_distributions_fn=lambda b: m_log.PARAM_DIST,
            x_train=x, y_train=y,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: m_log.construir_pipeline(x, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y, beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC(OOF): {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Logistica regularizada (elastic net, benchmark)", especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="0 + indicador (numericas), 'Sin dato' + one-hot (categoricas)",
            balanceo_info=resultado, hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === Logistica regularizada -- {espec} -- FIN ===")


def main() -> None:
    mu.RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"INICIO Ageo3/Bgeo3 bajo config ROBUSTA: CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, beta={BETA}. 6 combinaciones (3 algoritmos x 2 especificaciones).")
    correr_xgboost()
    correr_histgradientboosting()
    correr_logistica()
    log(f"FIN. Registro: {REGISTRO_CSV}")


if __name__ == "__main__":
    main()
