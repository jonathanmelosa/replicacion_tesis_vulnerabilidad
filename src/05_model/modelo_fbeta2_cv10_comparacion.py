"""
Version mas robusta de `modelo_fbeta2_comparacion.py`: mismo cambio de
criterio de umbral (F-beta, beta=2, en vez de F1 -- ver docstring de ese
script para la justificacion completa), pero con `CV_FOLDS=10` y
`N_ITER_BUSQUEDA=30` en vez de los 3/8 de la suite original -- 3
folds/8 iteraciones no es el estandar. Corrida mas pesada:
RandomizedSearchCV pasa de 8*3=24 fits por balanceo a 30*10=300 fits por
balanceo (12.5x), y `elegir_umbral_por_cv` (cross_val_predict) tambien pasa
de 3 a 10 folds. Imprime avance con timestamps (por balanceo, por
algoritmo/especificacion, por semilla) y usa `verbose=1` en
RandomizedSearchCV para que quede claro que el proceso sigue corriendo, no
colgado.

QUE NO CAMBIA: espacios de busqueda de hiperparametros (`PARAM_DIST`,
importados literalmente de los scripts originales), `SCORING="roc_auc"`
para balanceo/hiperparametros, `RANDOM_STATE=42`, `SEMILLAS=[42,1,2,3,4]`.
Alcance identico a `modelo_fbeta2_comparacion.py`: 3 algoritmos (XGBoost,
HistGradientBoosting, Logistica regularizada) x 4 especificaciones (A, B,
AgeoDMSP, BgeoDMSP).

Registro SEPARADO tanto del original (registro_modelos.csv, folds=3/F1)
como del de `modelo_fbeta2_comparacion.py` (registro_modelos_fbeta2.csv,
folds=3/F-beta=2) -- ninguno de los dos se toca.

OUTPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.xlsx
    data/processed/benchmark_resultados/fbeta2_cv10/{xgboost,histgradientboosting,logistica_regularizada}/
        metricas_multiples_semillas_modelo_{espec}.csv

COMO CORRER

    cd src/05_model && python -u modelo_fbeta2_cv10_comparacion.py
    (el -u evita que Python bufferee stdout -- importante para ver el
    avance en tiempo real si se redirige a un log)
"""

import sys
import time
from datetime import datetime

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb
import modelo_logistica_regularizada as m_log
import modelo_xgboost as m_xgb

BETA = 2.0
CV_FOLDS = 10
N_ITER_BUSQUEDA = 30

REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv"
REGISTRO_XLSX = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.xlsx"
OUTPUT_ROOT = mu.RESULTADOS_DIR / "fbeta2_cv10"

OBSERVACIONES = (
    f"Identico a modelo_fbeta2_comparacion.py (umbral por F-beta, beta={BETA}) "
    f"pero con CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA} en vez de "
    f"3/8 (3 folds/8 iteraciones no es el estandar). Balanceo/hiperparametros siguen por AUC-ROC, "
    f"RANDOM_STATE={mu.RANDOM_STATE}, SEMILLAS={mu.SEMILLAS} sin cambios. "
    f"Comparar contra registro_modelos.csv (F1, folds=3) y "
    f"registro_modelos_fbeta2.csv (F-beta={BETA}, folds=3) para separar el "
    f"efecto del criterio de umbral del efecto de folds/iteraciones."
)

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def correr_xgboost() -> None:
    out_dir = OUTPUT_ROOT / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mu.ESPECIFICACIONES_PRINCIPAL:
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        log(f"=== XGBoost -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_xgb.construir_pipeline(x_train, y_train, b),
            param_distributions_fn=lambda b: m_xgb.PARAM_DIST,
            x_train=x_train, y_train=y_train,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: m_xgb.construir_pipeline(x_train, y_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC: {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "scale_pos_weight": "auto (balanced)" if resultado["balanceo_elegido"] == "balanced" else 1.0}
        mu.registrar_resultado(
            algoritmo="XGBoost",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (enable_categorical)",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === XGBoost -- Modelo {espec} -- FIN ===")


def correr_histgradientboosting() -> None:
    out_dir = OUTPUT_ROOT / "histgradientboosting"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mu.ESPECIFICACIONES_PRINCIPAL:
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        log(f"=== HistGradientBoosting -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=m_hgb.construir_pipeline,
            param_distributions_fn=lambda b: m_hgb.PARAM_DIST,
            x_train=x_train, y_train=y_train,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: m_hgb.construir_pipeline(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC: {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="HistGradientBoosting (sklearn)",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (categorical_features='from_dtype')",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === HistGradientBoosting -- Modelo {espec} -- FIN ===")


def correr_logistica() -> None:
    out_dir = OUTPUT_ROOT / "logistica_regularizada"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mu.ESPECIFICACIONES_PRINCIPAL:
        train, test = mu.cargar_datos(espec)
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)

        log(f"=== Logistica regularizada -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_log.construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: m_log.PARAM_DIST,
            x_train=x_train, y_train=y_train,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: m_log.construir_pipeline(x_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC: {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Logistica regularizada (elastic net, benchmark)",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="0 + indicador (numericas), 'Sin dato' + one-hot (categoricas) -- ver docstring de modelo_utils.py",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === Logistica regularizada -- Modelo {espec} -- FIN ===")


def main() -> None:
    mu.RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"INICIO corrida robusta: CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, beta={BETA}. 12 combinaciones (3 algoritmos x 4 especificaciones).")
    correr_xgboost()
    correr_histgradientboosting()
    correr_logistica()
    log(f"FIN. Registro (no toca registro_modelos.csv ni registro_modelos_fbeta2.csv): {REGISTRO_CSV}")


if __name__ == "__main__":
    main()
