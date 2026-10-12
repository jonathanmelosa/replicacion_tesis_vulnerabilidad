"""
Comparacion DMSP-OLS con target de POBREZA MULTIDIMENSIONAL (IPM) en vez
de pobreza monetaria
el hallazgo de que DMSP-OLS no aporta es especifico de la
definicion monetaria de vulnerabilidad. Mismo criterio de umbral y
busqueda que la version "robusta" ya validada para pobreza monetaria
(`modelo_fbeta2_cv10_comparacion.py`): F-beta beta=2, CV_FOLDS=10,
N_ITER_BUSQUEDA=30
prefirio la config robusta desde el inicio para esta comparacion, con
`caffeinate -i` para evitar que el reposo de la Mac vuelva a inflar el
tiempo de pared como paso con la corrida anterior).

Alcance identico en estructura a `modelo_fbeta2_cv10_comparacion.py`: 3
algoritmos (XGBoost, HistGradientBoosting, Logistica regularizada) x 4
especificaciones -- pero las especificaciones son `Aipm`/`Bipm`/
`AipmgeoDMSP`/`BipmgeoDMSP` (target `pobre_ipm`, ver
`build_benchmark_train_test_ipm.py` y `construir_pipeline_geo_dmsp_ipm.py`),
no `A`/`B`/`AgeoDMSP`/`BgeoDMSP` (target `pobre_ingreso`). Las covariables
X son las MISMAS que el benchmark monetario -- unico factor que cambia es
el target.

Registro SEPARADO de todos los anteriores -- no toca ningun registro de
pobreza monetaria.

RIESGO DE TIEMPO CONOCIDO: Logistica regularizada (`saga`) ya tardo hasta
3h40min en una combinacion durante la corrida robusta de pobreza
monetaria, por mala suerte de la busqueda aleatoria de hiperparametros
(C/l1_ratio con convergencia lenta) -- puede volver a pasar aca, y el
dataset de entrenamiento IPM es ademas mas grande (6.474 hogares en
2010->2013 vs. 3.089 en el benchmark monetario), lo que si algo aumenta
el riesgo, no lo reduce.

OUTPUTS

    data/processed/benchmark_resultados/registro_modelos_ipm.csv
    data/processed/benchmark_resultados/registro_modelos_ipm.xlsx
    data/processed/benchmark_resultados/ipm/{xgboost,histgradientboosting,logistica_regularizada}/
        metricas_multiples_semillas_modelo_{espec}.csv

COMO CORRER

    cd src/05_model && caffeinate -i python -u modelo_ipm_comparacion.py
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
ESPECIFICACIONES_IPM = ["Aipm", "Bipm", "AipmgeoDMSP", "BipmgeoDMSP"]

REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_ipm.csv"
REGISTRO_XLSX = mu.RESULTADOS_DIR / "registro_modelos_ipm.xlsx"
OUTPUT_ROOT = mu.RESULTADOS_DIR / "ipm"

OBSERVACIONES = (
    f"Target = pobre_ipm (Indice de Pobreza Multidimensional, ver "
    f"src/04_features/build_ipm_multidimensional.py), no pobre_ingreso. "
    f"Covariables X identicas al benchmark monetario -- unico factor que "
    f"cambia es el target. Umbral por F-beta (beta={BETA}), "
    f"CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, "
    f"balanceo/hiperparametros por AUC-ROC, RANDOM_STATE={mu.RANDOM_STATE}, "
    f"SEMILLAS={mu.SEMILLAS} -- misma config robusta que "
    f"registro_modelos_fbeta2_cv10.csv (pobreza monetaria), para "
    f"comparabilidad directa entre targets."
)

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def correr_xgboost() -> None:
    out_dir = OUTPUT_ROOT / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES_IPM:
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

    for espec in ESPECIFICACIONES_IPM:
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

    for espec in ESPECIFICACIONES_IPM:
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
    log(f"INICIO comparacion IPM: CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, beta={BETA}. 12 combinaciones (3 algoritmos x 4 especificaciones).")
    correr_xgboost()
    correr_histgradientboosting()
    correr_logistica()
    log(f"FIN. Registro (no toca ningun registro de pobreza monetaria): {REGISTRO_CSV}")


if __name__ == "__main__":
    main()
