"""
Extension de `modelo_ipm_comparacion.py`: Random Forest y LightGBM bajo
target IPM (los 2 algoritmos de la suite original de 5 que no se habian
probado todavia con este target)
para ver si el efecto significativo de DMSP-OLS encontrado en XGBoost
aparece tambien en otros algoritmos basados en arboles, o es exclusivo
de XGBoost.

Misma config robusta que el resto de la comparacion IPM: F-beta beta=2,
CV_FOLDS=10, N_ITER_BUSQUEDA=30, RANDOM_STATE=42, SEMILLAS=[42,1,2,3,4].
Escribe en el MISMO registro que `modelo_ipm_comparacion.py`
(`registro_modelos_ipm.csv`, upsert por algoritmo+especificacion) --
agrega filas nuevas, no toca las de XGBoost/HistGB/Logistica ya
guardadas.

RIESGO DE TIEMPO: ni Random Forest ni LightGBM tienen el problema de
convergencia lenta que si tiene Logistica (`saga`) -- se espera un
tiempo comparable a XGBoost/HistGB (minutos por combinacion, no horas).

OUTPUTS

    data/processed/benchmark_resultados/registro_modelos_ipm.csv (filas nuevas)
    data/processed/benchmark_resultados/ipm/{random_forest,lightgbm}/
        metricas_multiples_semillas_modelo_{espec}.csv

COMO CORRER

    cd src/05_model && caffeinate -i python -u modelo_ipm_comparacion_rf_lgbm.py
"""

import time
from datetime import datetime

import modelo_utils as mu
import modelo_lightgbm as m_lgb
import modelo_random_forest as m_rf

BETA = 2.0
CV_FOLDS = 10
N_ITER_BUSQUEDA = 30
ESPECIFICACIONES_IPM = ["Aipm", "Bipm", "AipmgeoDMSP", "BipmgeoDMSP"]

REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_ipm.csv"
REGISTRO_XLSX = mu.RESULTADOS_DIR / "registro_modelos_ipm.xlsx"
OUTPUT_ROOT = mu.RESULTADOS_DIR / "ipm"

OBSERVACIONES = (
    f"Target = pobre_ipm. Extension de modelo_ipm_comparacion.py (que cubrio "
    f"XGBoost/HistGB/Logistica) -- Random Forest y LightGBM, los 2 algoritmos "
    f"de la suite de 5 que faltaban bajo este target. Misma config: F-beta "
    f"(beta={BETA}), CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, "
    f"balanceo/hiperparametros por AUC-ROC, RANDOM_STATE={mu.RANDOM_STATE}, "
    f"SEMILLAS={mu.SEMILLAS}."
)

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def correr_random_forest() -> None:
    out_dir = OUTPUT_ROOT / "random_forest"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES_IPM:
        train, test = mu.cargar_datos(espec)
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)

        log(f"=== Random Forest -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_rf.construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: m_rf.PARAM_DIST,
            x_train=x_train, y_train=y_train,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: m_rf.construir_pipeline(x_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC: {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Random Forest",
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
        log(f"  === Random Forest -- Modelo {espec} -- FIN ===")


def correr_lightgbm() -> None:
    out_dir = OUTPUT_ROOT / "lightgbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES_IPM:
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        log(f"=== LightGBM -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_lgb.construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: m_lgb.PARAM_DIST,
            x_train=x_train, y_train=y_train,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: m_lgb.construir_pipeline(x_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC: {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="LightGBM",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === LightGBM -- Modelo {espec} -- FIN ===")


def main() -> None:
    mu.RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"INICIO Random Forest + LightGBM bajo IPM: CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, beta={BETA}. 8 combinaciones (2 algoritmos x 4 especificaciones).")
    correr_random_forest()
    correr_lightgbm()
    log(f"FIN. Registro (agrega filas a): {REGISTRO_CSV}")


if __name__ == "__main__":
    main()
