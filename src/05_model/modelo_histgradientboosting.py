"""
HistGradientBoostingClassifier (sklearn) para prediccion de entrada a la
pobreza -- Modelo A (con ingreso/gasto) vs. Modelo B (sin) -- sobre la
particion temporal principal (train = 2010->2013, test = 2013->2016).
Parte de la suite de comparacion de algoritmos.

NOTA: reemplaza a `entrenar_benchmark.py` (version anterior a la suite,
cuando HistGB era el unico modelo entrenado, con hiperparametros fijos y
sin comparacion de balanceo). Ahora usa la misma arquitectura que el resto
de la suite: comparacion de 3 estrategias de balanceo por AUC-ROC en CV +
RandomizedSearchCV para hiperparametros.

Por que HistGradientBoosting en esta suite
--------------------------------------------------------------------------
Maneja NaN y categoricas nativamente (missingness estructural, ver
docstring de `modelo_utils.py`) -- mismo argumento que XGBoost/LightGBM,
las otras dos implementaciones de gradient boosting de la suite.
Columnas categoricas en dtype `category`, `categorical_features="from_dtype"`.

Ademas de las metricas estandar, reporta importancia de variables por
permutation importance (AUC en el conjunto de prueba, 10 repeticiones) --
mas costosa pero mas robusta que `feature_importances_`.
"""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

import modelo_utils as mu

OUTPUT_DIR = mu.RESULTADOS_DIR / "histgradientboosting"

PARAM_DIST = {
    "modelo__max_iter": randint(100, 400),
    "modelo__max_depth": randint(3, 8),
    "modelo__max_leaf_nodes": randint(15, 63),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__l2_regularization": loguniform(0.01, 10),
}


def construir_pipeline(balanceo: str, semilla: int = mu.RANDOM_STATE) -> ImbPipeline:
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = HistGradientBoostingClassifier(
        categorical_features="from_dtype", class_weight=class_weight, random_state=semilla,
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for espec in mu.ESPECIFICACIONES_PRINCIPAL:
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        print(f"\n=== HistGradientBoosting -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=construir_pipeline,
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        imp = permutation_importance(modelo_final, x_test, y_test, scoring="roc_auc", n_repeats=10, random_state=mu.RANDOM_STATE, n_jobs=-1)
        importancias = pd.DataFrame({
            "variable": x_test.columns,
            "importancia_media": imp.importances_mean,
            "importancia_std": imp.importances_std,
        }).sort_values("importancia_media", ascending=False)
        importancias.to_csv(OUTPUT_DIR / f"importancia_variables_modelo_{espec}.csv", index=False)

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: construir_pipeline(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        )
        multi["detalle"].to_csv(OUTPUT_DIR / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        print(f"  Hiperparametros: {resultado['mejores_params']}")
        print(f"  Umbral medio (5 semillas): {multi['resumen']['umbral_media']:.2f}")
        r = multi["resumen"]
        print(f"  AUC-ROC: {r['auc_roc']['media']:.3f} (IC95 {r['auc_roc']['ci95_low']:.3f}-{r['auc_roc']['ci95_high']:.3f})  Recall: {r['recall']['media']:.3f}  F1: {r['f1']['media']:.3f}")
        print(f"  Top 10 variables mas importantes:")
        print(importancias.head(10).to_string(index=False))

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
            observaciones="Reemplaza a entrenar_benchmark.py. Ahora con comparacion de balanceo y RandomizedSearchCV. Importancia por permutation importance (semilla 42, AUC, 10 repeticiones). Metricas = media +/- IC95 sobre 5 semillas. Umbral elegido por CV maximizando F1 en cada semilla.",
        )

    for espec in mu.ESPECIFICACIONES_ABLATION:
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        print(f"\n=== HistGradientBoosting -- Modelo {espec} (ablation: sin riqueza/servicios) ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=construir_pipeline,
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        imp = permutation_importance(modelo_final, x_test, y_test, scoring="roc_auc", n_repeats=10, random_state=mu.RANDOM_STATE, n_jobs=-1)
        importancias = pd.DataFrame({
            "variable": x_test.columns,
            "importancia_media": imp.importances_mean,
            "importancia_std": imp.importances_std,
        }).sort_values("importancia_media", ascending=False)
        importancias.to_csv(OUTPUT_DIR / f"importancia_variables_modelo_{espec}.csv", index=False)

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: construir_pipeline(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        )
        multi["detalle"].to_csv(OUTPUT_DIR / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        r = multi["resumen"]
        print(f"  AUC-ROC: {r['auc_roc']['media']:.3f} (IC95 {r['auc_roc']['ci95_low']:.3f}-{r['auc_roc']['ci95_high']:.3f})  precision_top10: {r['precision_top10']['media']:.3f}")

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
            observaciones=(
                "ABLATION: igual a la especificacion base, pero sin n_servicios_publicos_hogar "
                "ni n_bienes_durables_hogar (las dos variables que hacen a DMSP-OLS redundante, "
                "ver Seccion 5.3). Prueba si DMSP-OLS aporta cuando esas preguntas de la encuesta "
                "no estan disponibles. Holdout temporal identico a las especificaciones principales."
            ),
        )

    for espec in mu.ESPECIFICACIONES_CV_2010_2013:
        datos = mu.cargar_datos_cv(espec)
        x, y, cat_cols = mu.preparar_arboles_nativos(datos)

        print(f"\n=== HistGradientBoosting -- Modelo {espec} (CV dentro de 2010->2013, sin holdout temporal) ===")
        print(f"  n={x.shape[0]}, columnas={x.shape[1]}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=construir_pipeline,
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x, y_train=y,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        # Sin conjunto de prueba separado -- importancia por permutacion
        # calculada IN-SAMPLE sobre el mismo x/y usado para entrenar (el
        # estimador ya viene reentrenado sobre el 100% de x, ver
        # comparar_balanceo_y_tunear). Interpretar con cautela: mide que
        # tanto se apoya el ajuste en cada variable, no su aporte fuera de
        # muestra (no existe holdout en este ejercicio).
        imp = permutation_importance(modelo_final, x, y, scoring="roc_auc", n_repeats=10, random_state=mu.RANDOM_STATE, n_jobs=-1)
        importancias = pd.DataFrame({
            "variable": x.columns,
            "importancia_media": imp.importances_mean,
            "importancia_std": imp.importances_std,
        }).sort_values("importancia_media", ascending=False)
        importancias.to_csv(OUTPUT_DIR / f"importancia_variables_modelo_{espec}.csv", index=False)

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: construir_pipeline(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y,
        )
        multi["detalle"].to_csv(OUTPUT_DIR / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        r = multi["resumen"]
        print(f"  AUC-ROC (OOF): {r['auc_roc']['media']:.3f} (IC95 {r['auc_roc']['ci95_low']:.3f}-{r['auc_roc']['ci95_high']:.3f})")
        print(f"  Top 10 variables mas importantes (in-sample):")
        print(importancias.head(10).to_string(index=False))

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="HistGradientBoosting (sklearn)",
            especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y,
            multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (categorical_features='from_dtype')",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=(
                "PIPELINE 2 (exploratorio): ELCA + DMSP-OLS + ALOS PALSAR + Landsat 5 TM, "
                "restringido a la transicion 2010->2013. Sin holdout temporal -- metricas "
                "sobre probabilidades OUT-OF-FOLD dentro de la misma muestra, NO comparables "
                "cifra a cifra contra A/B/AgeoDMSP/BgeoDMSP. n_train=n_test porque no hay "
                "periodo de prueba separado. Importancia por permutacion in-sample (sin test)."
            ),
        )

    print(f"\nGuardado en: {OUTPUT_DIR}")
    print(f"Registro actualizado: {mu.REGISTRO_XLSX}")


if __name__ == "__main__":
    main()
