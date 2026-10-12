"""
BENCHMARK: regresion logistica con regularizacion elastic net para
prediccion de entrada a la pobreza -- Modelo A (con ingreso/gasto) vs.
Modelo B (sin) -- sobre la particion temporal principal (train =
2010->2013, test = 2013->2016).

Por que logistica regularizada como benchmark
--------------------------------------------------------------------------
Estandar de facto en la literatura de prediccion de pobreza/vulnerabilidad
(interpretable, coeficientes con signo e intuicion economica directa).
Con p ~ 165-169 covariables (mas tras one-hot) y n ~ 3.089 (train), sin
regularizar el modelo sobreajustaria -- de ahi la penalizacion. Se usa
`penalty="elasticnet"`, `solver="saga"`, con `C` y `l1_ratio` tuneados por
`RandomizedSearchCV` (ver `modelo_utils.py`) en vez de comprometerse a
priori con Lasso o Ridge puro.

Imputacion, balanceo de clases e hiperparametros: ver docstring de
`modelo_utils.py`
(0+indicador para missings, comparacion de 3 estrategias de balanceo
seleccionando por AUC-ROC en CV, RandomizedSearchCV para hiperparametros).
Para la estrategia "oversampling", `class_weight` NO se fija (evita doble
ajuste combinando remuestreo + reponderacion).
"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, uniform
from sklearn.linear_model import LogisticRegression

import modelo_utils as mu

OUTPUT_DIR = mu.RESULTADOS_DIR / "logistica_regularizada"

PARAM_DIST = {
    "modelo__C": loguniform(1e-3, 1e2),
    "modelo__l1_ratio": uniform(0, 1),
}


def construir_pipeline(x_train, balanceo: str, semilla: int = mu.RANDOM_STATE) -> ImbPipeline:
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for espec in mu.ESPECIFICACIONES_PRINCIPAL:
        train, test = mu.cargar_datos(espec)
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)

        print(f"\n=== Logistica regularizada -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        import pandas as pd
        coeficientes = pd.DataFrame({
            "variable": nombres_features,
            "coeficiente": modelo_final.coef_[0],
        }).sort_values("coeficiente", key=np.abs, ascending=False)
        coeficientes.to_csv(OUTPUT_DIR / f"coeficientes_modelo_{espec}.csv", index=False)

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: construir_pipeline(x_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        )
        multi["detalle"].to_csv(OUTPUT_DIR / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        print(f"  Hiperparametros: {resultado['mejores_params']}")
        print(f"  Umbral medio (5 semillas): {multi['resumen']['umbral_media']:.2f}")
        r = multi["resumen"]
        print(f"  AUC-ROC: {r['auc_roc']['media']:.3f} (IC95 {r['auc_roc']['ci95_low']:.3f}-{r['auc_roc']['ci95_high']:.3f})  Recall: {r['recall']['media']:.3f}  F1: {r['f1']['media']:.3f}")
        print(f"  Top 10 |coeficiente| mayor:")
        print(coeficientes.head(10).to_string(index=False))

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Logistica regularizada (elastic net, benchmark)",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="0 + indicador de faltante (numericas), 'Sin dato' + one-hot drop-first (categoricas), estandarizacion",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones="Modelo principal. C y l1_ratio tuneados por RandomizedSearchCV (AUC-ROC, CV). Metricas = media +/- IC95 sobre 5 semillas (balanceo/hiperparametros fijos, solo cambia el random_state del ajuste final). Umbral de clasificacion elegido por CV maximizando F1 en cada semilla. Coeficientes (semilla 42) en unidades estandarizadas, signo interpretable directamente.",
        )

    for espec in mu.ESPECIFICACIONES_ABLATION:
        train, test = mu.cargar_datos(espec)
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)

        print(f"\n=== Logistica regularizada -- Modelo {espec} (ablation: sin riqueza/servicios) ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        import pandas as pd
        coeficientes = pd.DataFrame({
            "variable": nombres_features,
            "coeficiente": modelo_final.coef_[0],
        }).sort_values("coeficiente", key=np.abs, ascending=False)
        coeficientes.to_csv(OUTPUT_DIR / f"coeficientes_modelo_{espec}.csv", index=False)

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: construir_pipeline(x_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        )
        multi["detalle"].to_csv(OUTPUT_DIR / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        r = multi["resumen"]
        print(f"  AUC-ROC: {r['auc_roc']['media']:.3f} (IC95 {r['auc_roc']['ci95_low']:.3f}-{r['auc_roc']['ci95_high']:.3f})  precision_top10: {r['precision_top10']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Logistica regularizada (elastic net, benchmark)",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="0 + indicador de faltante (numericas), 'Sin dato' + one-hot drop-first (categoricas), estandarizacion",
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
        x, y = mu.preparar_xy_crudo(datos)

        print(f"\n=== Logistica regularizada -- Modelo {espec} (CV dentro de 2010->2013, sin holdout temporal) ===")
        print(f"  n={x.shape[0]}, columnas={x.shape[1]}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x, y_train=y,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        import pandas as pd
        coeficientes = pd.DataFrame({
            "variable": nombres_features,
            "coeficiente": modelo_final.coef_[0],
        }).sort_values("coeficiente", key=np.abs, ascending=False)
        coeficientes.to_csv(OUTPUT_DIR / f"coeficientes_modelo_{espec}.csv", index=False)

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: construir_pipeline(x, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y,
        )
        multi["detalle"].to_csv(OUTPUT_DIR / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        r = multi["resumen"]
        print(f"  AUC-ROC (OOF): {r['auc_roc']['media']:.3f} (IC95 {r['auc_roc']['ci95_low']:.3f}-{r['auc_roc']['ci95_high']:.3f})")
        print(f"  Top 10 |coeficiente| mayor:")
        print(coeficientes.head(10).to_string(index=False))

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Logistica regularizada (elastic net, benchmark)",
            especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y,
            multi_resultado=multi,
            estrategia_imputacion="0 + indicador de faltante (numericas), 'Sin dato' + one-hot drop-first (categoricas), estandarizacion",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=(
                "PIPELINE 2 (exploratorio): ELCA + DMSP-OLS + ALOS PALSAR + Landsat 5 TM, "
                "restringido a la transicion 2010->2013. Sin holdout temporal -- metricas "
                "sobre probabilidades OUT-OF-FOLD dentro de la misma muestra, NO comparables "
                "cifra a cifra contra A/B/AgeoDMSP/BgeoDMSP. n_train=n_test porque no hay "
                "periodo de prueba separado. Coeficientes en unidades estandarizadas."
            ),
        )

    print(f"\nGuardado en: {OUTPUT_DIR}")
    print(f"Registro actualizado: {mu.REGISTRO_XLSX}")


if __name__ == "__main__":
    main()
