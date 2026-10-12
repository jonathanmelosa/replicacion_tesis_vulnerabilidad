"""
Random Forest para prediccion de entrada a la pobreza -- Modelo A (con
ingreso/gasto) vs. Modelo B (sin) -- sobre la particion temporal principal
(train = 2010->2013, test = 2013->2016). Parte de la suite de comparacion
de algoritmos.

Punto intermedio entre la logistica (lineal, benchmark) y los gradient
boosting (HistGB/XGBoost/LightGBM): captura no-linealidades e
interacciones sin especificarlas a mano, con menor varianza que un solo
arbol (promedio de arboles bootstrap independientes).

`RandomForestClassifier` de sklearn NO maneja NaN nativamente -- usa el
mismo preprocesador que la logistica (0+indicador, one-hot -- ver
`modelo_utils.construir_preprocesador`), sin estandarizacion (irrelevante
para arboles, invariantes a escala monotona).

Imputacion, balanceo de clases e hiperparametros: ver docstring de
`modelo_utils.py`
comparacion de 3 estrategias de balanceo por AUC-ROC en CV,
RandomizedSearchCV).
"""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier

import modelo_utils as mu

OUTPUT_DIR = mu.RESULTADOS_DIR / "random_forest"

PARAM_DIST = {
    "modelo__n_estimators": randint(200, 600),
    "modelo__max_depth": randint(3, 15),
    "modelo__min_samples_leaf": randint(2, 30),
    "modelo__max_features": uniform(0.2, 0.6),
}


def construir_pipeline(x_train, balanceo: str, semilla: int = mu.RANDOM_STATE) -> ImbPipeline:
    preprocesador = mu.construir_preprocesador(x_train, escalar=False)
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = RandomForestClassifier(class_weight=class_weight, random_state=semilla, n_jobs=-1)
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

        print(f"\n=== Random Forest -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        importancias = pd.DataFrame({
            "variable": nombres_features,
            "importancia": modelo_final.feature_importances_,
        }).sort_values("importancia", ascending=False)
        importancias.to_csv(OUTPUT_DIR / f"importancia_variables_modelo_{espec}.csv", index=False)

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
        print(f"  Top 10 variables mas importantes:")
        print(importancias.head(10).to_string(index=False))

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Random Forest",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="0 + indicador de faltante (numericas), 'Sin dato' + one-hot drop-first (categoricas)",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones="Hiperparametros tuneados por RandomizedSearchCV (AUC-ROC, CV). Metricas = media +/- IC95 sobre 5 semillas (balanceo/hiperparametros fijos). Umbral elegido por CV maximizando F1 en cada semilla.",
        )

    for espec in mu.ESPECIFICACIONES_CV_2010_2013:
        datos = mu.cargar_datos_cv(espec)
        x, y = mu.preparar_xy_crudo(datos)

        print(f"\n=== Random Forest -- Modelo {espec} (CV dentro de 2010->2013, sin holdout temporal) ===")
        print(f"  n={x.shape[0]}, columnas={x.shape[1]}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x, y_train=y,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        importancias = pd.DataFrame({
            "variable": nombres_features,
            "importancia": modelo_final.feature_importances_,
        }).sort_values("importancia", ascending=False)
        importancias.to_csv(OUTPUT_DIR / f"importancia_variables_modelo_{espec}.csv", index=False)

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: construir_pipeline(x, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y,
        )
        multi["detalle"].to_csv(OUTPUT_DIR / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        r = multi["resumen"]
        print(f"  AUC-ROC (OOF): {r['auc_roc']['media']:.3f} (IC95 {r['auc_roc']['ci95_low']:.3f}-{r['auc_roc']['ci95_high']:.3f})")
        print(f"  Top 10 variables mas importantes:")
        print(importancias.head(10).to_string(index=False))

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Random Forest",
            especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y,
            multi_resultado=multi,
            estrategia_imputacion="0 + indicador de faltante (numericas), 'Sin dato' + one-hot drop-first (categoricas)",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=(
                "PIPELINE 2 (exploratorio): ELCA + DMSP-OLS + ALOS PALSAR + Landsat 5 TM, "
                "restringido a la transicion 2010->2013. Sin holdout temporal -- metricas "
                "sobre probabilidades OUT-OF-FOLD dentro de la misma muestra, NO comparables "
                "cifra a cifra contra A/B/AgeoDMSP/BgeoDMSP. n_train=n_test porque no hay "
                "periodo de prueba separado."
            ),
        )

    print(f"\nGuardado en: {OUTPUT_DIR}")
    print(f"Registro actualizado: {mu.REGISTRO_XLSX}")


if __name__ == "__main__":
    main()
