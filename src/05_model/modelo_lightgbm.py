"""
LightGBM para prediccion de entrada a la pobreza -- Modelo A (con
ingreso/gasto) vs. Modelo B (sin) -- sobre la particion temporal principal
(train = 2010->2013, test = 2013->2016). Parte de la suite de comparacion
de algoritmos.

Igual que HistGB y XGBoost, LightGBM maneja NaN y categoricas nativamente
(columnas en dtype `category`, autodetectadas) -- NO se imputa
(missingness estructural, ver docstring de `modelo_utils.py`). Crecimiento
leaf-wise (en vez de level-wise como HistGB/XGBoost) -- puede capturar
interacciones mas profundas con el mismo presupuesto de arboles, a costa
de mayor riesgo de sobreajuste en muestras pequeñas (n~3.089); el espacio
de busqueda de `max_depth`/`num_leaves` restringe esto explicitamente.

Imputacion, balanceo e hiperparametros: ver docstring de `modelo_utils.py`

balanceo por AUC-ROC en CV, RandomizedSearchCV para hiperparametros).
`class_weight="balanced"` (misma API que sklearn) para la estrategia
"balanced"; None para "ninguno"/"oversampling".
"""

import pandas as pd
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint

import modelo_utils as mu

OUTPUT_DIR = mu.RESULTADOS_DIR / "lightgbm"

PARAM_DIST = {
    "modelo__n_estimators": randint(100, 500),
    "modelo__max_depth": randint(3, 8),
    "modelo__num_leaves": randint(15, 63),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__reg_lambda": loguniform(0.1, 10),
    "modelo__min_child_samples": randint(5, 50),
}


def construir_pipeline(x_train, balanceo: str, semilla: int = mu.RANDOM_STATE) -> ImbPipeline:
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = lgb.LGBMClassifier(class_weight=class_weight, random_state=semilla, n_jobs=-1, verbosity=-1)
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

        print(f"\n=== LightGBM -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        importancias = pd.DataFrame({
            "variable": x_train.columns,
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
            algoritmo="LightGBM",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (dtype category, autodeteccion)",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones="Hiperparametros tuneados por RandomizedSearchCV (AUC-ROC, CV). max_depth/num_leaves acotados por el riesgo de sobreajuste del crecimiento leaf-wise con n~3.089. Metricas = media +/- IC95 sobre 5 semillas. Umbral elegido por CV maximizando F1 en cada semilla.",
        )

    for espec in mu.ESPECIFICACIONES_CV_2010_2013:
        datos = mu.cargar_datos_cv(espec)
        x, y, cat_cols = mu.preparar_arboles_nativos(datos)

        print(f"\n=== LightGBM -- Modelo {espec} (CV dentro de 2010->2013, sin holdout temporal) ===")
        print(f"  n={x.shape[0]}, columnas={x.shape[1]}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x, y_train=y,
        )
        pipe = resultado["estimador"]

        modelo_final = pipe.named_steps["modelo"]
        importancias = pd.DataFrame({
            "variable": x.columns,
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
            algoritmo="LightGBM",
            especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y,
            multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (dtype category, autodeteccion)",
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
