"""
Analisis SHAP (SHapley Additive exPlanations) para los 5 algoritmos de la
suite, en las especificaciones CON DMSP-OLS de ambos targets (AgeoDMSP --
pobreza monetaria; AipmgeoDMSP -- IPM)
(2026-08-30) para profundizar en la pregunta abierta de los informes:
¿por que XGBoost capta un efecto de AUC significativo de DMSP-OLS bajo
IPM y HistGB no, si la comparacion de importancia de variables
(feature_importances_/permutation importance) no mostraba una diferencia
grande entre ambos?

SHAP tiene una ventaja sobre feature_importances_/permutation importance
para esta pregunta especifica: satisface propiedades de aditividad
(la suma de los SHAP values de un hogar reconstruye exactamente la
prediccion) y consistencia (si un modelo depende MAS de una variable,
su SHAP value nunca puede bajar) -- una comparacion mas rigurosa de
"cuanto pesa cada variable en cada prediccion individual" que las
alternativas ya usadas.

AMPLIADO (2026-09-02): la version anterior solo cubria XGBoost y
HistGradientBoosting (2 de 5 algoritmos), con una lista `COMBINACIONES`
hardcodeada -- el mismo patron de hardcodeo que dejo a Random Forest y
LightGBM fuera del bootstrap de significancia de IPM sin que nadie lo
notara. Ahora este script:
    1. Deriva los algoritmos a procesar de los registros de resultados
       (`algoritmos_presentes_en_registro`, en `algoritmos_suite.py`),
       no de una lista mantenida a mano -- cubre los 5 algoritmos
       automaticamente si estan en el registro.
    2. Agrega SHAP para Logistica regularizada ("también el análisis de shap values a la logística").
       Los modelos de arbol (XGBoost/HistGB/LightGBM/Random Forest) usan
       `shap.TreeExplainer` (exacto, rapido); Logistica -- al ser un
       modelo LINEAL -- usa `shap.LinearExplainer`, que para un modelo
       lineal calcula el SHAP value de forma cerrada como
       coeficiente x (x_i - media(x_i)) sobre los datos YA transformados
       por el preprocesador (imputacion + escalado + one-hot) -- ver
       docstring de `calcular_shap()` para el detalle exacto de por que
       el explainer y los datos de entrada difieren entre familias de
       algoritmo.

QUE HACE
---------------------------------------------------------------------
Para cada algoritmo presente en el registro de cada target (hasta 5 x 2
= 10 combinaciones):
    1. Reconstruye el modelo ganador (balanceo/hiperparametros ya
       encontrados, UN solo fit, sin repetir RandomizedSearchCV) y lo
       entrena sobre x_train.
    2. Calcula SHAP values sobre el conjunto de PRUEBA, con el explainer
       que corresponda a la familia del algoritmo (tree/linear).
    3. Reporta: |SHAP| medio por variable (ranking global), el ranking
       especifico de las 23 variables DMSP-OLS dentro de ese ranking, y
       la suma total de |SHAP| de DMSP-OLS como % del total de todas las
       variables (mide cuanto "peso" total tienen las 23 variables juntas
       en las predicciones, mas alla del ranking de la mejor variable
       individual).

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia.csv
    (ranking completo de variables por combinacion)
    data/processed/benchmark_resultados/diagnostico_shap_resumen_dmsp.csv
    (resumen: peso total y ranking de DMSP-OLS por combinacion)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap.py
"""

import numpy as np
import pandas as pd
import shap

import modelo_utils as mu
from algoritmos_suite import (
    algoritmos_presentes_en_registro,
    filtrar_params_modelo,
    preparar_x_y,
    resolver_algoritmo,
)

TARGETS = {
    "Monetaria": {
        "registro": mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv",
        "espec_con_dmsp": "AgeoDMSP",
    },
    "IPM": {
        "registro": mu.RESULTADOS_DIR / "registro_modelos_ipm.csv",
        "espec_con_dmsp": "AipmgeoDMSP",
    },
}


def entrenar(algoritmo_raw: str, espec: str, registro: pd.DataFrame):
    """Reconstruye y entrena el pipeline ganador. Devuelve el ESTIMADOR
    final (`pipe.named_steps["modelo"]`), el pipeline completo (para
    poder acceder a `named_steps["prep"]` en la familia
    `preprocesador_clasico`), y x_test/y_test crudos (sin transformar
    -- la transformacion para SHAP se hace en `calcular_shap`, distinta
    segun familia)."""
    fila = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)].iloc[0]
    balanceo = fila["balanceo_elegido"]
    params = filtrar_params_modelo(fila["hiperparametros"])

    train, test = mu.cargar_datos(espec)
    x_train, y_train, x_test, y_test, cat_cols = preparar_x_y(algoritmo_raw, train, test)

    pipe = resolver_algoritmo(algoritmo_raw)["construir_pipeline_fn"](x_train, y_train, balanceo, mu.RANDOM_STATE)
    if params:
        pipe.set_params(**params)
    pipe.fit(x_train, y_train)

    return pipe, x_train, x_test, y_test, cat_cols


def calcular_shap(algoritmo_raw: str, pipe, x_train: pd.DataFrame, x_test: pd.DataFrame, cat_cols: list):
    """Calcula SHAP values y devuelve (shap_values, nombres_variables,
    x_test_shap) -- x_test_shap es la matriz EXACTA que el explainer usa
    (nativa para arbol_nativo, o la salida del preprocesador para
    preprocesador_clasico) y hace falta ademas de shap_values/nombres
    para graficar (shap.summary_plot necesita el valor de cada variable,
    no solo su nombre y su shap value) -- devuelta aqui en vez de
    recalculada por cada script que grafica, para no duplicar esta
    logica de transformacion por familia de algoritmo.

    Familia `arbol_nativo` (XGBoost/HistGB/LightGBM): `shap.TreeExplainer`
    sobre el estimador directamente, con las mismas columnas nativas
    (dtype category) que uso el modelo -- excepto HistGB/LightGBM, cuyo
    TreeExplainer no soporta dtype category para estos wrappers y
    necesita las categoricas codificadas a enteros (`.cat.codes`) primero
    (verificado que reproduce el mismo split interno que usa el modelo).

    Familia `preprocesador_clasico` (Random Forest/Logistica): el modelo
    NO ve `x_test` crudo -- ve la salida de `pipe.named_steps["prep"]`
    (0+indicador escalado o no, one-hot). Por eso SHAP tambien debe
    calcularse sobre esos datos YA transformados, con los nombres de
    columna que produce el propio preprocesador
    (`get_feature_names_out()`, prefijos "num__"/"cat__"). Para Random
    Forest (arbol) se usa igual `TreeExplainer`; para Logistica (lineal)
    se usa `LinearExplainer`, que necesita ademas un conjunto de fondo
    (`x_train` transformado) para estimar la media de cada variable --
    el termino que SHAP resta a cada observacion antes de multiplicar
    por el coeficiente (SHAP_i = coef_i * (x_i - media(x_i)) para un
    modelo lineal, la formula cerrada que usa `LinearExplainer`)."""
    info = resolver_algoritmo(algoritmo_raw)
    modelo = pipe.named_steps["modelo"]

    if info["familia"] == "arbol_nativo":
        x_test_shap = x_test
        if info["necesita_codificar_categoricas_shap"]:
            x_test_shap = x_test.copy()
            for c in cat_cols:
                if c in x_test_shap.columns:
                    x_test_shap[c] = x_test_shap[c].cat.codes.astype(float).replace(-1, np.nan)
        nombres = x_test_shap.columns
        if info["shap_explainer"] == "tree":
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(x_test_shap)
        else:
            raise ValueError(f"Familia arbol_nativo con shap_explainer={info['shap_explainer']!r} no soportado")
    else:  # preprocesador_clasico
        prep = pipe.named_steps["prep"]
        x_test_shap = prep.transform(x_test)
        nombres = prep.get_feature_names_out()
        if hasattr(x_test_shap, "toarray"):
            x_test_shap = x_test_shap.toarray()
        if info["shap_explainer"] == "tree":
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(x_test_shap)
        elif info["shap_explainer"] == "linear":
            x_train_shap = prep.transform(x_train)
            if hasattr(x_train_shap, "toarray"):
                x_train_shap = x_train_shap.toarray()
            explainer = shap.LinearExplainer(modelo, x_train_shap)
            shap_values = explainer.shap_values(x_test_shap)
        else:
            raise ValueError(f"shap_explainer desconocido: {info['shap_explainer']!r}")

    if isinstance(shap_values, list):  # algunos wrappers devuelven [neg, pos]
        shap_values = shap_values[1]
    if np.ndim(shap_values) == 3:  # (n, features, clases) -- clase positiva
        shap_values = shap_values[:, :, 1]

    return shap_values, nombres, x_test_shap


def main() -> None:
    filas_ranking = []
    filas_resumen = []

    for target, cfg in TARGETS.items():
        algoritmos_crudos = algoritmos_presentes_en_registro(cfg["registro"])
        print(f"\n{'#'*70}\nTarget {target} -- algoritmos detectados en {cfg['registro'].name}: "
              f"{algoritmos_crudos}\n{'#'*70}")
        registro = pd.read_csv(cfg["registro"])
        espec = cfg["espec_con_dmsp"]

        for algoritmo_raw in algoritmos_crudos:
            nombre_algo = resolver_algoritmo(algoritmo_raw)["nombre_bonito"]

            # No todos los algoritmos tienen fila para `espec` (la
            # especificacion CON DMSP-OLS) -- ej. Random Forest en el
            # registro monetario robusto solo se corrio bajo "A" (sin
            # DMSP), nunca bajo "AgeoDMSP". Esto es un hueco real de los
            # datos (esa combinacion nunca se entreno en la config
            # robusta), no un error de este script -- se documenta
            # explicitamente en vez de fallar o de omitirlo en silencio.
            existe = ((registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)).any()
            if not existe:
                print(f"\n=== {nombre_algo} -- {target} ({espec}) === OMITIDO: no hay fila "
                      f"para esta combinacion en {cfg['registro'].name} (nunca se entreno "
                      f"{algoritmo_raw!r} bajo {espec!r} en la config robusta)")
                filas_resumen.append({
                    "algoritmo": nombre_algo, "target": target, "especificacion": espec,
                    "n_variables_totales": None, "n_variables_dmsp": None,
                    "n_variables_dmsp_con_peso_no_cero": None, "n_variables_dmsp_indicadores_inertes": None,
                    "pct_shap_dmsp": None, "mejor_rank_dmsp": None, "mediana_rank_dmsp": None,
                    "omitido_sin_fila_en_registro": True,
                })
                continue

            print(f"\n=== {nombre_algo} -- {target} ({espec}) ===")
            pipe, x_train, x_test, y_test, cat_cols = entrenar(algoritmo_raw, espec, registro)
            print(f"  Calculando SHAP sobre {x_test.shape[0]} hogares de test...")
            shap_values, nombres, _ = calcular_shap(algoritmo_raw, pipe, x_train, x_test, cat_cols)

            importancia_media = pd.Series(np.abs(shap_values).mean(axis=0), index=nombres).sort_values(ascending=False)
            ranking = importancia_media.rank(ascending=False)

            for var, val in importancia_media.items():
                filas_ranking.append({
                    "algoritmo": nombre_algo, "target": target, "variable": var,
                    "shap_abs_medio": round(val, 6), "rank": int(ranking[var]),
                })

            dmsp_vars = [c for c in nombres if "dmsp_" in c]
            peso_total = importancia_media.sum()
            peso_dmsp = importancia_media[dmsp_vars].sum()
            pct_dmsp = 100 * peso_dmsp / peso_total if peso_total > 0 else float("nan")
            mejor_rank_dmsp = int(ranking[dmsp_vars].min()) if dmsp_vars else None
            mediana_rank_dmsp = float(ranking[dmsp_vars].median()) if dmsp_vars else None

            # `n_variables_dmsp` (arriba) cuenta COLUMNAS del espacio de
            # features que contienen "dmsp_" en el nombre -- para la familia
            # arbol_nativo eso son las 23 variables reales 1 a 1, pero para
            # preprocesador_clasico (Logistica/Random Forest) el
            # ColumnTransformer expande cada variable numerica en 2 columnas
            # (valor + indicador de faltante, SimpleImputer add_indicator=True)
            # y la categorica dmsp_saturado en 2 dummies -- 23 variables
            # reales terminan contadas como 46 columnas, SIN que hayan
            # entrado mas variables. La mayoria de esas columnas extra son
            # estructuralmente inertes: el indicador de faltante es una
            # constante (0) si DMSP-OLS no tiene NaN para estos hogares, y
            # un SHAP value de una columna constante es exactamente 0. Para
            # tener una cifra COMPARABLE entre familias de algoritmo, se
            # cuenta tambien cuantas de esas columnas dmsp_* tienen peso
            # SHAP no nulo (se noto una discrepancia 23 vs. 46).
            UMBRAL_PESO_NULO = 1e-9
            dmsp_vars_con_peso = [c for c in dmsp_vars if importancia_media[c] > UMBRAL_PESO_NULO]
            n_dmsp_indicadores_inertes = sum(1 for c in dmsp_vars if "missingindicator" in c)

            print(f"  Peso total |SHAP| de las {len(dmsp_vars)} columnas DMSP-OLS "
                  f"({len(dmsp_vars_con_peso)} con peso no nulo): {pct_dmsp:.2f}% del total")
            print(f"  Mejor rank DMSP-OLS: {mejor_rank_dmsp}/{len(importancia_media)}  Mediana rank: {mediana_rank_dmsp}")
            print(f"  Top 5 variables (|SHAP| medio): {importancia_media.head(5).index.tolist()}")

            filas_resumen.append({
                "algoritmo": nombre_algo, "target": target, "especificacion": espec,
                "n_variables_totales": len(importancia_media),
                "n_variables_dmsp": len(dmsp_vars),
                "n_variables_dmsp_con_peso_no_cero": len(dmsp_vars_con_peso),
                "n_variables_dmsp_indicadores_inertes": n_dmsp_indicadores_inertes,
                "pct_shap_dmsp": round(pct_dmsp, 2),
                "mejor_rank_dmsp": mejor_rank_dmsp, "mediana_rank_dmsp": mediana_rank_dmsp,
                "omitido_sin_fila_en_registro": False,
            })

    ranking_df = pd.DataFrame(filas_ranking)
    resumen_df = pd.DataFrame(filas_resumen)

    ruta_ranking = mu.RESULTADOS_DIR / "diagnostico_shap_importancia.csv"
    ruta_resumen = mu.RESULTADOS_DIR / "diagnostico_shap_resumen_dmsp.csv"
    ranking_df.to_csv(ruta_ranking, index=False)
    resumen_df.to_csv(ruta_resumen, index=False)

    print("\n\n=== Resumen final ===")
    print(resumen_df.to_string(index=False))
    print(f"\nCSV ranking completo: {ruta_ranking}")
    print(f"CSV resumen: {ruta_resumen}")


if __name__ == "__main__":
    main()
