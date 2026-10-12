"""
algoritmos_suite.py
=====================================================================
Registro CENTRAL de los 5 algoritmos de la suite (XGBoost, HistGradient-
Boosting, LightGBM, Random Forest, Logistica regularizada) -- creado
porque varios scripts de diagnostico (`diagnostico_bootstrap_dmsp.py`,
`diagnostico_bootstrap_ipm.py`, `diagnostico_shap.py`) mantenian cada
uno su PROPIA lista fija de "que algoritmos incluir" (`ALGOS = {...}`,
`COMBINACIONES = [...]`), curada a mano. Cuando se agregaron Random
Forest y LightGBM al registro de resultados (2026-08-30), ese hardcodeo
hizo que quedaran invisibles para el bootstrap de significancia y para
SHAP sin que se notara.

PRINCIPIO DE DISEÑO: ningun script debe mantener su propia lista de
algoritmos. En su lugar, cada script debe:
    1. Leer que algoritmos existen REALMENTE en el registro de resultados
       correspondiente (`registro["algoritmo"].unique()`).
    2. Buscar cada uno en `ALGORITMOS_SUITE` (este modulo).
    3. Si un algoritmo del registro NO esta en `ALGORITMOS_SUITE`,
       FALLAR RUIDOSAMENTE (`KeyError` con mensaje explicito) en vez de
       omitirlo en silencio -- ver `resolver_algoritmo()`.
Asi, agregar un algoritmo nuevo a la suite requiere UN solo cambio (este
archivo), y olvidarse de registrarlo aqui rompe el script con un error
claro en vez de dejarlo silenciosamente incompleto.

QUE EXPONE
---------------------------------------------------------------------
`ALGORITMOS_SUITE`: dict {nombre_crudo_en_registro: dict con}
    - "nombre_bonito": para imprimir/graficar.
    - "familia": "arbol_nativo" (XGBoost/HistGB/LightGBM -- sin imputar,
      categoricas nativas) o "preprocesador_clasico" (Random Forest/
      Logistica -- imputacion + one-hot via ColumnTransformer).
    - "construir_pipeline_fn": callable(x_train, y_train, balanceo,
      semilla) -> Pipeline, firma UNIFORME aunque cada modulo original
      (modelo_xgboost.py etc.) tenga firmas distintas entre si.
    - "shap_explainer": "tree" o "linear" -- que Explainer de la libreria
      `shap` usar.

`preparar_x_y(algoritmo_raw, df)`: dado un DataFrame (train o test) y el
    nombre crudo del algoritmo, devuelve (x, y, cat_cols) usando la ruta
    de preparacion correcta segun la familia (reusa `preparar_arboles_
    nativos`/`preparar_xy_crudo` de `modelo_utils.py`, no duplica).

`resolver_algoritmo(nombre_crudo)`: lookup con error explicito si el
    algoritmo no esta registrado.

`algoritmos_presentes_en_registro(ruta_registro)`: lee un CSV de
    registro y devuelve la lista de algoritmos crudos unicos presentes
    -- el punto de entrada que reemplaza las listas hardcodeadas.
"""

from pathlib import Path

import pandas as pd

import modelo_histgradientboosting as m_hgb
import modelo_lightgbm as m_lgbm
import modelo_logistica_regularizada as m_log
import modelo_random_forest as m_rf
import modelo_utils as mu
import modelo_xgboost as m_xgb

ALGORITMOS_SUITE = {
    "XGBoost": {
        "nombre_bonito": "XGBoost",
        "familia": "arbol_nativo",
        "construir_pipeline_fn": lambda x_train, y_train, balanceo, semilla: m_xgb.construir_pipeline(
            x_train, y_train, balanceo, semilla=semilla
        ),
        "shap_explainer": "tree",
        "necesita_codificar_categoricas_shap": False,
    },
    "HistGradientBoosting (sklearn)": {
        "nombre_bonito": "HistGradientBoosting",
        "familia": "arbol_nativo",
        "construir_pipeline_fn": lambda x_train, y_train, balanceo, semilla: m_hgb.construir_pipeline(
            balanceo, semilla=semilla
        ),
        "shap_explainer": "tree",
        "necesita_codificar_categoricas_shap": True,
    },
    "LightGBM": {
        "nombre_bonito": "LightGBM",
        "familia": "arbol_nativo",
        "construir_pipeline_fn": lambda x_train, y_train, balanceo, semilla: m_lgbm.construir_pipeline(
            x_train, balanceo, semilla=semilla
        ),
        "shap_explainer": "tree",
        # a diferencia de HistGB, el booster de LightGBM guarda internamente
        # que columnas son categoricas y espera verlas con el MISMO dtype
        # category en prediccion (incl. via shap.TreeExplainer) que en
        # entrenamiento -- codificarlas a .cat.codes (float) antes de tiempo
        # rompe la prediccion ("train and valid dataset categorical_feature
        # do not match"). Se dejan sin codificar, igual que XGBoost.
        "necesita_codificar_categoricas_shap": False,
    },
    "Random Forest": {
        "nombre_bonito": "Random Forest",
        "familia": "preprocesador_clasico",
        "construir_pipeline_fn": lambda x_train, y_train, balanceo, semilla: m_rf.construir_pipeline(
            x_train, balanceo, semilla=semilla
        ),
        "shap_explainer": "tree",
        "necesita_codificar_categoricas_shap": False,
    },
    "Logistica regularizada (elastic net, benchmark)": {
        "nombre_bonito": "Logistica",
        "familia": "preprocesador_clasico",
        "construir_pipeline_fn": lambda x_train, y_train, balanceo, semilla: m_log.construir_pipeline(
            x_train, balanceo, semilla=semilla
        ),
        "shap_explainer": "linear",
        "necesita_codificar_categoricas_shap": False,
    },
}


def resolver_algoritmo(nombre_crudo: str) -> dict:
    """Lookup con error EXPLICITO si el algoritmo no esta registrado --
    en vez de dejarlo fuera en silencio (la causa raiz del problema que
    motivo este modulo)."""
    if nombre_crudo not in ALGORITMOS_SUITE:
        disponibles = ", ".join(repr(k) for k in ALGORITMOS_SUITE)
        raise KeyError(
            f"Algoritmo {nombre_crudo!r} no esta registrado en ALGORITMOS_SUITE "
            f"(src/05_model/algoritmos_suite.py). Algoritmos conocidos: {disponibles}. "
            f"Si es un algoritmo nuevo de la suite, agregalo a ALGORITMOS_SUITE antes "
            f"de continuar -- no se permite omitirlo en silencio."
        )
    return ALGORITMOS_SUITE[nombre_crudo]


def algoritmos_presentes_en_registro(ruta_registro: Path) -> list:
    """Lee un registro_modelos*.csv y devuelve los algoritmos crudos
    unicos presentes -- reemplaza cualquier lista `ALGOS = {...}`
    hardcodeada: el universo de algoritmos a procesar lo define SIEMPRE
    el registro de resultados, nunca una lista mantenida a mano en el
    script de diagnostico."""
    registro = pd.read_csv(ruta_registro)
    crudos = registro["algoritmo"].unique().tolist()
    for c in crudos:
        resolver_algoritmo(c)  # valida que todos esten registrados, falla ruidoso si no
    return crudos


def preparar_x_y(algoritmo_raw: str, train: pd.DataFrame, test: pd.DataFrame):
    """Prepara (x_train, y_train, x_test, y_test, cat_cols) con la ruta
    correcta segun la familia del algoritmo -- reusa `preparar_arboles_
    nativos`/`preparar_xy_crudo`/`alinear_columnas_categoricas` de
    `modelo_utils.py`, no duplica esa logica."""
    familia = resolver_algoritmo(algoritmo_raw)["familia"]
    if familia == "arbol_nativo":
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)
        return x_train, y_train, x_test, y_test, cat_cols
    else:
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)
        return x_train, y_train, x_test, y_test, []


def filtrar_params_modelo(hiperparametros_json: str) -> dict:
    """Las claves guardadas en `hiperparametros` incluyen una entrada
    decorativa extra (scale_pos_weight/class_weight, string descriptivo)
    ademas de los parametros reales del pipeline (prefijo `modelo__`) --
    solo estas ultimas son validas para `set_params`. (Copia literal de
    la funcion ya usada en diagnostico_bootstrap_dmsp.py -- se centraliza
    aqui para que los scripts que usan este modulo no la dupliquen.)"""
    import json
    d = json.loads(hiperparametros_json)
    return {k: v for k, v in d.items() if k.startswith("modelo__")}
