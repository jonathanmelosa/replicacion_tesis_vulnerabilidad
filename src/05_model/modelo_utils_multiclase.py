"""
modelo_utils_multiclase.py
=============================

Extension de `modelo_utils.py` (que NO se modifica) a las 4 categorias de
la matriz de transicion de pobreza (nunca_pobre/entra/sale/siempre_pobre),
sobre los datasets de `build_benchmark_train_test_4clases.py` -- ver ese
script para la motivacion y el diseno.

Reutiliza de `modelo_utils.py` (importado como `mu`, sin modificarlo) lo
que es mecanicamente identico sin importar el tipo de target:
`construir_preprocesador` (0+indicador/one-hot) y
`alinear_columnas_categoricas`. Todo lo que depende de que el target sea
binario (umbral de clasificacion, scale_pos_weight, AUC-ROC binario,
`registro_modelos.csv`) se reimplementa aqui para multiclase, en un
registro y unos archivos de salida SEPARADOS -- nunca se compara cifra a
cifra contra el benchmark binario (esquemas de evaluacion distintos).

Codificacion de Y_grupo
-----------------------
Se usa un mapeo ENTERO EXPLICITO (no LabelEncoder alfabetico, para que el
indice de cada clase sea auditable a simple vista en el codigo, decision
deliberada dado que varias metricas necesitan referirse puntualmente a la
clase "entra" y al par "entra"/"sale"):

    0 = nunca_pobre, 1 = entra, 2 = sale, 3 = siempre_pobre

Balanceo de clases en multiclase
---------------------------------
Mismas 3 estrategias que el binario (BALANCEOS): "balanced" (reweighting),
"ninguno" (baseline), "oversampling" (RandomOverSampler, balancea TODAS
las clases a la de mayor tamano -- soporta multiclase nativamente).
`class_weight="balanced"` es nativo en HistGB/RandomForest/LogisticRegression/
LightGBM para cualquier numero de clases. XGBoost NO tiene `class_weight`
(solo `scale_pos_weight`, binario) -- para "balanced" con XGBoost se
calcula `sample_weight` via `compute_sample_weight("balanced", y)` y se
pasa como fit_param (`modelo__sample_weight`) a traves de
`RandomizedSearchCV.fit(..., **fit_params)`; sklearn (>=0.24, aqui 1.6.1)
recorta automaticamente los arrays de fit_params a los indices de train de
cada fold via `_check_fit_params`, asi que el peso queda correctamente
alineado por fold -- NO se usa junto con oversampling (evita doble ajuste,
mismo criterio que ya aplica `modelo_logistica_regularizada.py` en el
binario).

Metrica de seleccion (RandomizedSearchCV / comparacion de balanceo):
`roc_auc_ovr` (AUC one-vs-rest promedio macro, scorer nativo de sklearn)
-- analogo multiclase de `SCORING="roc_auc"` en el binario, mismo
argumento (umbral-independiente).

Regla de decision y metricas de evaluacion
--------------------------------------------
En multiclase la prediccion natural es el argmax de las 4 probabilidades
-- NO hay un unico umbral que elegir por CV como en el binario (ver
docstring de `modelo_utils.py`, "Umbral de clasificacion"). Se reportan:
  - accuracy, balanced_accuracy, f1_macro, f1_weighted (desempeno global)
  - precision/recall/f1 POR CLASE (en particular para "entra", la clase
    de interes de la tesis)
  - auc_ovr_macro (promedio de las 4 AUC one-vs-rest)
  - auc_entra_vs_resto: AUC one-vs-rest especificamente para "entra"
    (P(entra) vs. el resto) -- el analogo mas cercano al AUC-ROC del
    benchmark binario, aunque NO es la misma cantidad.
  - auc_entra_vs_sale: AUC restringido a las filas de "entra"+"sale"
    (score = P(entra) en ese subconjunto) -- la metrica que responde
    directamente la pregunta de la Seccion 5.1 (que tan separable es la
    frontera entra/sale con estas covariables), no reportada en el
    binario porque "sale" no estaba en esos datos.
  - precision_top10_entra: analogo de `precision_top_k` del binario,
    aplicado a P(entra) sobre TODA la muestra de evaluacion.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu  # reutiliza construir_preprocesador / alinear_columnas_categoricas

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"
RESULTADOS_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "multiclase"
REGISTRO_CSV = RESULTADOS_DIR / "registro_modelos_multiclase.csv"
REGISTRO_XLSX = RESULTADOS_DIR / "registro_modelos_multiclase.xlsx"

COLS_NO_FEATURE = ["consecutivo", "consecutivo_c", "llave_compuesta", "Y_grupo"]

CATEGORIAS_Y_GRUPO = ["nunca_pobre", "entra", "sale", "siempre_pobre"]
GRUPO_A_ENTERO = {g: i for i, g in enumerate(CATEGORIAS_Y_GRUPO)}
IDX_ENTRA = GRUPO_A_ENTERO["entra"]
IDX_SALE = GRUPO_A_ENTERO["sale"]

# Pista principal: holdout temporal 2010->2013 / 2013->2016 (A4/B4 solo
# ELCA, +geoDMSP con DMSP-OLS -- unica fuente con cobertura real en ambas
# olas, ver construir_pipeline_geo_dmsp_4clases.py).
ESPECIFICACIONES_4CLASES_PRINCIPAL = ["A4", "B4", "A4geoDMSP", "B4geoDMSP"]
# Pista exploratoria: CV dentro de 2010->2013, sin holdout (ALOS PALSAR y
# Landsat 5 TM solo tienen datos reales en 2010, ver
# construir_pipeline_geo3_cv_4clases.py) -- NO comparable cifra a cifra
# contra la pista principal.
ESPECIFICACIONES_4CLASES_CV = ["A4geo3", "B4geo3"]

BALANCEOS = ["balanced", "ninguno", "oversampling"]
SCORING = "roc_auc_ovr"
RANDOM_STATE = 42
SEMILLAS = [42, 1, 2, 3, 4]

FRACCION_TOP_K = 0.10

METRICAS_RESUMEN = [
    "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
    "precision_entra", "recall_entra", "f1_entra",
    "auc_ovr_macro", "auc_entra_vs_resto", "auc_entra_vs_sale",
    "precision_top10_entra",
]

COLUMNAS_REGISTRO = [
    "algoritmo", "especificacion", "fecha_entrenamiento",
    "n_train", "n_test", "n_covariables_originales", "n_covariables_modelo",
    "distribucion_train", "distribucion_test",
    "balanceo_elegido", "auc_cv_balanced", "auc_cv_ninguno", "auc_cv_oversampling",
    "n_semillas",
] + [f"{m}_{suf}" for m in METRICAS_RESUMEN for suf in ("media", "std", "ci95_low", "ci95_high")] + [
    "estrategia_imputacion", "hiperparametros", "observaciones",
]


def cargar_datos_4clases(especificacion: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """especificacion en ESPECIFICACIONES_4CLASES_PRINCIPAL. Train =
    2010->2013, test = 2013->2016 (holdout temporal, igual convencion que
    mu.cargar_datos)."""
    train = pd.read_parquet(DATA_DIR / f"modelo_{especificacion}_2010_2013.parquet")
    test = pd.read_parquet(DATA_DIR / f"modelo_{especificacion}_2013_2016.parquet")
    return train, test


def cargar_datos_cv_4clases(especificacion: str) -> pd.DataFrame:
    """especificacion en ESPECIFICACIONES_4CLASES_CV. Un solo archivo, la
    transicion 2010->2013 completa (igual convencion que mu.cargar_datos_cv)."""
    return pd.read_parquet(DATA_DIR / f"modelo_{especificacion}_2010_2013.parquet")


def _codificar_y(y_grupo: pd.Series) -> pd.Series:
    y_str = y_grupo.astype(str)
    faltantes = set(y_str.unique()) - set(GRUPO_A_ENTERO)
    if faltantes:
        raise ValueError(f"Valores de Y_grupo no reconocidos: {faltantes}")
    return y_str.map(GRUPO_A_ENTERO).astype(int)


def preparar_arboles_nativos_multiclase(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Analogo de mu.preparar_arboles_nativos, con Y_grupo codificada a
    entero (ver docstring del modulo) en vez de la columna binaria Y."""
    y = _codificar_y(df["Y_grupo"])
    x = df.drop(columns=[c for c in COLS_NO_FEATURE if c in df.columns])
    cols_categoricas = x.select_dtypes(include="object").columns.tolist()
    for c in cols_categoricas:
        x[c] = x[c].astype("category")
    return x, y, cols_categoricas


def preparar_xy_crudo_multiclase(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Analogo de mu.preparar_xy_crudo -- para pasar a un Pipeline con
    imputacion/encoding/escalado (logistica, random forest)."""
    y = _codificar_y(df["Y_grupo"])
    x = df.drop(columns=[c for c in COLS_NO_FEATURE if c in df.columns])
    return x, y


def comparar_balanceo_y_tunear_multiclase(
    construir_pipeline_fn,
    param_distributions_fn,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fit_params_fn=None,
    cv_folds: int = mu.CV_FOLDS,
    n_iter_busqueda: int = mu.N_ITER_BUSQUEDA,
    verbose: int = 0,
    random_state: int = RANDOM_STATE,
    balanceos: list = BALANCEOS,
) -> dict:
    """Identico en estructura a mu.comparar_balanceo_y_tunear, pero
    `scoring=SCORING` ("roc_auc_ovr") y con soporte opcional de
    `fit_params_fn(balanceo) -> dict|None` (usado por XGBoost para pasar
    `modelo__sample_weight` en la estrategia "balanced" -- ver docstring
    del modulo). `balanceos` permite acotar la lista (usado por la red
    neuronal, que no soporta "balanced" -- ver pipeline_nn en
    modelo_multiclase_robusto_comparacion.py)."""
    resultados = {}
    mejor_balanceo, mejor_score = None, -np.inf
    mejor_estimador, mejor_params = None, {}

    for balanceo in balanceos:
        pipeline = construir_pipeline_fn(balanceo)
        param_dist = param_distributions_fn(balanceo)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        fit_params = fit_params_fn(balanceo) if fit_params_fn else None
        fit_params = fit_params or {}

        if param_dist:
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_dist, n_iter=n_iter_busqueda,
                scoring=SCORING, cv=cv, random_state=random_state, n_jobs=-1, refit=True,
                verbose=verbose,
            )
            search.fit(x_train, y_train, **fit_params)
            score, mejores_params, estimador = search.best_score_, search.best_params_, search.best_estimator_
        else:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(pipeline, x_train, y_train, scoring=SCORING, cv=cv, n_jobs=-1, params=fit_params or None)
            score, mejores_params = float(scores.mean()), {}
            estimador = pipeline.fit(x_train, y_train, **fit_params)

        resultados[balanceo] = {"auc_cv": round(float(score), 4), "mejores_params": mejores_params}
        if score > mejor_score:
            mejor_balanceo, mejor_score, mejor_estimador, mejor_params = balanceo, score, estimador, mejores_params

    return {
        "balanceo_elegido": mejor_balanceo,
        "auc_cv_por_balanceo": {b: resultados[b]["auc_cv"] for b in balanceos},
        "mejores_params": mejor_params,
        "estimador": mejor_estimador,
    }


def precision_top_k(y_true_bin, score, frac: float = FRACCION_TOP_K) -> float:
    y_arr = np.asarray(y_true_bin)
    score_arr = np.asarray(score)
    n_top = max(1, int(np.ceil(len(score_arr) * frac)))
    idx_top = np.argsort(score_arr)[::-1][:n_top]
    return float(y_arr[idx_top].mean())


def calcular_metricas_multiclase(y_true: np.ndarray, proba: np.ndarray, clases_estimador: list) -> dict:
    """`clases_estimador` = classes_ del estimador (orden de columnas de
    `proba`) -- se reindexa a CATEGORIAS_Y_GRUPO (0..3) para que
    IDX_ENTRA/IDX_SALE sean validos sin importar el orden interno que use
    cada libreria."""
    orden = [clases_estimador.index(i) for i in range(len(CATEGORIAS_Y_GRUPO))]
    proba = proba[:, orden]  # ahora proba[:, k] = P(clase k) para k en GRUPO_A_ENTERO

    y_pred = proba.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_entra": precision_score(y_true, y_pred, labels=[IDX_ENTRA], average="macro", zero_division=0),
        "recall_entra": recall_score(y_true, y_pred, labels=[IDX_ENTRA], average="macro", zero_division=0),
        "f1_entra": f1_score(y_true, y_pred, labels=[IDX_ENTRA], average="macro", zero_division=0),
        "auc_ovr_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(len(CATEGORIAS_Y_GRUPO)))),
        "auc_entra_vs_resto": roc_auc_score((y_true == IDX_ENTRA).astype(int), proba[:, IDX_ENTRA]),
        "precision_top10_entra": precision_top_k((y_true == IDX_ENTRA).astype(int), proba[:, IDX_ENTRA]),
    }

    mask_par = np.isin(y_true, [IDX_ENTRA, IDX_SALE])
    if mask_par.sum() > 0 and len(np.unique(y_true[mask_par])) == 2:
        y_par = (y_true[mask_par] == IDX_ENTRA).astype(int)
        score_par = proba[mask_par, IDX_ENTRA]
        metrics["auc_entra_vs_sale"] = roc_auc_score(y_par, score_par)
    else:
        metrics["auc_entra_vs_sale"] = float("nan")

    return metrics


def evaluar_multiples_semillas_multiclase(
    construir_pipeline_fn,
    mejores_params: dict,
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    fit_params_fn=None,
    semillas: list = SEMILLAS,
) -> dict:
    """Analogo de mu.evaluar_multiples_semillas: reentrena el modelo FINAL
    (balanceo/mejores_params ya elegidos) con cada semilla, mide sobre
    x_test/y_test (holdout temporal). Sin umbral -- decision por argmax
    (ver docstring del modulo)."""
    filas = []
    for semilla in semillas:
        pipe = construir_pipeline_fn(semilla)
        if mejores_params:
            pipe.set_params(**mejores_params)
        fit_params = fit_params_fn(semilla) if fit_params_fn else {}
        pipe.fit(x_train, y_train, **(fit_params or {}))

        proba_test = pipe.predict_proba(x_test)
        clases = list(pipe.classes_)
        metricas = calcular_metricas_multiclase(np.asarray(y_test), proba_test, clases)
        filas.append({"semilla": semilla, **metricas})

    detalle = pd.DataFrame(filas)
    resumen = _resumir_semillas(detalle, len(semillas))
    return {"detalle": detalle, "resumen": resumen}


def evaluar_cv_semillas_multiclase(
    construir_pipeline_fn,
    mejores_params: dict,
    x: pd.DataFrame, y: pd.Series,
    fit_params_fn=None,
    semillas: list = SEMILLAS,
    cv_folds: int = mu.CV_FOLDS,
) -> dict:
    """Analogo de mu.evaluar_cv_semillas -- para especificaciones SIN
    holdout (ESPECIFICACIONES_4CLASES_CV): probabilidades OUT-OF-FOLD
    (cross_val_predict) dentro de la misma muestra."""
    filas = []
    for semilla in semillas:
        pipe = construir_pipeline_fn(semilla)
        if mejores_params:
            pipe.set_params(**mejores_params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=semilla)
        fit_params = fit_params_fn(semilla) if fit_params_fn else None
        proba_oof = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba", n_jobs=-1, params=fit_params)

        pipe_ref = construir_pipeline_fn(semilla)
        if mejores_params:
            pipe_ref.set_params(**mejores_params)
        pipe_ref.fit(x, y, **(fit_params or {}))
        clases = list(pipe_ref.classes_)

        metricas = calcular_metricas_multiclase(np.asarray(y), proba_oof, clases)
        filas.append({"semilla": semilla, **metricas})

    detalle = pd.DataFrame(filas)
    resumen = _resumir_semillas(detalle, len(semillas))
    return {"detalle": detalle, "resumen": resumen}


def _resumir_semillas(detalle: pd.DataFrame, n: int) -> dict:
    t_mult = float(scipy_stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    resumen = {}
    for metrica in METRICAS_RESUMEN:
        vals = detalle[metrica].dropna()
        media = float(vals.mean()) if len(vals) else float("nan")
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        margen = t_mult * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        resumen[metrica] = {
            "media": round(media, 4), "std": round(std, 4),
            "ci95_low": round(media - margen, 4), "ci95_high": round(media + margen, 4),
        }
    return resumen


def sample_weight_balanced(y) -> np.ndarray:
    """Para XGBoost, que no tiene `class_weight` (ver docstring del
    modulo)."""
    return compute_sample_weight("balanced", y)


def registrar_resultado_multiclase(
    algoritmo: str,
    especificacion: str,
    x_train_shape: tuple,
    x_test_shape: tuple,
    n_covariables_originales: int,
    y_train: pd.Series,
    y_test: pd.Series,
    multi_resultado: dict,
    estrategia_imputacion: str,
    balanceo_info: dict,
    hiperparametros: dict,
    observaciones: str,
    registro_csv: Path = REGISTRO_CSV,
    registro_xlsx: Path = REGISTRO_XLSX,
) -> None:
    registro_csv.parent.mkdir(parents=True, exist_ok=True)

    def _distribucion(y):
        return json.dumps({CATEGORIAS_Y_GRUPO[k]: int((y == k).sum()) for k in range(len(CATEGORIAS_Y_GRUPO))})

    auc_cv = balanceo_info.get("auc_cv_por_balanceo", {})
    resumen = multi_resultado["resumen"]
    fila = {
        "algoritmo": algoritmo,
        "especificacion": especificacion,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_train": x_train_shape[0],
        "n_test": x_test_shape[0],
        "n_covariables_originales": n_covariables_originales,
        "n_covariables_modelo": x_train_shape[1],
        "distribucion_train": _distribucion(np.asarray(y_train)),
        "distribucion_test": _distribucion(np.asarray(y_test)),
        "balanceo_elegido": balanceo_info.get("balanceo_elegido", ""),
        "auc_cv_balanced": auc_cv.get("balanced", ""),
        "auc_cv_ninguno": auc_cv.get("ninguno", ""),
        "auc_cv_oversampling": auc_cv.get("oversampling", ""),
        "n_semillas": len(SEMILLAS),
        "estrategia_imputacion": estrategia_imputacion,
        "hiperparametros": json.dumps(hiperparametros, ensure_ascii=False, default=str),
        "observaciones": observaciones,
    }
    for metrica in METRICAS_RESUMEN:
        m = resumen[metrica]
        fila[f"{metrica}_media"] = m["media"]
        fila[f"{metrica}_std"] = m["std"]
        fila[f"{metrica}_ci95_low"] = m["ci95_low"]
        fila[f"{metrica}_ci95_high"] = m["ci95_high"]

    if registro_csv.exists():
        registro = pd.read_csv(registro_csv)
        registro = registro[~((registro["algoritmo"] == algoritmo) & (registro["especificacion"] == especificacion))]
        registro = pd.concat([registro, pd.DataFrame([fila])], ignore_index=True)
    else:
        registro = pd.DataFrame([fila])

    registro = registro[COLUMNAS_REGISTRO].sort_values(["algoritmo", "especificacion"]).reset_index(drop=True)
    registro.to_csv(registro_csv, index=False)
    _regenerar_excel(registro, registro_xlsx)


def _regenerar_excel(registro: pd.DataFrame, registro_xlsx: Path = REGISTRO_XLSX) -> None:
    with pd.ExcelWriter(registro_xlsx, engine="openpyxl") as writer:
        registro.to_excel(writer, sheet_name="Registro modelos multiclase", index=False)
        ws = writer.sheets["Registro modelos multiclase"]
        ws.freeze_panes = "A2"
        for col_cells in ws.columns:
            letra = col_cells[0].column_letter
            largo = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[letra].width = min(max(largo + 2, 10), 60)
        from openpyxl.styles import Font
        for cell in ws[1]:
            cell.font = Font(bold=True)
