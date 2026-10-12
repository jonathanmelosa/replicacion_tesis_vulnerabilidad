"""
diagnostico_curva_pr.py
=======================
Metricas de precision-recall para el test 2013->2016, con y sin DMSP-OLS,
porque, con desbalance de
clases y pensando en focalizacion, la curva precision-recall debe tener
mas protagonismo que el AUC-ROC.

Por cada definicion (monetaria / IPM) x especificacion (A / B) x
algoritmo, y para el modelo sin luz (`proba_base`) y con luz
(`proba_geo`):
  - ap: AUC-PR como average precision (sklearn); la referencia de un
    modelo al azar es la prevalencia del test.
  - recall_f2 / precision_f2 / f1_f2 / marcados_f2 y la matriz de
    confusion (vp_f2, fp_f2, fn_f2, vn_f2): metricas al umbral de
    clasificacion elegido por validacion cruzada maximizando F2
    (`umbral_clasificacion_media` del registro de cada especificacion), la
    regla de operacion del paper.
  - precision_k10 / recall_k10: si el programa atiende al 10% de los
    hogares con mayor riesgo predicho (el "decil de mayor riesgo"), que
    fraccion de ellos entra en pobreza y que fraccion de todos los que
    entran queda atendida. Unico presupuesto.
  - delta_ap e IC95% por bootstrap pareado por COMUNIDAD (2,000
    remuestras), con la misma regla de `diagnostico_bootstrap_cluster_dmsp.py`:
    `consecutivo_c == 8888888` (comunidad no identificada) cuenta como
    cluster de tamaño 1. Tambien IC95% del AP del modelo sin luz.

Ademas guarda los puntos de la curva precision-recall (umbral a umbral)
para graficarla sin recalcular.

Usa las predicciones ya generadas por `generar_predicciones_test_dmsp.py`
y `generar_predicciones_test_ipm.py` (una sola semilla, 42) -- no
reentrena nada.

INPUTS

    data/processed/benchmark_resultados/predicciones_test_dmsp_{A,B}.parquet
    data/processed/benchmark_resultados/predicciones_test_ipm_{Aipm,Bipm}.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_curva_pr.csv
    data/processed/benchmark_resultados/diagnostico_curva_pr_puntos.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_curva_pr.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import modelo_utils as mu

N_BOOT = 2000
SENTINEL_COMUNIDAD = 8888888
PRESUPUESTOS = [0.10]

ARCHIVOS = [
    ("monetaria", "A", "predicciones_test_dmsp_A.parquet"),
    ("monetaria", "B", "predicciones_test_dmsp_B.parquet"),
    ("ipm", "A", "predicciones_test_ipm_Aipm.parquet"),
    ("ipm", "B", "predicciones_test_ipm_Bipm.parquet"),
]
# Registro y nombres de especificacion (sin luz, con luz) de donde sale el
# umbral F2 de cada modelo.
REGISTROS = {
    "monetaria": ("registro_modelos_fbeta2_cv10.csv", {"A": ("A", "AgeoDMSP"), "B": ("B", "BgeoDMSP")}),
    "ipm": ("registro_modelos_ipm.csv", {"A": ("Aipm", "AipmgeoDMSP"), "B": ("Bipm", "BipmgeoDMSP")}),
}
NOMBRES_CORTOS = ["Logistica", "Random Forest", "XGBoost", "LightGBM", "HistGradientBoosting"]


def nombre_corto(algoritmo_registro: str) -> str:
    for corto in NOMBRES_CORTOS:
        if algoritmo_registro.startswith(corto):
            return corto
    raise ValueError(algoritmo_registro)


def umbrales_f2() -> dict:
    """{(definicion, especificacion, algoritmo, modelo): umbral}, con modelo
    'base' (sin luz) o 'geo' (con luz)."""
    umbrales = {}
    for definicion, (archivo, especs) in REGISTROS.items():
        registro = pd.read_csv(mu.RESULTADOS_DIR / archivo)
        registro["corto"] = registro["algoritmo"].map(nombre_corto)
        for espec, (espec_base, espec_geo) in especs.items():
            for modelo, espec_reg in [("base", espec_base), ("geo", espec_geo)]:
                for _, fila in registro[registro["especificacion"] == espec_reg].iterrows():
                    umbrales[(definicion, espec, fila["corto"], modelo)] = float(fila["umbral_clasificacion_media"])
    return umbrales


def metricas_umbral(y: np.ndarray, proba: np.ndarray, umbral: float) -> dict:
    marcado = proba >= umbral
    vp = int((marcado & (y == 1)).sum())
    fp = int((marcado & (y == 0)).sum())
    fn = int((~marcado & (y == 1)).sum())
    vn = int((~marcado & (y == 0)).sum())
    recall, precision = vp / (vp + fn), vp / (vp + fp)
    return {"recall": recall, "precision": precision,
            "f1": 2 * precision * recall / (precision + recall), "marcados": marcado.mean(),
            "vp": vp, "fp": fp, "fn": fn, "vn": vn}


def asignar_cluster_id(df: pd.DataFrame) -> pd.Series:
    """Igual que en `diagnostico_bootstrap_cluster_dmsp.py`: comunidad real
    si es valida; si es el centinela, un cluster unitario por hogar."""
    es_valido = df["consecutivo_c"] != SENTINEL_COMUNIDAD
    cluster_id = df["consecutivo_c"].astype(str).copy()
    cluster_id[~es_valido] = "singleton_" + df.loc[~es_valido, "consecutivo"].astype(str)
    return cluster_id


def precision_cobertura(y: np.ndarray, proba: np.ndarray, presupuesto: float):
    """Atiende a los ceil(presupuesto * n) hogares de mayor riesgo."""
    k = int(np.ceil(presupuesto * len(y)))
    atendidos = np.argsort(-proba, kind="stable")[:k]
    aciertos = y[atendidos].sum()
    return aciertos / k, aciertos / y.sum()


def bootstrap_ap(y: np.ndarray, p_base: np.ndarray, p_geo: np.ndarray, clusters: np.ndarray, rng):
    """Bootstrap pareado por cluster: devuelve (ap_base, delta_ap) por remuestra."""
    codigos, inverso = np.unique(clusters, return_inverse=True)
    indices = [np.flatnonzero(inverso == c) for c in range(len(codigos))]
    ap_base = np.empty(N_BOOT)
    delta = np.empty(N_BOOT)
    b = 0
    while b < N_BOOT:
        elegidos = rng.integers(0, len(codigos), size=len(codigos))
        idx = np.concatenate([indices[c] for c in elegidos])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ap_base[b] = average_precision_score(yb, p_base[idx])
        delta[b] = average_precision_score(yb, p_geo[idx]) - ap_base[b]
        b += 1
    return ap_base, delta


def main() -> None:
    rng = np.random.default_rng(mu.RANDOM_STATE)
    umbral_por_modelo = umbrales_f2()
    filas, puntos = [], []
    for definicion, especificacion, archivo in ARCHIVOS:
        pred = pd.read_parquet(mu.RESULTADOS_DIR / archivo)
        pred["cluster_id"] = asignar_cluster_id(pred)
        print(f"\n=== {definicion} {especificacion}: {pred['consecutivo'].nunique()} hogares, "
              f"{pred['cluster_id'].nunique()} clusters ===")

        for algoritmo, sub in pred.groupby("algoritmo"):
            y = sub["Y"].to_numpy().astype(int)
            p_base, p_geo = sub["proba_base"].to_numpy(), sub["proba_geo"].to_numpy()
            ap_base_boot, delta_boot = bootstrap_ap(y, p_base, p_geo, sub["cluster_id"].to_numpy(), rng)

            fila = {
                "definicion": definicion, "especificacion": especificacion, "algoritmo": algoritmo,
                "n_hogares": len(y), "prevalencia": round(y.mean(), 4),
                "auc_roc_base": round(roc_auc_score(y, p_base), 4),
                "auc_roc_geo": round(roc_auc_score(y, p_geo), 4),
                "ap_base": round(average_precision_score(y, p_base), 4),
                "ap_base_ci95_low": round(np.percentile(ap_base_boot, 2.5), 4),
                "ap_base_ci95_high": round(np.percentile(ap_base_boot, 97.5), 4),
                "ap_geo": round(average_precision_score(y, p_geo), 4),
            }
            fila["delta_ap"] = round(fila["ap_geo"] - fila["ap_base"], 4)
            fila["delta_ap_ci95_low"] = round(np.percentile(delta_boot, 2.5), 4)
            fila["delta_ap_ci95_high"] = round(np.percentile(delta_boot, 97.5), 4)
            fila["delta_ap_cruza_cero"] = bool(fila["delta_ap_ci95_low"] <= 0 <= fila["delta_ap_ci95_high"])
            for modelo, proba in [("base", p_base), ("geo", p_geo)]:
                umbral = umbral_por_modelo[(definicion, especificacion, algoritmo, modelo)]
                fila[f"umbral_f2_{modelo}"] = round(umbral, 4)
                for nombre, valor in metricas_umbral(y, proba, umbral).items():
                    fila[f"{nombre}_f2_{modelo}"] = round(valor, 4)
            for presupuesto in PRESUPUESTOS:
                etiqueta = f"k{int(presupuesto * 100)}"
                for modelo, proba in [("base", p_base), ("geo", p_geo)]:
                    prec, rec = precision_cobertura(y, proba, presupuesto)
                    fila[f"precision_{etiqueta}_{modelo}"] = round(prec, 4)
                    fila[f"recall_{etiqueta}_{modelo}"] = round(rec, 4)
            filas.append(fila)

            print(f"  {algoritmo:22s} AP base={fila['ap_base']:.3f} [{fila['ap_base_ci95_low']:.3f},{fila['ap_base_ci95_high']:.3f}] "
                  f"geo={fila['ap_geo']:.3f} delta={fila['delta_ap']:+.3f} "
                  f"[{fila['delta_ap_ci95_low']:+.3f},{fila['delta_ap_ci95_high']:+.3f}] "
                  f"F2(u={fila['umbral_f2_base']:.3f}): recall={fila['recall_f2_base']:.3f} precision={fila['precision_f2_base']:.3f} "
                  f"marcados={fila['marcados_f2_base']:.2f} | 10%: precision={fila['precision_k10_base']:.2f} recall={fila['recall_k10_base']:.2f}")

            for modelo, proba in [("base", p_base), ("geo", p_geo)]:
                precision, recall, umbrales = precision_recall_curve(y, proba)
                puntos.append(pd.DataFrame({
                    "definicion": definicion, "especificacion": especificacion, "algoritmo": algoritmo,
                    "modelo": modelo, "precision": precision, "recall": recall,
                    "umbral": np.append(umbrales, np.nan),
                }))

    out = pd.DataFrame(filas)
    out.to_csv(mu.RESULTADOS_DIR / "diagnostico_curva_pr.csv", index=False)
    pd.concat(puntos, ignore_index=True).to_csv(mu.RESULTADOS_DIR / "diagnostico_curva_pr_puntos.csv", index=False)
    print(f"\nGuardado: {mu.RESULTADOS_DIR / 'diagnostico_curva_pr.csv'} y diagnostico_curva_pr_puntos.csv")


if __name__ == "__main__":
    main()
