"""
Version IPM de `diagnostico_bootstrap_cluster_dmsp.py` -- mismo bootstrap
por comunidad (clusters, no hogares sueltos), aplicado al hallazgo
significativo de `diagnostico_bootstrap_ipm.py` (XGBoost, target IPM:
Aipm->AipmgeoDMSP +0.0128 p=0.002; Bipm->BipmgeoDMSP +0.0198 p<0.001).
Misma regla para `consecutivo_c` == 8888888 (comunidad no identificada):
clusters de tamaño 1.

Usa las predicciones ya generadas por `generar_predicciones_test_ipm.py`
-- no reentrena nada.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_bootstrap_cluster_ipm.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_bootstrap_cluster_ipm.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import modelo_utils as mu

N_BOOT = 2000
RNG = np.random.default_rng(mu.RANDOM_STATE)
SENTINEL_COMUNIDAD = 8888888


def asignar_cluster_id(df: pd.DataFrame) -> pd.Series:
    es_valido = df["consecutivo_c"] != SENTINEL_COMUNIDAD
    cluster_id = df["consecutivo_c"].astype(str).copy()
    cluster_id[~es_valido] = "singleton_" + df.loc[~es_valido, "consecutivo"].astype(str)
    return cluster_id


def bootstrap_cluster_delta_auc(df: pd.DataFrame, n_boot: int = N_BOOT):
    clusters_unicos = df["cluster_id"].unique()
    n_clusters = len(clusters_unicos)
    grupos = {c: sub for c, sub in df.groupby("cluster_id")}

    deltas = np.empty(n_boot)
    intentos = 0
    b = 0
    while b < n_boot:
        intentos += 1
        elegidos = RNG.choice(clusters_unicos, size=n_clusters, replace=True)
        partes = [grupos[c] for c in elegidos]
        pool = pd.concat(partes, ignore_index=True)
        y = pool["Y"].values
        if y.sum() == 0 or y.sum() == len(y):
            if intentos > n_boot * 20:
                raise RuntimeError("Demasiadas remuestras degeneradas (sin ambas clases) -- revisar cluster_id")
            continue
        auc_base = roc_auc_score(y, pool["proba_base"].values)
        auc_geo = roc_auc_score(y, pool["proba_geo"].values)
        deltas[b] = auc_geo - auc_base
        b += 1
    return deltas


def main() -> None:
    filas = []
    for base, geo in [("Aipm", "AipmgeoDMSP"), ("Bipm", "BipmgeoDMSP")]:
        pred = pd.read_parquet(mu.RESULTADOS_DIR / f"predicciones_test_ipm_{base}.parquet")
        pred["cluster_id"] = asignar_cluster_id(pred)
        n_clusters = pred["cluster_id"].nunique()
        print(f"\n=== {base} vs {geo} -- {pred['consecutivo'].nunique()} hogares, {n_clusters} clusters ===")

        for algoritmo in pred["algoritmo"].unique():
            sub = pred[pred["algoritmo"] == algoritmo].reset_index(drop=True)
            y = sub["Y"].values
            auc_base_obs = roc_auc_score(y, sub["proba_base"].values)
            auc_geo_obs = roc_auc_score(y, sub["proba_geo"].values)
            delta_obs = auc_geo_obs - auc_base_obs

            deltas = bootstrap_cluster_delta_auc(sub)
            ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
            p_valor = min(2 * min((deltas > 0).mean(), (deltas < 0).mean()), 1.0)

            print(f"  {algoritmo:22s} AUC {base}={auc_base_obs:.4f}  AUC {geo}={auc_geo_obs:.4f}  delta={delta_obs:+.4f}  IC95%(cluster)=[{ci_low:+.4f},{ci_high:+.4f}]  p={p_valor:.3f}  {'cruza 0' if ci_low<=0<=ci_high else 'NO cruza 0'}")

            filas.append({
                "algoritmo": algoritmo, "especificacion_base": base, "especificacion_geo": geo,
                "n_hogares": len(sub), "n_clusters": n_clusters,
                "auc_base": round(auc_base_obs, 4), "auc_geo": round(auc_geo_obs, 4),
                "delta_auc": round(delta_obs, 4),
                "ci95_low_cluster": round(ci_low, 4), "ci95_high_cluster": round(ci_high, 4),
                "p_valor_cluster": round(p_valor, 4), "cruza_cero": bool(ci_low <= 0 <= ci_high),
            })

    out = pd.DataFrame(filas)
    out_path = mu.RESULTADOS_DIR / "diagnostico_bootstrap_cluster_ipm.csv"
    out.to_csv(out_path, index=False)
    print("\n=== Resumen ===")
    print(out.to_string(index=False))
    print(f"\nCSV: {out_path}")


if __name__ == "__main__":
    main()
