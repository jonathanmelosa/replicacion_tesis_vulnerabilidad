"""
diagnostico_bootstrap_cluster_multiclase.py
=====================================
Bootstrap pareado por comunidad para los AUC por par del modelo multiclase
(Modelo B, prueba 2013->2016): intervalo de confianza de cada AUC por par
en B4 y del cambio al agregar DMSP-OLS (B4geoDMSP - B4).

Motivo: los intervalos de 5 semillas de
`auc_pares_multiclase_semillas.py` solo miden el azar del ENTRENAMIENTO, y
en logistica, HistGradientBoosting y LightGBM son degenerados (desviacion
0: con los hiperparametros elegidos no usan azar). La incertidumbre
relevante para afirmar que DMSP-OLS no mejora la separacion es la de la
muestra de prueba, que es la que mide este bootstrap. Misma convencion que
el bootstrap del modelo binario (`diagnostico_bootstrap_cluster_dmsp.py`):
se remuestrean comunidades completas (`consecutivo_c`) con reemplazo; los
hogares con comunidad no identificada (8888888) son clusters de tamano 1;
N_BOOT = 2000; IC95 por percentiles; p-valor bilateral
2 * min(P(delta > 0), P(delta < 0)).

Pareado: en cada remuestra se usan los MISMOS hogares para B4 y
B4geoDMSP, asi que la diferencia no mezcla la variacion de la muestra.

Usa las probabilidades de prueba ya guardadas (semilla 42) en
`multiclase/predicciones/` por `modelo_multiclase_robusto_comparacion.py`
-- no reentrena nada. Para B4 y B4geoDMSP son predicciones fuera de
muestra (hogares de 2013->2016 que el modelo no vio).

INPUTS
    data/processed/benchmark_resultados/multiclase/predicciones/*_{B4,B4geoDMSP}.parquet
    data/processed/benchmark_train_test/modelo_B4_2013_2016.parquet (consecutivo_c)

OUTPUTS
    data/processed/benchmark_resultados/multiclase/diagnostico_bootstrap_cluster_multiclase.csv
        (una fila por algoritmo x par: AUC B4 con IC95, AUC B4geoDMSP,
        delta con IC95 y p-valor)

COMO CORRER
    cd src/05_model && python -u diagnostico_bootstrap_cluster_multiclase.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
import modelo_utils_multiclase as mcu
from diagnostico_bootstrap_cluster_dmsp import SENTINEL_COMUNIDAD

N_BOOT = 2000
RNG = np.random.default_rng(mu.RANDOM_STATE)
GRUPOS = list(mcu.CATEGORIAS_Y_GRUPO)
PARES = list(itertools.combinations(GRUPOS, 2))
DIR_PRED = mcu.RESULTADOS_DIR / "predicciones"
RUTA_SALIDA = mcu.RESULTADOS_DIR / "diagnostico_bootstrap_cluster_multiclase.csv"


def auc_par(y_grupo: np.ndarray, proba_a: np.ndarray, a: str, b: str) -> float:
    mask = (y_grupo == a) | (y_grupo == b)
    y = (y_grupo[mask] == a).astype(int)
    if y.min() == y.max():
        return float("nan")
    return roc_auc_score(y, proba_a[mask])


def cargar_pareado() -> pd.DataFrame:
    """Una fila por algoritmo x hogar, con las probabilidades de B4 y de
    B4geoDMSP lado a lado y el id de cluster."""
    comunidad = pd.read_parquet(
        mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_B4_2013_2016.parquet",
        columns=["consecutivo", "consecutivo_c"],
    )
    piezas = {}
    for espec in ["B4", "B4geoDMSP"]:
        for ruta in DIR_PRED.glob(f"*_{espec}.parquet"):
            df = pd.read_parquet(ruta)
            piezas.setdefault(espec, []).append(df)
    base = pd.concat(piezas["B4"], ignore_index=True)
    geo = pd.concat(piezas["B4geoDMSP"], ignore_index=True)
    prob = [f"prob_{g}" for g in GRUPOS]
    df = base.merge(geo[["algoritmo", "consecutivo", "Y_grupo"] + prob],
                    on=["algoritmo", "consecutivo"], suffixes=("_base", "_geo"), validate="one_to_one")
    if not (df["Y_grupo_base"] == df["Y_grupo_geo"]).all():
        raise ValueError("Y_grupo distinto entre B4 y B4geoDMSP para el mismo hogar")
    df = df.rename(columns={"Y_grupo_base": "Y_grupo"}).drop(columns="Y_grupo_geo")
    df = df.merge(comunidad, on="consecutivo", how="left", validate="many_to_one")
    es_valido = df["consecutivo_c"].notna() & (df["consecutivo_c"] != SENTINEL_COMUNIDAD)
    df["cluster_id"] = df["consecutivo_c"].astype(str)
    df.loc[~es_valido, "cluster_id"] = "singleton_" + df.loc[~es_valido, "consecutivo"].astype(str)
    return df


def main() -> None:
    df = cargar_pareado()
    filas = []
    for algoritmo, sub in df.groupby("algoritmo", sort=False):
        sub = sub.reset_index(drop=True)
        y_grupo = sub["Y_grupo"].to_numpy()
        indices = {c: idx.to_numpy() for c, idx in sub.groupby("cluster_id").groups.items()}
        clusters = np.array(list(indices))
        print(f"=== {algoritmo}: {len(sub)} hogares, {len(clusters)} clusters ===", flush=True)

        # Remuestras de clusters (las mismas para todos los pares y ambas especificaciones).
        remuestras = [np.concatenate([indices[c] for c in RNG.choice(clusters, size=len(clusters), replace=True)])
                      for _ in range(N_BOOT)]

        for a, b in PARES:
            p_base = sub[f"prob_{a}_base"].to_numpy()
            p_geo = sub[f"prob_{a}_geo"].to_numpy()
            auc_base = auc_par(y_grupo, p_base, a, b)
            auc_geo = auc_par(y_grupo, p_geo, a, b)
            boot_base = np.array([auc_par(y_grupo[i], p_base[i], a, b) for i in remuestras])
            boot_geo = np.array([auc_par(y_grupo[i], p_geo[i], a, b) for i in remuestras])
            validas = ~np.isnan(boot_base) & ~np.isnan(boot_geo)
            deltas = (boot_geo - boot_base)[validas]
            ci_base = np.percentile(boot_base[validas], [2.5, 97.5])
            ci_delta = np.percentile(deltas, [2.5, 97.5])
            p_valor = min(2 * min((deltas > 0).mean(), (deltas < 0).mean()), 1.0)
            filas.append({
                "algoritmo": algoritmo, "par": f"{a}_vs_{b}",
                "n_hogares": len(sub), "n_clusters": len(clusters), "n_boot_validas": int(validas.sum()),
                "auc_B4": round(auc_base, 4),
                "auc_B4_ci95_low": round(ci_base[0], 4), "auc_B4_ci95_high": round(ci_base[1], 4),
                "auc_B4geoDMSP": round(auc_geo, 4),
                "delta_auc": round(auc_geo - auc_base, 4),
                "delta_ci95_low": round(ci_delta[0], 4), "delta_ci95_high": round(ci_delta[1], 4),
                "p_valor": round(p_valor, 4), "cruza_cero": bool(ci_delta[0] <= 0 <= ci_delta[1]),
            })

    out = pd.DataFrame(filas)
    out.to_csv(RUTA_SALIDA, index=False)
    print(out.to_string(index=False))
    print(f"\nGuardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
