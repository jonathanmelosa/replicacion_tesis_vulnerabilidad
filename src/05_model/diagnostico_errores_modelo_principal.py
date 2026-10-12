"""
diagnostico_errores_modelo_principal.py
=======================================
Analisis de errores del MODELO PRINCIPAL (elastic net, Modelo B, pobreza
monetaria, prueba 2013->2016) con dos reglas de seleccion:
  - umbral_f2 (regla principal del paper): marca a los hogares con
    probabilidad >= umbral elegido por validacion cruzada maximizando F2
    (`umbral_clasificacion_media` del elastic net B en
    `registro_modelos_fbeta2_cv10.csv`), la misma regla de
    `diagnostico_curva_pr.py`.
  - top10: el programa atiende al 10% de hogares con mayor riesgo
    predicho (complemento).

Reemplaza, para el cuerpo del paper, a `diagnostico_fp_vs_vn.py` y
`diagnostico_choques_fp_vs_vp.py`, que usan el Modelo A y los cinco
algoritmos. Esos scripts no se tocan.

Tres comparaciones:
  1. Falsos positivos (FP: atendidos que no son pobres en 2016) frente a
     verdaderos negativos (VN: no atendidos que no son pobres en 2016) en
     las 22 variables del nucleo SHAP, medidas en 2013. Ambos grupos
     tuvieron el mismo desenlace; solo los separa la decision del modelo.
  2. Verdaderos positivos (VP: atendidos que son pobres en 2016) frente a
     FP en los choques reportados en la ola de 2016, posteriores a la
     prediccion y que el modelo no usa.
  3. VP frente a FP en las variables del nucleo, medidas en 2013: si los
     FP son menos precarios que los VP, parte de la diferencia en choques
     puede reflejar esa diferencia de precariedad (cautela del texto).
  4. Respaldo de esa cautela: choques de 2016 por cuartil del indice de
     riqueza de 2013 entre los hogares que NO caen (si los mas precarios
     reportan mas choques aun sin caer, la precariedad confunde la
     comparacion 2).

Para cada variable numerica se reporta la media en cada grupo, la
diferencia y la diferencia estandarizada (diferencia / desviacion estandar
combinada), e IC95% de la diferencia por bootstrap por comunidad (2,000
remuestras, misma regla que `diagnostico_bootstrap_cluster_dmsp.py`:
`consecutivo_c == 8888888` cuenta como cluster de tamano 1). Promedios
simples, sin ponderar (igual que el resto de la evaluacion).

Usa las probabilidades ya guardadas por `generar_predicciones_test_dmsp.py`
(algoritmo "Logistica", columna `proba_base` = Modelo B sin DMSP); no
reentrena nada. Los valores de las variables se toman EN CRUDO de
`modelo_A_2013_2016.parquet` (mismos 3,191 hogares que el Modelo B; el A
incluye ademas el ingreso, util como descriptor).

INPUTS

    data/processed/benchmark_resultados/predicciones_test_dmsp_B.parquet
    data/processed/benchmark_resultados/diagnostico_shap_nucleo_perfil.csv
    data/processed/benchmark_train_test/modelo_A_2013_2016.parquet
    data/processed/choques_hogar_elca_longitudinal.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_errores_modelo_principal.csv
    (una fila por regla x comparacion x variable)
    data/processed/benchmark_resultados/diagnostico_errores_modelo_principal_conteos.csv
    (VP, FP, FN, VN por regla)
    data/processed/benchmark_resultados/diagnostico_errores_choques_por_riqueza.csv
    (choques 2016 por cuartil de riqueza 2013, hogares que no caen)

COMO CORRER

    cd src/05_model && python -u diagnostico_errores_modelo_principal.py
"""

import numpy as np
import pandas as pd

import modelo_utils as mu

ALGORITMO = "Logistica"
ALGORITMO_REGISTRO = "Logistica regularizada (elastic net, benchmark)"
ESPECIFICACION = "B"
REGISTRO_PATH = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv"
N_BOOT = 2000
SENTINEL_COMUNIDAD = 8888888
OLA_DESTINO = 3  # 2016

PRED_PATH = mu.RESULTADOS_DIR / "predicciones_test_dmsp_B.parquet"
NUCLEO_PATH = mu.RESULTADOS_DIR / "diagnostico_shap_nucleo_perfil.csv"
TEST_PATH = mu.DATA_DIR / "modelo_A_2013_2016.parquet"
CHOQUES_PATH = mu.PROJECT_ROOT / "data" / "processed" / "choques_hogar_elca_longitudinal.parquet"
COLS_NO_CHOQUE = ["consecutivo", "llave", "llave_n16", "ola", "zona"]

RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_errores_modelo_principal.csv"
RUTA_CONTEOS = mu.RESULTADOS_DIR / "diagnostico_errores_modelo_principal_conteos.csv"
RUTA_RIQUEZA = mu.RESULTADOS_DIR / "diagnostico_errores_choques_por_riqueza.csv"


def marcar_presupuesto(proba: np.ndarray, presupuesto: float) -> np.ndarray:
    """1 para los ceil(presupuesto * n) hogares de mayor riesgo (misma regla
    que `diagnostico_curva_pr.precision_cobertura`)."""
    k = int(np.ceil(presupuesto * len(proba)))
    marcado = np.zeros(len(proba), dtype=int)
    marcado[np.argsort(-proba, kind="stable")[:k]] = 1
    return marcado


def cluster_id(df: pd.DataFrame) -> pd.Series:
    es_valido = df["consecutivo_c"] != SENTINEL_COMUNIDAD
    ids = df["consecutivo_c"].astype(str).copy()
    ids[~es_valido] = "singleton_" + df.loc[~es_valido, "consecutivo"].astype(str)
    return ids


def ic_diferencia(valores: np.ndarray, en_a: np.ndarray, en_b: np.ndarray, clusters: np.ndarray, rng) -> tuple:
    """IC95% de media(A) - media(B) por bootstrap por cluster."""
    codigos, inverso = np.unique(clusters, return_inverse=True)
    indices = [np.flatnonzero(inverso == c) for c in range(len(codigos))]
    difs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([indices[c] for c in rng.integers(0, len(codigos), size=len(codigos))])
        v, a, bb = valores[idx], en_a[idx], en_b[idx]
        difs[b] = np.nanmean(v[a]) - np.nanmean(v[bb])
    return float(np.nanpercentile(difs, 2.5)), float(np.nanpercentile(difs, 97.5))


def comparar(df: pd.DataFrame, variables: list, grupo_a: str, grupo_b: str, regla: str, comparacion: str, rng) -> list:
    en_a = (df["celda"] == grupo_a).to_numpy()
    en_b = (df["celda"] == grupo_b).to_numpy()
    clusters = df["cluster_id"].to_numpy()
    filas = []
    for var in variables:
        valores = pd.to_numeric(df[var], errors="coerce").to_numpy(dtype=float)
        media_a, media_b = np.nanmean(valores[en_a]), np.nanmean(valores[en_b])
        sd = np.sqrt((np.nanvar(valores[en_a], ddof=1) + np.nanvar(valores[en_b], ddof=1)) / 2)
        lo, hi = ic_diferencia(valores, en_a, en_b, clusters, rng)
        filas.append({
            "regla": regla, "comparacion": comparacion, "variable": var,
            "grupo_a": grupo_a, "grupo_b": grupo_b,
            "media_a": round(media_a, 4), "media_b": round(media_b, 4),
            "diferencia_a_menos_b": round(media_a - media_b, 4),
            "diferencia_estandarizada": round((media_a - media_b) / sd, 3) if sd > 0 else np.nan,
            "ic95_low": round(lo, 4), "ic95_high": round(hi, 4),
            "ic_cruza_cero": bool(lo <= 0 <= hi),
            "n_a": int(en_a.sum()), "n_b": int(en_b.sum()),
        })
    return filas


def main() -> None:
    rng = np.random.default_rng(mu.RANDOM_STATE)

    pred = pd.read_parquet(PRED_PATH)
    pred = pred[pred["algoritmo"] == ALGORITMO][["consecutivo", "consecutivo_c", "Y", "proba_base"]]
    test = pd.read_parquet(TEST_PATH)
    nucleo = pd.read_csv(NUCLEO_PATH)["variable"].tolist()
    numericas = [v for v in nucleo if test[v].dtype != object]

    choques = pd.read_parquet(CHOQUES_PATH)
    # En 2016 hay `consecutivo` repetidos (hogares divididos), pero ninguno
    # pertenece al conjunto de prueba, que excluye a los hogares divididos;
    # se filtra antes de unir y la union valida que sea uno a uno.
    choques = choques[(choques["ola"] == OLA_DESTINO) & choques["consecutivo"].isin(pred["consecutivo"])]
    # Sufijo _2016: `tuvo_choque_economico_hogar` tambien esta en el nucleo,
    # medida en 2013; sin el sufijo las dos se confundirian al unir.
    choques = choques.rename(columns={c: f"{c}_2016" for c in choques.columns if c not in COLS_NO_CHOQUE})
    cols_choque = [c for c in choques.columns if c not in COLS_NO_CHOQUE]

    base = (pred.merge(test[["consecutivo", "Y"] + nucleo], on="consecutivo", validate="one_to_one", suffixes=("", "_test"))
                .merge(choques[["consecutivo"] + cols_choque], on="consecutivo", how="left", validate="one_to_one"))
    assert len(base) == len(pred) and (base["Y"] == base["Y_test"]).all(), "las predicciones no cuadran con el conjunto de prueba"
    base["cluster_id"] = cluster_id(base)
    print(f"{len(base)} hogares, prevalencia {base['Y'].mean():.3f}, sin choques 2016: {base[cols_choque[0]].isna().sum()}")

    registro = pd.read_csv(REGISTRO_PATH)
    umbral = float(registro[(registro["algoritmo"] == ALGORITMO_REGISTRO)
                            & (registro["especificacion"] == ESPECIFICACION)]["umbral_clasificacion_media"].iloc[0])
    proba = base["proba_base"].to_numpy()
    reglas = {
        "umbral_f2": (proba >= umbral).astype(int),
        "top10": marcar_presupuesto(proba, 0.10),
    }

    filas, conteos = [], []
    y = base["Y"].to_numpy()
    for regla, marcado in reglas.items():
        base["celda"] = np.select([(marcado == 1) & (y == 1), (marcado == 1) & (y == 0), (marcado == 0) & (y == 1)],
                                  ["VP", "FP", "FN"], default="VN")
        conteo = base["celda"].value_counts().to_dict()
        conteos.append({"regla": regla, "umbral": umbral if regla == "umbral_f2" else np.nan,
                        **{c: int(conteo.get(c, 0)) for c in ["VP", "FP", "FN", "VN"]}})
        print(f"\n=== {regla}: {conteos[-1]} ===")
        filas += comparar(base, numericas, "FP", "VN", regla, "FP_vs_VN_nucleo_2013", rng)
        filas += comparar(base, cols_choque, "VP", "FP", regla, "VP_vs_FP_choques_2016", rng)
        filas += comparar(base, numericas, "VP", "FP", regla, "VP_vs_FP_nucleo_2013", rng)

    no_caen = base[base["Y"] == 0].copy()
    no_caen["cuartil_riqueza_2013"] = pd.qcut(no_caen["riqueza_pca_hogar"], 4, labels=["Q1 (menor)", "Q2", "Q3", "Q4 (mayor)"])
    por_riqueza = (no_caen.groupby("cuartil_riqueza_2013", observed=True)
                   .agg(n=("Y", "size"), total_choques_2016=("total_choques_hogar_2016", "mean"),
                        algun_choque_2016=("tuvo_algun_choque_hogar_2016", "mean"))
                   .round(3).reset_index())
    por_riqueza.to_csv(RUTA_RIQUEZA, index=False)
    print("\n=== Hogares que no caen: choques 2016 por cuartil de riqueza 2013 ===")
    print(por_riqueza.to_string(index=False))

    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_SALIDA, index=False)
    pd.DataFrame(conteos).to_csv(RUTA_CONTEOS, index=False)

    pd.set_option("display.width", 250)
    for (regla, comp), sub in tabla.groupby(["regla", "comparacion"], sort=False):
        print(f"\n--- {regla}: {comp} ---")
        print(sub.drop(columns=["regla", "comparacion", "grupo_a", "grupo_b"])
              .sort_values("diferencia_estandarizada", key=abs, ascending=False).head(8).to_string(index=False))
    print(f"\nGuardado: {RUTA_SALIDA}\nGuardado: {RUTA_CONTEOS}\nGuardado: {RUTA_RIQUEZA}")


if __name__ == "__main__":
    main()
