"""
Analisis de heterogeneidad del aporte marginal de DMSP-OLS --
como complemento al efecto agregado (que ya se
establecio como no significativo, ver
`diagnostico_bootstrap_dmsp.py`/informe). La pregunta aca es distinta:
¿DMSP-OLS ayuda mas en subpoblaciones especificas, aunque el efecto
agregado sea nulo?

Usa las predicciones ya generadas por `generar_predicciones_test_dmsp.py`
-- no reentrena nada. Para cada eje de heterogeneidad, cada grupo, cada
algoritmo y cada par (A/AgeoDMSP, B/BgeoDMSP): AUC-ROC, recall, precision
y F1 dentro del grupo con y sin DMSP-OLS, delta, n de hogares y n de
positivos (para poder juzgar si el grupo tiene tamaño suficiente para que
las metricas sean informativas -- ver nota al pie de cada tabla).

AMPLIADO: AUC-ROC se estableció
como una metrica de magnitud pequeña e insuficiente por si sola para el
argumento de politica publica -- lo que importa para un ejercicio de
focalizacion es si DMSP-OLS identifica MAS hogares vulnerables (recall) y
a que costo en precision, no solo si mejora el ranking general. Se
calculan recall/precision/F1 por subgrupo aplicando el UMBRAL de
clasificacion ya elegido por validacion cruzada para cada
algoritmo/especificacion (`umbral_clasificacion_media` de
`registro_modelos_fbeta2_cv10.csv` -- el mismo umbral que reporta la
tabla principal de la tesis, no uno nuevo elegido para este ejercicio).

EJES DE HETEROGENEIDAD

    1. Zona (zona): Urbano / Rural.
    2. Cercania al umbral de pobreza (brecha_lp_ingreso, razon ingreso/LP):
       cuartiles -- el cuartil mas bajo son los hogares MAS cerca de caer
       en pobreza, donde el problema de clasificacion es mas dificil.
    3. Estrato socioeconomico verificado (estrato_verificado_hogar).
    4. Nivel de las variables de riqueza ELCA que hacen redundante a
       DMSP-OLS (n_servicios_publicos_hogar + n_bienes_durables_hogar,
       estandarizadas y sumadas -- "indice_riqueza_proxy"): terciles.
       Hipotesis a probar: si DMSP-OLS es puramente redundante con estas
       dos variables (ver Seccion 5.3 de la tesis), su aporte deberia ser
       parejo en los 3 terciles; si aporta MAS donde estas variables
       tienen menos variacion/poder discriminante (terciles extremos,
       donde casi todos los hogares tienen el mismo puntaje), seria
       evidencia de que la redundancia no es total.

NOTA IMPORTANTE: este es un analisis EXPLORATORIO (estimaciones puntuales,
sin bootstrap por celda) -- con subgrupos de pocos cientos de hogares y
~20% de tasa de positivos, las metricas por celda son ruidosas. Sirve para
detectar candidatos a investigar con mas rigor (bootstrap especifico de
esa celda), no para conclusiones definitivas por si solo.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_heterogeneidad_dmsp.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_heterogeneidad_dmsp.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import modelo_utils as mu
from algoritmos_suite import ALGORITMOS_SUITE

N_MIN_POSITIVOS = 20  # bajo este umbral, la celda se marca como poco confiable

# predicciones_test_dmsp_*.parquet identifica algoritmos por "nombre_bonito"
# (ej. "XGBoost", "Random Forest") -- registro_modelos_fbeta2_cv10.csv los
# identifica por su nombre CRUDO. Mapeo inverso construido UNA vez sobre el
# registro central de algoritmos_suite.py -- no se hardcodea de nuevo aqui.
NOMBRE_BONITO_A_CRUDO = {info["nombre_bonito"]: crudo for crudo, info in ALGORITMOS_SUITE.items()}


def cargar_umbrales(ruta_registro) -> dict:
    """dict {(nombre_bonito, especificacion): umbral_clasificacion_media}."""
    registro = pd.read_csv(ruta_registro)
    umbrales = {}
    for _, fila in registro.iterrows():
        nombre_bonito = ALGORITMOS_SUITE[fila["algoritmo"]]["nombre_bonito"]
        umbrales[(nombre_bonito, fila["especificacion"])] = fila["umbral_clasificacion_media"]
    return umbrales


def metricas_por_grupo(sub: pd.DataFrame, umbral_base: float, umbral_geo: float) -> dict:
    y = sub["Y"].values
    n = len(sub)
    n_pos = int(y.sum())
    if n_pos < 2 or n_pos > n - 2:
        return {
            "n": n, "n_pos": n_pos, "auc_base": np.nan, "auc_geo": np.nan, "delta": np.nan,
            "recall_base": np.nan, "recall_geo": np.nan, "delta_recall": np.nan,
            "precision_base": np.nan, "precision_geo": np.nan, "delta_precision": np.nan,
            "f1_base": np.nan, "f1_geo": np.nan, "delta_f1": np.nan,
            "confiable": False,
        }
    auc_base = roc_auc_score(y, sub["proba_base"].values)
    auc_geo = roc_auc_score(y, sub["proba_geo"].values)

    pred_base = (sub["proba_base"].values >= umbral_base).astype(int)
    pred_geo = (sub["proba_geo"].values >= umbral_geo).astype(int)
    recall_base = recall_score(y, pred_base, zero_division=0)
    recall_geo = recall_score(y, pred_geo, zero_division=0)
    precision_base = precision_score(y, pred_base, zero_division=0)
    precision_geo = precision_score(y, pred_geo, zero_division=0)
    f1_base = f1_score(y, pred_base, zero_division=0)
    f1_geo = f1_score(y, pred_geo, zero_division=0)

    return {
        "n": n, "n_pos": n_pos, "auc_base": round(auc_base, 4), "auc_geo": round(auc_geo, 4),
        "delta": round(auc_geo - auc_base, 4),
        "recall_base": round(recall_base, 4), "recall_geo": round(recall_geo, 4),
        "delta_recall": round(recall_geo - recall_base, 4),
        "precision_base": round(precision_base, 4), "precision_geo": round(precision_geo, 4),
        "delta_precision": round(precision_geo - precision_base, 4),
        "f1_base": round(f1_base, 4), "f1_geo": round(f1_geo, 4),
        "delta_f1": round(f1_geo - f1_base, 4),
        "confiable": n_pos >= N_MIN_POSITIVOS,
    }


def construir_ejes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["eje_zona"] = df["zona"]

    df["eje_brecha_lp"] = pd.qcut(df["brecha_lp_ingreso"], 4, labels=["Q1 (mas cerca de LP)", "Q2", "Q3", "Q4 (mas lejos)"])

    df["eje_estrato"] = "Estrato " + df["estrato_verificado_hogar"].astype("Int64").astype(str)

    riqueza_z = (
        (df["n_servicios_publicos_hogar"] - df["n_servicios_publicos_hogar"].mean()) / df["n_servicios_publicos_hogar"].std()
        + (df["n_bienes_durables_hogar"] - df["n_bienes_durables_hogar"].mean()) / df["n_bienes_durables_hogar"].std()
    )
    df["eje_riqueza_proxy"] = pd.qcut(riqueza_z, 3, labels=["Bajo (proxy menos discriminante)", "Medio", "Alto (proxy menos discriminante)"])
    return df


def main() -> None:
    umbrales = cargar_umbrales(mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv")
    filas = []
    for base, geo in [("A", "AgeoDMSP"), ("B", "BgeoDMSP")]:
        pred = pd.read_parquet(mu.RESULTADOS_DIR / f"predicciones_test_dmsp_{base}.parquet")
        pred = construir_ejes(pred)

        for algoritmo in pred["algoritmo"].unique():
            sub_algo = pred[pred["algoritmo"] == algoritmo]
            umbral_base = umbrales[(algoritmo, base)]
            umbral_geo = umbrales[(algoritmo, geo)]

            print(f"\n=== {algoritmo} -- {base} vs {geo} (umbral {base}={umbral_base}, umbral {geo}={umbral_geo}) ===")
            general = metricas_por_grupo(sub_algo, umbral_base, umbral_geo)
            print(f"  [Global] n={general['n']} n_pos={general['n_pos']} "
                  f"AUC {general['auc_base']}->{general['auc_geo']} (d={general['delta']})  "
                  f"Recall {general['recall_base']}->{general['recall_geo']} (d={general['delta_recall']})  "
                  f"Precision {general['precision_base']}->{general['precision_geo']} (d={general['delta_precision']})  "
                  f"F1 {general['f1_base']}->{general['f1_geo']} (d={general['delta_f1']})")
            filas.append({"algoritmo": algoritmo, "especificacion_base": base, "eje": "Global", "grupo": "Global", **general})

            for eje in ["eje_zona", "eje_brecha_lp", "eje_estrato", "eje_riqueza_proxy"]:
                for grupo, sub_grupo in sub_algo.groupby(eje, observed=True):
                    r = metricas_por_grupo(sub_grupo, umbral_base, umbral_geo)
                    marca = "" if r["confiable"] else "  [n_pos<20, POCO CONFIABLE]"
                    print(f"  [{eje}={grupo}] n={r['n']} n_pos={r['n_pos']} "
                          f"AUC d={r['delta']}  Recall d={r['delta_recall']}  "
                          f"Precision d={r['delta_precision']}  F1 d={r['delta_f1']}{marca}")
                    filas.append({"algoritmo": algoritmo, "especificacion_base": base, "eje": eje, "grupo": str(grupo), **r})

    out = pd.DataFrame(filas)
    out_path = mu.RESULTADOS_DIR / "diagnostico_heterogeneidad_dmsp.csv"
    out.to_csv(out_path, index=False)

    print("\n\n=== Candidatos con mayor |delta_recall| entre celdas confiables (n_pos>=20) ===")
    confiables = out[(out["confiable"]) & (out["eje"] != "Global")].copy()
    confiables["abs_delta_recall"] = confiables["delta_recall"].abs()
    print(confiables.sort_values("abs_delta_recall", ascending=False).head(15)[
        ["algoritmo", "especificacion_base", "eje", "grupo", "n", "n_pos",
         "delta_recall", "delta_precision", "delta_f1", "delta"]
    ].to_string(index=False))

    print(f"\nCSV: {out_path}")


if __name__ == "__main__":
    main()
