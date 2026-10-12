"""
robustez_techo_comunidad.py
===========================
Si la luz varia sobre todo entre
comunidades y el riesgo sobre todo entre vecinos, hay que justificar en
que margen se esperaba que la informacion geoespacial aportara; que no
aporte "parece obvio de antemano".

Esta prueba mide ese margen: cuanto podria aportar como MAXIMO cualquier
informacion de comunidad, y donde queda la luz frente a ese techo.

Hipotesis: ni siquiera la mejor informacion de comunidad mejora de forma
importante la prediccion de quien entra en pobreza (el margen territorial
esta casi agotado por la encuesta).

Parte 1 -- correlacion intraclase: nivel frente a entrada
---------------------------------------------------------
ICC(1) por comunidad (estimador ANOVA de
`diagnostico_variacion_dmsp_comunidad.icc_anova`) de:
  - nivel de pobreza: todos los hogares de cada ola (2010, 2013, 2016);
  - entrada en pobreza: el resultado Y de cada ventana.
Pobreza monetaria e IPM, total y por zona. Se excluyen hogares sin
comunidad identificada (8888888).

Parte 2 -- techo
----------------
Dentro de 2013->2016, validacion cruzada estratificada de 10 particiones x
5 semillas, cinco algoritmos, Modelos A y B, pobreza monetaria e IPM,
hiperparametros del paper. Variantes:
  - base: covariables de 2013 (referencia).
  - dmsp: + 23 variables de DMSP-OLS (archivo geoDMSP; sus
                 hiperparametros son los de la especificacion con luz).
  - tasa_previa: + tasa de entrada de la comunidad en 2010->2013 (la mejor
                 informacion de comunidad que podria tener un sistema real).
  - oraculo: + tasa de entrada de los VECINOS en el mismo periodo
                 2013->2016, calculada solo con la parte de entrenamiento
                 de cada particion y nunca con el propio hogar (en
                 entrenamiento, sin el propio hogar: leave-one-out). Ningun
                 sistema real tendria este dato: es el techo teorico.
Las tasas se encogen hacia la media de la zona con un estimador bayesiano
empirico beta-binomial (las comunidades tienen 3 o 4 hogares): tasa =
(k + m*mu) / (n + m), con mu la tasa de la zona y m = (1 - rho) / rho, rho
el ICC de la zona en los datos de origen (si rho <= 0, tasa = mu). Una
comunidad sin datos de origen recibe mu.

Muestra: se EXCLUYEN los hogares sin comunidad identificada: 600 bajo pobreza monetaria y 937 bajo IPM. Todas las
variantes se comparan sobre los mismos hogares. Para ver que la exclusion
no cambia el punto de partida, el modelo base tambien se corre con la
muestra completa (variante `base_muestra_completa`).

INPUTS

    data/processed/benchmark_train_test/modelo_{A,B,Aipm,Bipm}[geoDMSP]_{2010_2013,2013_2016}.parquet
    data/processed/benchmark_consolidado_elca_longitudinal.parquet
    data/processed/ipm_multidimensional_elca_longitudinal.parquet
    data/processed/benchmark_resultados/registro_modelos_{fbeta2_cv10,ipm}.csv

OUTPUTS (data/processed/benchmark_resultados/robustez_asesor_2026_10/)

    techo_icc.csv
    techo_predicciones.parquet
    techo_resultados.csv (diferencias frente a base con IC95%)
    techo_muestra.csv (AUC-PR del modelo base: muestra completa vs. submuestra)

COMO CORRER

    cd src/05_model && python -u robustez_techo_comunidad.py 2>&1 | tee../../logs/robustez_techo_comunidad.log
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import robustez_asesor_comun as rc
from diagnostico_variacion_dmsp_comunidad import icc_anova

VARIANTES = ["dmsp", "tasa_previa", "oraculo"]
COL_TASA = "tasa_entrada_comunidad"


# ---------------------------------------------------------------- parte 1
def parte_icc() -> pd.DataFrame:
    panel = pd.read_parquet(rc.PANEL_PATH, columns=["consecutivo", "ola", "consecutivo_c", "zona", "pobre_ingreso"])
    panel = panel[~panel.duplicated(["consecutivo", "ola"], keep=False)]
    ipm = pd.read_parquet(rc.IPM_PATH, columns=["consecutivo", "ola", "pobre_ipm"])
    ipm = ipm[~ipm.duplicated(["consecutivo", "ola"], keep=False)]
    panel = panel.merge(ipm, on=["consecutivo", "ola"], how="left")
    anio = {1: 2010, 2: 2013, 3: 2016}

    filas = []

    def agregar(medida, definicion, periodo, df, col):
        df = df[(df["consecutivo_c"] != rc.SENTINEL_COMUNIDAD) & df[col].notna()]
        for zona, sub in [("Total", df)] + list(df.groupby("zona")):
            filas.append({"medida": medida, "definicion": definicion, "periodo": periodo, "zona": zona,
                          "n_hogares": len(sub), "n_comunidades": sub["consecutivo_c"].nunique(),
                          "prevalencia": round(sub[col].astype(float).mean(), 4),
                          "icc": round(icc_anova(sub[col].astype(float), sub["consecutivo_c"]), 4)})

    for ola, sub in panel.groupby("ola"):
        agregar("nivel", "monetaria", str(anio[ola]), sub, "pobre_ingreso")
        agregar("nivel", "ipm", str(anio[ola]), sub, "pobre_ipm")
    for definicion, (_, especs) in rc.REGISTROS.items():
        for transicion in ["2010_2013", "2013_2016"]:
            agregar("entrada", definicion, transicion.replace("_", "-"), rc.cargar(especs["A"][0], transicion), "Y")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- parte 2
def prior_zona(df: pd.DataFrame) -> dict:
    """{zona: (mu, m)} del estimador beta-binomial (ver docstring)."""
    out = {}
    for zona, sub in df.groupby("zona"):
        mu = sub["Y"].mean()
        rho = icc_anova(sub["Y"].astype(float), sub["consecutivo_c"]) if sub["consecutivo_c"].nunique() > 1 else 0.0
        out[zona] = (mu, (1 - rho) / rho if rho > 0 else np.inf)
    return out


def tasa_encogida(k, n, mu, m):
    k, n = np.asarray(k, float), np.asarray(n, float)
    if np.isinf(m):
        return np.full(len(k), mu)
    return (k + m * mu) / (n + m)


def tasas_desde(origen: pd.DataFrame, destino: pd.DataFrame, excluir_propio: bool = False) -> np.ndarray:
    """Tasa encogida de la comunidad de cada fila de `destino`, calculada con
    el resultado de `origen`. Con `excluir_propio`, `destino` es `origen` y
    cada hogar se descuenta de su comunidad (leave-one-out)."""
    origen = origen[origen["consecutivo_c"] != rc.SENTINEL_COMUNIDAD]
    priors = prior_zona(origen)
    agg = origen.groupby("consecutivo_c")["Y"].agg(["sum", "size"])
    k = destino["consecutivo_c"].map(agg["sum"]).fillna(0).values
    n = destino["consecutivo_c"].map(agg["size"]).fillna(0).values
    if excluir_propio:
        k, n = k - destino["Y"].values, n - 1
    out = np.empty(len(destino))
    for zona in destino["zona"].unique():
        m_z = (destino["zona"] == zona).values
        mu, m = priors.get(zona, (origen["Y"].mean(), np.inf))
        out[m_z] = tasa_encogida(k[m_z], n[m_z], mu, m)
    return out


def oraculo_por_fold(df_tr: pd.DataFrame, df_va: pd.DataFrame):
    df_tr, df_va = df_tr.copy(), df_va.copy()
    df_va[COL_TASA] = tasas_desde(df_tr, df_va)
    df_tr[COL_TASA] = tasas_desde(df_tr, df_tr, excluir_propio=True)
    return df_tr, df_va


def main() -> None:
    print(f"Nucleos en paralelo: {rc.N_JOBS}", flush=True)
    print("=== Parte 1: correlacion intraclase ===", flush=True)
    icc = parte_icc()
    rc.guardar(icc, "techo_icc.csv")
    print(icc.to_string(index=False), flush=True)

    print("\n=== Parte 2: techo de comunidad ===", flush=True)
    piezas, claves, casos, muestra = [], [], [], []
    for definicion, (_, especs) in rc.REGISTROS.items():
        registro, algoritmos = rc.cargar_registro(definicion)
        for espec, (e_base, e_geo) in especs.items():
            completo = rc.cargar(e_base, "2013_2016")
            geo = rc.cargar(e_geo, "2013_2016")
            assert (completo["consecutivo"].values == geo["consecutivo"].values).all()
            con_com = (completo["consecutivo_c"] != rc.SENTINEL_COMUNIDAD).values
            base = completo[con_com].reset_index(drop=True)
            previo = rc.cargar(e_base, "2010_2013")
            datos = {
                "base": base,
                "dmsp": geo[con_com].reset_index(drop=True),
                "tasa_previa": base.assign(**{COL_TASA: tasas_desde(previo, base)}),
                "oraculo": base,
            }
            e_reg = {"base": e_base, "dmsp": e_geo, "tasa_previa": e_base, "oraculo": e_base}
            y, clusters = base["Y"].values, rc.cluster_id(base)
            sin_previa = (~base["consecutivo_c"].isin(previo["consecutivo_c"])).sum()
            print(f"{definicion} {espec}: {len(completo):,} hogares, {len(base):,} con comunidad; "
                  f"{sin_previa:,} en comunidades sin datos 2010-2013", flush=True)

            for algoritmo in algoritmos:
                corto = rc.nombre_corto(algoritmo)
                prom, por_semilla = {}, {}
                for variante in ["base"] + VARIANTES:
                    por_fold = oraculo_por_fold if variante == "oraculo" else None
                    por_semilla[variante] = rc.oof_semillas(algoritmo, e_reg[variante], registro, datos[variante], por_fold=por_fold)
                    prom[variante] = rc.promedio_semillas(por_semilla[variante])
                    for s, p in por_semilla[variante].items():
                        piezas.append(pd.DataFrame({"definicion": definicion, "especificacion": espec, "algoritmo": corto,
                                                    "variante": variante, "semilla": s,
                                                    "consecutivo": base["consecutivo"].values, "proba": p}))
                p_completo = rc.promedio_semillas(rc.oof_semillas(algoritmo, e_base, registro, completo))
                muestra.append({"definicion": definicion, "especificacion": espec, "algoritmo": corto,
                                "n_completa": len(completo), "ap_base_completa": round(average_precision_score(completo["Y"], p_completo), 4),
                                "prevalencia_completa": round(completo["Y"].mean(), 4),
                                "n_submuestra": len(base), "ap_base_submuestra": round(average_precision_score(y, prom["base"]), 4),
                                "prevalencia_submuestra": round(y.mean(), 4),
                                "ap_base_completa_en_submuestra": round(average_precision_score(y, p_completo[con_com]), 4)})
                for variante in VARIANTES:
                    claves.append({"definicion": definicion, "especificacion": espec, "algoritmo": corto, "variante": variante,
                                   **{f"var_{k}": v for k, v in rc.metricas_por_semilla(y, por_semilla[variante]).items()}})
                    casos.append((y, prom["base"], prom[variante], clusters))
                print(f"  {corto:22s} AP base {average_precision_score(y, prom['base']):.3f}  " + "  ".join(
                    f"{v}: {average_precision_score(y, prom[v]) - average_precision_score(y, prom['base']):+.3f}" for v in VARIANTES), flush=True)

    print("Bootstrap de las diferencias...", flush=True)
    resultados = pd.DataFrame([{**k, **r} for k, r in zip(claves, rc.comparar_varios(casos, rc.mu.RANDOM_STATE))])
    rc.guardar(pd.concat(piezas, ignore_index=True), "techo_predicciones.parquet")
    rc.guardar(resultados, "techo_resultados.csv")
    rc.guardar(pd.DataFrame(muestra), "techo_muestra.csv")
    print(resultados[["definicion", "especificacion", "algoritmo", "variante", "ap_ref", "ap_var", "delta_ap",
                      "delta_ap_ci95_low", "delta_ap_ci95_high"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
