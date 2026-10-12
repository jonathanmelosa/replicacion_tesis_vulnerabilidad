"""
robustez_hogares_no_repetidos.py
================================
Los mismos hogares pueden aparecer tanto en la muestra de entrenamiento
como en la de validacion.

Hecho: 1,920 de los 3,191 hogares del conjunto de prueba 2013->2016 (60%)
estan tambien en el de entrenamiento 2010->2013 (no pobres en 2010 y en
2013), con covariables y resultado de otro periodo. Los otros 1,271
salieron de la pobreza entre 2010 y 2013. La pregunta es si esa
repeticion infla el desempeño reportado.

Hipotesis: la repeticion no infla el desempeño del modelo principal (la
diferencia entre validar por fila y validar agrupando por hogar es
cercana a cero); en los arboles podria aparecer una diferencia pequeña.

Parte (a) -- prueba actual, por grupo
-------------------------------------
Con las predicciones del paper (`predicciones_test_dmsp_{A,B}.parquet`,
semilla 42, `generar_predicciones_test_dmsp.py`), separa los hogares
repetidos de los nuevos y reporta, por algoritmo y especificacion, con y
sin luz: tasa de entrada observada, riesgo predicho promedio, AUC-PR (con
IC95% por bootstrap por comunidad) y AUC-ROC. Los modelos con
reponderacion de clases no estan calibrados, asi que el riesgo predicho
se compara entre grupos (razon repetidos/nuevos), no contra la tasa.

Parte (b) -- ambos periodos juntos, tres formas de particionar
--------------------------------------------------------------
Se apilan 2010->2013 y 2013->2016 (6,280 filas; una variable
`periodo_2013` indica el periodo) y se evalua con validacion cruzada de 10
particiones x 5 semillas, de tres formas:
  - fila: StratifiedKFold; las dos filas de un hogar pueden quedar en
               lados distintos (lo que hoy ocurre entre train y test).
  - hogar: StratifiedGroupKFold por hogar; ningun hogar en los dos lados.
  - comunidad: StratifiedGroupKFold por comunidad: el modelo se evalua en comunidades que no vio.
               El grupo es la componente conexa de hogar y comunidad, para
               que tampoco se parta un hogar que cambio de comunidad entre
               olas; un hogar sin comunidad identificada (8888888) es su
               propio grupo.
Para A, B, AgeoDMSP y BgeoDMSP (pobreza monetaria), con los
hiperparametros del paper. Diferencias que se reportan, con IC95% por
bootstrap pareado (clusters = la componente hogar-comunidad):
  - fila - hogar: cuanto infla la repeticion de hogares.
  - fila - comunidad: cuanto depende de conocer la comunidad.
  - geo - base en cada forma: si el aporte de la luz depende de la forma
                        de particionar.

INPUTS

    data/processed/benchmark_train_test/modelo_{A,B,AgeoDMSP,BgeoDMSP}_{2010_2013,2013_2016}.parquet
    data/processed/benchmark_resultados/predicciones_test_dmsp_{A,B}.parquet
    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv

OUTPUTS (data/processed/benchmark_resultados/robustez_asesor_2026_10/)

    no_repetidos_a_por_grupo.csv
    no_repetidos_b_predicciones.parquet (oof por fila x algoritmo x espec x forma x semilla)
    no_repetidos_b_desempeno.csv (AUC-PR/AUC-ROC por forma)
    no_repetidos_b_diferencias.csv (diferencias con IC95%)

COMO CORRER

    cd src/05_model && python -u robustez_hogares_no_repetidos.py 2>&1 | tee../../logs/robustez_hogares_no_repetidos.log
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import robustez_asesor_comun as rc

ESPECS = ["A", "B", "AgeoDMSP", "BgeoDMSP"]
FORMAS = ["fila", "hogar", "comunidad"]
PARES_GEO = [("A", "AgeoDMSP"), ("B", "BgeoDMSP")]


def _ic_ap(y, p, clusters, semilla) -> tuple:
    ap_boot, _ = rc.dcp.bootstrap_ap(y, p, p, clusters, np.random.default_rng(semilla))
    return tuple(np.percentile(ap_boot, [2.5, 97.5]))


def parte_a() -> pd.DataFrame:
    hogares_train = set(rc.cargar("B", "2010_2013")["consecutivo"])
    filas, casos = [], []
    for base in ["A", "B"]:
        pred = pd.read_parquet(rc.mu.RESULTADOS_DIR / f"predicciones_test_dmsp_{base}.parquet")
        pred["grupo"] = np.where(pred["consecutivo"].isin(hogares_train), "repetido", "nuevo")
        for algoritmo, sub in pred.groupby("algoritmo"):
            for col, luz in [("proba_base", "sin luz"), ("proba_geo", "con luz")]:
                medias = {g: sub.loc[sub.grupo == g, col].mean() for g in ["repetido", "nuevo"]}
                for grupo in ["repetido", "nuevo", "todos"]:
                    s = sub if grupo == "todos" else sub[sub.grupo == grupo]
                    s = s.reset_index(drop=True)
                    y, p = s["Y"].values, s[col].values
                    casos.append((y, p, rc.cluster_id(s)))
                    filas.append({
                        "especificacion": base, "algoritmo": algoritmo, "luz": luz, "grupo": grupo,
                        "n_hogares": len(s), "tasa_entrada": round(y.mean(), 4),
                        "riesgo_predicho_medio": round(p.mean(), 4),
                        "razon_riesgo_repetido_nuevo": round(medias["repetido"] / medias["nuevo"], 4),
                        "ap": round(average_precision_score(y, p), 4),
                        "auc": round(roc_auc_score(y, p), 4),
                    })
    ics = rc.en_paralelo(rc.delayed(_ic_ap)(y, p, c, rc.mu.RANDOM_STATE + i) for i, (y, p, c) in enumerate(casos))
    for fila, (low, high) in zip(filas, ics):
        fila["ap_ci95_low"], fila["ap_ci95_high"] = round(low, 4), round(high, 4)
    tasas = pred.drop_duplicates("consecutivo").groupby("grupo")["Y"].mean()
    print(f"Tasa de entrada observada: repetidos {tasas['repetido']:.3f}, nuevos {tasas['nuevo']:.3f}", flush=True)
    return pd.DataFrame(filas)


def apilar(espec: str) -> pd.DataFrame:
    a = rc.cargar(espec, "2010_2013").assign(periodo_2013=0)
    b = rc.cargar(espec, "2013_2016").assign(periodo_2013=1)
    assert list(a.columns) == list(b.columns)
    return pd.concat([a, b], ignore_index=True)


def grupos_hogar_comunidad(df: pd.DataFrame) -> np.ndarray:
    """Componente conexa del grafo hogar--comunidad (union-find): ni un
    hogar ni una comunidad quedan partidos entre particiones."""
    padre = {}

    def raiz(x):
        padre.setdefault(x, x)
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for h, c in zip(df["consecutivo"], df["consecutivo_c"]):
        nodo_h = ("h", h)
        raiz(nodo_h)
        if c != rc.SENTINEL_COMUNIDAD:
            padre[raiz(nodo_h)] = raiz(("c", c))
    return np.array([str(raiz(("h", h))) for h in df["consecutivo"]])


def main() -> None:
    registro, algoritmos = rc.cargar_registro("monetaria")
    print(f"Nucleos en paralelo: {rc.N_JOBS}", flush=True)

    print("=== Parte (a): prueba actual por grupo ===", flush=True)
    rc.guardar(parte_a(), "no_repetidos_a_por_grupo.csv")

    print("\n=== Parte (b): ambos periodos juntos ===", flush=True)
    datos = {e: apilar(e) for e in ESPECS}
    ref = datos["B"]
    for e in ESPECS:
        assert (datos[e]["consecutivo"].values == ref["consecutivo"].values).all()
    grupos = {"fila": None, "hogar": ref["consecutivo"].astype(str).values, "comunidad": grupos_hogar_comunidad(ref)}
    clusters = grupos["comunidad"]
    print(f"Filas: {len(ref):,}; hogares: {ref['consecutivo'].nunique():,}; grupos hogar-comunidad: {len(set(clusters)):,}", flush=True)

    piezas = []
    for algoritmo in algoritmos:
        for espec in ESPECS:
            for forma in FORMAS:
                for semilla, p in rc.oof_semillas(algoritmo, espec, registro, datos[espec], grupos[forma]).items():
                    piezas.append(pd.DataFrame({
                        "algoritmo": rc.nombre_corto(algoritmo), "especificacion": espec, "forma": forma,
                        "semilla": semilla, "fila": np.arange(len(ref)), "proba": p,
                    }))
                print(f"  {rc.nombre_corto(algoritmo):22s} {espec:9s} {forma:9s} listo", flush=True)
    pred = pd.concat(piezas, ignore_index=True)
    rc.guardar(pred, "no_repetidos_b_predicciones.parquet")

    y = ref["Y"].values
    prom = pred.groupby(["algoritmo", "especificacion", "forma", "fila"])["proba"].mean()
    desempeno, claves, casos = [], [], []
    for algoritmo in pred["algoritmo"].unique():
        for espec in ESPECS:
            for forma in FORMAS:
                p = prom.loc[(algoritmo, espec, forma)].values
                por_semilla = {s: g.sort_values("fila")["proba"].values for s, g in
                               pred[(pred.algoritmo == algoritmo) & (pred.especificacion == espec) & (pred.forma == forma)].groupby("semilla")}
                desempeno.append({"algoritmo": algoritmo, "especificacion": espec, "forma": forma,
                                  "ap": round(average_precision_score(y, p), 4), "auc": round(roc_auc_score(y, p), 4),
                                  **rc.metricas_por_semilla(y, por_semilla)})
            for ref_forma, var_forma in [("hogar", "fila"), ("comunidad", "fila"), ("comunidad", "hogar")]:
                claves.append({"algoritmo": algoritmo, "especificacion": espec, "comparacion": f"{var_forma} - {ref_forma}"})
                casos.append((y, prom.loc[(algoritmo, espec, ref_forma)].values, prom.loc[(algoritmo, espec, var_forma)].values, clusters))
        for base, geo in PARES_GEO:
            for forma in FORMAS:
                claves.append({"algoritmo": algoritmo, "especificacion": base, "comparacion": f"luz ({forma})"})
                casos.append((y, prom.loc[(algoritmo, base, forma)].values, prom.loc[(algoritmo, geo, forma)].values, clusters))
    rc.guardar(pd.DataFrame(desempeno), "no_repetidos_b_desempeno.csv")
    print("Bootstrap de las diferencias...", flush=True)
    dif = pd.DataFrame([{**k, **r} for k, r in zip(claves, rc.comparar_varios(casos, rc.mu.RANDOM_STATE))])
    rc.guardar(dif, "no_repetidos_b_diferencias.csv")
    print("\n" + dif[["algoritmo", "especificacion", "comparacion", "ap_ref", "ap_var", "delta_ap",
                      "delta_ap_ci95_low", "delta_ap_ci95_high"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
