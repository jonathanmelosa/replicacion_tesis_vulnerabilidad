"""
robustez_encuesta_desactualizada.py
===================================
El paper compara "encuesta al
dia" con "encuesta al dia + luz", donde la luz no tiene nada que agregar.
Pero el paper afirma (Introduccion y Seccion 5.3) que la luz sirve "como
respaldo de informacion desactualizada entre rondas de encuesta", y eso no
esta probado. Esta prueba lo pone a prueba con el panel: predice la
entrada en pobreza 2013->2016 con la encuesta de 2010 (tres años de
rezago, la situacion habitual del SISBEN) y mide que parte de lo que se
pierde recupera la luz actualizada.

Hipotesis: desactualizar la encuesta tiene un costo predictivo medible y
la luz actualizada recupera solo una parte pequeña de ese costo, algo
mayor en la zona rural.

Diseño
------
Modelos: los del paper, entrenados con 2010->2013 (covariables 2010, luz
2010) con los hiperparametros del registro y las 5 semillas; no se
reestima nada. Poblacion: hogares no pobres en 2013 (conjunto de prueba
del paper). Cambia solo la informacion con la que se predice:

  escenario  encuesta  luz    modelo
  E1         2013      --     sin luz   (encuesta al dia; referencia)
  E1_luz     2013      2013   con luz   (lo que reporta el paper con luz)
  E2         2010      --     sin luz   (encuesta con 3 años de rezago)
  E3         2010      2013   con luz   (encuesta vieja + luz actualizada)
  E4         2010      2010   con luz   (encuesta vieja + luz igual de vieja)

Medidas (AUC-PR, IC95% por bootstrap pareado por comunidad):
  - perdida por no actualizar la encuesta: E1 - E2.
  - aporte de la luz actualizada a la encuesta vieja: E3 - E2.
  - valor de que la luz este al dia: E3 - E4.
  - parte recuperada: (E3 - E2) / (E1 - E2), con su IC95% del mismo
    bootstrap (se reporta solo si E1 - E2 > 0 en la cifra puntual).
Se reporta para pobreza monetaria e IPM, Modelos A y B, cinco algoritmos,
en total y por zona.

Covariables de 2010 de los hogares de prueba: las filas de la ola 2010 del
panel consolidado, con exactamente las columnas del archivo de
entrenamiento (las mismas transformaciones de `build_benchmark_train_test*.py`,
que solo seleccionan columnas). Para los hogares que tambien estan en
entrenamiento, se verifica que esas filas coincidan con las del archivo
de entrenamiento. La luz se toma de `variables_geoespaciales_unificadas.parquet`
con la misma funcion que el paper (`construir_pipeline_geo_dmsp.cargar_geoespacial_dmsp`).

Limites (declarados en el documento de plan): el modelo aprendio a predecir
a tres años y aqui usa informacion de seis años antes; la poblacion se
define con la pobreza de 2013, que un sistema con encuesta vieja no
conoceria; los hogares que salieron de la pobreza entran con la encuesta
de cuando eran pobres.

INPUTS

    data/processed/benchmark_train_test/modelo_{A,B,Aipm,Bipm}[geoDMSP]_{2010_2013,2013_2016}.parquet
    data/processed/benchmark_consolidado_elca_longitudinal.parquet (ola 2010)
    data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet
    data/processed/benchmark_resultados/registro_modelos_{fbeta2_cv10,ipm}.csv

OUTPUTS (data/processed/benchmark_resultados/robustez_asesor_2026_10/)

    desactualizada_predicciones.parquet
    desactualizada_escenarios.csv   (AUC-PR/AUC-ROC por escenario, total y por zona)
    desactualizada_resultados.csv   (diferencias y parte recuperada con IC95%)

COMO CORRER

    cd src/05_model && python -u robustez_encuesta_desactualizada.py 2>&1 | tee ../../logs/robustez_encuesta_desactualizada.log
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import robustez_asesor_comun as rc
from construir_pipeline_geo_dmsp import cargar_geoespacial_dmsp

ESCENARIOS = ["E1", "E1_luz", "E2", "E3", "E4"]
COMPARACIONES = [("E2", "E1", "perdida_encuesta_vieja"), ("E2", "E3", "aporte_luz_actualizada"), ("E4", "E3", "valor_luz_al_dia")]


def covariables_2010(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Covariables de 2010 de los hogares de `test`, con las columnas de
    `train` (sin Y) y el Y de `test`."""
    cols = [c for c in train.columns if c != "Y"]
    ola1 = rc.panel_ola(1)
    faltan = [c for c in cols if c not in ola1.columns]
    assert not faltan, f"columnas del archivo de entrenamiento que no estan en el panel: {faltan}"
    ola1 = ola1.set_index("consecutivo", drop=False)
    assert test["consecutivo"].isin(ola1.index).all(), "hay hogares de prueba sin fila en 2010"
    x10 = ola1.loc[test["consecutivo"], cols].reset_index(drop=True)
    distintos = {c: (str(x10[c].dtype), str(train[c].dtype)) for c in cols if x10[c].dtype != train[c].dtype}
    assert not distintos, f"tipos distintos a los del archivo de entrenamiento: {distintos}"
    x10["Y"] = test["Y"].values

    cmp_cols = [c for c in cols if c != "consecutivo"]
    comunes = train.set_index("consecutivo")
    comunes = comunes.loc[comunes.index.isin(test["consecutivo"]), cmp_cols]
    nuevas = x10.set_index("consecutivo").loc[comunes.index, cmp_cols]
    distintas = [c for c in cmp_cols if not (comunes[c].eq(nuevas[c]) | (comunes[c].isna() & nuevas[c].isna())).all()]
    assert not distintas, f"covariables 2010 reconstruidas que no coinciden con el archivo de entrenamiento: {distintas}"
    print(f"  covariables 2010: {len(x10):,} hogares; {len(comunes):,} verificados contra el archivo de entrenamiento", flush=True)
    return x10


def con_luz(x: pd.DataFrame, geo: pd.DataFrame, anio: int, columnas: list) -> pd.DataFrame:
    luz = geo[geo["ola"] == anio].drop(columns=["ola"])
    out = x.merge(luz, on="consecutivo", how="left", validate="one_to_one")
    assert len(out) == len(x)
    return out[columnas]


def main() -> None:
    print(f"Nucleos en paralelo: {rc.N_JOBS}", flush=True)
    geo = cargar_geoespacial_dmsp()
    piezas, escenarios, claves, casos, claves_rec, casos_rec = [], [], [], [], [], []

    for definicion, (_, especs) in rc.REGISTROS.items():
        registro, algoritmos = rc.cargar_registro(definicion)
        for espec, (e_base, e_geo) in especs.items():
            print(f"\n=== {definicion} {espec} ===", flush=True)
            train_b, test_b = rc.cargar(e_base, "2010_2013"), rc.cargar(e_base, "2013_2016")
            train_g, test_g = rc.cargar(e_geo, "2010_2013"), rc.cargar(e_geo, "2013_2016")
            x10 = covariables_2010(test_b, train_b)
            datos = {
                "E1": (e_base, train_b, test_b),
                "E1_luz": (e_geo, train_g, test_g),
                "E2": (e_base, train_b, x10),
                "E3": (e_geo, train_g, con_luz(x10, geo, 2013, list(train_g.columns))),
                "E4": (e_geo, train_g, con_luz(x10, geo, 2010, list(train_g.columns))),
            }
            y, zona, clusters = test_b["Y"].values, test_b["zona"].values, rc.cluster_id(test_b)

            for algoritmo in algoritmos:
                corto = rc.nombre_corto(algoritmo)
                planes = [(esc, s) for esc in ESCENARIOS for s in rc.SEMILLAS]
                ajustes = rc.en_paralelo(rc.delayed(rc.ajustar_y_predecir)(algoritmo, datos[esc][0], registro, datos[esc][1], datos[esc][2], s)
                                         for esc, s in planes)
                prom = {}
                for esc in ESCENARIOS:
                    por_semilla = {s: p for (e, s), p in zip(planes, ajustes) if e == esc}
                    prom[esc] = rc.promedio_semillas(por_semilla)
                    for s, p in por_semilla.items():
                        piezas.append(pd.DataFrame({"definicion": definicion, "especificacion": espec, "algoritmo": corto,
                                                    "escenario": esc, "semilla": s,
                                                    "consecutivo": test_b["consecutivo"].values, "proba": p}))
                    for z in ["Total"] + sorted(set(zona)):
                        m = np.ones(len(y), bool) if z == "Total" else zona == z
                        escenarios.append({"definicion": definicion, "especificacion": espec, "algoritmo": corto,
                                           "escenario": esc, "zona": z, "n_hogares": int(m.sum()),
                                           "prevalencia": round(y[m].mean(), 4),
                                           "ap": round(average_precision_score(y[m], prom[esc][m]), 4),
                                           "auc": round(roc_auc_score(y[m], prom[esc][m]), 4),
                                           **(rc.metricas_por_semilla(y, por_semilla) if z == "Total" else {})})

                for z in ["Total"] + sorted(set(zona)):
                    m = np.ones(len(y), bool) if z == "Total" else zona == z
                    yz, cz = y[m], clusters[m]
                    clave = {"definicion": definicion, "especificacion": espec, "algoritmo": corto, "zona": z}
                    for ref, var, nombre in COMPARACIONES:
                        claves.append({**clave, "medida": nombre, "referencia": ref, "variante": var})
                        casos.append((yz, prom[ref][m], prom[var][m], cz))
                    claves_rec.append({**clave, "medida": "parte_recuperada"})
                    casos_rec.append((yz, prom["E1"][m], prom["E2"][m], prom["E3"][m], cz))
                ap = {e: average_precision_score(y, prom[e]) for e in ESCENARIOS}
                print(f"  {corto:22s} " + "  ".join(f"{e}: {ap[e]:.3f}" for e in ESCENARIOS), flush=True)

    print("Bootstrap de las diferencias...", flush=True)
    resultados = [{**k, **r} for k, r in zip(claves, rc.comparar_varios(casos, rc.mu.RANDOM_STATE))]
    base_rec = rc.mu.RANDOM_STATE + len(casos)
    recuperadas = rc.en_paralelo(rc.delayed(parte_recuperada)(*caso, base_rec + i) for i, caso in enumerate(casos_rec))
    resultados += [{**k, **r} for k, r in zip(claves_rec, recuperadas)]
    rc.guardar(pd.concat(piezas, ignore_index=True), "desactualizada_predicciones.parquet")
    rc.guardar(pd.DataFrame(escenarios), "desactualizada_escenarios.csv")
    res = pd.DataFrame(resultados)
    rc.guardar(res, "desactualizada_resultados.csv")
    print(res[res.zona == "Total"].drop(columns=["referencia", "variante"], errors="ignore").to_string(index=False), flush=True)


def parte_recuperada(y, p1, p2, p3, clusters, semilla) -> dict:
    """(E3 - E2) / (E1 - E2) con IC95% por bootstrap pareado por comunidad
    (mismos remuestreos para los tres escenarios)."""
    rng = np.random.default_rng(semilla)
    ap = lambda yy, pp: average_precision_score(yy, pp)
    perdida = ap(y, p1) - ap(y, p2)
    punto = (ap(y, p3) - ap(y, p2)) / perdida if perdida > 0 else np.nan
    codigos, inverso = np.unique(clusters, return_inverse=True)
    indices = [np.flatnonzero(inverso == c) for c in range(len(codigos))]
    razones, b = [], 0
    while b < rc.N_BOOT:
        idx = np.concatenate([indices[c] for c in rng.integers(0, len(codigos), size=len(codigos))])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        den = ap(yb, p1[idx]) - ap(yb, p2[idx])
        razones.append((ap(yb, p3[idx]) - ap(yb, p2[idx])) / den if den > 0 else np.nan)
        b += 1
    razones = np.array(razones)
    validas = razones[~np.isnan(razones)]
    low, high = np.percentile(validas, [2.5, 97.5]) if len(validas) else (np.nan, np.nan)
    return {"n_hogares": len(y), "perdida_ap": round(perdida, 4),
            "parte_recuperada": round(punto, 4) if not np.isnan(punto) else np.nan,
            "parte_recuperada_ci95_low": round(low, 4), "parte_recuperada_ci95_high": round(high, 4),
            "pct_remuestras_sin_perdida": round(float(np.isnan(razones).mean()), 4)}


if __name__ == "__main__":
    main()
