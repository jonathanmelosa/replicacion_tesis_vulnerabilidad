"""
robustez_encuesta_desactualizada_diagnostico.py
===============================================
Diagnostico de una anomalia de `robustez_encuesta_desactualizada.py`: bajo
IPM, predecir la entrada 2013->2016 con la encuesta de 2010 (E2) da MAYOR
AUC-PR que con la de 2013 (E1), significativo en la logistica y XGBoost.

Por que NO es el cambio de medicion de barreras de salud del IPM: ese
cambio afecta el resultado y la seleccion de la muestra, que son identicos
en E1 y E2 (mismo modelo entrenado, mismos hogares, mismo resultado 2016);
entre las covariables no hay ningun indicador del IPM. Lo unico que
cambia entre E1 y E2 son las covariables (2013 frente a 2010).

Hipotesis: cambio de cobertura de las covariables entre olas. Varias
variables tienen tasas de dato faltante muy distintas en 2010 y 2013 para
los MISMOS hogares (p. ej. estrato verificado: 45% sin dato en 2010, 0% en
2013). El modelo aprendio con la estructura de faltantes de 2010; las
covariables 2010 de los hogares de prueba la comparten, las de 2013 no.

Partes
------
1. Cambio por covariable entre 2010 y 2013, en los mismos hogares de
   prueba: tasa de dato faltante en cada ola y diferencia de medias
   estandarizada (numericas). Regla para marcar una variable como
   "cambiada": |diferencia de faltantes| >= 0.10 o |DME| >= 0.25.
2. Prueba de la hipotesis: se repiten E1 y E2 SIN las variables cambiadas
   (se quitan del entrenamiento y de la prediccion), cinco algoritmos, 5
   semillas. Si la ventaja de E2 bajo IPM desaparece, el cambio de
   cobertura la explica. Pobreza monetaria entra como control.
   Resultado (2026-10-07): NO desaparece; la hipotesis se descarta.
3. Descomposicion por grupos, con las predicciones originales de E1 y E2
   (`desactualizada_predicciones.parquet`): hogares de prueba que salieron
   de la pobreza en el periodo anterior (no estan en el archivo de
   entrenamiento de su definicion) frente a los que no eran pobres en
   2010. Para E1 y E2: razon del riesgo predicho medio salio/no frente a
   la razon de tasas observada (cuanto separa el modelo a los dos grupos),
   y diferencia de AUC-PR E1 - E2 DENTRO de cada grupo con IC95% por
   bootstrap por comunidad (cuanto aporta la encuesta al dia una vez se
   descuenta la separacion entre grupos). Explicacion que soporta: la
   encuesta 2010 se tomo cuando los hogares que salieron eran pobres, asi
   que "recuerda" su historia y los separa mejor del resto; bajo IPM la
   encuesta al dia no gana dentro de los grupos (las privaciones cambian
   lento) y domina la historia; bajo pobreza monetaria la encuesta al dia
   gana dentro de los grupos y domina.



INPUTS

    los mismos de robustez_encuesta_desactualizada.py
    data/processed/benchmark_resultados/robustez_asesor_2026_10/desactualizada_escenarios.csv (E1/E2 originales, referencia)
    data/processed/benchmark_resultados/robustez_asesor_2026_10/desactualizada_predicciones.parquet

OUTPUTS (data/processed/benchmark_resultados/robustez_asesor_2026_10/)

    diagnostico_cambio_covariables.csv
    diagnostico_e1_e2_sin_cambiadas.csv
    diagnostico_e1_e2_por_grupo.csv

COMO CORRER

    cd src/05_model && python -u robustez_encuesta_desactualizada_diagnostico.py 2>&1 | tee../../logs/robustez_encuesta_desactualizada_diagnostico.log
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import robustez_asesor_comun as rc
from robustez_encuesta_desactualizada import covariables_2010

UMBRAL_FALTANTES = 0.10
UMBRAL_DME = 0.25
NO_COVARIABLES = {"Y", "consecutivo", "consecutivo_c", "llave_compuesta"}


def cambio_covariables(x10: pd.DataFrame, x13: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c in [c for c in x13.columns if c not in NO_COVARIABLES]:
        a, b = x10[c], x13[c]
        fila = {"variable": c, "faltante_2010": round(a.isna().mean(), 4), "faltante_2013": round(b.isna().mean(), 4)}
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            a, b = a.astype(float), b.astype(float)
            s = np.sqrt((a.var() + b.var()) / 2)
            fila["dme"] = round((b.mean() - a.mean()) / s, 4) if s > 0 else np.nan
        filas.append(fila)
    out = pd.DataFrame(filas)
    out["dif_faltantes"] = out["faltante_2013"] - out["faltante_2010"]
    out["cambiada"] = (out["dif_faltantes"].abs() >= UMBRAL_FALTANTES) | (out["dme"].abs() >= UMBRAL_DME)
    return out


def por_grupo() -> pd.DataFrame:
    """Parte 3: descomposicion de E1 - E2 entre hogares que salieron de la
    pobreza en el periodo anterior y los demas (ver docstring)."""
    pred = pd.read_parquet(rc.OUT_DIR / "desactualizada_predicciones.parquet")
    pred = pred[pred["escenario"].isin(["E1", "E2"])]
    filas, casos = [], []
    for definicion, (_, especs) in rc.REGISTROS.items():
        e_base = especs["A"][0]
        test = rc.cargar(e_base, "2013_2016")[["consecutivo", "consecutivo_c", "Y"]]
        test["salio"] = ~test["consecutivo"].isin(rc.cargar(e_base, "2010_2013")["consecutivo"])
        razon_obs = test.loc[test.salio, "Y"].mean() / test.loc[~test.salio, "Y"].mean()
        sub = pred[pred["definicion"] == definicion]
        prom = sub.groupby(["especificacion", "algoritmo", "escenario", "consecutivo"])["proba"].mean().unstack("escenario")
        for (espec, algoritmo), w in prom.groupby(level=[0, 1]):
            w = test.merge(w.droplevel([0, 1]), left_on="consecutivo", right_index=True, validate="one_to_one")
            fila = {"definicion": definicion, "especificacion": espec, "algoritmo": algoritmo,
                    "n_salio": int(w.salio.sum()), "n_no_pobre_previo": int((~w.salio).sum()),
                    "tasa_salio": round(w.loc[w.salio, "Y"].mean(), 4), "tasa_no_pobre_previo": round(w.loc[~w.salio, "Y"].mean(), 4),
                    "razon_tasas_observada": round(razon_obs, 4)}
            for esc in ["E1", "E2"]:
                fila[f"razon_riesgo_predicho_{esc}"] = round(w.loc[w.salio, esc].mean() / w.loc[~w.salio, esc].mean(), 4)
            filas.append(fila)
            for grupo, m in [("salio", w.salio.values), ("no_pobre_previo", ~w.salio.values)]:
                g = w[m].reset_index(drop=True)
                casos.append((g["Y"].values, g["E2"].values, g["E1"].values, rc.cluster_id(g)))
    dentro = rc.comparar_varios(casos, rc.mu.RANDOM_STATE + 1000)
    for i, fila in enumerate(filas):
        for j, grupo in enumerate(["salio", "no_pobre_previo"]):
            r = dentro[2 * i + j]
            fila[f"ap_e1_{grupo}"], fila[f"ap_e2_{grupo}"] = r["ap_var"], r["ap_ref"]
            fila[f"dif_e1_menos_e2_{grupo}"] = r["delta_ap"]
            fila[f"dif_{grupo}_ci95_low"], fila[f"dif_{grupo}_ci95_high"] = r["delta_ap_ci95_low"], r["delta_ap_ci95_high"]
    return pd.DataFrame(filas)


def main() -> None:
    print(f"Nucleos en paralelo: {rc.N_JOBS}", flush=True)
    cambios, casos, claves = [], [], []
    ref = pd.read_csv(rc.OUT_DIR / "desactualizada_escenarios.csv")

    for definicion, (_, especs) in rc.REGISTROS.items():
        registro, algoritmos = rc.cargar_registro(definicion)
        for espec, (e_base, _) in especs.items():
            print(f"\n=== {definicion} {espec} ===", flush=True)
            train, test = rc.cargar(e_base, "2010_2013"), rc.cargar(e_base, "2013_2016")
            x10 = covariables_2010(test, train)

            tabla = cambio_covariables(x10, test).assign(definicion=definicion, especificacion=espec)
            cambios.append(tabla)
            quitar = tabla.loc[tabla["cambiada"], "variable"].tolist()
            print(f"  variables cambiadas: {len(quitar)} de {len(tabla)}", flush=True)

            sin = {k: df.drop(columns=quitar) for k, df in [("train", train), ("E1", test), ("E2", x10)]}
            y, clusters = test["Y"].values, rc.cluster_id(test)
            for algoritmo in algoritmos:
                corto = rc.nombre_corto(algoritmo)
                planes = [(esc, s) for esc in ["E1", "E2"] for s in rc.SEMILLAS]
                ajustes = rc.en_paralelo(rc.delayed(rc.ajustar_y_predecir)(algoritmo, e_base, registro, sin["train"], sin[esc], s)
                                         for esc, s in planes)
                prom = {esc: rc.promedio_semillas({s: p for (e, s), p in zip(planes, ajustes) if e == esc}) for esc in ["E1", "E2"]}
                orig = ref[(ref.definicion == definicion) & (ref.especificacion == espec) & (ref.algoritmo == corto)
                           & (ref.zona == "Total")].set_index("escenario")["ap"]
                claves.append({"definicion": definicion, "especificacion": espec, "algoritmo": corto,
                               "n_variables_quitadas": len(quitar),
                               "ap_e1_original": orig["E1"], "ap_e2_original": orig["E2"],
                               "dif_e1_menos_e2_original": round(orig["E1"] - orig["E2"], 4)})
                casos.append((y, prom["E2"], prom["E1"], clusters))
                print(f"  {corto:22s} original E1-E2 {orig['E1'] - orig['E2']:+.3f}   sin cambiadas E1-E2 "
                      f"{average_precision_score(y, prom['E1']) - average_precision_score(y, prom['E2']):+.3f}", flush=True)

    print("\nBootstrap de las diferencias...", flush=True)
    res = pd.DataFrame([{**k, **{f"sin_cambiadas_{c}": v for c, v in r.items()}}
                        for k, r in zip(claves, rc.comparar_varios(casos, rc.mu.RANDOM_STATE))])
    res = res.rename(columns={"sin_cambiadas_ap_ref": "ap_e2_sin_cambiadas", "sin_cambiadas_ap_var": "ap_e1_sin_cambiadas",
                              "sin_cambiadas_delta_ap": "dif_e1_menos_e2_sin_cambiadas"})
    rc.guardar(pd.concat(cambios, ignore_index=True), "diagnostico_cambio_covariables.csv")
    rc.guardar(res, "diagnostico_e1_e2_sin_cambiadas.csv")
    print(res[["definicion", "especificacion", "algoritmo", "dif_e1_menos_e2_original", "dif_e1_menos_e2_sin_cambiadas",
               "sin_cambiadas_delta_ap_ci95_low", "sin_cambiadas_delta_ap_ci95_high"]].to_string(index=False), flush=True)

    print("\n=== Descomposicion por grupos ===", flush=True)
    grupos = por_grupo()
    rc.guardar(grupos, "diagnostico_e1_e2_por_grupo.csv")
    print(grupos.drop(columns=[c for c in grupos.columns if c.startswith(("n_", "tasa_", "ap_"))]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
