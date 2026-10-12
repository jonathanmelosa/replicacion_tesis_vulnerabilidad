"""
robustez_historia_hogar.py
==========================
¿Aprovecha el modelo la estructura de panel?

Hoy el panel se usa para construir el resultado y separar los periodos;
las covariables son solo los niveles de la ola base. Esta prueba agrega
al modelo lo que el panel sabe del pasado del hogar y mide cuanto mejora
la prediccion.

Hipotesis: la historia del hogar mejora la prediccion mas que la luz
nocturna, sobre todo porque distingue a quienes acaban de salir de la
pobreza (1,271 de los 3,191 hogares de prueba; tasa de entrada 32.7%
frente a 11.5% de los demas).

Diseño
------
  - Poblacion: los 3,191 hogares no pobres en 2013 (pobreza monetaria);
    resultado: pobre en 2016. Modelos A y B, cinco algoritmos,
    hiperparametros del paper (`registro_modelos_fbeta2_cv10.csv`).
  - Validacion cruzada estratificada de 10 particiones x 5 semillas
    DENTRO de 2013->2016 (cada hogar aparece una sola vez). No se puede
    evaluar hacia adelante: 2010 es la primera ola de la ELCA y no hay
    historia con la cual entrenar en 2010->2013.
  - Variantes, cada una agregada a las covariables de 2013:
      * base: sin historia (referencia).
      * estado_previo: `pobre_ingreso_2010` (0/1) y `brecha_lp_ingreso_2010`.
                       Viene del ingreso; en el Modelo B se reporta de
                       todas formas como
                       lo que ganaria un sistema que guarda el historial de
                       pobreza del hogar.
      * cambios: `cambio_<v>` = valor 2013 - valor 2010 de cada
                       variable NUMERICA y NO MONETARIA del nucleo SHAP de
                       la especificacion (`nucleo_por_especificacion.csv`,
                       columna en_A / en_B). Es la variante limpia para el
                       Modelo B.
      * ambas: estado_previo + cambios.
  - Diferencia de AUC-PR frente a base, con IC95% por bootstrap pareado
    por comunidad, en toda la muestra y por grupo (salio de la pobreza en
    2013 / no pobre en 2010).

INPUTS

    data/processed/benchmark_train_test/modelo_{A,B}_2013_2016.parquet
    data/processed/benchmark_consolidado_elca_longitudinal.parquet (ola 2010)
    data/processed/benchmark_resultados/nucleo_por_especificacion.csv
    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv

OUTPUTS (data/processed/benchmark_resultados/robustez_asesor_2026_10/)

    historia_variables.csv (que variables entra en cada variante)
    historia_predicciones.parquet
    historia_resultados.csv (diferencias con IC95%, total y por grupo)

COMO CORRER

    cd src/05_model && python -u robustez_historia_hogar.py 2>&1 | tee../../logs/robustez_historia_hogar.log
"""

import numpy as np
import pandas as pd

import robustez_asesor_comun as rc

VARIANTES = ["base", "estado_previo", "cambios", "ambas"]
MONETARIAS = {"ingreso_percapita_hogar_real", "gasto_percapita_hogar_real", "brecha_lp_ingreso", "brecha_lp_gasto"}


def variables_cambio(espec: str, df: pd.DataFrame) -> list:
    nucleo = pd.read_csv(rc.NUCLEO_PATH)
    candidatas = nucleo.loc[nucleo[f"en_{espec}"], "variable"]
    return [v for v in candidatas if v not in MONETARIAS and v in df.columns
            and pd.api.types.is_numeric_dtype(df[v]) and not pd.api.types.is_bool_dtype(df[v])]


def construir_variantes(espec: str) -> tuple[dict, pd.DataFrame]:
    test = rc.cargar(espec, "2013_2016")
    cambios = variables_cambio(espec, test)
    ola1 = rc.panel_ola(1, ["pobre_ingreso", "brecha_lp_ingreso"] + cambios).set_index("consecutivo")
    falta = set(test["consecutivo"]) - set(ola1.index)
    assert not falta, f"{len(falta)} hogares de prueba sin fila en 2010"
    previo = ola1.loc[test["consecutivo"]]

    estado = pd.DataFrame({
        "pobre_ingreso_2010": previo["pobre_ingreso"].astype(float).values,
        "brecha_lp_ingreso_2010": previo["brecha_lp_ingreso"].values,
    })
    delta = pd.DataFrame({f"cambio_{v}": test[v].values - previo[v].astype(float).values for v in cambios})
    base = test.reset_index(drop=True)
    variantes = {
        "base": base,
        "estado_previo": pd.concat([base, estado], axis=1),
        "cambios": pd.concat([base, delta], axis=1),
        "ambas": pd.concat([base, estado, delta], axis=1),
    }
    lista = pd.DataFrame([{"especificacion": espec, "variante": "estado_previo", "variable": c} for c in estado.columns]
                         + [{"especificacion": espec, "variante": "cambios", "variable": c} for c in delta.columns])
    salio = estado["pobre_ingreso_2010"].values == 1
    print(f"Modelo {espec}: {len(base):,} hogares; salieron de la pobreza en 2013: {salio.sum():,}; "
          f"variables de cambio: {len(delta.columns)}", flush=True)
    return variantes, lista


def main() -> None:
    registro, algoritmos = rc.cargar_registro("monetaria")
    print(f"Nucleos en paralelo: {rc.N_JOBS}", flush=True)
    piezas, listas, claves, casos = [], [], [], []

    for espec in ["A", "B"]:
        variantes, lista = construir_variantes(espec)
        listas.append(lista)
        base = variantes["base"]
        y, clusters = base["Y"].values, rc.cluster_id(base)
        salio = variantes["estado_previo"]["pobre_ingreso_2010"].values == 1
        for algoritmo in algoritmos:
            corto = rc.nombre_corto(algoritmo)
            prom, por_semilla = {}, {}
            for variante in VARIANTES:
                por_semilla[variante] = rc.oof_semillas(algoritmo, espec, registro, variantes[variante])
                prom[variante] = rc.promedio_semillas(por_semilla[variante])
                for s, p in por_semilla[variante].items():
                    piezas.append(pd.DataFrame({"especificacion": espec, "algoritmo": corto,
                                                "variante": variante, "semilla": s,
                                                "consecutivo": base["consecutivo"].values, "proba": p}))
            for variante in VARIANTES[1:]:
                for grupo, m in [("todos", np.ones(len(y), bool)), ("salio_2013", salio), ("no_pobre_2010", ~salio)]:
                    extra = ({f"var_{k}": v for k, v in rc.metricas_por_semilla(y, por_semilla[variante]).items()}
                             if grupo == "todos" else {})
                    claves.append({"especificacion": espec, "algoritmo": corto, "variante": variante, "grupo": grupo, **extra})
                    casos.append((y[m], prom["base"][m], prom[variante][m], clusters[m]))
            print(f"  {espec} {corto:22s} ajustes listos", flush=True)

    print("Bootstrap de las diferencias...", flush=True)
    resultados = pd.DataFrame([{**k, **r} for k, r in zip(claves, rc.comparar_varios(casos, rc.mu.RANDOM_STATE))])
    rc.guardar(pd.concat(listas, ignore_index=True), "historia_variables.csv")
    rc.guardar(pd.concat(piezas, ignore_index=True), "historia_predicciones.parquet")
    rc.guardar(resultados, "historia_resultados.csv")
    print(resultados[resultados.grupo == "todos"][["especificacion", "algoritmo", "variante", "ap_ref", "ap_var", "delta_ap",
                                                   "delta_ap_ci95_low", "delta_ap_ci95_high"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
