"""
sensibilidad_shap_nucleo.py
=====================================
Pruebas de sensibilidad del NUCLEO SHAP y de su DIRECCION (Seccion 5.2 de
`main.tex`). Los criterios con que se definen ambos (masa SHAP del 80%,
presencia en >= 5 de 10 combinaciones, |rho| >= 0.10, proporcion de signo
>= 0.8) son elecciones del analista; este script muestra cuanto cambian
las conclusiones si se mueven, para poder afirmar que no dependen de esas
elecciones o para saber en cuales si.

Tres ejercicios (todos sobre resultados ya calculados, sin reentrenar)

  1. Nucleo (fuente Monetaria 2010->2013, la de la tabla del nucleo):
     malla masa acumulada {0.70, 0.80, 0.90} x combinaciones minimas
     {4, 5, 6} de 10. Para cada celda: tamano del nucleo, cuantas de las
     22 variables base conserva, y cuantas de las 7 universales (nucleo
     en las 4 fuentes definicion x ventana con esos mismos parametros).
  2. Nucleo por especificacion: el nucleo con el Modelo A (incluye
     ingreso/gasto) y con el Modelo B (sin ellos) por separado, con
     mayoria simple de 5 (>= 3 de 5 combinaciones). Motivo: una variable
     exclusiva del Modelo A (ingreso, gasto y sus brechas) solo puede
     sumar 5 de 10 en la regla original, es decir, necesitaria estar en
     TODAS las combinaciones del A; la regla la penaliza por diseno.
  3. Direccion: malla |rho| minimo {0.05, 0.10, 0.15, 0.20} x proporcion
     minima de signo {0.70, 0.80, 0.90} (minimo de 3 combinaciones
     claras por fuente). Para cada celda, cuantas de las variables
     numericas del nucleo de 22 quedan "consistentes" en las 4 fuentes, y
     en cuantas celdas de la malla cada variable es consistente.

INPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia_ab.csv
    data/processed/benchmark_resultados/diagnostico_shap_importancia_2013_2016.csv
    data/processed/benchmark_resultados/diagnostico_shap_importancia_ipm.csv
    data/processed/benchmark_resultados/diagnostico_shap_importancia_ipm_2013_2016.csv
    data/processed/benchmark_resultados/diagnostico_shap_nucleo_perfil.csv
    data/processed/benchmark_resultados/diagnostico_shap_signo.csv

OUTPUTS

    data/processed/benchmark_resultados/sensibilidad_nucleo_shap.csv
    data/processed/benchmark_resultados/nucleo_por_especificacion.csv
    data/processed/benchmark_resultados/sensibilidad_signo_nucleo.csv
    data/processed/benchmark_resultados/sensibilidad_signo_nucleo_por_variable.csv

COMO CORRER

    cd src/05_model && python sensibilidad_shap_nucleo.py
"""

import itertools
import re

import numpy as np
import pandas as pd

import modelo_utils as mu
from comparar_nucleos_shap import FUENTES

R = mu.RESULTADOS_DIR
RUTA_NUCLEO_BASE = R / "diagnostico_shap_nucleo_perfil.csv"
RUTA_SIGNO = R / "diagnostico_shap_signo.csv"

MASAS = [0.70, 0.80, 0.90]
MIN_COMBINACIONES = [4, 5, 6]
RHOS = [0.05, 0.10, 0.15, 0.20]
PROPORCIONES = [0.70, 0.80, 0.90]
MIN_CLARAS = 3
UNIVERSALES = [
    "material_pisos_hogar", "nivel_educ_max_hogar", "nivel_educ_ordinal_jefe",
    "personas_por_cuarto_hogar", "razon_dependencia_demografica",
    "riqueza_pca_hogar", "tvip_puntaje_directo_hogar",
]
CATEGORICAS_NUCLEO = {"material_pisos_hogar", "estado_civil_jefe"}
FUENTE_TABLA = "Monetaria_2010-2013(test 2013-2016)"


def _mapa_variable_base(parquet_categoricas):
    categoricas = pd.read_parquet(parquet_categoricas).select_dtypes(include="object").columns.tolist()
    categoricas_por_largo = sorted(categoricas, key=len, reverse=True)

    def variable_base(columna: str) -> str:
        columna = re.sub(r"^(num__|cat__)", "", columna)
        columna = re.sub(r"^missingindicator_", "", columna)
        for c in categoricas_por_largo:
            if columna == c or columna.startswith(c + "_"):
                return c
        return columna

    return variable_base


def conjuntos_80(cfg: dict, masa: float, espec_prefijo: str = None) -> dict:
    """{combinacion: set de variables base dentro de la `masa` acumulada}.
    Misma logica que `diagnostico_shap_nucleo_perfil.py`, con la masa como
    parametro y, opcionalmente, solo las combinaciones de una especificacion
    (`espec_prefijo`: 'A' o 'B'; empieza por esa letra)."""
    shap_raw = pd.read_csv(cfg["shap_csv"])
    shap_raw["base"] = shap_raw["variable"].map(_mapa_variable_base(cfg["parquet_categoricas"]))
    if espec_prefijo:
        shap_raw = shap_raw[shap_raw["especificacion"].str.startswith(espec_prefijo)]
    salida = {}
    for (algoritmo, espec), g in shap_raw.groupby(["algoritmo", "especificacion"]):
        agregado = g.groupby("base")["shap_abs_medio"].sum().sort_values(ascending=False)
        acumulada = agregado.cumsum() / agregado.sum()
        n = int((acumulada < masa).sum() + 1)
        salida[f"{algoritmo}_{espec}"] = set(agregado.index[:n])
    return salida


def nucleo(cfg: dict, masa: float, minimo: int, espec_prefijo: str = None) -> set:
    conj = conjuntos_80(cfg, masa, espec_prefijo)
    conteo = pd.Series({v: sum(v in c for c in conj.values()) for v in set().union(*conj.values())})
    return set(conteo[conteo >= minimo].index)


def ejercicio_nucleo(base22: set) -> pd.DataFrame:
    filas = []
    for masa, minimo in itertools.product(MASAS, MIN_COMBINACIONES):
        nucleos = {nombre: nucleo(cfg, masa, minimo) for nombre, cfg in FUENTES.items()}
        en_4 = set.intersection(*nucleos.values())
        filas.append({
            "masa": masa, "min_combinaciones": minimo,
            "n_nucleo_tabla": len(nucleos[FUENTE_TABLA]),
            "de_las_22_conserva": len(base22 & nucleos[FUENTE_TABLA]),
            "n_en_las_4_fuentes": len(en_4),
            "de_las_7_universales_en_las_4": len(set(UNIVERSALES) & en_4),
            # Nombres (2026-10-01): el texto cita cuales sobreviven a los
            # criterios mas estrictos, no solo cuantas.
            "universales_en_las_4": ";".join(sorted(set(UNIVERSALES) & en_4)),
        })
    return pd.DataFrame(filas)


def ejercicio_por_especificacion(base22: set) -> pd.DataFrame:
    cfg = FUENTES[FUENTE_TABLA]
    nucleo_a = nucleo(cfg, 0.80, 3, "A")
    nucleo_b = nucleo(cfg, 0.80, 3, "B")
    filas = [{"variable": v, "en_nucleo_22": v in base22, "en_A": v in nucleo_a, "en_B": v in nucleo_b}
             for v in sorted(base22 | nucleo_a | nucleo_b)]
    return pd.DataFrame(filas)


def ejercicio_signo(nucleo_vars: list) -> tuple:
    signo = pd.read_csv(RUTA_SIGNO)
    cols = signo[signo["tipo"] == "columna"].copy()
    cols["v"] = cols["variable"].str.replace("^num__", "", regex=True)
    numericas = [v for v in nucleo_vars if v not in CATEGORICAS_NUCLEO]

    def consistente(v: str, rho: float, prop: float) -> bool:
        signos = set()
        for _, g in cols[cols["v"] == v].groupby("fuente"):
            n_neg = int((g["corr_spearman"] <= -rho).sum())
            n_pos = int((g["corr_spearman"] >= rho).sum())
            m = n_neg + n_pos
            if m < MIN_CLARAS or max(n_neg, n_pos) / m < prop:
                return False
            signos.add("-" if n_neg >= n_pos else "+")
        return len(signos) == 1

    filas, por_variable = [], {v: 0 for v in numericas}
    for rho, prop in itertools.product(RHOS, PROPORCIONES):
        ok = [v for v in numericas if consistente(v, rho, prop)]
        for v in ok:
            por_variable[v] += 1
        filas.append({"rho_min": rho, "prop_min": prop, "n_consistentes": len(ok), "de": len(numericas)})
    celdas = len(RHOS) * len(PROPORCIONES)
    por_var = pd.DataFrame({"variable": list(por_variable), "celdas_consistente": list(por_variable.values()),
                            "de_celdas": celdas}).sort_values("celdas_consistente", ascending=False)
    return pd.DataFrame(filas), por_var


def main() -> None:
    base = pd.read_csv(RUTA_NUCLEO_BASE)
    base22 = set(base["variable"])

    print("=== 1. Nucleo: malla masa x minimo de combinaciones ===")
    sens = ejercicio_nucleo(base22)
    sens.to_csv(R / "sensibilidad_nucleo_shap.csv", index=False)
    print(sens.to_string(index=False))

    print("\n=== 2. Nucleo por especificacion (A con ingreso / B sin ingreso; >=3 de 5) ===")
    por_espec = ejercicio_por_especificacion(base22)
    por_espec.to_csv(R / "nucleo_por_especificacion.csv", index=False)
    print(por_espec.to_string(index=False))
    print(f"    nucleo A: {int(por_espec.en_A.sum())} | nucleo B: {int(por_espec.en_B.sum())} | "
          f"de las 22, en A: {int((por_espec.en_nucleo_22 & por_espec.en_A).sum())}, "
          f"en B: {int((por_espec.en_nucleo_22 & por_espec.en_B).sum())}, en ambos: "
          f"{int((por_espec.en_nucleo_22 & por_espec.en_A & por_espec.en_B).sum())}")

    print("\n=== 3. Direccion: malla |rho| x proporcion de signo ===")
    celdas, por_var = ejercicio_signo(base["variable"].tolist())
    celdas.to_csv(R / "sensibilidad_signo_nucleo.csv", index=False)
    por_var.to_csv(R / "sensibilidad_signo_nucleo_por_variable.csv", index=False)
    print(celdas.to_string(index=False))
    print()
    print(por_var.to_string(index=False))


if __name__ == "__main__":
    main()
