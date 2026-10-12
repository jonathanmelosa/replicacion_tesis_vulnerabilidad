"""
Construccion de la matriz de entrenamiento/prueba del modelo benchmark de
prediccion de ENTRADA a la pobreza.

Poblacion y outcome (ver metodologia, punto 1)
--------------------------------------------------
El modelo se entrena y evalua exclusivamente sobre hogares NO pobres en
la ola base (un hogar ya pobre no puede "entrar" en pobreza, esa es la
categoria "siempre pobre"). Outcome:

    Y_{i,t+1} = 1[hogar i es pobre en t+1 | no pobre en t]

Particion temporal (punto 2) y emparejamiento de hogares entre olas
--------------------------------------------------------------------------
  - **Principal**: entrenar con 2010->2013 (covariables ola 1, outcome
    ola 2), evaluar con 2013->2016 (covariables ola 2, outcome ola 3) --
    holdout temporal hacia adelante.
  - **Reversa** (robustez): mismos 2 conjuntos, roles de train/test
    invertidos.
  - **Pooled** (robustez): concatenacion de ambos conjuntos con dummy de
    periodo, para k-fold agrupado por hogar (no se hace la validacion
    cruzada en este script, se deja el dataset listo con la columna
    `consecutivo` para agrupar).

Emparejamiento de hogares: se usa **`consecutivo`, solo matches 1 a 1**
-- MISMA politica ya confirmada y usada en las matrices de transicion de
`build_pobreza_desagregaciones.py` (`construir_matriz_transicion`): los
hogares que se DIVIDIERON entre olas (consecutivo duplicado en la ola
inicial o final) se EXCLUYEN, para no inventar un mapeo 1 a 1 arbitrario
entre sub-hogares. Se reporta el numero de hogares excluidos por esta
razon.

Ingreso como covariable: Modelo A vs. Modelo B (punto 3)
--------------------------------------------------------------------------
Se generan DOS versiones de cada conjunto de covariables:
  - **Modelo A**: incluye `ingreso_percapita_hogar_real`,
    `gasto_percapita_hogar_real`, `brecha_lp_ingreso`, `brecha_lp_gasto`
    (nivel/brecha a la LP en la ola base) -- enfoque de vulnerabilidad a
    la pobreza (Chaudhuri, Jalan y Suryahadi, 2002).
  - **Modelo B**: excluye esas 4 columnas, solo covariables no
    monetarias (Personas, Comunidades, Niños, Choques, Hogar).
No se excluyen `pobre_ingreso`/`pobre_extremo_ingreso`/`pobre_gasto`/
`pobre_extremo_gasto`/`lp`/`li`/`concuerdan_ingreso_gasto` de NINGUN
conjunto de covariables porque estas son directamente el resultado (o
subproducto directo del resultado) de la ola BASE -- usarlas como
covariable no séria fuga de outcome del periodo siguiente (`Y` es la ola
FINAL), pero `pobre_ingreso` de la ola base ya esta fijo en 0 para toda
la muestra (poblacion no-pobre), asi que no aporta varianza -- se
eliminan de las covariables por ser constantes, no por fuga.

Output
------
Para cada transicion (`2010_2013`, `2013_2016`) y cada especificacion
(`A` con ingreso, `B` sin ingreso): un parquet en
`data/processed/benchmark_train_test/`, con columnas de identidad
(`consecutivo`, `zona`), las covariables de la ola base, y `Y` (outcome).
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSOLIDADO_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_consolidado_elca_longitudinal.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"

COLS_MONETARIAS_MODELO_A = [
    "ingreso_percapita_hogar_real", "gasto_percapita_hogar_real",
    "brecha_lp_ingreso", "brecha_lp_gasto",
]
# Constantes dentro de la poblacion no-pobre (por construccion, ver docstring)
# o subproducto directo del label de la ola base -- se excluyen de covariables
# en AMBAS especificaciones (A y B), no son parte de la comparacion A/B.
COLS_EXCLUIR_SIEMPRE = [
    "pobre_ingreso", "pobre_extremo_ingreso", "pobre_gasto", "pobre_extremo_gasto",
    "lp", "li", "concuerdan_ingreso_gasto",
    "ingreso_percapita_hogar", "gasto_percapita_hogar",  # nominales, se usa la version real en Modelo A
    "llave", "llave_n16",  # llaves de OTRAS olas (100% NaN en la ola base de esta transicion), no consecutivo
    "ola",  # constante dentro de cada transicion, no aporta varianza
]
COLS_ID = ["consecutivo", "llave", "llave_n16", "ola", "zona"]


def construir_transicion(consolidado: pd.DataFrame, ola_ini: int, ola_fin: int) -> tuple[pd.DataFrame, dict]:
    ini = consolidado[consolidado["ola"] == ola_ini].copy()
    fin = consolidado[consolidado["ola"] == ola_fin][["consecutivo", "pobre_ingreso"]].dropna(subset=["pobre_ingreso"])

    ini_unicos = ini[~ini["consecutivo"].duplicated(keep=False)]
    fin_unicos = fin[~fin["consecutivo"].duplicated(keep=False)]

    n_excluidos = (
        ini["consecutivo"].nunique() - ini_unicos["consecutivo"].nunique()
        + fin["consecutivo"].nunique() - fin_unicos["consecutivo"].nunique()
    )

    panel = ini_unicos.merge(fin_unicos, on="consecutivo", suffixes=("", "_fin"))

    n_total_no_pobre_base = (panel["pobre_ingreso"] == False).sum()  # noqa: E712
    panel = panel[panel["pobre_ingreso"] == False].copy()  # noqa: E712
    panel["Y"] = panel["pobre_ingreso_fin"].astype(int)

    stats = {
        "ola_ini": ola_ini, "ola_fin": ola_fin,
        "n_hogares_ola_ini": len(ini), "n_hogares_ola_fin": len(fin),
        "n_excluidos_por_division": n_excluidos,
        "n_panel_1a1": len(ini_unicos.merge(fin_unicos, on="consecutivo")),
        "n_no_pobre_base": n_total_no_pobre_base,
        "tasa_entrada_pobreza": panel["Y"].mean(),
    }
    return panel, stats


def separar_covariables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_base = [c for c in panel.columns if c not in ("pobre_ingreso_fin", "Y")]
    covariables = panel[cols_base].drop(columns=[c for c in COLS_EXCLUIR_SIEMPRE if c in cols_base])

    cols_modelo_b = [c for c in covariables.columns if c not in COLS_MONETARIAS_MODELO_A]
    modelo_a = covariables.copy()
    modelo_a["Y"] = panel["Y"].values
    modelo_b = covariables[cols_modelo_b].copy()
    modelo_b["Y"] = panel["Y"].values
    return modelo_a, modelo_b


def main() -> None:
    consolidado = pd.read_parquet(CONSOLIDADO_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transiciones = {"2010_2013": (1, 2), "2013_2016": (2, 3)}
    resumen = []

    for nombre, (ola_ini, ola_fin) in transiciones.items():
        panel, stats = construir_transicion(consolidado, ola_ini, ola_fin)
        modelo_a, modelo_b = separar_covariables(panel)

        modelo_a.to_parquet(OUTPUT_DIR / f"modelo_A_{nombre}.parquet", index=False)
        modelo_b.to_parquet(OUTPUT_DIR / f"modelo_B_{nombre}.parquet", index=False)

        stats["n_covariables_modelo_A"] = modelo_a.shape[1] - 1
        stats["n_covariables_modelo_B"] = modelo_b.shape[1] - 1
        resumen.append({"transicion": nombre, **stats})

        print(f"\n=== Transicion {nombre} (ola {ola_ini} -> ola {ola_fin}) ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    pd.DataFrame(resumen).to_csv(OUTPUT_DIR / "resumen_construccion.csv", index=False)
    print(f"\nGuardado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
