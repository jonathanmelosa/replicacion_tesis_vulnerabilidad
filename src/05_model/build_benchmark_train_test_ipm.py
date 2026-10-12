"""
Version del benchmark de prediccion de entrada a la pobreza que usa el
Indice de Pobreza Multidimensional (`pobre_ipm`, ver
`src/04_features/build_ipm_multidimensional.py`) como outcome, en vez de
pobreza monetaria (`pobre_ingreso`)
para probar si el hallazgo de que DMSP-OLS no aporta se sostiene bajo una
definicion de vulnerabilidad distinta.

Espeja exactamente la logica de `build_benchmark_train_test.py` (mismo
emparejamiento 1 a 1 por `consecutivo`, misma poblacion "no pobre en la
ola base", mismo Modelo A/B de covariables) -- unico cambio: la columna
de pobreza usada es `pobre_ipm` (de
`ipm_multidimensional_elca_longitudinal.parquet`, mergeada por
consecutivo+ola) en vez de `pobre_ingreso` (ya presente en el
consolidado). Las covariables X son IDENTICAS al benchmark monetario --
deliberado: asi el unico factor que cambia entre este benchmark y el
original es la definicion del target, no el conjunto de features (aisla
la comparacion). NO se agregan `ipm_score` ni las columnas `priv_*` como
covariables (evita features casi-tautologicas con el propio target).

Especificaciones de salida: `Aipm`/`Bipm` (mismo criterio Modelo A/B con
ingreso vs. sin ingreso que el benchmark monetario) -- nombradas distinto
de "A"/"B" para no pisar los archivos del benchmark original.

OUTPUTS

    data/processed/benchmark_train_test/modelo_{Aipm,Bipm}_{2010_2013,2013_2016}.parquet

COMO CORRER

    python build_benchmark_train_test_ipm.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSOLIDADO_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_consolidado_elca_longitudinal.parquet"
IPM_PATH = PROJECT_ROOT / "data" / "processed" / "ipm_multidimensional_elca_longitudinal.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"

COLS_MONETARIAS_MODELO_A = [
    "ingreso_percapita_hogar_real", "gasto_percapita_hogar_real",
    "brecha_lp_ingreso", "brecha_lp_gasto",
]
# Igual que build_benchmark_train_test.py, mas las columnas propias del
# target monetario (que ya no es el outcome aqui, pero seguirian siendo
# "casi el resultado" de la ola base y no aportan varianza util) y las
# del IPM que no deben entrar como covariable (ver docstring).
COLS_EXCLUIR_SIEMPRE = [
    "pobre_ingreso", "pobre_extremo_ingreso", "pobre_gasto", "pobre_extremo_gasto",
    "lp", "li", "concuerdan_ingreso_gasto",
    "ingreso_percapita_hogar", "gasto_percapita_hogar",
    "llave", "llave_n16", "ola",
    "ipm_score", "pobre_ipm_fin",
]


def construir_transicion(consolidado: pd.DataFrame, ipm: pd.DataFrame, ola_ini: int, ola_fin: int) -> tuple[pd.DataFrame, dict]:
    ini = consolidado[consolidado["ola"] == ola_ini].copy()
    ipm_ini = ipm[ipm["ola"] == ola_ini][["consecutivo", "pobre_ipm"]]
    ipm_fin = ipm[ipm["ola"] == ola_fin][["consecutivo", "pobre_ipm"]].dropna(subset=["pobre_ipm"])

    # BUG evitado: `consecutivo` tiene duplicados DENTRO de una misma ola
    # (hogares que se dividieron -- 532 casos en ola 2).
    # Mergear ANTES de deduplicar multiplicaria filas
    # (producto cartesiano dentro de cada grupo duplicado). Se deduplica
    # cada lado del merge primero -- los duplicados se excluyen de todas
    # formas un par de lineas mas abajo, asi que el valor que les quede
    # (si sobrevivieran) no importa.
    ini_unicos = ini[~ini["consecutivo"].duplicated(keep=False)]
    ipm_ini_unicos = ipm_ini[~ipm_ini["consecutivo"].duplicated(keep=False)]
    ini_unicos = ini_unicos.merge(ipm_ini_unicos, on="consecutivo", how="left")
    fin_unicos = ipm_fin[~ipm_fin["consecutivo"].duplicated(keep=False)]

    n_excluidos = (
        ini["consecutivo"].nunique() - ini_unicos["consecutivo"].nunique()
        + ipm_fin["consecutivo"].nunique() - fin_unicos["consecutivo"].nunique()
    )

    panel = ini_unicos.merge(fin_unicos, on="consecutivo", suffixes=("", "_fin"))

    n_total_no_pobre_base = (panel["pobre_ipm"] == False).sum()  # noqa: E712
    panel = panel[panel["pobre_ipm"] == False].copy()  # noqa: E712
    panel["Y"] = panel["pobre_ipm_fin"].astype(int)

    stats = {
        "ola_ini": ola_ini, "ola_fin": ola_fin,
        "n_hogares_ola_ini": len(ini), "n_hogares_ola_fin": len(ipm_fin),
        "n_excluidos_por_division": n_excluidos,
        "n_panel_1a1": len(ini_unicos.merge(fin_unicos, on="consecutivo")),
        "n_no_pobre_ipm_base": n_total_no_pobre_base,
        "tasa_entrada_ipm": panel["Y"].mean(),
    }
    return panel, stats


def separar_covariables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_base = [c for c in panel.columns if c not in ("pobre_ipm", "pobre_ipm_fin", "Y")]
    covariables = panel[cols_base].drop(columns=[c for c in COLS_EXCLUIR_SIEMPRE if c in cols_base])

    cols_modelo_b = [c for c in covariables.columns if c not in COLS_MONETARIAS_MODELO_A]
    modelo_a = covariables.copy()
    modelo_a["Y"] = panel["Y"].values
    modelo_b = covariables[cols_modelo_b].copy()
    modelo_b["Y"] = panel["Y"].values
    return modelo_a, modelo_b


def main() -> None:
    consolidado = pd.read_parquet(CONSOLIDADO_PATH)
    ipm = pd.read_parquet(IPM_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transiciones = {"2010_2013": (1, 2), "2013_2016": (2, 3)}
    resumen = []

    for nombre, (ola_ini, ola_fin) in transiciones.items():
        panel, stats = construir_transicion(consolidado, ipm, ola_ini, ola_fin)
        modelo_a, modelo_b = separar_covariables(panel)

        modelo_a.to_parquet(OUTPUT_DIR / f"modelo_Aipm_{nombre}.parquet", index=False)
        modelo_b.to_parquet(OUTPUT_DIR / f"modelo_Bipm_{nombre}.parquet", index=False)

        stats["n_covariables_modelo_Aipm"] = modelo_a.shape[1] - 1
        stats["n_covariables_modelo_Bipm"] = modelo_b.shape[1] - 1
        resumen.append({"transicion": nombre, **stats})

        print(f"\n=== Transicion IPM {nombre} (ola {ola_ini} -> ola {ola_fin}) ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    pd.DataFrame(resumen).to_csv(OUTPUT_DIR / "resumen_construccion_ipm.csv", index=False)
    print(f"\nGuardado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
