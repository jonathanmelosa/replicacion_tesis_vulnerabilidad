"""
Chequeo directo de la hipotesis de "estabilidad temporal" planteada en el
informe IPM (Seccion 6.4/7): el modelo con DMSP-OLS ajusta peor en
entrenamiento (2010->2013) pero generaliza mejor a 2013->2016 en TODAS
las semillas probadas -- hipotesis propuesta: la señal DMSP-OLS
(luminosidad nocturna, ligada a infraestructura fisica) es mas ESTABLE
entre olas que los proxies ELCA correlacionados con ella
(n_servicios_publicos_hogar, n_bienes_durables_hogar), y por eso un
modelo que se apoya en ella generaliza mejor de un periodo a otro.

QUE HACE

    Para los hogares emparejados 1 a 1 entre 2010 y 2013 (mismo criterio
    de exclusion de hogares divididos que el resto del proyecto), mide
    2 metricas de estabilidad temporal para cada variable:

    1. Correlacion test-retest: correlacion de Pearson entre el valor en
       2010 y el valor en 2013 del MISMO hogar. Mas cerca de 1 = mas
       estable (el valor de un hogar predice bien su propio valor 3 años
       despues); mas cerca de 0 = mas volatil/ruidoso entre olas.
    2. Cambio absoluto medio estandarizado: |x_2013 - x_2010| / std(x)
       pooled -- cuantos desvios estandar cambia, en promedio, el valor
       de un hogar entre olas (mas bajo = mas estable).

    No reentrena ningun modelo -- es un chequeo puramente descriptivo/
    correlacional sobre los datos ya construidos.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_variabilidad_temporal_dmsp.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_variabilidad_temporal_dmsp.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

VARIABLES_DMSP = [
    "dmsp_stable_lights", "dmsp_avg_vis",
]
VARIABLES_ELCA_PROXY = [
    "n_servicios_publicos_hogar", "n_bienes_durables_hogar",
]


def cargar_emparejados_1a1(hogar: pd.DataFrame, ola_ini: int, ola_fin: int) -> pd.DataFrame:
    """Mismo criterio de emparejamiento que build_benchmark_train_test.py
    -- excluye hogares con consecutivo duplicado en cualquiera de las 2
    olas (hogares divididos)."""
    ini = hogar[hogar["ola"] == ola_ini]
    fin = hogar[hogar["ola"] == ola_fin]
    ini_unicos = ini[~ini["consecutivo"].duplicated(keep=False)]
    fin_unicos = fin[~fin["consecutivo"].duplicated(keep=False)]
    return ini_unicos.merge(fin_unicos, on="consecutivo", suffixes=("_ini", "_fin"))


def main() -> None:
    print("Cargando hogar, hogar_features y variables geoespaciales...")
    hogar = pd.read_parquet(DATA_DIR / "hogar_elca_longitudinal_clean.parquet")
    # n_servicios_publicos_hogar / n_bienes_durables_hogar viven en
    # hogar_features_elca_longitudinal.parquet (build_hogar_features.py),
    # NO en hogar_elca_longitudinal_clean.parquet.
    hogar_features = pd.read_parquet(DATA_DIR / "hogar_features_elca_longitudinal.parquet")[
        ["consecutivo", "ola"] + VARIABLES_ELCA_PROXY
    ]
    geo = pd.read_parquet(DATA_DIR / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet")
    geo = geo[geo["es_split"] == 0]

    geo_2010 = geo[geo["ola"] == 2010][["consecutivo"] + VARIABLES_DMSP].add_suffix("_ini").rename(columns={"consecutivo_ini": "consecutivo"})
    geo_2013 = geo[geo["ola"] == 2013][["consecutivo"] + VARIABLES_DMSP].add_suffix("_fin").rename(columns={"consecutivo_fin": "consecutivo"})

    hf_ini_unicos = hogar_features[hogar_features["ola"] == 1]
    hf_ini_unicos = hf_ini_unicos[~hf_ini_unicos["consecutivo"].duplicated(keep=False)].add_suffix("_ini").rename(columns={"consecutivo_ini": "consecutivo"})
    hf_fin_unicos = hogar_features[hogar_features["ola"] == 2]
    hf_fin_unicos = hf_fin_unicos[~hf_fin_unicos["consecutivo"].duplicated(keep=False)].add_suffix("_fin").rename(columns={"consecutivo_fin": "consecutivo"})

    panel = cargar_emparejados_1a1(hogar, 1, 2)
    panel = panel.merge(geo_2010, on="consecutivo", how="inner").merge(geo_2013, on="consecutivo", how="inner")
    panel = panel.merge(hf_ini_unicos, on="consecutivo", how="left").merge(hf_fin_unicos, on="consecutivo", how="left")
    print(f"n hogares emparejados 2010->2013 con DMSP-OLS en ambas olas: {len(panel)}")

    filas = []
    for var in VARIABLES_DMSP + VARIABLES_ELCA_PROXY:
        col_ini, col_fin = f"{var}_ini", f"{var}_fin"
        if col_ini not in panel.columns or col_fin not in panel.columns:
            print(f"AVISO: {var} no encontrada en ambas olas, se omite")
            continue
        x0 = pd.to_numeric(panel[col_ini], errors="coerce")
        x1 = pd.to_numeric(panel[col_fin], errors="coerce")
        validos = x0.notna() & x1.notna()
        x0, x1 = x0[validos], x1[validos]

        corr_test_retest = x0.corr(x1)
        std_pooled = pd.concat([x0, x1]).std()
        cambio_abs_std = (x1 - x0).abs().mean() / std_pooled if std_pooled > 0 else float("nan")

        tipo = "DMSP-OLS" if var.startswith("dmsp_") else "Proxy ELCA"
        filas.append({
            "variable": var, "tipo": tipo, "n": int(validos.sum()),
            "corr_test_retest_2010_2013": round(corr_test_retest, 4),
            "cambio_absoluto_medio_estandarizado": round(cambio_abs_std, 4),
        })
        print(f"  [{tipo}] {var}: corr(2010,2013)={corr_test_retest:.4f}  cambio_std={cambio_abs_std:.4f}  (n={int(validos.sum())})")

    out = pd.DataFrame(filas)
    out_path = DATA_DIR / "benchmark_resultados" / "diagnostico_variabilidad_temporal_dmsp.csv"
    out.to_csv(out_path, index=False)

    print("\n=== Comparacion directa: promedio por tipo de variable ===")
    print(out.groupby("tipo")[["corr_test_retest_2010_2013", "cambio_absoluto_medio_estandarizado"]].mean())
    print(f"\nCSV: {out_path}")


if __name__ == "__main__":
    main()
