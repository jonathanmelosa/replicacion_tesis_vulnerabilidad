"""
06_control_calidad.py
=======================

Control de calidad final del panel de variables Landsat 5 TM: cobertura,
valores faltantes, rangos físicos plausibles, tasa de cobertura de nubes.
Último script del sub-pipeline.

QUÉ HACE

    1. Carga variables_landsat.parquet (output de 05_unir_variables.py).
    2. Cobertura general para las olas 2010 y 2013.
    3. Valores faltantes por variable, por ronda.
    4. Rangos físicos plausibles: bandas de reflectancia en
       aproximadamente [-0.05, 1.2] (la corrección atmosférica de USGS
       puede dar valores ligeramente fuera de [0,1] en superficies muy
       oscuras o muy brillantes — rango de referencia amplio, no
       estricto), índices normalizados en (-1.2, 1.2).
    5. Exporta el reporte final.

INPUTS

    data/processed/variables_landsat.parquet

OUTPUTS

    logs/reporte_control_calidad.txt

CÓMO CORRER

    python 06_control_calidad.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "variables_landsat": _HERE.parent / "data" / "processed" / "variables_landsat.parquet",
    "rondas_cubiertas": [2010, 2013],

    "rangos_plausibles": {
        "l5_sr_b1": (-0.05, 1.2), "l5_sr_b2": (-0.05, 1.2), "l5_sr_b3": (-0.05, 1.2),
        "l5_sr_b4": (-0.05, 1.2), "l5_sr_b5": (-0.05, 1.2), "l5_sr_b7": (-0.05, 1.2),
        "l5_ndvi": (-1.2, 1.2), "l5_ndbi": (-1.2, 1.2), "l5_bsi": (-1.2, 1.2),
        "l5_ndwi": (-1.2, 1.2), "l5_mndwi": (-1.2, 1.2),
    },

    "logs_dir": _HERE.parent / "logs",
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)


def cargar_variables(cfg: dict) -> pd.DataFrame:
    """Lee variables_landsat.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["variables_landsat"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró la tabla de variables Landsat 5.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO: corre primero 05_unir_variables.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"Variables cargadas: {len(df):,} filas × {df.shape[1]} columnas")
    return df


def calcular_cobertura(df: pd.DataFrame, cfg: dict) -> dict:
    """Cuenta, por ronda cubierta, cuántos hogares tienen variables l5_* no-nulas."""
    cols_l5 = [c for c in df.columns if c.startswith("l5_")]
    resultado = {"por_ola": {}, "n_hogares_unicos": int(df["id_ola"].nunique())}
    for ola in cfg["rondas_cubiertas"]:
        grupo = df[df["ola"] == ola]
        n_con_datos = int(grupo[cols_l5].notna().any(axis=1).sum()) if cols_l5 and not grupo.empty else 0
        resultado["por_ola"][int(ola)] = {"n_total": len(grupo), "n_con_datos": n_con_datos}
    return resultado


def calcular_faltantes(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calcula el % de NaN de cada columna l5_*, por ronda cubierta."""
    cols_l5 = [c for c in df.columns if c.startswith("l5_")]
    filas = []
    for ola in cfg["rondas_cubiertas"]:
        grupo = df[df["ola"] == ola]
        n = len(grupo)
        for col in cols_l5:
            pct_faltante = 100 * grupo[col].isna().sum() / n if n else np.nan
            filas.append({"variable": col, "ola": int(ola), "pct_faltante": pct_faltante})
    largo = pd.DataFrame(filas)
    return largo.pivot(index="variable", columns="ola", values="pct_faltante") if not largo.empty else pd.DataFrame()


def verificar_rangos(df: pd.DataFrame, cfg: dict) -> list:
    """Señala, para cada variable en CONFIG['rangos_plausibles'], cuántos hogares caen fuera del rango de referencia."""
    hallazgos = []
    for col, (minimo, maximo) in cfg["rangos_plausibles"].items():
        if col not in df.columns:
            continue
        n_fuera = len(df[(df[col] < minimo) | (df[col] > maximo)].dropna(subset=[col]))
        if n_fuera > 0:
            hallazgos.append(f"{col}: {n_fuera:,} hogares fuera del rango de referencia [{minimo}, {maximo}]")
    return hallazgos


def generar_reporte(cobertura: dict, faltantes: pd.DataFrame, hallazgos: list, cfg: dict) -> str:
    """Genera el texto del reporte de control de calidad final."""
    sep = "=" * 65
    lines = [sep, "REPORTE DE CONTROL DE CALIDAD FINAL — VARIABLES LANDSAT 5 TM", sep, "",
             "── 1. COBERTURA GENERAL ──────────────────────────────────", "",
             f"  Hogares únicos (id_ola) en el panel: {cobertura['n_hogares_unicos']:,}",
             f"  Rondas cubiertas por diseño: {cfg['rondas_cubiertas']}", ""]
    for ola, c in sorted(cobertura["por_ola"].items()):
        lines.append(f"  Ronda {ola}: {c['n_con_datos']:,}/{c['n_total']:,} hogares con variables Landsat 5")
        if c["n_con_datos"] < 0.9 * c["n_total"]:
            lines.append(
                "    NOTA: cobertura considerablemente menor al 90% — a diferencia de DMSP-OLS/ALOS"
                " PALSAR (compuestos globales sin huecos), Landsat 5 SÍ puede tener hogares sin ningún"
                " píxel válido en todo el año si la nubosidad fue persistente — ver la mediana anual"
                " en 03_series_temporales.py."
            )

    lines += ["", "── 2. VALORES FALTANTES POR VARIABLE (%) ─────────────────", ""]
    if not faltantes.empty:
        for var, fila in faltantes.iterrows():
            valores = "  ".join(f"{int(ola)}: {pct:.1f}%" for ola, pct in fila.items())
            lines.append(f"    {var}: {valores}")

    lines += ["", "── 3. RANGOS FÍSICOS PLAUSIBLES ──────────────────────────", ""]
    if not hallazgos:
        lines.append("  OK — todas las variables dentro de rango.")
    else:
        for h in hallazgos:
            lines.append(f"    - {h}")

    lines += ["", sep]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  06_control_calidad.py")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando variables Landsat 5 ───────────────────────────")
    df = cargar_variables(cfg)

    log.info("\n── Calculando cobertura general ───────────────────────────")
    cobertura = calcular_cobertura(df, cfg)
    for ola, c in sorted(cobertura["por_ola"].items()):
        log.info(f"  Ronda {ola}: {c['n_con_datos']:,}/{c['n_total']:,} con datos")

    log.info("\n── Calculando valores faltantes por variable ─────────────")
    faltantes = calcular_faltantes(df, cfg)

    log.info("\n── Verificando rangos físicos plausibles ─────────────────")
    hallazgos = verificar_rangos(df, cfg)
    if hallazgos:
        log.warning(f"  {len(hallazgos)} variable(s) con valores fuera del rango de referencia.")
    else:
        log.info("  OK — sin valores fuera de rango.")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    reporte = generar_reporte(cobertura, faltantes, hallazgos, cfg)
    (Path(cfg["logs_dir"]) / "reporte_control_calidad.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
