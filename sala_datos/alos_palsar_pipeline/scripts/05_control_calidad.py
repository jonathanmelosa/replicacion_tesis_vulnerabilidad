"""
05_control_calidad.py
=======================

Control de calidad final del panel de variables ALOS PALSAR: cobertura,
valores faltantes, rangos físicos plausibles. Último script del
sub-pipeline.

QUÉ HACE

    1. Carga variables_alos.parquet (output de 04_unir_variables.py).
    2. Cobertura general para la ola 2010 (la única cubierta por diseño).
    3. Valores faltantes por variable.
    4. Rangos físicos plausibles: HH_db/HV_db típicamente en [-30, 5] dB
       sobre superficie terrestre (referencia amplia, no un límite
       estricto — SAR tiene outliers legítimos, igual que Sentinel-1);
       indice_normalizado en (-1, 1).
    5. Exporta el reporte final.

INPUTS

    data/processed/variables_alos.parquet

OUTPUTS

    logs/reporte_control_calidad.txt

CÓMO CORRER

    python 05_control_calidad.py
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
    "variables_alos": _HERE.parent / "data" / "processed" / "variables_alos.parquet",
    "ronda_cubierta": 2010,

    # Rango de referencia AMPLIO, no un límite estricto (ver docstring).
    "rangos_plausibles": {
        "alos_hh_db": (-35, 10),
        "alos_hv_db": (-40, 5),
        "alos_indice_normalizado": (-1.01, 1.01),
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
    """Lee variables_alos.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["variables_alos"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró la tabla de variables ALOS PALSAR.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO: corre primero 04_unir_variables.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"Variables cargadas: {len(df):,} filas × {df.shape[1]} columnas")
    return df


def calcular_cobertura(df: pd.DataFrame, cfg: dict) -> dict:
    """Cuenta cuántos hogares de la ola cubierta tienen variables alos_* no-nulas."""
    cols_alos = [c for c in df.columns if c.startswith("alos_")]
    grupo = df[df["ola"] == cfg["ronda_cubierta"]]
    n_con_datos = int(grupo[cols_alos].notna().any(axis=1).sum()) if cols_alos and not grupo.empty else 0
    return {"n_total": len(grupo), "n_con_datos": n_con_datos, "n_hogares_unicos": int(df["id_ola"].nunique())}


def calcular_faltantes(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Calcula el % de NaN de cada columna alos_*, para la ola cubierta."""
    cols_alos = [c for c in df.columns if c.startswith("alos_")]
    grupo = df[df["ola"] == cfg["ronda_cubierta"]]
    n = len(grupo)
    return pd.Series({col: 100 * grupo[col].isna().sum() / n if n else np.nan for col in cols_alos})


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


def generar_reporte(cobertura: dict, faltantes: pd.Series, hallazgos: list, cfg: dict) -> str:
    """Genera el texto del reporte de control de calidad final."""
    sep = "=" * 65
    lines = [
        sep, "REPORTE DE CONTROL DE CALIDAD FINAL — VARIABLES ALOS PALSAR", sep, "",
        "── 1. COBERTURA GENERAL (OLA 2010) ───────────────────────", "",
        f"  Hogares únicos (id_ola) en el panel: {cobertura['n_hogares_unicos']:,}",
        f"  Ola {cfg['ronda_cubierta']}: {cobertura['n_con_datos']:,}/{cobertura['n_total']:,} hogares con variables ALOS",
        "", "── 2. VALORES FALTANTES POR VARIABLE (%) ─────────────────", "",
    ]
    for var, pct in faltantes.items():
        lines.append(f"    {var}: {pct:.1f}%")

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
    log.info("  05_control_calidad.py")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando variables ALOS PALSAR ─────────────────────────")
    df = cargar_variables(cfg)

    log.info("\n── Calculando cobertura general ───────────────────────────")
    cobertura = calcular_cobertura(df, cfg)
    log.info(f"  Ola {cfg['ronda_cubierta']}: {cobertura['n_con_datos']:,}/{cobertura['n_total']:,} con datos")

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
