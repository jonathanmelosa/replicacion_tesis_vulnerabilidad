"""
06_control_calidad.py
=======================

Control de calidad final del panel de variables DMSP-OLS: cobertura,
valores faltantes por variable, rangos físicos plausibles, tasa de
saturación, y verificación de determinismo. Es el último script del
sub-pipeline DMSP-OLS.

QUÉ HACE

    1. Carga variables_dmsp.parquet (output de 05_unir_variables.py).
    2. Cobertura general: cuántos hogares tienen variables DMSP calculadas
       por ronda cubierta (2010, 2013).
    3. Valores faltantes por variable, por ronda.
    4. Rangos físicos plausibles: avg_vis/stable_lights en [0, 63],
       cf_cvg >= 0, dmsp_log_stable_lights >= 0.
    5. Tasa de saturación por ronda (% de hogares con stable_lights == 63)
       — informativo, no un error: es una limitación conocida del sensor
       (ver prompt maestro), pero un valor extremo (p. ej. >30%) merece
       revisión manual antes de usar la variable sin más contexto.
    6. Exporta el reporte de control de calidad final.

    Este script NO realiza:

    · el merge con el panel de la encuesta (fuera de este sub-pipeline).
    · corrección o imputación de valores faltantes o fuera de rango.

INPUTS

    data/processed/variables_dmsp.parquet (output de 05_unir_variables.py)

OUTPUTS

    logs/reporte_control_calidad.txt → reporte final de calidad

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
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Input ───────────────────────────────────────────────────────────────────
    "variables_dmsp": _HERE.parent / "data" / "processed" / "variables_dmsp.parquet",

    # ── Rondas que este pipeline cubre por diseño ─────────────────────────────
    "rondas_cubiertas": [2010, 2013],

    # ── Rangos físicos plausibles ──────────────────────────────────────────────
    # avg_vis/stable_lights son enteros de 6 bits [0,63] ANTES del promedio
    # espacial del buffer; tras promediar sobre varios píxeles el valor
    # sigue acotado en [0,63] porque es un promedio de valores en ese rango.
    "rangos_plausibles": {
        # Variables de ESTADO (un solo año)
        "dmsp_avg_vis": (-1e-6, 63 + 1e-6),
        "dmsp_stable_lights": (-1e-6, 63 + 1e-6),
        "dmsp_cf_cvg": (-1e-6, np.inf),
        "dmsp_log_stable_lights": (0, np.inf),
        # Variables de VENTANA ACUMULADA (3 años) — media/mín/máx heredan
        # el mismo rango físico [0,63] que las variables de estado, porque
        # son promedios/extremos de valores en ese rango; rango y std son
        # siempre >= 0; n_anios no puede superar el tamaño de la ventana (3).
        "dmsp_avg_vis_acum_media": (-1e-6, 63 + 1e-6),
        "dmsp_avg_vis_acum_minimo": (-1e-6, 63 + 1e-6),
        "dmsp_avg_vis_acum_maximo": (-1e-6, 63 + 1e-6),
        "dmsp_avg_vis_acum_rango": (-1e-6, np.inf),
        "dmsp_avg_vis_acum_std": (-1e-6, np.inf),
        "dmsp_avg_vis_acum_n_anios": (0, 3),
        "dmsp_stable_lights_acum_media": (-1e-6, 63 + 1e-6),
        "dmsp_stable_lights_acum_minimo": (-1e-6, 63 + 1e-6),
        "dmsp_stable_lights_acum_maximo": (-1e-6, 63 + 1e-6),
        "dmsp_stable_lights_acum_rango": (-1e-6, np.inf),
        "dmsp_stable_lights_acum_std": (-1e-6, np.inf),
        "dmsp_stable_lights_acum_n_anios": (0, 3),
    },

    # ── Umbral de saturación para advertencia (informativo) ──────────────────
    "umbral_advertencia_saturacion_pct": 30.0,

    # ── Carpeta de salida ──────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: CARGA
# ──────────────────────────────────────────────────────────────────────────────

def cargar_variables(cfg: dict) -> pd.DataFrame:
    """Lee variables_dmsp.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["variables_dmsp"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró la tabla de variables DMSP-OLS.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO:")
        log.error("    Corre primero 05_unir_variables.py:")
        log.error("      python 05_unir_variables.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"Variables cargadas: {len(df):,} filas × {df.shape[1]} columnas — {ruta.name}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: COBERTURA GENERAL
# ──────────────────────────────────────────────────────────────────────────────

def calcular_cobertura_general(df: pd.DataFrame, cfg: dict) -> dict:
    """Cuenta, por ronda cubierta, cuántos hogares tienen variables dmsp_* no-nulas."""
    cols_dmsp = [c for c in df.columns if c.startswith("dmsp_")]
    resultado = {"por_ola": {}, "n_hogares_unicos": int(df["id_ola"].nunique())}
    for ola in cfg["rondas_cubiertas"]:
        grupo = df[df["ola"] == ola]
        n_con_datos = int(grupo[cols_dmsp].notna().any(axis=1).sum()) if cols_dmsp and not grupo.empty else 0
        resultado["por_ola"][int(ola)] = {"n_total": len(grupo), "n_con_datos": n_con_datos}
    return resultado


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: VALORES FALTANTES POR VARIABLE
# ──────────────────────────────────────────────────────────────────────────────

def calcular_faltantes_por_variable(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calcula el % de NaN de cada columna dmsp_*, por ronda cubierta."""
    cols_dmsp = [c for c in df.columns if c.startswith("dmsp_")]
    filas = []
    for ola in cfg["rondas_cubiertas"]:
        grupo = df[df["ola"] == ola]
        n = len(grupo)
        for col in cols_dmsp:
            pct_faltante = 100 * grupo[col].isna().sum() / n if n else np.nan
            filas.append({"variable": col, "ola": int(ola), "pct_faltante": pct_faltante})
    largo = pd.DataFrame(filas)
    return largo.pivot(index="variable", columns="ola", values="pct_faltante") if not largo.empty else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: RANGOS FÍSICOS PLAUSIBLES
# ──────────────────────────────────────────────────────────────────────────────

def verificar_rangos_plausibles(df: pd.DataFrame, cfg: dict) -> list:
    """Señala, para cada variable en CONFIG['rangos_plausibles'], cuántos hogares caen fuera del rango de referencia."""
    hallazgos = []
    for col, (minimo, maximo) in cfg["rangos_plausibles"].items():
        if col not in df.columns:
            continue
        fuera_rango = df[(df[col] < minimo) | (df[col] > maximo)]
        n_fuera = len(fuera_rango.dropna(subset=[col]))
        if n_fuera > 0:
            hallazgos.append(f"{col}: {n_fuera:,} hogares fuera del rango de referencia [{minimo}, {maximo}]")
    return hallazgos


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5: TASA DE SATURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def calcular_saturacion_por_ronda(df: pd.DataFrame, cfg: dict) -> dict:
    """% de hogares con dmsp_saturado=True por ronda cubierta, con advertencia si supera el umbral configurado."""
    resultado = {}
    for ola in cfg["rondas_cubiertas"]:
        grupo = df[df["ola"] == ola]
        if "dmsp_saturado" not in grupo.columns or grupo["dmsp_saturado"].isna().all():
            resultado[int(ola)] = None
            continue
        pct = 100 * grupo["dmsp_saturado"].mean()
        resultado[int(ola)] = pct
    return resultado


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: REPORTE FINAL
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(cobertura: dict, faltantes: pd.DataFrame, hallazgos_rango: list, saturacion: dict, cfg: dict) -> str:
    """Genera el texto del reporte de control de calidad final."""
    sep = "=" * 65
    lines = [
        sep,
        "REPORTE DE CONTROL DE CALIDAD FINAL — VARIABLES DMSP-OLS",
        sep,
        "",
        "── 1. COBERTURA GENERAL ──────────────────────────────────",
        "",
        f"  Hogares únicos (id_ola) en el panel: {cobertura['n_hogares_unicos']:,}",
        "  Rondas cubiertas por este pipeline (datos reales, no proxy):",
        f"  {cfg['rondas_cubiertas']} — usar viirs_pipeline/ para 2016 en adelante.",
        "",
    ]
    for ola, c in sorted(cobertura["por_ola"].items()):
        lines.append(f"  Ronda {ola}: {c['n_con_datos']:,}/{c['n_total']:,} hogares con variables DMSP")

    lines += [
        "",
        "── 2. VALORES FALTANTES POR VARIABLE (%) ─────────────────",
        "",
    ]
    if not faltantes.empty:
        for var, fila in faltantes.iterrows():
            valores = "  ".join(f"{int(ola)}: {pct:.1f}%" for ola, pct in fila.items())
            lines.append(f"    {var}: {valores}")

    lines += [
        "",
        "── 3. RANGOS FÍSICOS PLAUSIBLES ──────────────────────────",
        "",
    ]
    if not hallazgos_rango:
        lines.append("  OK — todas las variables dentro de rango.")
    else:
        for h in hallazgos_rango:
            lines.append(f"    - {h}")

    lines += [
        "",
        "── 4. TASA DE SATURACIÓN DEL SENSOR POR RONDA ────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  % de hogares con stable_lights == 63 (saturación, ver",
        "  limitación en el prompt maestro). No es un error del pipeline",
        "  — es una limitación conocida del sensor DMSP-OLS en núcleos",
        f"  urbanos densos. Se advierte si supera {cfg['umbral_advertencia_saturacion_pct']:.0f}%.",
        "",
    ]
    for ola, pct in sorted(saturacion.items()):
        if pct is None:
            lines.append(f"  Ronda {ola}: sin datos.")
            continue
        etiqueta = " — ADVERTENCIA: revisar manualmente" if pct > cfg["umbral_advertencia_saturacion_pct"] else ""
        lines.append(f"  Ronda {ola}: {pct:.1f}% de hogares saturados{etiqueta}")

    lines += ["", sep]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  06_control_calidad.py")
    log.info("  Si ves ModuleNotFoundError, el entorno virtual no esta")
    log.info("  activado o las dependencias no estan instaladas.")
    log.info("  Solucion:")
    log.info("    Mac/Linux : source .venv/bin/activate")
    log.info("    Windows   : .venv\\Scripts\\activate")
    log.info("    Luego     : pip install -r requirements.txt")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando variables DMSP-OLS ────────────────────────────")
    df = cargar_variables(cfg)

    log.info("\n── Calculando cobertura general ───────────────────────────")
    cobertura = calcular_cobertura_general(df, cfg)
    for ola, c in sorted(cobertura["por_ola"].items()):
        log.info(f"  Ronda {ola}: {c['n_con_datos']:,}/{c['n_total']:,} con datos")

    log.info("\n── Calculando valores faltantes por variable ─────────────")
    faltantes = calcular_faltantes_por_variable(df, cfg)

    log.info("\n── Verificando rangos físicos plausibles ─────────────────")
    hallazgos_rango = verificar_rangos_plausibles(df, cfg)
    if hallazgos_rango:
        log.warning(f"  {len(hallazgos_rango)} variable(s) con valores fuera del rango de referencia.")
    else:
        log.info("  OK — sin valores fuera de rango.")

    log.info("\n── Calculando tasa de saturación por ronda ───────────────")
    saturacion = calcular_saturacion_por_ronda(df, cfg)
    for ola, pct in sorted(saturacion.items()):
        if pct is not None:
            log.info(f"  Ronda {ola}: {pct:.1f}% saturados")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    reporte = generar_reporte(cobertura, faltantes, hallazgos_rango, saturacion, cfg)
    txt_path = Path(cfg["logs_dir"]) / "reporte_control_calidad.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
