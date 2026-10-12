"""
03_series_temporales.py
=========================

Separa la tabla cruda hogar×ronda×año (extraccion_buffer_landsat.parquet)
en las dos familias de variables del pipeline: ESTADO (año exacto de la
ola, con las 10 bandas/índices completos) y VENTANA ACUMULADA
(estadísticos sobre los 3 años, calculados solo sobre un subconjunto de
4 índices clave — ver CONFIG['columnas_acumulado'] — para mantener el
panel final manejable).

QUÉ HACE

    1. Carga extraccion_buffer_landsat.parquet.
    2. Separa en:
         a. Estado: fila con año == año de estado de cada ronda, para
            TODAS las bandas/índices (renombradas con prefijo l5_).
         b. Ventana acumulada: estadísticos sobre los 3 años, solo para
            ndvi, ndbi, bsi, tc_brillo (los más directamente
            interpretables para "cambio del entorno construido/vegetación"
            — ver justificación en la tesis, Sección 3.3.7).
    3. Une ambos subconjuntos en un panel hogar×ronda.
    4. Verifica consistencia aritmética.
    5. Exporta el panel y un reporte de calidad.

INPUTS

    data/processed/extraccion_buffer_landsat.parquet

OUTPUTS

    data/processed/series_temporales_landsat.parquet
    logs/reporte_series_temporales.txt

CÓMO CORRER

    python 03_series_temporales.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from utils.landsat5_utils import calcular_estadisticos_ventana_acumulada  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "extraccion_buffer": _HERE.parent / "data" / "processed" / "extraccion_buffer_landsat.parquet",

    "ventanas_por_ronda": {
        2010: {"anio_estado": 2010, "anios_acumulado": [2008, 2009, 2010]},
        2013: {"anio_estado": 2013, "anios_acumulado": [2011, 2012, 2013]},
    },

    # Todas las bandas/índices que se conservan en el ESTADO.
    "columnas_estado": [
        "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7",
        "ndvi", "evi", "savi", "ndbi", "ndwi", "mndwi", "bsi",
        "tc_brillo", "tc_verdor", "tc_humedad",
    ],

    # Subconjunto sobre el que se calculan estadísticos de VENTANA
    # ACUMULADA — ver justificación en el docstring del módulo.
    "columnas_acumulado": ["ndvi", "ndbi", "bsi", "tc_brillo"],

    "prefijo_variables": "l5",

    "output_dir": _HERE.parent / "data" / "processed",
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


def cargar_extraccion(cfg: dict) -> pd.DataFrame:
    """Lee extraccion_buffer_landsat.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["extraccion_buffer"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró la tabla de extracción cruda.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO: corre primero 02_extraccion_buffer.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"Extracción cargada: {len(df):,} filas (hogar×ronda×año) — {ruta.name}")
    return df


def extraer_estado(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filtra a la fila de cada hogar×ronda cuyo año coincide con el año de estado de esa ronda."""
    prefijo = cfg["prefijo_variables"]
    partes = []
    for ronda, info_ronda in cfg["ventanas_por_ronda"].items():
        parte = df[(df["ola"] == ronda) & (df["anio"] == info_ronda["anio_estado"])].copy()
        partes.append(parte)
    estado = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

    renombres = {c: f"{prefijo}_{c.lower()}" for c in cfg["columnas_estado"] if c in estado.columns}
    estado = estado.rename(columns=renombres).drop(columns=["anio"])
    log.info(f"  Estado: {len(estado):,} filas (hogar×ronda, una por año de ola)")
    return estado


def calcular_ventana_acumulada(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Agrupa por hogar×ronda y calcula, para cada columna en CONFIG['columnas_acumulado'], los estadísticos de la ventana de 3 años."""
    prefijo_base = cfg["prefijo_variables"]
    filas = []
    for (consecutivo, id_ola, ola), grupo in df.groupby(["consecutivo", "id_ola", "ola"]):
        fila = {"consecutivo": consecutivo, "id_ola": id_ola, "ola": ola}
        for col in cfg["columnas_acumulado"]:
            if col not in grupo.columns:
                continue
            prefijo = f"{prefijo_base}_{col.lower()}_acum"
            fila.update(calcular_estadisticos_ventana_acumulada(grupo, col, prefijo))
        filas.append(fila)
    acumulado = pd.DataFrame(filas)
    log.info(f"  Ventana acumulada: {len(acumulado):,} filas (hogar×ronda) × {acumulado.shape[1]} columnas")
    return acumulado


def verificar_consistencia(panel: pd.DataFrame, cfg: dict) -> list:
    """Verifica rango = max-min para cada variable acumulada."""
    problemas = []
    prefijo_base = cfg["prefijo_variables"]
    for col in cfg["columnas_acumulado"]:
        prefijo = f"{prefijo_base}_{col.lower()}_acum"
        cols_necesarias = [f"{prefijo}_maximo", f"{prefijo}_minimo", f"{prefijo}_rango"]
        if not all(c in panel.columns for c in cols_necesarias):
            continue
        validos = panel.dropna(subset=cols_necesarias)
        for _, fila in validos.iterrows():
            rango_calc = fila[f"{prefijo}_maximo"] - fila[f"{prefijo}_minimo"]
            if abs(fila[f"{prefijo}_rango"] - rango_calc) > 1e-6:
                problemas.append(f"consecutivo={fila['consecutivo']} ola={fila['ola']} {prefijo}: rango != max-min")
    return problemas


def generar_reporte(panel: pd.DataFrame, problemas: list, cfg: dict) -> str:
    """Genera el texto del reporte de calidad."""
    sep = "=" * 65
    lines = [sep, "REPORTE DE CALIDAD — SERIES TEMPORALES LANDSAT 5 TM", sep, "",
             "── 1. COBERTURA DEL PANEL POR RONDA ──────────────────────", ""]
    for ola, grupo in panel.groupby("ola"):
        col_n = f"{cfg['prefijo_variables']}_ndvi_acum_n_anios"
        n_prom = grupo[col_n].mean() if col_n in grupo.columns else float("nan")
        lines.append(f"  Ronda {ola}: {len(grupo):,} hogares — años en ventana acumulada (promedio): {n_prom:.2f} de 3")

    lines += ["", "── 2. VERIFICACIÓN DE CONSISTENCIA ───────────────────────", ""]
    if not problemas:
        lines.append(f"  OK — sin problemas de consistencia en {len(panel):,} hogares×ronda.")
    else:
        lines.append(f"  {len(problemas)} problema(s) encontrado(s):")
        for p in problemas[:20]:
            lines.append(f"    - {p}")

    lines += ["", "── 3. ESTADÍSTICOS DESCRIPTIVOS (VARIABLES CLAVE) ────────", ""]
    prefijo = cfg["prefijo_variables"]
    cols_clave = [f"{prefijo}_ndvi", f"{prefijo}_ndbi", f"{prefijo}_bsi", f"{prefijo}_tc_brillo",
                  f"{prefijo}_ndvi_acum_media", f"{prefijo}_ndvi_acum_tendencia"]
    cols_clave = [c for c in cols_clave if c in panel.columns]
    resumen = panel[cols_clave].describe().T[["mean", "std", "min", "max"]]
    for col, fila in resumen.iterrows():
        lines.append(f"    {col}: media={fila['mean']:.3f}  std={fila['std']:.3f}  min={fila['min']:.3f}  max={fila['max']:.3f}")

    lines += ["", sep]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  03_series_temporales.py")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando extracción cruda ──────────────────────────────")
    df = cargar_extraccion(cfg)

    log.info("\n── Separando en estado y ventana acumulada ───────────────")
    estado = extraer_estado(df, cfg)
    acumulado = calcular_ventana_acumulada(df, cfg)

    panel = estado.merge(acumulado, on=["consecutivo", "id_ola", "ola"], how="left")
    log.info(f"  Panel final: {len(panel):,} filas × {panel.shape[1]} columnas")

    log.info("\n── Verificando consistencia aritmética ───────────────────")
    problemas = verificar_consistencia(panel, cfg)
    if problemas:
        log.warning(f"  {len(problemas)} problema(s) de consistencia encontrado(s).")
    else:
        log.info("  OK — sin problemas de consistencia.")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "series_temporales_landsat.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(f"  Panel exportado: {out_path} ({len(panel):,} filas)")

    reporte = generar_reporte(panel, problemas, cfg)
    (Path(cfg["logs_dir"]) / "reporte_series_temporales.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
