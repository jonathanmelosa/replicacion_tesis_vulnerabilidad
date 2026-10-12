"""
03_series_temporales.py
=========================

Limpia la tabla cruda hogar×ronda×año (extraccion_buffer_dmsp.parquet,
output de 02_extraccion_buffer.py) y la separa en las DOS familias de
variables del pipeline (ver "Estado vs. ventana acumulada" en
01_construir_coleccion_dmsp.py):

    1. ESTADO: el valor del año exacto de la ola (una fila por hogar×ronda).
    2. VENTANA ACUMULADA: estadísticos (media, tendencia, crecimiento
       acumulado, etc.) calculados sobre los 3 años de la ventana de cada
       ronda (una fila por hogar×ronda, agregando 3 observaciones anuales).

QUÉ HACE

    1. Carga extraccion_buffer_dmsp.parquet (hogar × ronda × año).
    2. Limpia avg_vis/stable_lights (utils/dmsp_utils.py -- por ahora solo
       recorte de negativos; DMSP-OLS ya viene truncado a [0,63] por el
       propio producto de NOAA, así que no hace falta winsorizar como en
       VIIRS).
    3. Separa la tabla en:
         a. Estado: filtra a la fila donde año == año de estado de esa
            ronda (ver CONFIG['ventanas_por_ronda']).
         b. Ventana acumulada: agrupa por hogar×ronda y calcula
            estadísticos sobre las 3 filas (utils/dmsp_utils.py::
            calcular_estadisticos_ventana_acumulada) para avg_vis y para
            stable_lights por separado.
    4. Une ambos subconjuntos en un solo panel hogar×ronda.
    5. Verifica consistencia aritmética (rango = max-min, cv = std/media)
       sobre el 100% de las filas.
    6. Exporta el panel y un reporte de calidad.

    Este script NO realiza:

    · extracción de datos de GEE (02_extraccion_buffer.py).
    · saturación, log-transform, ni comparación entre rondas
      (04_indicadores_derivados.py).
    · merge con el panel de encuesta (05_unir_variables.py).

INPUTS

    data/processed/extraccion_buffer_dmsp.parquet (output de 02_extraccion_buffer.py)

OUTPUTS

    data/processed/series_temporales_dmsp.parquet → panel hogar×ronda (estado + ventana acumulada)
    logs/reporte_series_temporales.txt → diagnóstico

CÓMO CORRER

    python 03_series_temporales.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

_HERE = Path(__file__).resolve().parent            # .../dmsp_ols_pipeline/scripts
sys.path.insert(0, str(_HERE.parent))               # .../dmsp_ols_pipeline

from utils.dmsp_utils import calcular_estadisticos_ventana_acumulada  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Input: output de 02_extraccion_buffer.py ──────────────────────────────
    "extraccion_buffer": _HERE.parent / "data" / "processed" / "extraccion_buffer_dmsp.parquet",

    # ── Ventanas por ronda (idéntico a 01/02) ─────────────────────────────────
    "ventanas_por_ronda": {
        2010: {"anio_estado": 2010, "anios_acumulado": [2008, 2009, 2010]},
        2013: {"anio_estado": 2013, "anios_acumulado": [2011, 2012, 2013]},
    },

    # ── Variables sobre las que se calculan estadísticos de ventana acumulada ──
    "columnas_acumulado": ["avg_vis", "stable_lights"],

    # ── Carpetas de salida ──────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: CARGA Y LIMPIEZA
# ──────────────────────────────────────────────────────────────────────────────

def cargar_extraccion(cfg: dict) -> pd.DataFrame:
    """Lee extraccion_buffer_dmsp.parquet. Termina el proceso con diagnóstico si no existe."""
    ruta = Path(cfg["extraccion_buffer"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró la tabla de extracción cruda.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO:")
        log.error("    Corre primero 02_extraccion_buffer.py:")
        log.error("      python 02_extraccion_buffer.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"Extracción cargada: {len(df):,} filas (hogar×ronda×año) — {ruta.name}")
    return df


def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Recorta avg_vis/stable_lights a >= 0. A diferencia de VIIRS, DMSP-OLS
    ya entrega el producto truncado a [0,63] desde NOAA, así que no se
    winsoriza por techo — solo se documenta si aparecieran negativos
    inesperados (artefacto, no debería ocurrir con este producto).
    """
    df = df.copy()
    conteos = {}
    for col in ["avg_vis", "stable_lights"]:
        n_neg = int((df[col] < 0).sum())
        conteos[f"n_negativos_{col}"] = n_neg
        df[col] = df[col].clip(lower=0)
        log.info(f"  {col}: {n_neg:,} filas negativas recortadas a 0 ({n_neg / len(df):.2%})")
    return df, conteos


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: SEPARACIÓN EN ESTADO Y VENTANA ACUMULADA
# ──────────────────────────────────────────────────────────────────────────────

def extraer_estado(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Filtra a la fila de cada hogar×ronda cuyo año coincide con el año de
    estado de esa ronda (el último año de su ventana acumulada).
    """
    partes = []
    for ronda, info_ronda in cfg["ventanas_por_ronda"].items():
        anio_estado = info_ronda["anio_estado"]
        parte = df[(df["ola"] == ronda) & (df["anio"] == anio_estado)].copy()
        partes.append(parte)
    estado = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    estado = estado.rename(columns={
        "avg_vis": "dmsp_avg_vis", "stable_lights": "dmsp_stable_lights", "cf_cvg": "dmsp_cf_cvg",
    })
    estado = estado.drop(columns=["anio"])
    log.info(f"  Estado: {len(estado):,} filas (hogar×ronda, una por año de ola)")
    return estado


def calcular_ventana_acumulada(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Agrupa por hogar×ronda y calcula, para cada columna en
    CONFIG['columnas_acumulado'], los estadísticos de la ventana de 3 años
    (utils/dmsp_utils.py::calcular_estadisticos_ventana_acumulada).
    """
    filas = []
    for (consecutivo, id_ola, ola), grupo in df.groupby(["consecutivo", "id_ola", "ola"]):
        fila = {"consecutivo": consecutivo, "id_ola": id_ola, "ola": ola}
        for col in cfg["columnas_acumulado"]:
            prefijo = f"dmsp_{col}_acum"
            fila.update(calcular_estadisticos_ventana_acumulada(grupo, col, prefijo))
        filas.append(fila)

    acumulado = pd.DataFrame(filas)
    log.info(f"  Ventana acumulada: {len(acumulado):,} filas (hogar×ronda) × {acumulado.shape[1]} columnas")
    return acumulado


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: VERIFICACIÓN DE CONSISTENCIA ARITMÉTICA
# ──────────────────────────────────────────────────────────────────────────────

def verificar_consistencia(panel: pd.DataFrame, cfg: dict) -> list:
    """
    Verifica, sobre TODAS las filas, que rango = max-min y cv = std/media
    para cada variable acumulada. Un problema aquí indica un error en
    calcular_estadisticos_ventana_acumulada(), no un problema de los datos.
    """
    problemas = []
    for col in cfg["columnas_acumulado"]:
        prefijo = f"dmsp_{col}_acum"
        validos = panel.dropna(subset=[f"{prefijo}_maximo", f"{prefijo}_minimo", f"{prefijo}_rango"])
        for _, fila in validos.iterrows():
            rango_calc = fila[f"{prefijo}_maximo"] - fila[f"{prefijo}_minimo"]
            if abs(fila[f"{prefijo}_rango"] - rango_calc) > 1e-6:
                problemas.append(
                    f"consecutivo={fila['consecutivo']} ola={fila['ola']} {prefijo}: "
                    f"rango ({fila[f'{prefijo}_rango']:.4f}) != max-min ({rango_calc:.4f})"
                )
            media, std, cv = fila[f"{prefijo}_media"], fila[f"{prefijo}_std"], fila.get(f"{prefijo}_cv")
            if media != 0 and pd.notna(cv):
                cv_calc = std / media
                if abs(cv - cv_calc) > 1e-6:
                    problemas.append(
                        f"consecutivo={fila['consecutivo']} ola={fila['ola']} {prefijo}: "
                        f"cv ({cv:.4f}) != std/media ({cv_calc:.4f})"
                    )
    return problemas


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(panel: pd.DataFrame, conteos_limpieza: dict, problemas: list, cfg: dict) -> str:
    """Genera el texto del reporte de calidad."""
    sep = "=" * 65
    lines = [
        sep,
        "REPORTE DE CALIDAD — SERIES TEMPORALES DMSP-OLS (ESTADO + ACUMULADO)",
        sep,
        "",
        "── 1. LIMPIEZA APLICADA ───────────────────────────────────",
        "",
    ]
    for k, v in conteos_limpieza.items():
        lines.append(f"  {k}: {v:,}")

    lines += [
        "",
        "── 2. COBERTURA DEL PANEL POR RONDA ──────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Cuántos hogares×ronda tienen estado y ventana acumulada",
        "  calculados, y cuántos años de los 3 esperados tuvo en promedio",
        "  cada hogar en su ventana (n_anios < 3 indica que falta algún",
        "  año de la ventana para ese hogar — revisar 02 para esa ronda).",
        "",
    ]
    for ola, grupo in panel.groupby("ola"):
        col_n = f"dmsp_stable_lights_acum_n_anios"
        n_prom = grupo[col_n].mean() if col_n in grupo.columns else float("nan")
        lines.append(f"  Ronda {ola}: {len(grupo):,} hogares — años en ventana acumulada (promedio): {n_prom:.2f} de 3")

    lines += [
        "",
        "── 3. VERIFICACIÓN DE CONSISTENCIA ARITMÉTICA ────────────",
        "",
    ]
    if not problemas:
        lines.append(f"  OK — sin problemas de consistencia en {len(panel):,} hogares×ronda.")
    else:
        lines.append(f"  {len(problemas)} problema(s) encontrado(s):")
        for p in problemas[:20]:
            lines.append(f"    - {p}")
        if len(problemas) > 20:
            lines.append(f"    ... y {len(problemas) - 20} más (ver log completo).")

    lines += [
        "",
        "── 4. ESTADÍSTICOS DESCRIPTIVOS DEL PANEL ────────────────",
        "",
    ]
    cols_dmsp = [c for c in panel.columns if c.startswith("dmsp_")]
    resumen = panel[cols_dmsp].describe().T[["mean", "std", "min", "max"]]
    for col, fila in resumen.iterrows():
        lines.append(f"    {col}: media={fila['mean']:.3f}  std={fila['std']:.3f}  min={fila['min']:.3f}  max={fila['max']:.3f}")

    lines += ["", sep]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  03_series_temporales.py")
    log.info("  Si ves ModuleNotFoundError, el entorno virtual no esta")
    log.info("  activado o las dependencias no estan instaladas.")
    log.info("  Solucion:")
    log.info("    Mac/Linux : source .venv/bin/activate")
    log.info("    Windows   : .venv\\Scripts\\activate")
    log.info("    Luego     : pip install -r requirements.txt")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando extracción cruda ──────────────────────────────")
    df = cargar_extraccion(cfg)

    log.info("\n── Limpiando avg_vis / stable_lights ─────────────────────")
    df, conteos_limpieza = limpiar_datos(df)

    log.info("\n── Separando en estado y ventana acumulada ───────────────")
    estado = extraer_estado(df, cfg)
    acumulado = calcular_ventana_acumulada(df, cfg)

    panel = estado.merge(acumulado, on=["consecutivo", "id_ola", "ola"], how="left")
    log.info(f"  Panel final: {len(panel):,} filas × {panel.shape[1]} columnas")

    log.info("\n── Verificando consistencia aritmética ───────────────────")
    problemas = verificar_consistencia(panel, cfg)
    if problemas:
        log.warning(f"  {len(problemas)} problema(s) de consistencia encontrado(s) — ver reporte.")
    else:
        log.info("  OK — sin problemas de consistencia.")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "series_temporales_dmsp.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(f"  Panel exportado: {out_path} ({len(panel):,} filas)")

    reporte = generar_reporte(panel, conteos_limpieza, problemas, cfg)
    txt_path = Path(cfg["logs_dir"]) / "reporte_series_temporales.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
