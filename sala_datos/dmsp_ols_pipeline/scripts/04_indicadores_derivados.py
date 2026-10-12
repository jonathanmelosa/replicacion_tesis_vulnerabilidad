"""
04_indicadores_derivados.py
=============================

Agrega al panel de estado + ventana acumulada (series_temporales_dmsp.parquet,
output de 03_series_temporales.py) los indicadores derivados que faltan:
bandera de saturación y transformación logarítmica sobre las variables de
ESTADO, y el crecimiento ENTRE OLAS (2010→2013, distinto de la ventana
acumulada dentro de cada ola — ver nota de nomenclatura más abajo).

QUÉ HACE

    1. Carga series_temporales_dmsp.parquet.
    2. Carga las coordenadas de los hogares (reutilizadas de
       sentinel1_pipeline/) para clasificar movimiento entre 2010 y 2013.
    3. Calcula, sobre la variable de ESTADO (dmsp_stable_lights):
         a. dmsp_saturado = stable_lights >= 63.
         b. dmsp_log_stable_lights = log(stable_lights + 1).
    4. Calcula el crecimiento ENTRE OLAS 2010→2013 (comparando el estado
       de una ola contra el estado de la otra — dos puntos, no una
       ventana):
         a. Distancia Haversine entre las coordenadas del hogar en ambas
            rondas.
         b. Si no se movió: dmsp_crecimiento_entre_olas (razón de cambio).
         c. Si se movió: dmsp_cambio_nivel_migracion (diferencia de nivel).
         d. dmsp_hogar_se_movio: bandera booleana.
    5. Exporta el panel enriquecido y un reporte de calidad.

    NOMENCLATURA — TRES MEDIDAS DE CAMBIO DISTINTAS, NO CONFUNDIR
        dmsp_stable_lights_acum_crecimiento_acumulado (de 03): cambio
            DENTRO de la ventana de 3 años de UNA sola ola (p. ej.
            2008→2010 para la ola 2010). Mide el "proceso reciente" previo
            a esa ola específica.
        dmsp_crecimiento_entre_olas (de este script): cambio del ESTADO
            entre la ola 2010 y la ola 2013 —un salto de 3 años, pero
            usando solo los dos puntos de estado, no toda la ventana—.
            Es conceptualmente parecido a lo que ya calcula
            viirs_pipeline/04_indicadores_derivados.py entre rondas
            consecutivas, aplicado aquí porque, a diferencia de VIIRS/S1,
            2010 y 2013 SÍ son años reales y distintos en DMSP-OLS (ver
            prompt maestro).
        Las dos son válidas y miden cosas distintas: la primera es
        "¿este lugar se venía iluminando antes de ESTA ola?"; la segunda
        es "¿este lugar está más o menos iluminado en 2013 que en 2010?".

    DECISIÓN: se excluyen explícitamente los hogares divididos
    (consecutivo no único dentro de una ronda) del cálculo de
    crecimiento entre olas — mismo criterio ya corregido en
    viirs_pipeline/scripts/04_indicadores_derivados.py.

INPUTS

    data/processed/series_temporales_dmsp.parquet (output de 03_series_temporales.py)
    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet

OUTPUTS

    data/processed/indicadores_derivados_dmsp.parquet → panel hogar×ronda enriquecido
    logs/reporte_indicadores_derivados.txt → diagnóstico

CÓMO CORRER

    python 04_indicadores_derivados.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent            # .../dmsp_ols_pipeline/scripts
sys.path.insert(0, str(_HERE.parent))               # .../dmsp_ols_pipeline

from utils.dmsp_utils import marcar_saturacion, calcular_log_stable_lights  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Inputs ─────────────────────────────────────────────────────────────────
    "series_temporales": _HERE.parent / "data" / "processed" / "series_temporales_dmsp.parquet",
    "coordenadas_hogares": (
        _HERE.parent.parent
        / "sentinel1_pipeline" / "data" / "processed" / "coordenadas_hogares_s1.parquet"
    ),

    # ── Clasificación de movimiento entre rondas ─────────────────────────────
    "umbral_movimiento_m": 100,

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
# MÓDULO 1: CARGA
# ──────────────────────────────────────────────────────────────────────────────

def cargar_panel(cfg: dict) -> pd.DataFrame:
    """Lee series_temporales_dmsp.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["series_temporales"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró el panel de series temporales.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO:")
        log.error("    Corre primero 03_series_temporales.py:")
        log.error("      python 03_series_temporales.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"Panel cargado: {len(df):,} filas (hogar×ronda) — {ruta.name}")
    return df


def cargar_coordenadas(cfg: dict) -> pd.DataFrame:
    """Lee solo consecutivo/ola/lat/lon de coordenadas_hogares_s1.parquet."""
    ruta = Path(cfg["coordenadas_hogares"])
    if not ruta.exists():
        log.error(f"ERROR: no se encontró la tabla de coordenadas en {ruta.resolve()}")
        sys.exit(1)
    return pd.read_parquet(ruta)[["consecutivo", "ola", "lat", "lon"]]


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: INDICADORES DE NIVEL SOBRE EL ESTADO
# ──────────────────────────────────────────────────────────────────────────────

def calcular_indicadores_de_nivel(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega dmsp_saturado y dmsp_log_stable_lights sobre la variable de estado."""
    panel = panel.copy()
    panel["dmsp_saturado"] = marcar_saturacion(panel["dmsp_stable_lights"])
    panel["dmsp_log_stable_lights"] = calcular_log_stable_lights(panel["dmsp_stable_lights"])
    return panel


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: CRECIMIENTO ENTRE OLAS (2010 → 2013)
# ──────────────────────────────────────────────────────────────────────────────

def calcular_distancia_haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distancia en metros entre dos puntos (lat, lon) — fórmula de Haversine."""
    radio_tierra_m = 6_371_000
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    return radio_tierra_m * 2 * np.arcsin(np.sqrt(a))


def calcular_crecimiento_entre_olas(panel: pd.DataFrame, coordenadas: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Calcula dmsp_crecimiento_entre_olas / dmsp_cambio_nivel_migracion /
    dmsp_hogar_se_movio comparando el ESTADO 2010 contra el ESTADO 2013
    (ver nota de nomenclatura en el docstring del módulo).

    Excluye explícitamente los hogares divididos (consecutivo no único
    dentro de una ronda) — no comparables 1 a 1 entre 2010 y 2013.
    """
    if 2010 not in panel["ola"].values or 2013 not in panel["ola"].values:
        log.info("  Panel no tiene ambas rondas (2010 y 2013) — no se calcula crecimiento entre olas.")
        return pd.DataFrame(columns=["consecutivo", "ola", "dmsp_crecimiento_entre_olas",
                                      "dmsp_cambio_nivel_migracion", "dmsp_hogar_se_movio",
                                      "dmsp_distancia_movimiento_m"])

    panel_2010 = panel[panel["ola"] == 2010].copy()
    panel_2013 = panel[panel["ola"] == 2013].copy()
    coords_2010 = coordenadas[coordenadas["ola"] == 2010].copy()
    coords_2013 = coordenadas[coordenadas["ola"] == 2013].copy()

    # 'consecutivo' puede llegar como int64 en 'panel' (viene de un parquet
    # de extracción GEE) y como str en 'coordenadas' (o viceversa, según el
    # dtype que haya inferido pandas al leer el CSV de origen). Sin esto,
    # las comparaciones de más abajo (sets, .index, 'in') nunca encuentran
    # coincidencia aunque el valor 'sea el mismo' -- consecutivos_comunes
    # queda vacío en silencio y el crecimiento entre olas sale en 0 hogares
    # para TODOS, no por falta real de pares válidos.
    for df_ in (panel_2010, panel_2013, coords_2010, coords_2013):
        df_["consecutivo"] = df_["consecutivo"].astype(str)

    dup_2010 = set(panel_2010.loc[panel_2010["consecutivo"].duplicated(keep=False), "consecutivo"])
    dup_2013 = set(panel_2013.loc[panel_2013["consecutivo"].duplicated(keep=False), "consecutivo"])
    consecutivos_ambiguos = dup_2010 | dup_2013

    val_2010_s = panel_2010.set_index("consecutivo")["dmsp_stable_lights"]
    val_2013_s = panel_2013.set_index("consecutivo")["dmsp_stable_lights"]
    coords_2010_s = coords_2010.set_index("consecutivo")[["lat", "lon"]]
    coords_2013_s = coords_2013.set_index("consecutivo")[["lat", "lon"]]

    consecutivos_comunes = sorted((set(val_2010_s.index) & set(val_2013_s.index)) - consecutivos_ambiguos)
    n_excluidos = len((set(val_2010_s.index) & set(val_2013_s.index)) & consecutivos_ambiguos)
    if n_excluidos:
        log.info(f"  {n_excluidos:,} hogares excluidos por ser hogares divididos (consecutivo no único en 2010 o 2013).")

    filas = []
    for consecutivo in consecutivos_comunes:
        if consecutivo not in coords_2010_s.index or consecutivo not in coords_2013_s.index:
            continue
        lat1, lon1 = coords_2010_s.loc[consecutivo]
        lat2, lon2 = coords_2013_s.loc[consecutivo]
        distancia_m = calcular_distancia_haversine_m(lat1, lon1, lat2, lon2)
        se_movio = bool(distancia_m >= cfg["umbral_movimiento_m"])

        val_2010 = val_2010_s.loc[consecutivo]
        val_2013 = val_2013_s.loc[consecutivo]

        crecimiento = np.nan
        cambio_nivel = np.nan
        if se_movio:
            cambio_nivel = val_2013 - val_2010
        elif val_2010 != 0:
            crecimiento = (val_2013 - val_2010) / val_2010

        filas.append({
            "consecutivo": consecutivo,
            "ola": 2013,
            "dmsp_crecimiento_entre_olas": crecimiento,
            "dmsp_cambio_nivel_migracion": cambio_nivel,
            "dmsp_hogar_se_movio": se_movio,
            "dmsp_distancia_movimiento_m": distancia_m,
        })

    # pd.DataFrame([]) (lista vacía) no tiene columnas -- si ningún
    # consecutivo tuvo par válido, el merge posterior con 'panel' en
    # main() fallaría con KeyError('consecutivo'/'ola') en vez de
    # simplemente dejar estas columnas en NaN para todos los hogares.
    columnas_crecimiento = ["consecutivo", "ola", "dmsp_crecimiento_entre_olas",
                             "dmsp_cambio_nivel_migracion", "dmsp_hogar_se_movio",
                             "dmsp_distancia_movimiento_m"]
    df_crecimiento = pd.DataFrame(filas, columns=columnas_crecimiento)
    log.info(f"  Crecimiento entre olas 2010→2013 calculado: {len(df_crecimiento):,} hogares.")
    if len(df_crecimiento) == 0 and len(consecutivos_comunes) > 0:
        log.warning(
            "  0 hogares con crecimiento calculado pese a tener consecutivos comunes -- "
            "revisar si 'consecutivo' tiene el mismo tipo de dato (str vs. int) en "
            "series_temporales_dmsp.parquet y en coordenadas_hogares_s1.parquet."
        )
    return df_crecimiento


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(panel: pd.DataFrame) -> str:
    """Genera el texto del reporte de indicadores derivados."""
    sep = "=" * 65
    lines = [
        sep,
        "REPORTE — INDICADORES DERIVADOS DMSP-OLS",
        sep,
        "",
        "── 1. SATURACIÓN DEL SENSOR (ESTADO) POR RONDA ───────────",
        "",
    ]
    for ola, grupo in panel.groupby("ola"):
        pct_sat = 100 * grupo["dmsp_saturado"].mean()
        lines.append(f"  Ronda {ola}: {pct_sat:.1f}% de hogares saturados (n={len(grupo):,})")

    lines += [
        "",
        "── 2. CRECIMIENTO ENTRE OLAS 2010→2013 (DOS PUNTOS DE ESTADO) ─",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Cuántos hogares tienen dmsp_crecimiento_entre_olas (no se",
        "  movieron) vs. dmsp_cambio_nivel_migracion (sí se movieron).",
        "  NO confundir con la ventana acumulada de 3 años dentro de cada",
        "  ola (ver 03_series_temporales.py) — esto compara los dos",
        "  puntos de estado (2010 vs. 2013), no una tendencia interna.",
        "",
    ]
    subset = panel[panel["ola"] == 2013]
    if "dmsp_hogar_se_movio" in subset.columns and subset["dmsp_hogar_se_movio"].notna().any():
        n_total = subset["dmsp_hogar_se_movio"].notna().sum()
        n_movidos = int(subset["dmsp_hogar_se_movio"].sum())
        lines.append(
            f"  2010 → 2013: {n_total:,} hogares con par válido — "
            f"{n_total - n_movidos:,} sin moverse (crecimiento), {n_movidos:,} se movieron (cambio de nivel)"
        )
    else:
        lines.append("  Sin datos suficientes para calcular crecimiento.")

    lines += [
        "",
        "── 3. ESTADÍSTICOS DESCRIPTIVOS (VARIABLES CLAVE) ────────",
        "",
    ]
    cols = [
        "dmsp_avg_vis", "dmsp_stable_lights", "dmsp_log_stable_lights",
        "dmsp_stable_lights_acum_media", "dmsp_stable_lights_acum_tendencia",
        "dmsp_stable_lights_acum_crecimiento_acumulado",
        "dmsp_crecimiento_entre_olas", "dmsp_cambio_nivel_migracion", "dmsp_distancia_movimiento_m",
    ]
    cols = [c for c in cols if c in panel.columns]
    resumen = panel[cols].describe().T[["count", "mean", "std", "min", "max"]]
    for col, fila in resumen.iterrows():
        lines.append(
            f"    {col}: n={int(fila['count']):,}  media={fila['mean']:.3f}  "
            f"std={fila['std']:.3f}  min={fila['min']:.3f}  max={fila['max']:.3f}"
        )

    lines += ["", sep]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  04_indicadores_derivados.py")
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

    log.info("\n── Cargando panel y coordenadas ───────────────────────────")
    panel = cargar_panel(cfg)
    coordenadas = cargar_coordenadas(cfg)

    log.info("\n── Calculando indicadores de nivel sobre el estado ───────")
    panel = calcular_indicadores_de_nivel(panel)

    log.info("\n── Calculando crecimiento entre olas 2010→2013 ───────────")
    df_crecimiento = calcular_crecimiento_entre_olas(panel, coordenadas, cfg)
    panel = panel.merge(df_crecimiento, on=["consecutivo", "ola"], how="left")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "indicadores_derivados_dmsp.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(f"  Panel exportado: {out_path} ({len(panel):,} filas × {panel.shape[1]} columnas)")

    reporte = generar_reporte(panel)
    txt_path = Path(cfg["logs_dir"]) / "reporte_indicadores_derivados.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
