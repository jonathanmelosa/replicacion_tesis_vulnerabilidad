"""
04_indicadores_derivados.py
=============================

Agrega al panel de estado + ventana acumulada (series_temporales_landsat.parquet)
el crecimiento ENTRE OLAS 2010→2013 para las dos variables de estado más
interpretables (l5_ndvi: vegetación; l5_ndbi: área construida) — mismo
principio que dmsp_ols_pipeline/scripts/04_indicadores_derivados.py y la
misma nota de nomenclatura: esto es DISTINTO de la ventana acumulada de
3 años dentro de cada ola (03_series_temporales.py).

QUÉ HACE

    1. Carga series_temporales_landsat.parquet.
    2. Carga las coordenadas de los hogares (reutilizadas de
       sentinel1_pipeline/) para clasificar movimiento entre 2010 y 2013.
    3. Para l5_ndvi y l5_ndbi: calcula el cambio entre el estado 2010 y
       el estado 2013 (diferencia absoluta, no razón — NDVI/NDBI pueden
       cruzar por 0, así que una razón no es interpretable aquí, a
       diferencia de las luces nocturnas en dmsp_ols_pipeline).
         a. Si no se movió: l5_ndvi_cambio_entre_olas / l5_ndbi_cambio_entre_olas.
         b. Si se movió: mismas columnas, pero conceptualmente miden
            "cambio de entorno al mudarse", no "cambio del lugar" — se
            deja la MISMA columna (a diferencia de DMSP-OLS/VIIRS) porque
            NDVI/NDBI ya son diferencias acotadas en [-2,2] sin importar
            si el hogar se movió o no; se agrega l5_hogar_se_movio como
            bandera para que el usuario decida si excluir esos casos.
    4. Exporta el panel enriquecido y un reporte de calidad.

    DECISIÓN: se excluyen los hogares divididos (consecutivo no único
    dentro de una ronda) del cálculo — mismo criterio que en los demás
    pipelines nuevos/corregidos de esta revisión.

INPUTS

    data/processed/series_temporales_landsat.parquet
    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet

OUTPUTS

    data/processed/indicadores_derivados_landsat.parquet
    logs/reporte_indicadores_derivados.txt

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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "series_temporales": _HERE.parent / "data" / "processed" / "series_temporales_landsat.parquet",
    "coordenadas_hogares": (
        _HERE.parent.parent
        / "sentinel1_pipeline" / "data" / "processed" / "coordenadas_hogares_s1.parquet"
    ),
    "umbral_movimiento_m": 100,
    "columnas_cambio_entre_olas": ["l5_ndvi", "l5_ndbi"],
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


def cargar_panel(cfg: dict) -> pd.DataFrame:
    """Lee series_temporales_landsat.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["series_temporales"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró el panel de series temporales.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO: corre primero 03_series_temporales.py")
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


def calcular_distancia_haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distancia en metros entre dos puntos (lat, lon) — fórmula de Haversine."""
    radio_tierra_m = 6_371_000
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    return radio_tierra_m * 2 * np.arcsin(np.sqrt(a))


def calcular_cambio_entre_olas(panel: pd.DataFrame, coordenadas: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calcula el cambio 2010→2013 (diferencia absoluta) para cada columna en CONFIG['columnas_cambio_entre_olas'], excluyendo hogares divididos."""
    if 2010 not in panel["ola"].values or 2013 not in panel["ola"].values:
        log.info("  Panel no tiene ambas rondas — no se calcula cambio entre olas.")
        return pd.DataFrame(columns=["consecutivo", "ola", "l5_hogar_se_movio", "l5_distancia_movimiento_m"])

    panel_2010 = panel[panel["ola"] == 2010]
    panel_2013 = panel[panel["ola"] == 2013]
    coords_2010 = coordenadas[coordenadas["ola"] == 2010]
    coords_2013 = coordenadas[coordenadas["ola"] == 2013]

    dup_2010 = set(panel_2010.loc[panel_2010["consecutivo"].duplicated(keep=False), "consecutivo"])
    dup_2013 = set(panel_2013.loc[panel_2013["consecutivo"].duplicated(keep=False), "consecutivo"])
    consecutivos_ambiguos = dup_2010 | dup_2013

    panel_2010_s = panel_2010.set_index("consecutivo")
    panel_2013_s = panel_2013.set_index("consecutivo")
    coords_2010_s = coords_2010.set_index("consecutivo")[["lat", "lon"]]
    coords_2013_s = coords_2013.set_index("consecutivo")[["lat", "lon"]]

    consecutivos_comunes = sorted((set(panel_2010_s.index) & set(panel_2013_s.index)) - consecutivos_ambiguos)
    n_excluidos = len((set(panel_2010_s.index) & set(panel_2013_s.index)) & consecutivos_ambiguos)
    if n_excluidos:
        log.info(f"  {n_excluidos:,} hogares excluidos por ser hogares divididos.")

    filas = []
    for consecutivo in consecutivos_comunes:
        if consecutivo not in coords_2010_s.index or consecutivo not in coords_2013_s.index:
            continue
        lat1, lon1 = coords_2010_s.loc[consecutivo]
        lat2, lon2 = coords_2013_s.loc[consecutivo]
        distancia_m = calcular_distancia_haversine_m(lat1, lon1, lat2, lon2)
        se_movio = bool(distancia_m >= cfg["umbral_movimiento_m"])

        fila = {"consecutivo": consecutivo, "ola": 2013, "l5_hogar_se_movio": se_movio, "l5_distancia_movimiento_m": distancia_m}
        for col in cfg["columnas_cambio_entre_olas"]:
            if col not in panel_2010_s.columns or col not in panel_2013_s.columns:
                continue
            v10, v13 = panel_2010_s.loc[consecutivo, col], panel_2013_s.loc[consecutivo, col]
            fila[f"{col}_cambio_entre_olas"] = (v13 - v10) if pd.notna(v10) and pd.notna(v13) else np.nan
        filas.append(fila)

    df_cambio = pd.DataFrame(filas)
    log.info(f"  Cambio entre olas 2010→2013 calculado: {len(df_cambio):,} hogares.")
    return df_cambio


def generar_reporte(panel: pd.DataFrame, cfg: dict) -> str:
    """Genera el texto del reporte de indicadores derivados."""
    sep = "=" * 65
    lines = [sep, "REPORTE — INDICADORES DERIVADOS LANDSAT 5 TM", sep, "",
             "── 1. CAMBIO ENTRE OLAS 2010→2013 ────────────────────────", ""]
    subset = panel[panel["ola"] == 2013]
    if "l5_hogar_se_movio" in subset.columns and subset["l5_hogar_se_movio"].notna().any():
        n_total = subset["l5_hogar_se_movio"].notna().sum()
        n_movidos = int(subset["l5_hogar_se_movio"].sum())
        lines.append(f"  2010 → 2013: {n_total:,} hogares con par válido — {n_movidos:,} se movieron ({100*n_movidos/n_total:.1f}%)")
    else:
        lines.append("  Sin datos suficientes.")

    lines += ["", "── 2. ESTADÍSTICOS DESCRIPTIVOS ──────────────────────────", ""]
    cols = [f"{c}_cambio_entre_olas" for c in cfg["columnas_cambio_entre_olas"]]
    cols = [c for c in cols if c in panel.columns]
    if cols:
        resumen = panel[cols].describe().T[["count", "mean", "std", "min", "max"]]
        for col, fila in resumen.iterrows():
            lines.append(f"    {col}: n={int(fila['count']):,}  media={fila['mean']:.3f}  std={fila['std']:.3f}  min={fila['min']:.3f}  max={fila['max']:.3f}")

    lines += ["", sep]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  04_indicadores_derivados.py")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando panel y coordenadas ───────────────────────────")
    panel = cargar_panel(cfg)
    coordenadas = cargar_coordenadas(cfg)

    log.info("\n── Calculando cambio entre olas 2010→2013 ────────────────")
    df_cambio = calcular_cambio_entre_olas(panel, coordenadas, cfg)
    panel = panel.merge(df_cambio, on=["consecutivo", "ola"], how="left")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "indicadores_derivados_landsat.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(f"  Panel exportado: {out_path} ({len(panel):,} filas × {panel.shape[1]} columnas)")

    reporte = generar_reporte(panel, cfg)
    (Path(cfg["logs_dir"]) / "reporte_indicadores_derivados.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
