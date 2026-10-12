"""
dmsp_utils.py
==============
Funciones reutilizables específicas de DMSP-OLS (luces nocturnas,
generación anterior a VIIRS), usadas por los scripts 01 en adelante del
pipeline DMSP-OLS.

Ver docs/prompts/prompt_dmsp_ols_pipeline.md para el fundamento teórico y
las decisiones metodológicas detrás de cada función (por qué este pipeline
existe además de viirs_pipeline/, por qué NO hay ventana proxy para
2010/2013, por qué se usa un buffer de mayor radio que VIIRS/S1, etc.).
"""
from __future__ import annotations

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging

# ── Librerías externas ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Colección anual DMSP-OLS Nighttime Lights de NOAA en GEE. Cobertura real
# confirmada contra el catálogo de Earth Engine: 1992-2014 (composites
# anuales). En años donde dos satélites coincidieron en operación, el
# catálogo puede tener más de una imagen para el mismo año — este pipeline
# promedia todas las imágenes que caen en el año objetivo (ver
# construir_imagen_anual()) en vez de asumir una sola.
COLECCION_DMSP_OLS = "NOAA/DMSP-OLS/NIGHTTIME_LIGHTS"

# 'avg_vis'      : promedio del número digital de la banda visible, SIN
#                  filtrar eventos efímeros (incendios, quemas, luna) —
#                  variable cruda.
# 'stable_lights': igual que avg_vis pero con eventos efímeros removidos —
#                  la variable de contenido económico recomendada (más
#                  cercana en espíritu a lo que mide VIIRS avg_rad).
# 'cf_cvg'       : número de observaciones libres de nubes usadas en el
#                  compuesto anual — variable de CALIDAD del dato, no de
#                  actividad económica.
BANDAS_DMSP = ["avg_vis", "stable_lights", "cf_cvg"]

# Resolución nativa real del producto gridded en GEE (verificado contra el
# catálogo: 927.67 m, no los ~2.7 km que a veces se citan para el
# instrumento OLS crudo antes de la reproyección a grilla). Se usa como
# 'scale' en reduceRegions.
ESCALA_NATIVA_M = 927.67

# Techo de saturación del sensor: stable_lights y avg_vis son enteros de 6
# bits (0-63); el valor 63 indica saturación del sensor sobre fuentes muy
# brillantes (núcleos urbanos densos), no que la fuente real tenga
# exactamente esa intensidad — ver prompt maestro, sección "Limitaciones".
VALOR_SATURACION = 63


def construir_imagen_anual(ee_module, anio: int, bandas: list = BANDAS_DMSP):
    """
    Filtra COLECCION_DMSP_OLS al año calendario `anio` y retorna UNA sola
    imagen (promedio de todas las que existan para ese año — normalmente
    una, pero el catálogo puede tener dos en años de transición entre
    satélites, ver docstring del módulo).

    A diferencia de VIIRS/Sentinel-1 (colecciones con muchas imágenes
    dentro de UNA ventana que se agregan en estadísticos), aquí cada año
    calendario produce como máximo una imagen. La agregación temporal de
    este pipeline no ocurre "dentro de un año" sino ENTRE años: cada ronda
    tiene una ventana acumulada de 3 años (ver
    CONFIG['ventanas_por_ronda'] y 03_series_temporales.py), así que esta
    función se llama una vez por cada año de esa ventana, no una sola vez
    por ronda.
    """
    coleccion = (
        ee_module.ImageCollection(COLECCION_DMSP_OLS)
        .filterDate(f"{anio}-01-01", f"{anio}-12-31")
        .select(bandas)
    )
    return coleccion.mean()


def contar_imagenes_anio(ee_module, anio: int) -> int:
    """Cuenta cuántas imágenes componen el año `anio` en la colección (normalmente 1, a veces 2)."""
    return (
        ee_module.ImageCollection(COLECCION_DMSP_OLS)
        .filterDate(f"{anio}-01-01", f"{anio}-12-31")
        .size()
        .getInfo()
    )


def construir_geometria_buffer(ee_module, lat: float, lon: float, radio_m: float):
    """
    Construye un círculo de `radio_m` metros alrededor de (lat, lon).

    Radio por defecto en CONFIG: igual a ESCALA_NATIVA_M (~928 m), mismo
    criterio ya usado en viirs_pipeline (buffer ≈ resolución nativa del
    sensor) — ver justificación completa en el prompt maestro, sección
    "Escala espacial: por qué un buffer más grande que VIIRS".
    """
    return ee_module.Geometry.Point([lon, lat]).buffer(radio_m)


def construir_feature_collection_hogares(ee_module, hogares_df, geometria_fn):
    """
    Construye una ee.FeatureCollection con un Feature por hogar (mismo
    patrón que viirs_pipeline/utils/viirs_utils.py). Las propiedades son
    'id_ola', 'ola' y 'consecutivo' — necesarias para re-identificar cada
    hogar en el resultado de reduceRegions.
    """
    features = [
        ee_module.Feature(
            geometria_fn(row["lat"], row["lon"]),
            {
                "id_ola": str(row["id_ola"]),
                "ola": int(row["ola"]),
                "consecutivo": str(row["consecutivo"]),
            },
        )
        for _, row in hogares_df.iterrows()
    ]
    return ee_module.FeatureCollection(features)


def extraer_valores_lote(ee_module, imagen, fc_hogares, cfg: dict):
    """
    Promedia espacialmente las bandas de `imagen` (UNA sola imagen anual,
    no una colección) dentro de la geometría de cada hogar de
    `fc_hogares`, con una sola llamada reduceRegions server-side para todo
    el lote.

    A diferencia de VIIRS/S1 (reduceRegions una vez POR IMAGEN dentro de
    un .map() sobre la colección), aquí solo hay una imagen por año/ronda,
    así que basta una sola llamada — mismo patrón que
    sentinel1_pipeline/scripts/05_metricas_textura.py (que también reduce
    una sola imagen ya compuesta, no una colección).
    """
    escala = cfg.get("escala_reduccion_m", ESCALA_NATIVA_M)
    tile_scale = cfg.get("tile_scale", 4)
    return imagen.reduceRegions(
        collection=fc_hogares,
        reducer=ee_module.Reducer.mean(),
        scale=escala,
        tileScale=tile_scale,
    )


def marcar_saturacion(serie_stable_lights: pd.Series) -> pd.Series:
    """
    Bandera booleana: True si stable_lights == VALOR_SATURACION (63, el
    máximo del rango de 6 bits del sensor). Ver limitación documentada en
    el prompt maestro: en núcleos urbanos densos el sensor se satura y
    pierde capacidad discriminante — esta bandera permite excluir o tratar
    por separado esos hogares en la etapa de modelado, en vez de asumir
    que 63 es un valor de radiancia real y comparable al resto de la
    distribución.

    Se compara con `>= VALOR_SATURACION` (no `==`) porque el producto ya
    viene truncado a [0, 63]; un valor de 63 exacto tras el promedio
    espacial del buffer solo ocurre cuando TODO el buffer está saturado.
    """
    return serie_stable_lights >= VALOR_SATURACION


def calcular_log_stable_lights(serie_stable_lights: pd.Series) -> pd.Series:
    """log(stable_lights + 1) — misma transformación y misma justificación que viirs_log_rad_media."""
    return np.log(serie_stable_lights + 1)


def calcular_pendiente_temporal(grupo: pd.DataFrame, columna_anio: str, columna_valor: str) -> float:
    """
    Pendiente (mínimos cuadrados) de `columna_valor` a lo largo de los
    años de la ventana acumulada de un hogar×ronda, en unidades de
    columna_valor POR AÑO calendario.

    Misma lógica que viirs_utils.py::calcular_pendiente_temporal, adaptada
    a años en vez de meses (aquí la ventana tiene 3 observaciones, una por
    año — ver "estado vs. ventana acumulada" en el prompt maestro).
    Retorna NaN si el grupo tiene menos de 2 observaciones.
    """
    grupo_ordenado = grupo.sort_values(columna_anio)
    x = grupo_ordenado[columna_anio].to_numpy(dtype=float)
    y = grupo_ordenado[columna_valor].to_numpy(dtype=float)
    valido = ~np.isnan(y)
    if valido.sum() < 2:
        return float("nan")
    pendiente, _ = np.polyfit(x[valido], y[valido], 1)
    return float(pendiente)


def calcular_estadisticos_ventana_acumulada(grupo: pd.DataFrame, columna_valor: str, prefijo: str) -> dict:
    """
    Calcula, para UN hogar×ronda, los estadísticos de `columna_valor`
    (ya limpia) sobre TODOS los años de su ventana acumulada de 3 años
    (ver CONFIG['ventanas_por_ronda'] — 2008-2010 incl. para la ronda
    2010; 2011-2013 incl. para la ronda 2013).

    A diferencia de viirs_pipeline (12 observaciones mensuales por
    ventana), aquí solo hay 3 observaciones anuales — se calculan menos
    estadísticos (sin percentiles, poco informativos con n=3) pero se
    agrega la variable más relevante para "cambios acumulados":
    `<prefijo>_crecimiento_acumulado`, la variación relativa entre el
    primer y el último año de la ventana (no solo la pendiente) — más
    fácil de interpretar en la narrativa de la tesis ("cuánto cambió la
    luz nocturna en los 3 años previos a la ola") que una pendiente en
    unidades de DN/año.
    """
    grupo = grupo.dropna(subset=[columna_valor])
    n_obs = int(len(grupo))
    if n_obs == 0:
        return {
            f"{prefijo}_media": np.nan, f"{prefijo}_std": np.nan,
            f"{prefijo}_minimo": np.nan, f"{prefijo}_maximo": np.nan,
            f"{prefijo}_rango": np.nan, f"{prefijo}_cv": np.nan,
            f"{prefijo}_tendencia": np.nan, f"{prefijo}_crecimiento_acumulado": np.nan,
            f"{prefijo}_n_anios": 0,
        }

    serie = grupo[columna_valor]
    media = serie.mean()
    std = serie.std() if n_obs > 1 else 0.0
    minimo = serie.min()
    maximo = serie.max()

    crecimiento_acumulado = np.nan
    if n_obs >= 2:
        grupo_ordenado = grupo.sort_values("anio")
        valor_inicial = grupo_ordenado[columna_valor].iloc[0]
        valor_final = grupo_ordenado[columna_valor].iloc[-1]
        if valor_inicial != 0:
            crecimiento_acumulado = (valor_final - valor_inicial) / valor_inicial

    return {
        f"{prefijo}_media": media,
        f"{prefijo}_std": std,
        f"{prefijo}_minimo": minimo,
        f"{prefijo}_maximo": maximo,
        f"{prefijo}_rango": maximo - minimo,
        f"{prefijo}_cv": (std / media) if media != 0 else np.nan,
        f"{prefijo}_tendencia": calcular_pendiente_temporal(grupo, "anio", columna_valor),
        f"{prefijo}_crecimiento_acumulado": crecimiento_acumulado,
        f"{prefijo}_n_anios": n_obs,
    }
