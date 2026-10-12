"""
alos_utils.py
==============
Funciones reutilizables específicas de ALOS PALSAR (radar banda-L, mosaico
anual de JAXA), usadas por los scripts 01 en adelante de este pipeline.

Ver docs/prompts/prompt_alos_palsar_pipeline.md para el fundamento teórico
completo y la explicación pedagógica en la tesis, Sección 3.3.7.
"""
from __future__ import annotations

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging

# ── Librerías externas ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Mosaico global anual PALSAR/PALSAR-2 de JAXA en GEE. Cobertura real
# confirmada contra el catálogo: 2007-2010 (ALOS-1/PALSAR), hueco
# 2011-2014, 2015 en adelante (ALOS-2/PALSAR-2) — este pipeline SOLO usa
# 2008-2010 (ronda 2010; ver "Por qué únicamente la ola 2010" más abajo).
COLECCION_ALOS_PALSAR = "JAXA/ALOS/PALSAR/YEARLY/SAR"

# 'HH': polarización horizontal-horizontal (transmite y recibe horizontal).
# 'HV': polarización horizontal-vertical (cruzada) — más sensible a volumen
#       de vegetación/dispersión de volumen, análoga en espíritu a VH en
#       Sentinel-1.
BANDAS_ALOS = ["HH", "HV"]

# Resolución nativa del mosaico global de 25 m (0.8 arco-segundo) —
# fuente: documentación JAXA EORC del producto. Se usa como referencia
# para 'scale' en reduceRegion, aunque el buffer de extracción (ver
# CONFIG['radio_buffer_m']) se fija en 250 m para ser directamente
# comparable con Sentinel-1 (mismo dominio físico — radar de la banda
# construida —, mismo criterio de error de georreferenciación de la
# encuesta ya usado en sentinel1_pipeline/), no con la resolución nativa
# de ALOS PALSAR en sí.
ESCALA_NATIVA_M = 25

# Constante de calibración JAXA para convertir el número digital (DN) del
# mosaico a gamma-0 en decibeles: gamma0_dB = 20*log10(DN) - 83.0
# Fuente: documentación de calibración del mosaico global PALSAR de JAXA
# EORC (Shimada et al., 2009, "PALSAR Radiometric and Geometric
# Calibration", IEEE TGRS) — mismo factor usado en tutoriales oficiales de
# Google Earth Engine para este dataset. A diferencia de Sentinel-1 (que
# entrega VV/VH ya en dB), ALOS PALSAR en GEE entrega DN crudo — esta
# conversión es OBLIGATORIA antes de cualquier otro cálculo.
_FACTOR_CALIBRACION_DB = 83.0


def calibrar_a_gamma0_db(imagen):
    """
    Convierte las bandas HH/HV (DN crudo) del mosaico ALOS PALSAR a
    gamma-0 en decibeles: 20*log10(DN) - 83.0.

    DN=0 (fuera de la franja de cobertura o sin dato) produce log10(0) =
    -infinito — se enmascara explícitamente ANTES de la conversión
    (updateMask(DN > 0)) para que esos píxeles queden como 'sin dato'
    (masked) en vez de -infinito, que rompería cualquier reduceRegion
    posterior en vez de simplemente promediar sobre los píxeles válidos.
    """
    hh = imagen.select("HH")
    hv = imagen.select("HV")

    hh_valido = hh.updateMask(hh.gt(0))
    hv_valido = hv.updateMask(hv.gt(0))

    hh_db = hh_valido.log10().multiply(20).subtract(_FACTOR_CALIBRACION_DB).rename("HH_db")
    hv_db = hv_valido.log10().multiply(20).subtract(_FACTOR_CALIBRACION_DB).rename("HV_db")

    return imagen.addBands([hh_db, hv_db], overwrite=True)


def agregar_bandas_derivadas(imagen):
    """
    Agrega bandas derivadas de HH_db/HV_db, replicando EXACTAMENTE la
    metodología ya validada en sentinel1_pipeline/utils/sentinel1_utils.py
    (ver la nota metodológica completa ahí sobre por qué unas bandas se
    calculan en dB y otras en escala lineal) — solo cambia la nomenclatura
    (hh_hv en vez de vv_vh, porque ALOS PALSAR es HH/HV, no VV/VH).

    hh_hv_diff = HH_db - HV_db (resta en dB — estable, interpretación física directa)
    hh_hv_ratio, hv_hh_ratio, indice_normalizado, log_hh_hv: sobre potencia LINEAL
        (deshaciendo dB -> lineal ANTES de dividir, por la misma razón de
        estabilidad numérica ya documentada para Sentinel-1).
    """
    hh_db = imagen.select("HH_db")
    hv_db = imagen.select("HV_db")

    hh_hv_diff = hh_db.subtract(hv_db).rename("hh_hv_diff")

    _LN_10 = 2.302585092994046
    hh_lin = hh_db.divide(10).multiply(_LN_10).exp()
    hv_lin = hv_db.divide(10).multiply(_LN_10).exp()

    hh_hv_ratio = hh_lin.divide(hv_lin).rename("hh_hv_ratio")
    hv_hh_ratio = hv_lin.divide(hh_lin).rename("hv_hh_ratio")
    indice_normalizado = (
        hh_lin.subtract(hv_lin).divide(hh_lin.add(hv_lin)).rename("indice_normalizado")
    )
    log_hh_hv = (
        hh_hv_ratio.log().rename("log_hh_hv").updateMask(hh_hv_ratio.gt(0))
    )

    return imagen.addBands([hh_hv_diff, hh_hv_ratio, hv_hh_ratio, indice_normalizado, log_hh_hv])


def construir_imagen_anual(ee_module, anio: int):
    """
    Filtra COLECCION_ALOS_PALSAR al año calendario `anio`, calibra a
    gamma-0 dB y agrega las bandas derivadas. Retorna UNA sola imagen
    (el mosaico global de JAXA ya es un compuesto anual — no hay
    'ventana con varias imágenes' que promediar dentro de un año, igual
    que DMSP-OLS).
    """
    coleccion = (
        ee_module.ImageCollection(COLECCION_ALOS_PALSAR)
        .filterDate(f"{anio}-01-01", f"{anio}-12-31")
        .select(BANDAS_ALOS)
    )
    imagen = coleccion.mosaic()  # mosaico global: no hay traslape real entre "imágenes" del mismo año
    imagen = calibrar_a_gamma0_db(imagen)
    return agregar_bandas_derivadas(imagen)


def contar_imagenes_anio(ee_module, anio: int) -> int:
    """Cuenta cuántas imágenes (teselas) componen el año `anio` en el mosaico global."""
    return (
        ee_module.ImageCollection(COLECCION_ALOS_PALSAR)
        .filterDate(f"{anio}-01-01", f"{anio}-12-31")
        .size()
        .getInfo()
    )


def construir_geometria_buffer(ee_module, lat: float, lon: float, radio_m: float):
    """Construye un círculo de `radio_m` metros alrededor de (lat, lon) — 250 m por defecto, ver ESCALA_NATIVA_M."""
    return ee_module.Geometry.Point([lon, lat]).buffer(radio_m)


def construir_feature_collection_hogares(ee_module, hogares_df, geometria_fn):
    """Construye una ee.FeatureCollection con un Feature por hogar (mismo patrón que dmsp_utils.py/viirs_utils.py)."""
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
    """Promedia espacialmente las bandas de `imagen` dentro de la geometría de cada hogar de `fc_hogares`."""
    escala = cfg.get("escala_reduccion_m", ESCALA_NATIVA_M)
    tile_scale = cfg.get("tile_scale", 4)
    return imagen.reduceRegions(
        collection=fc_hogares,
        reducer=ee_module.Reducer.mean(),
        scale=escala,
        tileScale=tile_scale,
    )


def calcular_pendiente_temporal(grupo: pd.DataFrame, columna_anio: str, columna_valor: str) -> float:
    """Pendiente (mínimos cuadrados) de `columna_valor` a lo largo de los años de la ventana acumulada. Ver dmsp_utils.py (misma lógica)."""
    grupo_ordenado = grupo.sort_values(columna_anio)
    x = grupo_ordenado[columna_anio].to_numpy(dtype=float)
    y = grupo_ordenado[columna_valor].to_numpy(dtype=float)
    valido = ~np.isnan(y)
    if valido.sum() < 2:
        return float("nan")
    pendiente, _ = np.polyfit(x[valido], y[valido], 1)
    return float(pendiente)


def calcular_estadisticos_ventana_acumulada(grupo: pd.DataFrame, columna_valor: str, prefijo: str) -> dict:
    """Estadísticos de `columna_valor` sobre la ventana de 3 años de un hogar. Misma lógica que dmsp_utils.py."""
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
