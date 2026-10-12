"""
landsat5_utils.py
===================
Funciones reutilizables específicas de Landsat 5 TM (óptico, generación
anterior a Sentinel-2), usadas por los scripts 01 en adelante de este
pipeline.

Los índices derivados replican EXACTAMENTE las fórmulas ya validadas en
sentinel2_pipeline/utils/sentinel2_utils.py (mismo nombre de índice,
misma fórmula), sustituyendo únicamente las bandas de entrada por sus
equivalentes de Landsat 5 TM — ver el mapeo de bandas más abajo. Esto
garantiza que 'ndvi' de Landsat 5 y 'ndvi' de Sentinel-2 sean
comparables en su definición, aunque los sensores difieran (ver
discusión de continuidad Landsat-Sentinel en la tesis, Sección 3.3.7).

Ver docs/prompts/prompt_landsat5_pipeline.md para el fundamento teórico.
"""
from __future__ import annotations

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging

# ── Librerías externas ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Colección Landsat 5 TM, Collection 2, Tier 1, Nivel 2 (reflectancia de
# superficie ya corregida atmosféricamente por USGS) — el análogo directo
# de COPERNICUS/S2_SR_HARMONIZED para Landsat 5. Cobertura real: marzo de
# 1984 a noviembre de 2011 (fecha de la última escena TM utilizable antes
# de que fallara el sensor SLC; el satélite se retiró formalmente en
# 2013, pero dejó de adquirir imágenes de calidad utilizable antes) —
# verificar la fecha exacta de corte contra el catálogo de USGS/GEE al
# correr el pipeline, no asumir solo este comentario.
COLECCION_L5_SR = "LANDSAT/LT05/C02/T1_L2"

# Mapeo de bandas Landsat 5 TM -> mismo rol espectral que Sentinel-2:
#   SR_B1 (azul)   ~ B2 en S2
#   SR_B2 (verde)  ~ B3 en S2
#   SR_B3 (rojo)   ~ B4 en S2
#   SR_B4 (NIR)    ~ B8 en S2
#   SR_B5 (SWIR1)  ~ B11 en S2
#   SR_B7 (SWIR2)  ~ B12 en S2
# (SR_B6 es la banda térmica — no se usa aquí, mismo criterio que S2 no
# usa sus bandas de vapor de agua/cirrus para estos índices).
BANDAS_REFLECTANCIA = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
BANDA_QA = "QA_PIXEL"

# Factores de escala oficiales de USGS Collection 2 Level-2 Surface
# Reflectance: reflectancia_real = DN * 0.0000275 - 0.2. Fuente: USGS
# Landsat Collection 2 Level-2 Science Product Guide. Verificar contra la
# documentación vigente antes de usar — USGS ha cambiado factores de
# escala entre Collection 1 y Collection 2.
FACTOR_MULT_REFLECTANCIA = 0.0000275
FACTOR_ADD_REFLECTANCIA = -0.2

# Bits de QA_PIXEL (Collection 2) a enmascarar: bit 3 = nube, bit 4 =
# sombra de nube. Se combinan en una sola máscara bit a bit — mismo
# principio que la máscara de nubes de Sentinel-2, pero aquí con la banda
# de calidad propia de Landsat en vez de una colección de probabilidad de
# nube separada (Landsat Collection 2 no tiene un producto equivalente a
# COPERNICUS/S2_CLOUD_PROBABILITY).
_BIT_NUBE = 1 << 3
_BIT_SOMBRA_NUBE = 1 << 4

# Caja delimitadora amplia de Colombia continental + insular (lon, lat),
# usada SOLO para acotar la colección ANTES de construir el compuesto
# anual — sin este filtro, .median() sobre la colección global (decenas
# de miles de escenas/año en años de mucho archivo) excede el límite de
# memoria de un solo request a GEE ('User memory limit exceeded'), un
# error de escala del cómputo, no evidencia de falta de datos.
COLOMBIA_BBOX = [-79.5, -4.5, -66.5, 13.5]

INDICES_DERIVADOS = [
    "ndvi", "evi", "savi", "ndbi", "ndwi", "mndwi", "bsi",
    "tc_brillo", "tc_verdor", "tc_humedad",
]


def enmascarar_nubes(imagen):
    """
    Enmascara píxeles de nube/sombra de nube usando la banda QA_PIXEL de
    Collection 2, ANTES de escalar a reflectancia — mismo orden de
    operaciones que sentinel2_pipeline (máscara primero, escala después
    no importa el orden en realidad, pero se mantiene la simetría).
    """
    qa = imagen.select(BANDA_QA)
    mascara_despejado = (
        qa.bitwiseAnd(_BIT_NUBE).eq(0)
        .And(qa.bitwiseAnd(_BIT_SOMBRA_NUBE).eq(0))
    )
    return imagen.updateMask(mascara_despejado)


def escalar_reflectancia(imagen):
    """Aplica los factores de escala oficiales de USGS Collection 2 Level-2 a las bandas de reflectancia."""
    escalada = (
        imagen.select(BANDAS_REFLECTANCIA)
        .multiply(FACTOR_MULT_REFLECTANCIA)
        .add(FACTOR_ADD_REFLECTANCIA)
    )
    return escalada.copyProperties(imagen, imagen.propertyNames())


def agregar_indices_derivados(imagen):
    """
    Agrega los mismos 10 índices que sentinel2_pipeline/utils/sentinel2_utils.py
    (ver ese archivo para las fórmulas completas y las notas metodológicas
    sobre estabilidad numérica de las divisiones), sustituyendo B2,B3,B4,
    B8,B11,B12 (Sentinel-2) por SR_B1,SR_B2,SR_B3,SR_B4,SR_B5,SR_B7
    (Landsat 5) respectivamente.

    TASSELED CAP — FUENTE
        Coeficientes de Crist, E.P. (1985), "A TM Tasseled Cap equivalent
        transformation for reflectance factor data", Remote Sensing of
        Environment, 17(3), 301-306 — el set de coeficientes estándar y
        más citado para Landsat TM (distinto del usado para Sentinel-2 en
        sentinel2_pipeline, que son coeficientes de Nedkov 2017
        específicos para S2 — NO son intercambiables entre sensores).
    """
    azul = imagen.select("SR_B1")
    verde = imagen.select("SR_B2")
    rojo = imagen.select("SR_B3")
    nir = imagen.select("SR_B4")
    swir1 = imagen.select("SR_B5")
    swir2 = imagen.select("SR_B7")

    ndvi = nir.subtract(rojo).divide(nir.add(rojo)).rename("ndvi")
    evi = (
        nir.subtract(rojo).multiply(2.5)
        .divide(nir.add(rojo.multiply(6)).subtract(azul.multiply(7.5)).add(1))
        .rename("evi")
    )
    savi = nir.subtract(rojo).multiply(1.5).divide(nir.add(rojo).add(0.5)).rename("savi")
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("ndbi")
    ndwi = verde.subtract(nir).divide(verde.add(nir)).rename("ndwi")
    mndwi = verde.subtract(swir1).divide(verde.add(swir1)).rename("mndwi")
    bsi = (
        swir1.add(rojo).subtract(nir.add(azul))
        .divide(swir1.add(rojo).add(nir.add(azul)))
        .rename("bsi")
    )

    # Tasseled Cap (Crist, 1985) sobre TM1,TM2,TM3,TM4,TM5,TM7.
    tc_brillo = (
        azul.multiply(0.2043).add(verde.multiply(0.4158)).add(rojo.multiply(0.5524))
        .add(nir.multiply(0.5741)).add(swir1.multiply(0.3124)).add(swir2.multiply(0.2303))
        .rename("tc_brillo")
    )
    tc_verdor = (
        azul.multiply(-0.1603).add(verde.multiply(-0.2819)).add(rojo.multiply(-0.4934))
        .add(nir.multiply(0.7940)).add(swir1.multiply(-0.0002)).add(swir2.multiply(-0.1446))
        .rename("tc_verdor")
    )
    tc_humedad = (
        azul.multiply(0.0315).add(verde.multiply(0.2021)).add(rojo.multiply(0.3102))
        .add(nir.multiply(0.1594)).add(swir1.multiply(-0.6806)).add(swir2.multiply(-0.6109))
        .rename("tc_humedad")
    )

    return imagen.addBands([ndvi, evi, savi, ndbi, ndwi, mndwi, bsi, tc_brillo, tc_verdor, tc_humedad])


def construir_imagen_anual(ee_module, anio: int):
    """
    Filtra COLECCION_L5_SR al año calendario `anio`, enmascara nubes,
    escala a reflectancia real, agrega los índices derivados, y reduce
    TODAS las escenas del año a UNA sola imagen (mediana por píxel).

    A diferencia de DMSP-OLS/ALOS PALSAR (que ya vienen como un compuesto
    anual pre-hecho), Landsat 5 entrega escenas individuales (revisita de
    16 días) — este pipeline construye su propio compuesto anual con
    mediana (más robusta a nubes residuales no enmascaradas que la media),
    mismo principio que sentinel1_pipeline/sentinel2_pipeline usan para
    sus ventanas temporales, pero aquí acotado a un año calendario en vez
    de la ventana proxy compartida de esos pipelines.
    """
    region_colombia = ee_module.Geometry.Rectangle(COLOMBIA_BBOX)
    coleccion = (
        ee_module.ImageCollection(COLECCION_L5_SR)
        .filterDate(f"{anio}-01-01", f"{anio}-12-31")
        .filterBounds(region_colombia)
    )

    def procesar_escena(imagen):
        enmascarada = enmascarar_nubes(imagen)
        # copyProperties() en escalar_reflectancia() devuelve ee.Element, no
        # ee.Image (comportamiento documentado de la API de GEE) — hay que
        # recastear explícitamente o los .select() de agregar_indices_derivados()
        # fallan con 'Element' object has no attribute 'select'.
        escalada = ee_module.Image(escalar_reflectancia(enmascarada))
        return agregar_indices_derivados(escalada)

    coleccion_procesada = coleccion.map(procesar_escena)
    return coleccion_procesada.median()


def contar_escenas_anio(ee_module, anio: int) -> int:
    """Cuenta cuántas escenas Landsat 5 hay en el año `anio` (sobre todo el catálogo, sin filtrar por geometría)."""
    return (
        ee_module.ImageCollection(COLECCION_L5_SR)
        .filterDate(f"{anio}-01-01", f"{anio}-12-31")
        .size()
        .getInfo()
    )


def construir_geometria_buffer(ee_module, lat: float, lon: float, radio_m: float):
    """Construye un círculo de `radio_m` metros alrededor de (lat, lon) — 250 m por defecto, ver CONFIG."""
    return ee_module.Geometry.Point([lon, lat]).buffer(radio_m)


def construir_feature_collection_hogares(ee_module, hogares_df, geometria_fn):
    """Construye una ee.FeatureCollection con un Feature por hogar (mismo patrón que los demás pipelines nuevos)."""
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
    escala = cfg.get("escala_reduccion_m", 30)
    tile_scale = cfg.get("tile_scale", 4)
    return imagen.reduceRegions(
        collection=fc_hogares,
        reducer=ee_module.Reducer.mean(),
        scale=escala,
        tileScale=tile_scale,
    )


def calcular_pendiente_temporal(grupo: pd.DataFrame, columna_anio: str, columna_valor: str) -> float:
    """Pendiente (mínimos cuadrados) de `columna_valor` a lo largo de los años de la ventana acumulada."""
    grupo_ordenado = grupo.sort_values(columna_anio)
    x = grupo_ordenado[columna_anio].to_numpy(dtype=float)
    y = grupo_ordenado[columna_valor].to_numpy(dtype=float)
    valido = ~np.isnan(y)
    if valido.sum() < 2:
        return float("nan")
    pendiente, _ = np.polyfit(x[valido], y[valido], 1)
    return float(pendiente)


def calcular_estadisticos_ventana_acumulada(grupo: pd.DataFrame, columna_valor: str, prefijo: str) -> dict:
    """Estadísticos de `columna_valor` sobre la ventana de 3 años de un hogar. Misma lógica que dmsp_utils.py/alos_utils.py."""
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
