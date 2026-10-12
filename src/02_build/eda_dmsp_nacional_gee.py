"""
eda_dmsp_nacional_gee.py
=========================
Reemplaza a eda_dmsp_nacional_municipios.py: extrae el raster DMSP-OLS 2010
para TODO Colombia desde la MISMA fuente que ya usa el resto de la tesis
para `dmsp_stable_lights` -- Google Earth Engine, colección
`NOAA/DMSP-OLS/NIGHTTIME_LIGHTS`, banda `stable_lights`, satelite F18 (ver
prompts/prompt_dmsp_ols_pipeline.md) -- en vez del dataset alternativo de
Zenodo ("CCNL 1992-2013") usado en la primera version de este pipeline.

POR QUE SE CAMBIO DE FUENTE (2026-09-12)
    La primera version de este mapa territorial usaba el dataset abierto
    de Zenodo (DOI 10.5281/zenodo.6644980) porque la descarga DIRECTA del
    sitio de NOAA/EOG redirige a un login (verificado con curl). Pero la variable `dmsp_stable_lights` que
    ya usa el resto de la tesis NUNCA se descargo de ahi -- se extrajo via
    Google Earth Engine (ver prompts/prompt_dmsp_ols_pipeline.md), que es
    de acceso publico con cuenta de servicio (sin el login que bloqueaba la
    descarga directa). Usar la MISMA fuente exacta para el fondo de este
    mapa (en vez del sustituto de Zenodo, un procesamiento distinto con
    escala distinta, ~0-128 en vez de 0-63) elimina la advertencia de "no
    intercambiable" que tenia la version anterior.

METODO DE EXTRACCION -- mosaico de tiles, no descarga directa (corregido
2026-09-12, mismo dia de la primera version)
    El primer intento usaba `Image.getDownloadURL()` (descarga sincrona de
    GeoTIFF), pero la cuenta de servicio disponible devolvio
    "PERMISSION_DENIED: earthengine.thumbnails.create" -- verificado que
    es un problema de PERMISOS, no de tamano de la region (fallo igual con
    un area de prueba de 2km). En su lugar se usa `Image.sampleRectangle()`
    (lee pixeles directamente, permiso distinto que si esta habilitado),
    que tiene un limite de 262,144 pixeles por llamada -- Colombia completa
    a resolucion nativa son ~4.08 millones de pixeles, asi que se pide en
    tiles de 480x480 (230,400 pixeles, bajo el limite) y se ensambla el
    mosaico localmente con numpy antes de escribir el GeoTIFF final con
    rasterio. Grid nativo verificado empiricamente: 120 pixeles/grado
    (rejilla de 30 arco-segundos). ~20 tiles para toda Colombia, corre en
    menos de 1 minuto.

INPUTS
    Credenciales de cuenta de servicio de Google Earth Engine (ya creadas
    para el pipeline de Sentinel-1, reutilizadas aqui via CONFIG -- ver
    CONFIG['gee_key_file']).

OUTPUTS
    data/interim/geo_referencia/dmsp_nacional_gee_2010.tif
        (raster GeoTIFF, banda stable_lights, escala 0-63 verificada,
        cobertura Colombia completa, resolucion nativa ~927.67m, EPSG:4326)

COMO CORRER
    cd src/02_build && python eda_dmsp_nacional_gee.py

DESPUES DE CORRER ESTE SCRIPT
    Volver a correr eda_dmsp_fondo_mapa.py para regenerar
    fondo_dmsp_colombia_2010.npz con la escala correcta (ya apunta a este
    archivo por defecto).
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESIS_MECA_DIR = PROJECT_ROOT.parent  # .../Documentos/Tesis_MECA/ (hermano de este repo)

CONFIG = {
    # ── Credenciales de Earth Engine ───────────────────────────────────────
    # Reutiliza la cuenta de servicio ya creada y registrada para el
    # pipeline de Sentinel-1 (sentinel1_pipeline/), en vez de duplicar
    # credenciales -- ver scripts/01_inicializar_gee.py de ese pipeline.
    "gee_service_account": "mi-servidor-osm@plasma-minutia-492021-k7.iam.gserviceaccount.com",
    "gee_key_file": TESIS_MECA_DIR / "sentinel1_pipeline" / "credenciales" / "gee_service_account.json",
    "gee_project": "plasma-minutia-492021-k7",

    # ── Fuente DMSP-OLS (identica a la que genera dmsp_stable_lights) ──────
    "coleccion": "NOAA/DMSP-OLS/NIGHTTIME_LIGHTS",
    "banda": "stable_lights",
    "anio": 2010,
    "satelite_esperado": "F18",  # ver prompts/prompt_dmsp_ols_pipeline.md

    # ── Extension de Colombia (identica a la usada en todo el mapa territorial) ──
    "bbox": {"xmin": -82, "xmax": -66.5, "ymin": -4.5, "ymax": 13.8},
    "escala_m": 927.67,  # resolucion nativa de DMSP-OLS
    "px_por_grado": 120,  # verificado empiricamente (grid nativo de 30 arco-segundos)
    "tam_tile_px": 480,  # 480**2=230,400 <= 262,144 (limite de sampleRectangle), con margen

    # ── Salida ──────────────────────────────────────────────────────────────
    "output_path": PROJECT_ROOT / "data" / "interim" / "geo_referencia" / "dmsp_nacional_gee_2010.tif",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)


def conectar_gee(cfg: dict):
    """Inicializa Earth Engine con la cuenta de servicio configurada."""
    ruta_key = Path(cfg["gee_key_file"])
    if not ruta_key.exists():
        log.error("-" * 62)
        log.error(f"ERROR: no se encontro el archivo de credenciales de GEE en {ruta_key}")
        log.error("SOLUCION: confirma que sentinel1_pipeline/credenciales/gee_service_account.json")
        log.error("existe en esa ubicacion, o ajusta CONFIG['gee_key_file'] en este script.")
        log.error("-" * 62)
        sys.exit(1)

    try:
        import ee
    except ModuleNotFoundError:
        log.error("-" * 62)
        log.error("ERROR: el paquete 'earthengine-api' no esta instalado.")
        log.error("SOLUCION: pip install earthengine-api")
        log.error("-" * 62)
        sys.exit(1)

    try:
        credenciales = ee.ServiceAccountCredentials(cfg["gee_service_account"], str(ruta_key))
        ee.Initialize(credenciales, project=cfg["gee_project"])
    except Exception as e:
        log.error("-" * 62)
        log.error(f"ERROR al inicializar Earth Engine: {e}")
        log.error("SOLUCION: revisa que la cuenta de servicio siga registrada en")
        log.error("https://code.earthengine.google.com/register y que la API de")
        log.error("Earth Engine este habilitada en el proyecto de Google Cloud.")
        log.error("-" * 62)
        sys.exit(1)

    log.info(f"Earth Engine inicializado con la cuenta: {cfg['gee_service_account']}")
    return ee


def obtener_imagen_dmsp(ee_module, cfg: dict):
    """
    Filtra la coleccion DMSP-OLS por el anio configurado y valida que la
    imagen encontrada sea del satelite esperado (F18) -- varios satelites
    DMSP pueden cubrir el mismo anio calendario con sensibilidades
    distintas, y el resto de la tesis usa especificamente F18 para 2010.
    """
    anio = cfg["anio"]
    coleccion = (
        ee_module.ImageCollection(cfg["coleccion"])
        .filterDate(f"{anio}-01-01", f"{anio + 1}-01-01")
    )
    ids = coleccion.aggregate_array("system:index").getInfo()
    log.info(f"Imagenes encontradas para {anio}: {ids}")

    if not ids:
        log.error("-" * 62)
        log.error(f"ERROR: no se encontraron imagenes de {cfg['coleccion']} para {anio}.")
        log.error("SOLUCION: verifica el nombre de la coleccion y el rango de anios")
        log.error("cubierto (DMSP-OLS: 1992-2014, ver prompt_dmsp_ols_pipeline.md).")
        log.error("-" * 62)
        sys.exit(1)

    ids_f18 = [i for i in ids if cfg["satelite_esperado"] in i]
    if not ids_f18:
        log.error("-" * 62)
        log.error(f"ERROR: ninguna imagen de {anio} corresponde al satelite {cfg['satelite_esperado']}.")
        log.error(f"Imagenes disponibles: {ids}")
        log.error("SOLUCION: revisa cual satelite usar para este anio (puede que")
        log.error("el resto de la tesis use otro) y ajusta CONFIG['satelite_esperado'].")
        log.error("-" * 62)
        sys.exit(1)
    if len(ids_f18) > 1:
        log.warning(f"Mas de una imagen F18 para {anio}: {ids_f18} -- se usa la primera.")

    imagen_id = ids_f18[0]
    log.info(f"Usando imagen: {imagen_id}")
    return ee_module.Image(f"{cfg['coleccion']}/{imagen_id}").select(cfg["banda"])


def extraer_raster_por_mosaico(ee_module, imagen, cfg: dict) -> None:
    """
    Extrae el raster completo de Colombia en un mosaico de tiles via
    `sampleRectangle` (en vez de `getDownloadURL`, que devolvio
    'PERMISSION_DENIED: earthengine.thumbnails.create' con la cuenta de
    servicio disponible -- verificado que el problema es de permisos, no
    de tamano de region: incluso una region de 2km fallo igual).
    `sampleRectangle` usa un permiso distinto (lectura de pixeles, no
    generacion de thumbnail/descarga) que si esta habilitado, pero tiene
    un limite de 262,144 pixeles por llamada -- se pide por tiles y se
    ensambla el mosaico localmente.

    Grid nativo verificado: 120 pixeles/grado (rejilla de 30 arco-segundos,
    ~927.67m), EPSG:4326 implicito.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    bbox = cfg["bbox"]
    px_por_grado = cfg["px_por_grado"]
    xmin, xmax, ymin, ymax = bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]

    nx = round((xmax - xmin) * px_por_grado)
    ny = round((ymax - ymin) * px_por_grado)
    tam_tile = cfg["tam_tile_px"]  # pixeles por lado, debe cumplir tam_tile**2 <= 262144

    log.info(f"Extension total: {nx} x {ny} pixeles ({nx * ny:,} pixeles totales)")
    log.info(f"Extrayendo en tiles de {tam_tile}x{tam_tile} pixeles...")

    mosaico = np.zeros((ny, nx), dtype=np.float32)
    ix_starts = list(range(0, nx, tam_tile))
    iy_starts = list(range(0, ny, tam_tile))
    n_tiles = len(ix_starts) * len(iy_starts)
    contador = 0

    for iy0 in iy_starts:
        iy1 = min(iy0 + tam_tile, ny)
        for ix0 in ix_starts:
            ix1 = min(ix0 + tam_tile, nx)
            contador += 1

            lon0 = xmin + ix0 / px_por_grado
            lon1 = xmin + ix1 / px_por_grado
            lat0 = ymin + iy0 / px_por_grado
            lat1 = ymin + iy1 / px_por_grado
            region = ee_module.Geometry.Rectangle([lon0, lat0, lon1, lat1])

            try:
                rect = imagen.sampleRectangle(region=region, defaultValue=0)
                tile = np.array(rect.get(cfg["banda"]).getInfo(), dtype=np.float32)
            except Exception as e:
                log.error("-" * 62)
                log.error(f"ERROR extrayendo el tile ({ix0},{iy0}): {e}")
                log.error("SOLUCION: revisa CONFIG['tam_tile_px'] (debe cumplir")
                log.error("tam_tile_px**2 <= 262144, el limite de sampleRectangle) y")
                log.error("que la conexion a internet no se haya interrumpido.")
                log.error("-" * 62)
                sys.exit(1)

            alto_esperado, ancho_esperado = iy1 - iy0, ix1 - ix0
            # sampleRectangle SIEMPRE incluye ambos bordes (N celdas pedidas
            # -> N+1 puntos de muestra) -- se descarta siempre el punto
            # extra final (fila y columna), tambien en el ultimo tile: ese
            # punto corresponde al borde derecho/inferior exacto del bbox,
            # que ya queda fuera de la grilla de nx*ny pixeles objetivo.
            tile = tile[:alto_esperado, :ancho_esperado]
            # BUG CORREGIDO 2026-09-13: sampleRectangle devuelve cada tile con
            # la fila 0 = borde NORTE (lat1), orden estandar norte-arriba de
            # una imagen (verificado empiricamente contra una consulta directa
            # a GEE sobre Bogota). El resto de este bucle asume que la fila 0
            # de cada tile es el borde SUR (lat0) -- sin este flip, el
            # contenido de cada tile de 4 grados queda espejado verticalmente
            # DENTRO de su propio bloque de latitud (el flip global de mas
            # abajo no lo corrige, solo voltea el mosaico ya mal ensamblado).
            # Sintoma detectado: el brillo real de Bogota (deberia saturar en
            # 63) aparecia en 0 en su coordenada exacta, y el valor 63
            # aparecia en su lugar en el punto reflejado dentro del mismo
            # tile (~6.35N en vez de 4.65N, para el tile de latitud
            # [3.5,7.5)) -- confirmado punto por punto contra una consulta
            # aislada a GEE antes de aplicar este fix.
            tile = tile[::-1, :]
            mosaico[iy0:iy1, ix0:ix1] = tile

            if contador % 5 == 0 or contador == n_tiles:
                log.info(f"  Tile {contador}/{n_tiles} completado")

    # DMSP-OLS se distribuye con origen en la esquina SUPERIOR izquierda
    # (norte-oeste) en la convencion raster estandar -- voltear el eje y
    # (el mosaico se construyo de sur a norte) antes de escribir.
    mosaico = np.flipud(mosaico)

    transform = from_origin(xmin, ymax, 1 / px_por_grado, 1 / px_por_grado)
    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype=mosaico.dtype, crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(mosaico, 1)

    log.info(f"Guardado: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    log.info(f"Rango de valores: min={mosaico.min():.1f}, max={mosaico.max():.1f}")


def main() -> None:
    ee = conectar_gee(CONFIG)
    imagen = obtener_imagen_dmsp(ee, CONFIG)
    extraer_raster_por_mosaico(ee, imagen, CONFIG)
    log.info("\nListo. Siguiente paso: correr eda_dmsp_fondo_mapa.py apuntando")
    log.info(f"RASTER_PATH a {CONFIG['output_path']} para regenerar el fondo.")


if __name__ == "__main__":
    main()
