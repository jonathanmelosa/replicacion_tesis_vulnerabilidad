"""
eda_dmsp_fondo_mapa.py
=======================
Precomputa el fondo de "brillo nocturno continuo" para el mapa territorial
final (mapa_territorial_transicion_divipola.py), a partir del raster DMSP-OLS
2010 completo generado por eda_dmsp_nacional_gee.py.

POR QUE UN ARCHIVO PRECOMPUTADO
--------------------------------------------------------------------
El script final se debe correr en la sala de computo de la universidad, sin
poder depurar nada ahi ("no puedo probar nada"). El procesamiento (recorte +
2 pasadas de blur gaussiano + escala log) toma varios segundos y depende de
rasterio/scipy funcionando correctamente -- una superficie de fallo
innecesaria si el mapa final solo necesita el resultado visual, no el
raster crudo. Por eso este paso se corre AHORA (con internet y tiempo para
verificar), y su salida (un array ya listo, liviano) es la que el script
final simplemente carga con `numpy.load`, sin tocar internet ni reprocesar
nada en la sala.

FUENTE DEL RASTER
    Version anterior de este script usaba el dataset alternativo de Zenodo
    ("CCNL 1992-2013", DOI 10.5281/zenodo.6644980) porque la descarga
    directa de NOAA/EOG exige login. Pero
    `dmsp_stable_lights` (la variable que usa el RESTO de la tesis) nunca
    se descargo de ahi -- se extrajo via Google Earth Engine, de acceso
    publico con cuenta de servicio. Ahora este script lee el raster
    generado por `eda_dmsp_nacional_gee.py` (MISMA fuente/coleccion/banda/
    satelite que `dmsp_stable_lights`, escala 0-63), eliminando la
    advertencia de "no intercambiable" que tenia la version con Zenodo
    (que llegaba a ~0-128, otro procesamiento del mismo sensor).

METODO
    1. Recortar el raster a la extension de Colombia (con margen).
    2. Dos pasadas de desenfoque gaussiano (sigma=1.2 "nucleo nitido" +
       sigma=4.5 "halo difuso", sumadas con peso 1.0/0.6) para simular el
       "bloom" fotografico de una foto satelital real de luces nocturnas.
    3. Escala log1p (revela ciudades medianas, no solo las 3-4 mas grandes,
       dado que la distribucion real de DMSP es muy asimetrica).

INPUTS
    data/interim/geo_referencia/dmsp_nacional_gee_2010.tif
        (generado por eda_dmsp_nacional_gee.py -- correr ese script PRIMERO,
        requiere credenciales de Google Earth Engine)

OUTPUTS
    data/interim/geo_referencia/fondo_dmsp_colombia_2010.npz
        arrays: "compuesto_log" (grid 2D), "extent" ([xmin, xmax, ymin, ymax])

COMO CORRER
    # 1. Generar el raster (requiere credenciales GEE, ver ese script):
    cd src/02_build && python eda_dmsp_nacional_gee.py
    # 2. Correr este script:
    python eda_dmsp_fondo_mapa.py
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RASTER_PATH = PROJECT_ROOT / "data" / "interim" / "geo_referencia" / "dmsp_nacional_gee_2010.tif"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "geo_referencia" / "fondo_dmsp_colombia_2010.npz"

# Extension de Colombia con margen -- identica a la usada en toda la serie
# de pruebas de estilo del mapa territorial (artifact 8ae3a4d2...).
XMIN, XMAX, YMIN, YMAX = -82, -66.5, -4.5, 13.8


def main() -> None:
    if not RASTER_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el raster en {RASTER_PATH}. "
            "Descargarlo primero (ver docstring del modulo, seccion COMO CORRER)."
        )

    with rasterio.open(RASTER_PATH) as src:
        assert str(src.crs) == "EPSG:4326", f"CRS inesperado: {src.crs} (se esperaba EPSG:4326)"
        window = from_bounds(XMIN, YMIN, XMAX, YMAX, transform=src.transform)
        banda = src.read(1, window=window)
        print(f"Ventana recortada: {banda.shape}, min={banda.min()}, max={banda.max()}")

    banda = banda.astype(float)
    banda[banda < 0] = 0  # nodata / valores negativos a 0

    banda_blur = gaussian_filter(banda, sigma=1.2)
    halo = gaussian_filter(banda, sigma=4.5)
    compuesto = banda_blur * 1.0 + halo * 0.6
    compuesto_log = np.log1p(compuesto).astype(np.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        compuesto_log=compuesto_log,
        extent=np.array([XMIN, XMAX, YMIN, YMAX]),
    )
    print(f"Guardado: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
