"""
descargar_gadm.py
=================
Descarga los límites administrativos de Colombia de GADM 4.1 (niveles 0 y 1)
que usa el mapa del artículo (`graf_mapa_territorial_tramas.py`). La licencia
de GADM no permite redistribuir sus archivos, por eso no vienen en el paquete.

OUTPUT

    data/interim/geo_referencia/gadm41_COL_{0,1}.{shp,shx,dbf,prj,cpg}

CÓMO CORRER

    python src/descargar_gadm.py
"""
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_COL_shp.zip"
DESTINO = Path(__file__).resolve().parents[1] / "data" / "interim" / "geo_referencia"


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    with urlopen(URL) as respuesta:
        contenido = respuesta.read()
    with zipfile.ZipFile(io.BytesIO(contenido)) as zf:
        for nombre in zf.namelist():
            if nombre.startswith(("gadm41_COL_0.", "gadm41_COL_1.")):
                (DESTINO / nombre).write_bytes(zf.read(nombre))
                print(f"Guardado: {DESTINO / nombre}")


if __name__ == "__main__":
    main()
