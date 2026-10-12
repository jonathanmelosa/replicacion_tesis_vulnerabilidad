"""
01_construir_coleccion_landsat.py
===================================

Construye y valida, para cada uno de los 6 años que necesita el pipeline
(2008-2010 para la ronda 2010, 2011-2013 para la ronda 2013), el
compuesto anual Landsat 5 TM (mediana de escenas del año, enmascaradas
por nubes, con índices derivados). Confirma que funciona sobre datos
reales antes de escalar a miles de hogares.

QUÉ HACE

    1. Se conecta a Earth Engine.
    2. Lee la tabla de hogares (reutiliza coordenadas_hogares_s1.parquet).
    3. Para cada ronda y cada uno de sus 3 años: cuenta escenas
       disponibles y prueba la extracción real (máscara de nubes +
       escalado + índices) sobre el buffer de un hogar de muestra.
    4. Exporta un reporte de diagnóstico.

    Este script NO realiza extracción a gran escala (02), separación en
    estado/ventana acumulada (03), ni preparación de coordenadas.

VENTANAS: ESTADO Y VENTANA ACUMULADA (igual que dmsp_ols_pipeline)

    Ronda 2010 → años 2008, 2009, 2010 (estado = 2010, el último).
    Ronda 2013 → años 2011, 2012, 2013 (estado = 2013, el último).

    A diferencia de DMSP-OLS/ALOS PALSAR (compuestos anuales pre-hechos
    por el proveedor), aquí CADA año requiere construir un compuesto
    propio (mediana de escenas de ese año, enmascaradas por nubes) — ver
    utils/landsat5_utils.py::construir_imagen_anual.

INPUTS

    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet
    credenciales/gee_service_account.json

OUTPUTS

    logs/reporte_coleccion_landsat.txt

CÓMO CORRER

    python 01_construir_coleccion_landsat.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from utils.gee_utils import verificar_archivo_credenciales, inicializar_gee  # noqa: E402
from utils.io_utils import cargar_hogares  # noqa: E402
from utils.landsat5_utils import (  # noqa: E402
    construir_imagen_anual,
    contar_escenas_anio,
    construir_geometria_buffer,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "gee_service_account": "mi-servidor-osm@plasma-minutia-492021-k7.iam.gserviceaccount.com",
    "gee_key_file": _HERE.parent / "credenciales" / "gee_service_account.json",
    "gee_project": "plasma-minutia-492021-k7",

    "coordenadas_hogares": (
        _HERE.parent.parent
        / "sentinel1_pipeline" / "data" / "processed" / "coordenadas_hogares_s1.parquet"
    ),

    "ventanas_por_ronda": {
        2010: {"anio_estado": 2010, "anios_acumulado": [2008, 2009, 2010]},
        2013: {"anio_estado": 2013, "anios_acumulado": [2011, 2012, 2013]},
    },

    # Buffer de 250 m — mismo criterio que S1/S2 (error de georreferenciación
    # de la encuesta), no la resolución nativa de 30 m de Landsat TM.
    "radio_buffer_m": 250,
    "escala_reduccion_m": 30,

    "output_dir": _HERE.parent / "logs",
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


def conectar_gee(cfg: dict):
    """Verifica credenciales e inicializa GEE. Termina el proceso si falla."""
    ruta_key = Path(cfg["gee_key_file"])
    try:
        credenciales_json = verificar_archivo_credenciales(ruta_key)
        ee = inicializar_gee(credenciales_json["client_email"], ruta_key, project=cfg.get("gee_project"))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)
    return ee


def generar_reporte(resultados: dict, cfg: dict) -> str:
    """Genera el texto del reporte de diagnóstico de la colección Landsat 5."""
    sep = "=" * 65
    lines = [sep, "REPORTE DE COLECCIÓN — LANDSAT 5 TM (ESTADO + VENTANA ACUMULADA)", sep, "",
             "── 1. ESCENAS DISPONIBLES POR RONDA × AÑO ────────────────", ""]
    for ronda, info_ronda in sorted(cfg["ventanas_por_ronda"].items()):
        lines.append(f"  Ronda {ronda}:")
        for anio in info_ronda["anios_acumulado"]:
            etiqueta = " [ESTADO]" if anio == info_ronda["anio_estado"] else " [acumulado]"
            r = resultados.get((ronda, anio), {})
            lines.append(f"    Año {anio}{etiqueta}: {r.get('n_escenas', 'N/D')} escena(s) en el catálogo (antes de filtrar por nubes/geometría)")
        lines.append("")

    lines += ["── 2. PRUEBA DE EXTRACCIÓN REAL POR RONDA × AÑO ─────────", ""]
    for (ronda, anio), r in sorted(resultados.items()):
        prueba = r.get("prueba_extraccion")
        lines.append(f"  Ronda {ronda}, año {anio} (hogar de muestra: id_ola={r.get('id_ola_muestra')})")
        if prueba is None:
            lines.append("    Sin datos válidos para ese año/hogar — prueba omitida.")
        else:
            for k, v in prueba.items():
                lines.append(f"    {k}: {v}")
        lines.append("")

    lines += [
        "  INTERPRETACIÓN",
        "  Bandas de reflectancia (SR_B*) deben estar entre 0 y 1",
        "  aproximadamente. ndvi/evi/savi/ndbi/ndwi/mndwi/bsi deben estar",
        "  mayormente entre -1 y 1. Un valor None indica que la mediana",
        "  anual no tuvo ningún píxel válido (todas las escenas nubladas",
        "  en ese punto todo el año) — revisable en la ventana acumulada",
        "  del hogar en 03_series_temporales.py (n_anios < 3).",
        "", sep,
    ]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  01_construir_coleccion_landsat.py")
    log.info("=" * 62)

    cfg = CONFIG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("\n── Conectando a Earth Engine ──────────────────────────────")
    ee = conectar_gee(cfg)

    log.info("\n── Cargando tabla de hogares ──────────────────────────────")
    hogares = cargar_hogares(cfg["coordenadas_hogares"])
    hogares = hogares[hogares["coord_valida_ola"] == 1]

    resultados = {}
    for ronda, info_ronda in sorted(cfg["ventanas_por_ronda"].items()):
        log.info(f"\n── Ronda {ronda} ───────────────────────────────────────")
        if ronda not in hogares["ola"].unique():
            log.warning(f"  Ronda {ronda} no está en la tabla de hogares. Se omite.")
            continue
        hogares_ronda = hogares[hogares["ola"] == ronda]
        hogar_muestra = hogares_ronda.iloc[0]

        for anio in info_ronda["anios_acumulado"]:
            etiqueta = "ESTADO" if anio == info_ronda["anio_estado"] else "acumulado"
            log.info(f"  Año {anio} [{etiqueta}]")
            try:
                n_escenas = contar_escenas_anio(ee, anio)
                imagen = construir_imagen_anual(ee, anio)
            except Exception as e:
                log.error(f"    ERROR al construir el compuesto anual {anio}: {e}")
                resultados[(ronda, anio)] = {"n_escenas": 0, "prueba_extraccion": None, "id_ola_muestra": hogar_muestra["id_ola"]}
                continue
            log.info(f"    Escenas en el catálogo (año completo): {n_escenas}")

            try:
                geometria = construir_geometria_buffer(ee, hogar_muestra["lat"], hogar_muestra["lon"], cfg["radio_buffer_m"])
                prueba_extraccion = imagen.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=geometria, scale=cfg["escala_reduccion_m"],
                ).getInfo()
                log.info(f"    Prueba de extracción: {prueba_extraccion}")
            except Exception as e:
                log.warning(f"    No se pudo probar extracción para {ronda}/{anio}: {e}")
                prueba_extraccion = None

            resultados[(ronda, anio)] = {"n_escenas": n_escenas, "prueba_extraccion": prueba_extraccion, "id_ola_muestra": hogar_muestra["id_ola"]}

    log.info("\n── Exportando resultados ─────────────────────────────────")
    reporte = generar_reporte(resultados, cfg)
    (out_dir / "reporte_coleccion_landsat.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
