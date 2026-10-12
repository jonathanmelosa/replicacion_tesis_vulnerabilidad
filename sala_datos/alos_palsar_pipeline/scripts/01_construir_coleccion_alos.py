"""
01_construir_coleccion_alos.py
================================

Construye y valida las imágenes anuales ALOS PALSAR necesarias para la
ventana acumulada de la ola 2010 (2008, 2009, 2010 — ver "Por qué
únicamente la ola 2010" abajo). Confirma que la calibración a gamma-0 dB
y las bandas derivadas funcionan sobre datos reales antes de escalar a
miles de hogares.

QUÉ HACE

    1. Se conecta a Earth Engine.
    2. Lee la tabla de hogares (reutiliza coordenadas_hogares_s1.parquet).
    3. Para cada uno de los 3 años (2008, 2009, 2010): cuenta cuántas
       imágenes lo componen y prueba la extracción real (calibración +
       bandas derivadas) sobre el buffer del primer hogar de la ola 2010.
    4. Exporta un reporte de diagnóstico por año.

    Este script NO realiza extracción a gran escala (02), cómputo de
    estadísticos (03), ni preparación de coordenadas.

POR QUÉ ÚNICAMENTE LA OLA 2010

    El mosaico ALOS PALSAR/PALSAR-2 de JAXA tiene un hueco confirmado
    2011-2014 (ALOS-1 dejó de operar; ALOS-2 no empieza hasta 2015) — no
    cubre 2013 en absoluto, así que este pipeline no puede construir la
    ventana acumulada de la ola 2013 (necesitaría 2011, 2012, 2013).
    Se documenta como limitación explícita en la tesis, Sección 3.3.7,
    en vez de generar una ventana incompleta o silenciosamente rellenada.

INPUTS

    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet
    credenciales/gee_service_account.json

OUTPUTS

    logs/reporte_coleccion_alos.txt

CÓMO CORRER

    python 01_construir_coleccion_alos.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from utils.gee_utils import verificar_archivo_credenciales, inicializar_gee  # noqa: E402
from utils.io_utils import cargar_hogares  # noqa: E402
from utils.alos_utils import (  # noqa: E402
    construir_imagen_anual,
    contar_imagenes_anio,
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

    # Solo la ronda 2010 (ver "Por qué únicamente la ola 2010" arriba).
    "ronda_objetivo": 2010,
    "anio_estado": 2010,
    "anios_acumulado": [2008, 2009, 2010],

    # Buffer de 250 m — mismo criterio que Sentinel-1 (dominio físico
    # comparable, error de georreferenciación de la encuesta), no la
    # resolución nativa de 25 m de ALOS PALSAR. Ver ESCALA_NATIVA_M en
    # utils/alos_utils.py.
    "radio_buffer_m": 250,
    "escala_reduccion_m": 25,

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
    """Genera el texto del reporte de diagnóstico de la colección ALOS PALSAR."""
    sep = "=" * 65
    lines = [
        sep,
        "REPORTE DE COLECCIÓN — ALOS PALSAR (SOLO OLA 2010, VENTANA 2008-2010)",
        sep,
        "",
        "── 1. DISPONIBILIDAD DE IMAGEN POR AÑO ───────────────────",
        "",
    ]
    for anio in cfg["anios_acumulado"]:
        etiqueta = " [ESTADO]" if anio == cfg["anio_estado"] else " [acumulado]"
        r = resultados.get(anio, {})
        lines.append(f"  Año {anio}{etiqueta}: {r.get('n_imagenes_anio', 'N/D')} tesela(s)")

    lines += [
        "",
        "── 2. PRUEBA DE EXTRACCIÓN REAL (CALIBRACIÓN + BANDAS DERIVADAS) ─",
        "",
    ]
    for anio, r in sorted(resultados.items()):
        prueba = r.get("prueba_extraccion")
        lines.append(f"  Año {anio} (hogar de muestra: id_ola={r.get('id_ola_muestra')})")
        if prueba is None:
            lines.append("    Sin imagen para ese año — prueba omitida.")
        else:
            for k, v in prueba.items():
                lines.append(f"    {k}: {v}")
        lines.append("")

    lines += [
        "  INTERPRETACIÓN",
        "  HH_db/HV_db deben tener valores negativos del orden de -5 a -25",
        "  dB (retrodispersión típica sobre superficie terrestre en banda",
        "  L). indice_normalizado debe estar en (-1,1). Un valor None en",
        "  cualquier banda _db indica que DN=0 en ese punto (fuera de la",
        "  franja de cobertura del mosaico para ese año).",
        "",
        sep,
    ]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  01_construir_coleccion_alos.py")
    log.info("  Si ves ModuleNotFoundError, el entorno virtual no esta")
    log.info("  activado o las dependencias no estan instaladas.")
    log.info("  Solucion:")
    log.info("    Mac/Linux : source .venv/bin/activate")
    log.info("    Windows   : .venv\\Scripts\\activate")
    log.info("    Luego     : pip install -r requirements.txt")
    log.info("=" * 62)

    cfg = CONFIG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("\n── Conectando a Earth Engine ──────────────────────────────")
    ee = conectar_gee(cfg)

    log.info("\n── Cargando tabla de hogares ──────────────────────────────")
    hogares = cargar_hogares(cfg["coordenadas_hogares"])
    hogares = hogares[hogares["coord_valida_ola"] == 1]

    if cfg["ronda_objetivo"] not in hogares["ola"].unique():
        log.error(f"ERROR: la ronda {cfg['ronda_objetivo']} no está en la tabla de hogares.")
        sys.exit(1)
    hogares_ronda = hogares[hogares["ola"] == cfg["ronda_objetivo"]]
    hogar_muestra = hogares_ronda.iloc[0]
    log.info(f"  Hogares en la ronda {cfg['ronda_objetivo']}: {len(hogares_ronda):,}")

    resultados = {}
    for anio in cfg["anios_acumulado"]:
        etiqueta = "ESTADO" if anio == cfg["anio_estado"] else "acumulado"
        log.info(f"\n── Año {anio} [{etiqueta}] ─────────────────────────────")
        try:
            n_imagenes_anio = contar_imagenes_anio(ee, anio)
            imagen = construir_imagen_anual(ee, anio)
        except Exception as e:
            log.error(f"  ERROR al construir la imagen anual {anio}: {e}")
            resultados[anio] = {"n_imagenes_anio": 0, "prueba_extraccion": None, "id_ola_muestra": hogar_muestra["id_ola"]}
            continue

        log.info(f"  Teselas que componen el año {anio}: {n_imagenes_anio}")
        if n_imagenes_anio == 0:
            log.error(f"  ERROR: no hay ninguna imagen ALOS PALSAR para el año {anio} sobre este territorio.")
            resultados[anio] = {"n_imagenes_anio": 0, "prueba_extraccion": None, "id_ola_muestra": hogar_muestra["id_ola"]}
            continue

        try:
            geometria = construir_geometria_buffer(ee, hogar_muestra["lat"], hogar_muestra["lon"], cfg["radio_buffer_m"])
            prueba_extraccion = imagen.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geometria, scale=cfg["escala_reduccion_m"],
            ).getInfo()
            log.info(f"  Prueba de extracción: {prueba_extraccion}")
        except Exception as e:
            log.warning(f"  No se pudo probar extracción para {anio}: {e}")
            prueba_extraccion = None

        resultados[anio] = {"n_imagenes_anio": n_imagenes_anio, "prueba_extraccion": prueba_extraccion, "id_ola_muestra": hogar_muestra["id_ola"]}

    log.info("\n── Exportando resultados ─────────────────────────────────")
    reporte = generar_reporte(resultados, cfg)
    (out_dir / "reporte_coleccion_alos.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
