"""
01_construir_coleccion_dmsp.py
================================

Construye y valida, para cada año calendario que necesita el pipeline, la
imagen anual DMSP-OLS correspondiente. Confirma que la extracción funciona
sobre datos reales de GEE antes de escalar a miles de hogares.

QUÉ HACE

    1. Se conecta a Earth Engine (utils/gee_utils.py).
    2. Lee la tabla de hogares (reutiliza coordenadas_hogares_s1.parquet
       del pipeline Sentinel-1 — ver utils/io_utils.py).
    3. Para cada ronda en CONFIG['ventanas_por_ronda']:
         a. Determina los 3 años de su ventana acumulada (ver "Estado vs.
            ventana acumulada" más abajo) — para 2010: 2008,2009,2010;
            para 2013: 2011,2012,2013.
         b. Para cada uno de esos años: cuenta cuántas imágenes lo
            componen en el catálogo (normalmente 1, a veces 2) y prueba
            la extracción real sobre el buffer del primer hogar de la
            ronda.
    4. Exporta un reporte de diagnóstico por ronda × año.

    Este script NO realiza:

    · extracción de valores por hogar a gran escala (responsabilidad de
      02_extraccion_buffer.py).
    · cálculo de estadísticos de estado ni de ventana acumulada
      (03_series_temporales.py).
    · preparación de coordenadas — reutiliza el output de
      sentinel1_pipeline/scripts/00_preparar_coordenadas.py.

ESTADO VS. VENTANA ACUMULADA — POR QUÉ CADA RONDA NECESITA 3 AÑOS, NO 1

    Este pipeline construye DOS tipos de variables para cada ronda (ver
    docs/prompts/prompt_dmsp_ols_pipeline.md, sección "Estado vs. cambios
    acumulados", y la explicación pedagógica completa en la tesis,
    Sección 3.3.7):

    1. VARIABLES DE ESTADO: el valor de luces nocturnas en el año exacto
       de la ola (el "corte transversal" — cómo se ve el entorno en ese
       momento). Un solo año: 2010 para la ronda 2010, 2013 para la ronda
       2013.

    2. VARIABLES DE VENTANA ACUMULADA: cómo cambió el entorno en los 3
       años que llevaron hasta esa ola (el "proceso" — si el entorno se
       venía iluminando/oscureciendo, y con qué velocidad). Requiere el
       año de la ola MÁS los dos años inmediatamente anteriores:
       - Ronda 2010 → ventana 2008, 2009, 2010 (año de la ola incluido).
       - Ronda 2013 → ventana 2011, 2012, 2013 (año de la ola incluido).

    El año de estado SIEMPRE es el último año de la ventana acumulada de
    su propia ronda — por eso 02_extraccion_buffer.py extrae los 3 años
    de la ventana de una sola vez, y 03_series_temporales.py deriva tanto
    el "estado" (la fila del último año) como los estadísticos de "ventana
    acumulada" (agregando las 3 filas) de esa misma extracción, sin
    volver a consultar GEE.

    POR QUÉ 3 AÑOS Y NO MÁS: con años calendario completos (no meses como
    en VIIRS), una ventana de 3 años ya captura una tendencia con 3 puntos
    (el mínimo para ajustar una pendiente con algo de robustez) sin
    alejarse demasiado del momento de la ola — una ventana más larga
    diluiría la interpretación de "proceso reciente, previo a la ola".

INPUTS

    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet
    credenciales/gee_service_account.json

OUTPUTS

    logs/reporte_coleccion_dmsp.txt → diagnóstico de la colección por ronda × año

TIEMPO ESTIMADO

    Menos de 2 minutos en total (6 años × 1 hogar de prueba cada uno).

CÓMO CORRER

    python 01_construir_coleccion_dmsp.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # .../dmsp_ols_pipeline/scripts
sys.path.insert(0, str(_HERE.parent))               # .../dmsp_ols_pipeline

# ── Librerías externas / internas ───────────────────────────────────────────────
from utils.gee_utils import verificar_archivo_credenciales, inicializar_gee  # noqa: E402
from utils.io_utils import cargar_hogares  # noqa: E402
from utils.dmsp_utils import (  # noqa: E402
    construir_imagen_anual,
    contar_imagenes_anio,
    construir_geometria_buffer,
    ESCALA_NATIVA_M,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Credenciales de GEE ────────────────────────────────────────────────────
    "gee_service_account": "mi-servidor-osm@plasma-minutia-492021-k7.iam.gserviceaccount.com",
    "gee_key_file": _HERE.parent / "credenciales" / "gee_service_account.json",
    "gee_project": "plasma-minutia-492021-k7",

    # ── Input: tabla de hogares (reutilizada del pipeline Sentinel-1) ────────
    "coordenadas_hogares": (
        _HERE.parent.parent
        / "sentinel1_pipeline" / "data" / "processed" / "coordenadas_hogares_s1.parquet"
    ),

    # ── Ventanas por ronda: año de estado + ventana acumulada de 3 años ──────
    # Ver "ESTADO VS. VENTANA ACUMULADA" en el docstring del módulo.
    # 'anios_acumulado' SIEMPRE incluye a 'anio_estado' como su último año.
    "ventanas_por_ronda": {
        2010: {"anio_estado": 2010, "anios_acumulado": [2008, 2009, 2010]},
        2013: {"anio_estado": 2013, "anios_acumulado": [2011, 2012, 2013]},
    },

    # ── Buffer espacial para la prueba de extracción (mismo valor que usará 02) ─
    "radio_buffer_m": ESCALA_NATIVA_M,
    "escala_reduccion_m": ESCALA_NATIVA_M,

    # ── Carpeta de salida ──────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: CONEXIÓN
# ──────────────────────────────────────────────────────────────────────────────

def conectar_gee(cfg: dict):
    """Verifica credenciales e inicializa GEE. Termina el proceso si falla."""
    ruta_key = Path(cfg["gee_key_file"])
    try:
        credenciales_json = verificar_archivo_credenciales(ruta_key)
        ee = inicializar_gee(
            credenciales_json["client_email"], ruta_key, project=cfg.get("gee_project")
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)
    return ee


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: REPORTE DE DIAGNÓSTICO
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(resultados: dict, cfg: dict) -> str:
    """Genera el texto del reporte de diagnóstico de la colección DMSP-OLS."""
    sep = "=" * 65
    lines = [
        sep,
        "REPORTE DE COLECCIÓN — DMSP-OLS (ESTADO + VENTANA ACUMULADA 2010/2013)",
        sep,
        "",
        "── 1. DISPONIBILIDAD DE IMAGEN POR RONDA × AÑO ───────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Cuántas imágenes componen cada año dentro de la ventana",
        "  acumulada de cada ronda (normalmente 1; 2 en años de transición",
        "  entre satélites, que este pipeline promedia). El año de estado",
        "  de cada ronda es el ÚLTIMO de su ventana (marcado [ESTADO]).",
        "",
    ]
    for ronda, info_ronda in sorted(cfg["ventanas_por_ronda"].items()):
        anio_estado = info_ronda["anio_estado"]
        lines.append(f"  Ronda {ronda}:")
        for anio in info_ronda["anios_acumulado"]:
            etiqueta = " [ESTADO]" if anio == anio_estado else " [acumulado]"
            r = resultados.get((ronda, anio), {})
            lines.append(f"    Año {anio}{etiqueta}: {r.get('n_imagenes_anio', 'N/D')} imagen(es)")
        lines.append("")

    lines += [
        "── 2. PRUEBA DE EXTRACCIÓN REAL POR RONDA × AÑO ─────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Confirma que la extracción por buffer funciona con datos",
        f"  reales: reduce (media) cada imagen anual sobre el buffer de",
        f"  {cfg['radio_buffer_m']:.0f} m de un hogar de muestra.",
        "",
    ]
    for (ronda, anio), r in sorted(resultados.items()):
        prueba = r.get("prueba_extraccion")
        lines.append(f"  Ronda {ronda}, año {anio} (hogar de muestra: id_ola={r.get('id_ola_muestra')})")
        if prueba is None:
            lines.append("    Sin imagen para ese año — prueba omitida.")
        else:
            lines.append(f"    avg_vis      : {prueba.get('avg_vis')}")
            lines.append(f"    stable_lights: {prueba.get('stable_lights')}")
            lines.append(f"    cf_cvg       : {prueba.get('cf_cvg')}")
        lines.append("")

    lines += [
        "  INTERPRETACIÓN",
        "  avg_vis y stable_lights son enteros de 6 bits en [0, 63]. Un",
        "  valor de 63 indica saturación del sensor, no necesariamente que",
        "  la actividad económica real sea igual en todos los píxeles",
        "  saturados — ver utils/dmsp_utils.py::marcar_saturacion, usada",
        "  en 03_series_temporales.py.",
        "",
        sep,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  01_construir_coleccion_dmsp.py")
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

    log.info("\n── Cargando tabla de hogares (reutilizada de Sentinel-1) ─")
    hogares = cargar_hogares(cfg["coordenadas_hogares"])
    hogares = hogares[hogares["coord_valida_ola"] == 1]  # solo coordenadas válidas

    resultados = {}
    for ronda, info_ronda in sorted(cfg["ventanas_por_ronda"].items()):
        log.info(f"\n── Ronda {ronda} ───────────────────────────────────────")

        if ronda not in hogares["ola"].unique():
            log.warning(f"  Ronda {ronda} no está presente en la tabla de hogares. Se omite.")
            continue

        hogares_ronda = hogares[hogares["ola"] == ronda]
        hogar_muestra = hogares_ronda.iloc[0]

        for anio in info_ronda["anios_acumulado"]:
            etiqueta = "ESTADO" if anio == info_ronda["anio_estado"] else "acumulado"
            log.info(f"  Año {anio} [{etiqueta}] — hogares: {len(hogares_ronda):,}")

            try:
                n_imagenes_anio = contar_imagenes_anio(ee, anio)
                imagen = construir_imagen_anual(ee, anio)
            except Exception as e:
                log.error(f"  ERROR al construir la imagen anual {anio} (ronda {ronda}): {e}")
                log.error("  CAUSAS PROBABLES: cuota de GEE agotada, o problema de red.")
                resultados[(ronda, anio)] = {"n_imagenes_anio": 0, "prueba_extraccion": None, "id_ola_muestra": hogar_muestra["id_ola"]}
                continue

            log.info(f"    Imágenes que componen el año {anio}: {n_imagenes_anio}")
            if n_imagenes_anio == 0:
                log.error(
                    f"    ERROR: no hay ninguna imagen DMSP-OLS para el año {anio}. "
                    "Revisa CONFIG['ventanas_por_ronda'] (rango real de cobertura: 1992-2014)."
                )
                resultados[(ronda, anio)] = {"n_imagenes_anio": 0, "prueba_extraccion": None, "id_ola_muestra": hogar_muestra["id_ola"]}
                continue

            try:
                geometria = construir_geometria_buffer(
                    ee, hogar_muestra["lat"], hogar_muestra["lon"], cfg["radio_buffer_m"]
                )
                prueba_extraccion = imagen.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometria,
                    scale=cfg["escala_reduccion_m"],
                ).getInfo()
                log.info(f"    Prueba de extracción: {prueba_extraccion}")
            except Exception as e:
                log.warning(f"    No se pudo probar extracción para {ronda}/{anio}: {e}")
                prueba_extraccion = None

            resultados[(ronda, anio)] = {
                "n_imagenes_anio": n_imagenes_anio,
                "prueba_extraccion": prueba_extraccion,
                "id_ola_muestra": hogar_muestra["id_ola"],
            }

    if not resultados:
        log.error("ERROR: ninguna ronda pudo procesarse. Revisa los mensajes anteriores.")
        sys.exit(1)

    log.info("\n── Exportando resultados ─────────────────────────────────")
    reporte = generar_reporte(resultados, cfg)
    txt_path = out_dir / "reporte_coleccion_dmsp.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
