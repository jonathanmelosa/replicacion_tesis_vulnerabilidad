"""
00_inicializar_gee.py
======================

Verifica que la conexión a Google Earth Engine (GEE) mediante cuenta de
servicio esté correctamente configurada, ANTES de correr cualquier script
de extracción DMSP-OLS. Es una prueba de humo: si este script falla,
ningún script 01-05 puede funcionar.

QUÉ HACE

    1. Verifica que el archivo de credenciales (JSON de cuenta de
       servicio) exista y tenga la estructura correcta.
    2. Verifica que earthengine-api esté instalado.
    3. Inicializa la sesión de GEE con esas credenciales.
    4. Hace una consulta mínima real a la colección DMSP-OLS Nighttime
       Lights (NOAA/DMSP-OLS/NIGHTTIME_LIGHTS) sobre un punto de prueba en
       Colombia, para confirmar que la sesión no solo se "inicializó" sino
       que puede leer datos.
    5. Verifica que la primera imagen disponible tenga las bandas que el
       resto del pipeline espera ('avg_vis', 'stable_lights', 'cf_cvg').
    6. Guarda un archivo de estado (JSON) que confirma la fecha y hora de
       la última verificación exitosa.

    Este script NO realiza:

    · descarga ni procesamiento de imágenes DMSP-OLS (responsabilidad de
      01_construir_coleccion_dmsp.py en adelante).
    · la lógica de conexión en sí, que vive en utils/gee_utils.py.
    · preparación de coordenadas de hogares — este pipeline reutiliza
      directamente el output de sentinel1_pipeline/ (ver
      utils/io_utils.py), no tiene su propio script de coordenadas.

INPUTS

    credenciales/gee_service_account.json (cuenta de servicio de Google Cloud)

OUTPUTS

    logs/estado_gee.json          → última verificación exitosa (para diagnóstico)
    logs/reporte_conexion_gee.txt → reporte legible de esta verificación

TIEMPO ESTIMADO

    Menos de 10 segundos.

CÓMO CORRER

    python 00_inicializar_gee.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../dmsp_ols_pipeline/scripts
sys.path.insert(0, str(_HERE.parent))             # .../dmsp_ols_pipeline

# ── Librerías externas / internas ───────────────────────────────────────────────
from utils.gee_utils import (  # noqa: E402
    verificar_archivo_credenciales,
    inicializar_gee,
    probar_conexion,
    verificar_bandas_coleccion,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Credenciales de la cuenta de servicio de Google Cloud / Earth Engine ──
    # Misma cuenta de servicio que sentinel1_pipeline/, sentinel2_pipeline/
    # y viirs_pipeline/ (un solo proyecto de GEE puede leer cualquier
    # colección pública, incluida DMSP-OLS de NOAA).
    "gee_service_account": "mi-servidor-osm@plasma-minutia-492021-k7.iam.gserviceaccount.com",
    "gee_key_file": _HERE.parent / "credenciales" / "gee_service_account.json",
    "gee_project": "plasma-minutia-492021-k7",

    # ── Prueba de conectividad ────────────────────────────────────────────────
    # Colección anual DMSP-OLS Nighttime Lights. Cobertura real verificada
    # contra el catálogo de Earth Engine: 1992-2014 (composites anuales).
    # Un año dentro de ese rango siempre debe existir. Punto de prueba:
    # Bogotá (4.65, -74.1), igual que en sentinel1/sentinel2/viirs.
    "prueba_conexion": {
        "coleccion": "NOAA/DMSP-OLS/NIGHTTIME_LIGHTS",
        "fecha_inicio": "2010-01-01",
        "fecha_fin": "2010-12-31",
        "lon": -74.1,
        "lat": 4.65,
    },

    # ── Bandas que el resto del pipeline necesita ──────────────────────────────
    # 'avg_vis'       : promedio crudo de la banda visible, sin filtrar
    #                   eventos efímeros (incendios, quemas).
    # 'stable_lights' : versión filtrada (variable de contenido económico
    #                   recomendada).
    # 'cf_cvg'        : cobertura de observaciones libres de nubes —
    #                   variable de calidad, no económica.
    "bandas_esperadas": ["avg_vis", "stable_lights", "cf_cvg"],

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
# MÓDULO 1: VERIFICACIÓN Y CONEXIÓN
# ──────────────────────────────────────────────────────────────────────────────

def conectar_gee(cfg: dict):
    """Orquesta la verificación de credenciales y la inicialización de GEE. Termina el proceso si falla."""
    ruta_key = Path(cfg["gee_key_file"])

    try:
        credenciales_json = verificar_archivo_credenciales(ruta_key)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)

    correo_config = cfg["gee_service_account"]
    correo_json = credenciales_json["client_email"]
    if correo_config != correo_json:
        log.warning(
            f"  CONFIG['gee_service_account'] ('{correo_config}') no coincide con "
            f"'client_email' dentro del JSON ('{correo_json}')."
        )
        log.warning("  Se usará el correo del JSON, que es la fuente de verdad.")
        correo_efectivo = correo_json
    else:
        correo_efectivo = correo_config

    try:
        ee = inicializar_gee(correo_efectivo, ruta_key, project=cfg.get("gee_project"))
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    return ee


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: REPORTE DE DIAGNÓSTICO
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(cfg: dict, resultado_prueba: dict, bandas_presentes: list) -> str:
    """Genera el texto del reporte de conexión a GEE."""
    sep = "=" * 65
    prueba = cfg["prueba_conexion"]
    lines = [
        sep,
        "REPORTE DE CONEXIÓN — GOOGLE EARTH ENGINE (DMSP-OLS)",
        sep,
        "",
        "── 1. ESTADO DE LA CONEXIÓN ───────────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Confirma que las credenciales de la cuenta de servicio son",
        "  válidas y que Earth Engine aceptó la inicialización de sesión.",
        "",
        f"  Cuenta de servicio: {cfg['gee_service_account']}",
        f"  Archivo de credenciales: {Path(cfg['gee_key_file']).resolve()}",
        "  Estado: CONECTADO — OK",
        "",
        "── 2. PRUEBA DE LECTURA DE DATOS ─────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Una inicialización exitosa (sección 1) NO garantiza que se",
        "  puedan leer datos. Esta sección hace una consulta real a",
        f"  {prueba['coleccion']} para confirmarlo.",
        "",
        f"  Colección consultada: {prueba['coleccion']}",
        f"  Ventana de fechas: {prueba['fecha_inicio']} a {prueba['fecha_fin']}",
        f"  Punto de prueba: lon={prueba['lon']}, lat={prueba['lat']}",
        f"  Imágenes encontradas: {resultado_prueba['n_imagenes']:,}",
        f"  Tiempo de respuesta: {resultado_prueba['duracion_segundos']} s",
        "",
        "  INTERPRETACIÓN",
        "  DMSP-OLS Nighttime Lights cubre 1992-2014 en composites",
        "  anuales — para un año dentro de ese rango se espera 1 imagen",
        "  (a veces 2, en años de transición entre satélites; ver",
        "  utils/dmsp_utils.py). Un valor de 0 con año y punto correctos",
        "  indicaría un problema de permisos o de nombre de colección.",
        "",
        "── 3. VERIFICACIÓN DE BANDAS ─────────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Confirma que la imagen tiene las bandas que el resto del",
        "  pipeline va a usar.",
        "",
        f"  Bandas esperadas: {cfg['bandas_esperadas']}",
        f"  Bandas encontradas: {bandas_presentes}",
        "  Estado: OK — todas las bandas esperadas están presentes.",
        "",
        "  INTERPRETACIÓN",
        "  'stable_lights' es la variable de radiancia recomendada",
        "  (eventos efímeros ya filtrados por NOAA). 'avg_vis' es la",
        "  versión cruda, sin filtrar. 'cf_cvg' es una variable de",
        "  calidad del dato (cobertura libre de nubes), no debe",
        "  interpretarse como actividad económica.",
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
    log.info("  00_inicializar_gee.py")
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

    log.info("\n── Verificando credenciales y conectando a GEE ───────────")
    ee = conectar_gee(cfg)

    log.info("\n── Probando lectura de datos DMSP-OLS ────────────────────")
    prueba = cfg["prueba_conexion"]
    try:
        resultado_prueba = probar_conexion(
            ee,
            coleccion=prueba["coleccion"],
            fecha_inicio=prueba["fecha_inicio"],
            fecha_fin=prueba["fecha_fin"],
            lon=prueba["lon"],
            lat=prueba["lat"],
        )
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    log.info(
        f"  {resultado_prueba['n_imagenes']:,} imágenes encontradas en "
        f"{resultado_prueba['duracion_segundos']} s — conexión verificada."
    )

    log.info("\n── Verificando bandas de la colección ────────────────────")
    try:
        bandas_presentes = verificar_bandas_coleccion(
            ee,
            coleccion=prueba["coleccion"],
            fecha_inicio=prueba["fecha_inicio"],
            fecha_fin=prueba["fecha_fin"],
            bandas_esperadas=cfg["bandas_esperadas"],
        )
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("\n── Exportando resultados ─────────────────────────────────")
    estado = {
        "conectado": True,
        "cuenta_servicio": cfg["gee_service_account"],
        "ultima_verificacion_utc": datetime.now(timezone.utc).isoformat(),
        "prueba_conexion": resultado_prueba,
        "bandas_presentes": bandas_presentes,
    }
    estado_path = out_dir / "estado_gee.json"
    estado_path.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"  Estado exportado: {estado_path}")

    reporte = generar_reporte(cfg, resultado_prueba, bandas_presentes)
    txt_path = out_dir / "reporte_conexion_gee.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
