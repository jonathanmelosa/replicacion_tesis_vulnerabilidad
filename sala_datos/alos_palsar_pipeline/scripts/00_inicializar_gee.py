"""
00_inicializar_gee.py
======================

Verifica que la conexión a Google Earth Engine (GEE) mediante cuenta de
servicio esté correctamente configurada, ANTES de correr cualquier script
de extracción ALOS PALSAR. Es una prueba de humo: si este script falla,
ningún script 01-05 puede funcionar.

QUÉ HACE

    1. Verifica que el archivo de credenciales exista y tenga la
       estructura correcta.
    2. Verifica que earthengine-api esté instalado.
    3. Inicializa la sesión de GEE.
    4. Hace una consulta mínima real al mosaico anual PALSAR/PALSAR-2 de
       JAXA (JAXA/ALOS/PALSAR/YEARLY/SAR) sobre un punto de prueba en
       Colombia, para confirmar lectura real de datos.
    5. Verifica que la primera imagen disponible tenga las bandas HH/HV.
    6. Guarda un archivo de estado (JSON).

    Este script NO realiza descarga ni procesamiento a gran escala, ni
    preparación de coordenadas (reutiliza sentinel1_pipeline/).

INPUTS

    credenciales/gee_service_account.json

OUTPUTS

    logs/estado_gee.json
    logs/reporte_conexion_gee.txt

CÓMO CORRER

    python 00_inicializar_gee.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

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
    "gee_service_account": "mi-servidor-osm@plasma-minutia-492021-k7.iam.gserviceaccount.com",
    "gee_key_file": _HERE.parent / "credenciales" / "gee_service_account.json",
    "gee_project": "plasma-minutia-492021-k7",

    # Mosaico anual PALSAR. Cobertura real confirmada 2007-2010 (ALOS-1),
    # hueco 2011-2014, 2015 en adelante (ALOS-2) — este pipeline solo usa
    # 2008-2010 (ver 01_construir_coleccion_alos.py).
    "prueba_conexion": {
        "coleccion": "JAXA/ALOS/PALSAR/YEARLY/SAR",
        "fecha_inicio": "2010-01-01",
        "fecha_fin": "2010-12-31",
        "lon": -74.1,
        "lat": 4.65,
    },

    "bandas_esperadas": ["HH", "HV"],
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
        correo_efectivo = credenciales_json["client_email"]
        ee = inicializar_gee(correo_efectivo, ruta_key, project=cfg.get("gee_project"))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)
    return ee


def generar_reporte(cfg: dict, resultado_prueba: dict, bandas_presentes: list) -> str:
    """Genera el texto del reporte de conexión a GEE."""
    sep = "=" * 65
    prueba = cfg["prueba_conexion"]
    return "\n".join([
        sep,
        "REPORTE DE CONEXIÓN — GOOGLE EARTH ENGINE (ALOS PALSAR)",
        sep,
        "",
        "── 1. ESTADO DE LA CONEXIÓN ───────────────────────────────",
        "",
        f"  Cuenta de servicio: {cfg['gee_service_account']}",
        f"  Archivo de credenciales: {Path(cfg['gee_key_file']).resolve()}",
        "  Estado: CONECTADO — OK",
        "",
        "── 2. PRUEBA DE LECTURA DE DATOS ─────────────────────────",
        "",
        f"  Colección consultada: {prueba['coleccion']}",
        f"  Ventana de fechas: {prueba['fecha_inicio']} a {prueba['fecha_fin']}",
        f"  Punto de prueba: lon={prueba['lon']}, lat={prueba['lat']}",
        f"  Imágenes encontradas: {resultado_prueba['n_imagenes']:,}",
        f"  Tiempo de respuesta: {resultado_prueba['duracion_segundos']} s",
        "",
        "  INTERPRETACIÓN",
        "  El mosaico global PALSAR es anual: para una ventana de un año",
        "  se espera al menos 1 imagen (tesela) cubriendo Bogotá. 0",
        "  imágenes con año dentro de 2007-2010 indicaría un problema de",
        "  permisos o de nombre de colección, no falta de cobertura real.",
        "",
        "── 3. VERIFICACIÓN DE BANDAS ─────────────────────────────",
        "",
        f"  Bandas esperadas: {cfg['bandas_esperadas']}",
        f"  Bandas encontradas: {bandas_presentes}",
        "  Estado: OK — todas las bandas esperadas están presentes.",
        "",
        "  INTERPRETACIÓN",
        "  HH y HV son número digital (DN) CRUDO, no gamma-0 en dB — la",
        "  calibración a dB ocurre en utils/alos_utils.py::calibrar_a_gamma0_db,",
        "  no en este script. Ver esa función para la fórmula y su fuente.",
        "",
        sep,
    ])


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

    log.info("\n── Probando lectura de datos ALOS PALSAR ─────────────────")
    prueba = cfg["prueba_conexion"]
    try:
        resultado_prueba = probar_conexion(
            ee, coleccion=prueba["coleccion"], fecha_inicio=prueba["fecha_inicio"],
            fecha_fin=prueba["fecha_fin"], lon=prueba["lon"], lat=prueba["lat"],
        )
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)
    log.info(f"  {resultado_prueba['n_imagenes']:,} imágenes encontradas en {resultado_prueba['duracion_segundos']} s.")

    log.info("\n── Verificando bandas de la colección ────────────────────")
    try:
        bandas_presentes = verificar_bandas_coleccion(
            ee, coleccion=prueba["coleccion"], fecha_inicio=prueba["fecha_inicio"],
            fecha_fin=prueba["fecha_fin"], bandas_esperadas=cfg["bandas_esperadas"],
        )
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("\n── Exportando resultados ─────────────────────────────────")
    estado = {
        "conectado": True, "cuenta_servicio": cfg["gee_service_account"],
        "ultima_verificacion_utc": datetime.now(timezone.utc).isoformat(),
        "prueba_conexion": resultado_prueba, "bandas_presentes": bandas_presentes,
    }
    (out_dir / "estado_gee.json").write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")

    reporte = generar_reporte(cfg, resultado_prueba, bandas_presentes)
    (out_dir / "reporte_conexion_gee.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
