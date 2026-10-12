"""
00_inicializar_gee.py
======================

Verifica que la conexión a Google Earth Engine esté correctamente
configurada, ANTES de correr cualquier script de extracción Landsat 5.

QUÉ HACE

    1-3. Verifica credenciales, earthengine-api, inicializa sesión (igual
    que los demás pipelines nuevos).
    4. Consulta real a LANDSAT/LT05/C02/T1_L2 sobre un punto de prueba en
       Colombia para confirmar lectura de datos.
    5. Verifica que la primera escena tenga las bandas de reflectancia y
       la banda de calidad QA_PIXEL.
    6. Guarda estado.

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
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "gee_service_account": "mi-servidor-osm@plasma-minutia-492021-k7.iam.gserviceaccount.com",
    "gee_key_file": _HERE.parent / "credenciales" / "gee_service_account.json",
    "gee_project": "plasma-minutia-492021-k7",

    "prueba_conexion": {
        "coleccion": "LANDSAT/LT05/C02/T1_L2",
        "fecha_inicio": "2010-01-01",
        "fecha_fin": "2010-12-31",
        "lon": -74.1,
        "lat": 4.65,
    },

    "bandas_esperadas": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"],
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


def generar_reporte(cfg: dict, resultado_prueba: dict, bandas_presentes: list) -> str:
    """Genera el texto del reporte de conexión a GEE."""
    sep = "=" * 65
    prueba = cfg["prueba_conexion"]
    return "\n".join([
        sep, "REPORTE DE CONEXIÓN — GOOGLE EARTH ENGINE (LANDSAT 5 TM)", sep, "",
        "── 1. ESTADO DE LA CONEXIÓN ───────────────────────────────", "",
        f"  Cuenta de servicio: {cfg['gee_service_account']}",
        "  Estado: CONECTADO — OK", "",
        "── 2. PRUEBA DE LECTURA DE DATOS ─────────────────────────", "",
        f"  Colección consultada: {prueba['coleccion']}",
        f"  Ventana de fechas: {prueba['fecha_inicio']} a {prueba['fecha_fin']}",
        f"  Imágenes (escenas) encontradas: {resultado_prueba['n_imagenes']:,}",
        f"  Tiempo de respuesta: {resultado_prueba['duracion_segundos']} s", "",
        "  INTERPRETACIÓN",
        "  Landsat 5 revisita cada 16 días — para una ventana de un año",
        "  se esperan del orden de 15-23 escenas SOBRE UN PUNTO (antes de",
        "  filtrar por nubes), no 1 como en DMSP-OLS/ALOS PALSAR (esos son",
        "  compuestos anuales pre-hechos; Landsat 5 no lo es).", "",
        "── 3. VERIFICACIÓN DE BANDAS ─────────────────────────────", "",
        f"  Bandas esperadas: {cfg['bandas_esperadas']}",
        f"  Bandas encontradas: {bandas_presentes}",
        "  Estado: OK — todas las bandas esperadas están presentes.", "",
        sep,
    ])


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  00_inicializar_gee.py")
    log.info("  Si ves ModuleNotFoundError, activa el entorno virtual e instala requirements.txt")
    log.info("=" * 62)

    cfg = CONFIG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("\n── Conectando a Earth Engine ──────────────────────────────")
    ee = conectar_gee(cfg)

    log.info("\n── Probando lectura de datos Landsat 5 ───────────────────")
    prueba = cfg["prueba_conexion"]
    try:
        resultado_prueba = probar_conexion(
            ee, coleccion=prueba["coleccion"], fecha_inicio=prueba["fecha_inicio"],
            fecha_fin=prueba["fecha_fin"], lon=prueba["lon"], lat=prueba["lat"],
        )
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)
    log.info(f"  {resultado_prueba['n_imagenes']:,} escenas encontradas en {resultado_prueba['duracion_segundos']} s.")

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
