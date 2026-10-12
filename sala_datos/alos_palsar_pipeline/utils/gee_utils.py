"""
gee_utils.py
============
Funciones reutilizables para conectarse a Google Earth Engine (GEE) mediante
una cuenta de servicio, y para verificar que la conexión realmente funciona.

Estas funciones las usan 00_inicializar_gee.py (como prueba de humo) y todos
los scripts posteriores del pipeline (01 en adelante) que necesitan una
sesión de GEE ya inicializada.

Copia independiente de la versión usada en sentinel1_pipeline/,
sentinel2_pipeline/ y viirs_pipeline/ — los pipelines son autocontenidos a
propósito (ver decisión de arquitectura), así que este archivo NO se
importa entre pipelines. Si corriges un bug aquí, revisa si aplica también
en los demás utils/gee_utils.py.

No se importa `ee` a nivel de módulo: earthengine-api es una dependencia
pesada y con requisitos de red/autenticación que no todos los entornos
tienen disponibles (p. ej. para correr solo pruebas unitarias de otras
partes del pipeline). Se importa de forma perezosa dentro de cada función.
"""
# from __future__ import annotations: permite usar 'str | None' como type
# hint incluso en Python 3.9 (la sala de cómputo puede no tener 3.10+).
from __future__ import annotations

# ── Librería estándar ──────────────────────────────────────────────────────────
import json               # para leer y validar el archivo de credenciales
import logging             # logger de este módulo
import time                # para medir la duración de la prueba de conexión
from pathlib import Path   # manejo de rutas

log = logging.getLogger(__name__)

# Campos que debe tener un JSON de cuenta de servicio de Google Cloud válido.
# Si falta alguno, el archivo casi seguro no es una llave de cuenta de
# servicio (podría ser un archivo OAuth de usuario, que no sirve aquí).
_CAMPOS_REQUERIDOS_JSON = ["type", "client_email", "private_key", "project_id"]


def verificar_archivo_credenciales(ruta_json: Path) -> dict:
    """
    Verifica que el archivo de credenciales de la cuenta de servicio exista,
    sea JSON válido y tenga la estructura esperada de una llave de cuenta de
    servicio de Google Cloud (no un token OAuth de usuario ni un archivo
    corrupto).

    Retorna el contenido del JSON ya parseado (dict) si todo está bien.
    Termina el proceso con un mensaje diagnóstico si algo falla — sin
    credenciales válidas ningún script del pipeline puede continuar.
    """
    ruta_json = Path(ruta_json)

    if not ruta_json.exists():
        raise FileNotFoundError(
            "\n" + "-" * 62 + "\n"
            f"ERROR: no se encontró el archivo de credenciales de GEE.\n"
            f"  Se buscó en: {ruta_json.resolve()}\n"
            "  CÓMO SOLUCIONARLO:\n"
            "    1. Crea una cuenta de servicio en Google Cloud Console\n"
            "       (IAM & Admin > Service Accounts) con el rol\n"
            "       'Earth Engine Resource Writer' o similar.\n"
            "    2. Genera una llave JSON para esa cuenta de servicio.\n"
            "    3. Regístrala en https://code.earthengine.google.com/register\n"
            "       (Earth Engine requiere registrar la cuenta de servicio\n"
            "       antes de poder usarla).\n"
            f"    4. Copia el archivo .json descargado a:\n"
            f"         {ruta_json.resolve()}\n"
            "    5. Si le pusiste otro nombre, actualiza CONFIG['gee_key_file']\n"
            "       en 00_inicializar_gee.py.\n"
            + "-" * 62
        )

    try:
        contenido = json.loads(ruta_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            "\n" + "-" * 62 + "\n"
            f"ERROR: {ruta_json.name} no es un JSON válido ({e}).\n"
            "  CÓMO SOLUCIONARLO:\n"
            "    Vuelve a descargar la llave desde Google Cloud Console\n"
            "    (IAM & Admin > Service Accounts > tu cuenta > Keys > Add Key).\n"
            "    No edites el archivo manualmente: un JSON de credenciales\n"
            "    corrupto casi siempre se debe a una copia/pegado incompleto.\n"
            + "-" * 62
        ) from e

    faltantes = [c for c in _CAMPOS_REQUERIDOS_JSON if c not in contenido]
    if faltantes:
        raise ValueError(
            "\n" + "-" * 62 + "\n"
            f"ERROR: {ruta_json.name} no tiene la estructura de una llave de\n"
            f"  cuenta de servicio de Google Cloud. Faltan los campos: {faltantes}\n"
            "  CAUSA PROBABLE: descargaste un archivo de credenciales OAuth\n"
            "  de usuario (client_secret_*.json) en vez de una llave de\n"
            "  cuenta de servicio.\n"
            "  CÓMO SOLUCIONARLO:\n"
            "    En Google Cloud Console ve a IAM & Admin > Service Accounts,\n"
            "    entra a tu cuenta de servicio, pestaña 'Keys', y usa\n"
            "    'Add Key > Create new key > JSON'.\n"
            + "-" * 62
        )

    log.info(f"  Credenciales verificadas: {ruta_json.name} (cuenta: {contenido['client_email']})")
    return contenido


def inicializar_gee(client_email: str, ruta_key: Path, project: str | None = None):
    """
    Inicializa la sesión de Google Earth Engine usando una cuenta de
    servicio y retorna el módulo `ee` ya inicializado, listo para hacer
    consultas.

    Parámetros
    ----------
    client_email : el correo de la cuenta de servicio (client_email dentro
        del JSON de credenciales). Debe coincidir con el que aparece en el
        propio archivo de credenciales; se valida por separado en
        verificar_archivo_credenciales().
    ruta_key : ruta al archivo .json de la cuenta de servicio.
    project : ID del proyecto de Google Cloud asociado a Earth Engine.
        Necesario en cuentas de servicio creadas después de que GEE exigió
        proyecto de Cloud obligatorio (2022+). Si tu proyecto ya está
        embebido en el JSON de credenciales (project_id), puedes dejarlo
        en None.

    Lanza RuntimeError con diagnóstico si earthengine-api no está instalado
    o si la inicialización falla (credenciales no registradas en GEE,
    cuenta sin permisos, sin conexión a internet, etc.).
    """
    try:
        import ee  # import perezoso: earthengine-api es pesado y no siempre está instalado
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "\n" + "-" * 62 + "\n"
            "ERROR: el paquete 'earthengine-api' no está instalado.\n"
            "  CÓMO SOLUCIONARLO:\n"
            "    source .venv/bin/activate   (Mac/Linux)\n"
            "    .venv\\Scripts\\activate      (Windows)\n"
            "    pip install -r requirements.txt\n"
            "  o directamente:\n"
            "    pip install earthengine-api\n"
            + "-" * 62
        ) from e

    try:
        credenciales = ee.ServiceAccountCredentials(client_email, str(ruta_key))
        if project:
            ee.Initialize(credenciales, project=project)
        else:
            ee.Initialize(credenciales)
    except Exception as e:
        raise RuntimeError(
            "\n" + "-" * 62 + "\n"
            f"ERROR al inicializar Earth Engine con la cuenta '{client_email}': {e}\n"
            "  CAUSAS PROBABLES Y CÓMO SOLUCIONARLAS:\n"
            "    1. La cuenta de servicio no está registrada en Earth Engine.\n"
            "       Regístrala en https://code.earthengine.google.com/register\n"
            "       (elige 'Google Cloud project' y selecciona el proyecto\n"
            f"       dueño de la cuenta '{client_email}').\n"
            "    2. El proyecto de Cloud no tiene la API de Earth Engine\n"
            "       habilitada. En Google Cloud Console: APIs & Services >\n"
            "       Enable APIs > busca 'Earth Engine API' > Enable.\n"
            "    3. No hay conexión a internet en esta máquina — revisa la\n"
            "       red de la sala de cómputo (puede requerir proxy/VPN).\n"
            "    4. CONFIG['gee_project'] no coincide con el proyecto real\n"
            "       de la cuenta de servicio — revisa 'project_id' dentro\n"
            "       del propio archivo .json de credenciales.\n"
            + "-" * 62
        ) from e

    log.info(f"  Earth Engine inicializado correctamente con la cuenta: {client_email}")
    return ee


def probar_conexion(
    ee_module,
    coleccion: str,
    fecha_inicio: str,
    fecha_fin: str,
    lon: float,
    lat: float,
) -> dict:
    """
    Hace una consulta mínima a GEE para confirmar que la sesión realmente
    puede leer datos (no solo que ee.Initialize() no lanzó error — eso NO
    garantiza que las consultas después funcionen, p. ej. por falta de
    permisos sobre la colección específica).

    Cuenta cuántas imágenes de `coleccion` hay en una ventana de fechas y
    alrededor de un punto de prueba (por defecto, un punto dentro de
    Colombia). Retorna un diccionario con el conteo y el tiempo que tardó
    la consulta, útil para diagnosticar lentitud de red.

    Lanza RuntimeError con diagnóstico si la consulta falla.
    """
    t0 = time.time()
    try:
        punto = ee_module.Geometry.Point([lon, lat])
        coleccion_ee = (
            ee_module.ImageCollection(coleccion)
            .filterDate(fecha_inicio, fecha_fin)
            .filterBounds(punto)
        )
        n_imagenes = coleccion_ee.size().getInfo()  # única llamada que realmente viaja a los servidores de GEE
    except Exception as e:
        raise RuntimeError(
            "\n" + "-" * 62 + "\n"
            f"ERROR al consultar la colección '{coleccion}' en GEE: {e}\n"
            "  CAUSAS PROBABLES:\n"
            "    1. La cuenta de servicio no tiene permiso para leer esta\n"
            "       colección pública — esto es raro, pero revisa que el\n"
            "       nombre de la colección esté bien escrito.\n"
            "    2. Se agotó la cuota de peticiones de la cuenta — espera\n"
            "       unos minutos y reintenta.\n"
            "    3. Problema de red intermitente — reintenta el script.\n"
            + "-" * 62
        ) from e

    duracion = time.time() - t0
    return {"n_imagenes": n_imagenes, "duracion_segundos": round(duracion, 2)}


def verificar_bandas_coleccion(
    ee_module,
    coleccion: str,
    fecha_inicio: str,
    fecha_fin: str,
    bandas_esperadas: list[str],
) -> list[str]:
    """
    Verifica que la primera imagen disponible de `coleccion` en la ventana
    dada tenga las bandas listadas en `bandas_esperadas`.

    A diferencia de Sentinel-1/2 (bandas de radar/óptico bien documentadas
    y estables desde hace años), colecciones históricas de NOAA en GEE
    (VIIRS DNB, DMSP-OLS) han tenido variantes con nombres de banda
    ligeramente distintos entre versiones del producto. Por eso este
    pipeline NO asume los nombres de banda: los confirma en tiempo de
    ejecución contra una imagen real.

    Retorna la lista de bandas realmente presentes en la primera imagen de
    la colección filtrada. Lanza RuntimeError con diagnóstico si la
    colección está vacía en esa ventana (no se puede verificar nada) o si
    faltan bandas esperadas.
    """
    try:
        coleccion_ee = ee_module.ImageCollection(coleccion).filterDate(fecha_inicio, fecha_fin)
        primera = coleccion_ee.first()
        info = primera.getInfo()  # única llamada que viaja a los servidores de GEE
    except Exception as e:
        raise RuntimeError(
            "\n" + "-" * 62 + "\n"
            f"ERROR al leer la primera imagen de '{coleccion}' entre "
            f"{fecha_inicio} y {fecha_fin}: {e}\n"
            "  CAUSAS PROBABLES:\n"
            "    1. El nombre de la colección cambió en el catálogo de GEE.\n"
            "       Búscalo en https://developers.google.com/earth-engine/datasets\n"
            "    2. La ventana de fechas no tiene ninguna imagen disponible.\n"
            + "-" * 62
        ) from e

    if info is None:
        raise RuntimeError(
            "\n" + "-" * 62 + "\n"
            f"ERROR: la colección '{coleccion}' no tiene ninguna imagen entre "
            f"{fecha_inicio} y {fecha_fin}.\n"
            "  CÓMO SOLUCIONARLO:\n"
            "    Revisa que la ventana de fechas esté dentro del rango de\n"
            "    cobertura real de la colección (verificalo en el catálogo de\n"
            "    GEE). Si la ventana es correcta, revisa el nombre exacto de\n"
            "    la colección — puede haber cambiado de versión.\n"
            + "-" * 62
        )

    bandas_presentes = [b["id"] for b in info.get("bands", [])]
    faltantes = [b for b in bandas_esperadas if b not in bandas_presentes]
    if faltantes:
        raise RuntimeError(
            "\n" + "-" * 62 + "\n"
            f"ERROR: la colección '{coleccion}' no tiene las bandas esperadas: {faltantes}\n"
            f"  Bandas encontradas en la primera imagen: {bandas_presentes}\n"
            "  CÓMO SOLUCIONARLO:\n"
            "    Actualiza CONFIG['bandas_esperadas'] en 00_inicializar_gee.py\n"
            "    con los nombres reales listados arriba, y revisa que los\n"
            "    scripts 01 en adelante usen esos mismos nombres de banda.\n"
            + "-" * 62
        )

    log.info(f"  Bandas verificadas en '{coleccion}': {bandas_presentes}")
    return bandas_presentes
