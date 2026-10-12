"""
02_descarga_fotos_GSV.py
========================
Descarga las imágenes de Google Street View correspondientes a los panoramas
identificados por 01_analisis_cobertura_gsv.py.

RESPONSABILIDAD EXCLUSIVA
    Transformar el inventario de panoramas (hogar × ola × radio) en un conjunto
    organizado, reproducible y completamente trazable de imágenes descargadas.

    Este script NO realiza:
    · Validaciones de coordenadas (responsabilidad de script 00).
    · Consultas a la Metadata API (responsabilidad de script 01).
    · Cálculos espaciales — heading y distancia vienen del inventario.

INPUT
    gsv/inventario_panos.csv
    (output de 01_analisis_cobertura_gsv.py)

OUTPUTS
    gsv/fotos/{ola}/          → imágenes descargadas
    gsv/registro_descargas.csv → log de intentos (Output 2)
    gsv/resumen_panel_gsv.csv  → resumen por hogar × ola (Output 3)
    gsv/reporte_descarga.txt   → estadísticas del proceso

PARÁMETROS DE DESCARGA
    size:    640 × 640 px   (máximo sin surcharge en la GSV Static API)
    fov:     90°
    pitch:   0° (configurable en CONFIG)
    heading: ángulo pano → hogar, almacenado en el inventario

ESCENARIO DE DESCARGA (CONFIG["escenario"])
    1 → Escenario 1: 1 imagen por panorama.
        Cámara apunta desde el panorama hacia el hogar (gsv_heading).
        Cobertura mínima. Usa ≈ 28 571 imágenes del crédito de $200.
    2 → Escenario 2: 2 imágenes por panorama.
        Agrega la vista opuesta (gsv_heading + 180°) para capturar el
        entorno completo de la calle. Duplica el costo y el espacio en disco.
    Cambia CONFIG["escenario"] antes de correr. Para añadir el segundo ángulo
    después de haber corrido el Escenario 1, pon escenario=2 y vuelve a
    ejecutar: el script es idempotente y omite las imágenes ya descargadas.

IDEMPOTENCIA
    Antes de descargar cada imagen, el script verifica si el archivo ya existe
    y es un JPEG válido. Si existe, se registra como ya_descargada=True y se
    omite sin hacer llamadas a la API.
    Si existe un registro previo de fallo, el parámetro reintentar_fallidas
    controla si se reintenta o se conserva el resultado previo.

CÓMO CORRER
    export GSV_API_KEY="tu_clave_aqui"
    python 02_descarga_fotos_GSV.py
"""

import os
import sys
import time
import signal
import logging
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent   # carpeta GoogleStreetView/

CONFIG = {
    # ── Input ──────────────────────────────────────────────────────────────────
    # Inventario generado por 01_analisis_cobertura_gsv.py.
    "input_inventario": _HERE / "gsv" / "inventario_panos.csv",

    # ── API key ────────────────────────────────────────────────────────────────
    # Recomendado: variable de entorno GSV_API_KEY.
    # Nunca commitear la clave directamente en el repositorio.
    "api_key": os.environ.get("GSV_API_KEY", "AQUI_TU_API_KEY"),

    # ── Parámetros de imagen ───────────────────────────────────────────────────
    # Estos parámetros pueden modificarse sin necesidad de volver a consultar
    # la Metadata API ni de modificar la arquitectura del pipeline.
    "img_size": "640x640",   # máximo sin costo adicional en la GSV Static API
    "fov":      90,           # campo de visión en grados (50–120; 90 es el estándar)
    "pitch":    0,            # ángulo vertical (0=horizontal; -10 mira levemente hacia abajo)
    # heading proviene del inventario (ángulo desde el pano hacia el hogar).
    # El número de imágenes por panorama depende de CONFIG["escenario"] arriba.

    # ── Escenario de descarga ──────────────────────────────────────────────────
    # 1 = solo la imagen frontal (heading)           → 1 imagen por panorama
    # 2 = frontal + opuesta (heading y heading+180°) → 2 imágenes por panorama
    # Correr el script de nuevo con escenario=2 agrega solo las imágenes faltantes.
    "escenario": 1,

    # ── Subconjunto de descarga ────────────────────────────────────────────────
    # True  → descarga SOLO las fotografías únicas de hogares con coordenadas
    #          válidas presentes en las tres olas del panel balanceado.
    #          Usa las columnas es_foto_unica=1 & en_panel_balanceado=1 del
    #          inventario generado por la versión actualizada de script 01.
    #          Sin duplicaciones entre radios ni entre olas (Reglas 2.1 y 2.2).
    # False → descarga todos los panoramas OK del inventario (comportamiento
    #          original: deduplicación manual por radio mínimo, sin filtro de panel).
    # Requiere inventario generado con 01_analisis_cobertura_gsv.py actualizado.
    # Si las columnas no existen en el inventario, se activa el modo False automáticamente.
    "solo_panel_balanceado": True,

    # ── Control de descarga ────────────────────────────────────────────────────
    "max_workers":       10,    # descargas simultáneas; reducir a 5 si hay errores de red
    "timeout_s":         30,    # segundos máximos por imagen antes de declarar timeout
    "max_reintentos":     3,    # intentos antes de marcar como fallo
    "pausa_error_s":      2,    # segundos de espera entre reintentos fallidos

    # ── Idempotencia ───────────────────────────────────────────────────────────
    # True  → reintentar imágenes que fallaron en sesiones anteriores
    # False → conservar el resultado de fallo previo sin volver a intentar
    "reintentar_fallidas": True,

    # ── Salida ─────────────────────────────────────────────────────────────────
    "output_dir": _HERE / "gsv",
}


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

_GSV_IMG_URL      = "https://maps.googleapis.com/maps/api/streetview"  # endpoint de imágenes
_JPEG_MIN_BYTES   = 10_000    # imágenes de error/placeholder suelen pesar < 10 KB
_JPEG_MAGIC       = b"\xff\xd8"  # los primeros 2 bytes de cualquier JPEG válido
_GUARDAR_CADA_N   = 50        # persistir el registro a disco cada N descargas completadas


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
# MÓDULO 1: CARGA DEL INVENTARIO
# ──────────────────────────────────────────────────────────────────────────────

def cargar_inventario(cfg: dict) -> pd.DataFrame:
    """
    Lee el inventario de panoramas (output de script 01) y retorna los
    panoramas elegibles para descarga.

    Modos de selección — controlado por CONFIG["solo_panel_balanceado"]:

    Modo A: solo_panel_balanceado=True  (recomendado)
        Requiere inventario generado con la versión actualizada de script 01,
        que incluye las columnas es_foto_unica y en_panel_balanceado.
        Selecciona únicamente filas donde:
            es_foto_unica = 1       → asignación canónica del pano_id para ese
                                      hogar (primera ola elegible + radio mínimo).
            en_panel_balanceado = 1 → hogar presente en las tres olas (2010,
                                      2013, 2016) con coordenadas válidas.
        No se realiza deduplicación adicional: es_foto_unica ya garantiza que
        cada pano_id aparece una sola vez por hogar en todo el panel.

    Modo B: solo_panel_balanceado=False  (comportamiento original)
        Filtra por status==OK y pano_id válido.
        Deduplica manualmente por (consecutivo, ola, pano_id) conservando el
        radio mínimo (panorama más cercano). Incluye todos los hogares del
        inventario sin restricción de panel balanceado.

    Fallback automático:
        Si solo_panel_balanceado=True pero las columnas es_foto_unica /
        en_panel_balanceado no existen en el inventario (versión anterior de
        script 01), se activa el Modo B con una advertencia.

    Retorna:
        DataFrame con una fila por (consecutivo, ola, pano_id) elegible.
    """
    path = Path(cfg["input_inventario"])
    if not path.exists():
        log.error("-" * 62)
        log.error("ERROR: Inventario de panoramas no encontrado.")
        log.error(f"Buscado en: {path.resolve()}")
        log.error("SOLUCION: Ejecuta primero el script anterior:")
        log.error("  python 01_analisis_cobertura_gsv.py")
        log.error("Ese script genera: gsv/inventario_panos.csv")
        log.error("Si el archivo existe pero en otra carpeta, actualiza")
        log.error("CONFIG['input_inventario'] en este script con la ruta correcta.")
        log.error("-" * 62)
        sys.exit(1)

    inv = pd.read_csv(
        path,
        dtype={"consecutivo": str, "llave": str, "llave_n16": str},
        low_memory=False,
    )
    log.info(f"Inventario cargado: {len(inv):,} filas ({path.name})")

    if "status" not in inv.columns or "pano_id" not in inv.columns:
        log.error("-" * 62)
        log.error("ERROR: El inventario no tiene las columnas 'status' y/o 'pano_id'.")
        log.error("Columnas presentes: " + str(list(inv.columns)))
        log.error("El archivo gsv/inventario_panos.csv esta incompleto o corrompido.")
        log.error("SOLUCION: Borra gsv/inventario_panos.csv y el cache")
        log.error("gsv/resultados_api_cache.csv, luego vuelve a ejecutar:")
        log.error("  python 01_analisis_cobertura_gsv.py")
        log.error("-" * 62)
        sys.exit(1)

    solo_bal    = cfg.get("solo_panel_balanceado", False)
    tiene_flags = ("en_panel_balanceado" in inv.columns and
                   "es_foto_unica" in inv.columns)

    # ── Modo A: filtrar por columnas de unicidad y panel balanceado ───────────
    if solo_bal and tiene_flags:
        log.info("  Modo: solo_panel_balanceado=True "
                 "(es_foto_unica=1 & en_panel_balanceado=1)")
        elegibles = inv[
            (inv["es_foto_unica"] == 1) &
            (inv["en_panel_balanceado"] == 1)
        ].copy()

        # Garantía adicional: pano_id no nulo (no debería ocurrir, pero es defensivo)
        sin_pano  = elegibles["pano_id"].isna() | (elegibles["pano_id"].astype(str) == "")
        elegibles = elegibles[~sin_pano].copy()

        if elegibles.empty:
            log.warning("-" * 62)
            log.warning("AVISO: No hay filas con es_foto_unica=1 & en_panel_balanceado=1.")
            log.warning("Causas posibles:")
            log.warning("  · El inventario fue generado con una version anterior de script 01.")
            log.warning("    Solucion: borra gsv/inventario_panos.csv y re-ejecuta script 01.")
            log.warning("  · Ningún hogar está presente en las tres olas con coord. válidas.")
            log.warning("-" * 62)
            sys.exit(0)

        log.info(f"  → {len(elegibles):,} panoramas únicos del panel balanceado")
        if "ola" in elegibles.columns:
            for ola, grp in elegibles.groupby("ola"):
                log.info(f"    Ola {ola}: {len(grp):,} panoramas")
        return elegibles.reset_index(drop=True)

    # ── Fallback / Modo B: filtrado y deduplicación estándar ─────────────────
    if solo_bal and not tiene_flags:
        log.warning("-" * 62)
        log.warning("AVISO: solo_panel_balanceado=True pero el inventario no tiene")
        log.warning("las columnas 'es_foto_unica' / 'en_panel_balanceado'.")
        log.warning("Esto indica un inventario generado con una version anterior de script 01.")
        log.warning("SOLUCION recomendada: borra gsv/inventario_panos.csv y re-ejecuta:")
        log.warning("  python 01_analisis_cobertura_gsv.py")
        log.warning("Por ahora se aplica el filtrado estandar (Modo B).")
        log.warning("-" * 62)

    log.info("  Modo: todos los panoramas OK "
             "(status=OK, pano_id válido, radio mínimo por hogar × ola × pano_id)")
    elegibles = inv[inv["status"] == "OK"].copy()
    sin_pano  = elegibles["pano_id"].isna() | (elegibles["pano_id"].astype(str) == "")
    elegibles = elegibles[~sin_pano].copy()

    if elegibles.empty:
        log.warning("-" * 62)
        log.warning("AVISO: No hay panoramas con status OK en el inventario.")
        dist_statuses = inv["status"].value_counts().to_dict() if "status" in inv.columns else {}
        log.warning(f"Distribucion de statuses en el inventario: {dist_statuses}")
        log.warning("Causas posibles:")
        log.warning("  · ZERO_RESULTS: no existe foto de calle en los radios configurados.")
        log.warning("  · REQUEST_DENIED: la API key no tiene acceso — re-ejecuta script 01.")
        log.warning("  · SIN_COORD_VALIDA: ningun hogar tenia coordenadas validas — revisa script 00.")
        log.warning("Sin imagenes que descargar. El script termina aqui.")
        log.warning("-" * 62)
        sys.exit(0)

    # Deduplicación manual: conserva el radio más pequeño por (hogar, ola, pano_id)
    elegibles = elegibles.sort_values("radio_m").drop_duplicates(
        subset=["consecutivo", "ola", "pano_id"], keep="first"
    )

    log.info(f"  → {len(elegibles):,} panoramas únicos (deduplicación por radio mínimo)")
    if "ola" in elegibles.columns:
        for ola, grp in elegibles.groupby("ola"):
            log.info(f"    Ola {ola}: {len(grp):,} panoramas")

    return elegibles.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: CONSTRUCCIÓN DEL PLAN DE DESCARGA
# ──────────────────────────────────────────────────────────────────────────────

def construir_plan(inv_ok: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Construye la tabla de tareas de descarga a partir del inventario elegible.

    Por cada (consecutivo, ola, pano_id) se genera 1 o 2 tareas según
    CONFIG["escenario"]:
    1. heading     → cámara apunta desde el panorama hacia el hogar
    2. heading+180 → cámara apunta en la dirección opuesta (solo escenario 2)

    El heading proviene del inventario (calculado en script 01 como ángulo
    azimutal desde el panorama hacia el hogar). No se recalcula aquí.

    Nomenclatura de archivos:
        {consecutivo}_{ola}_{pano_id}_{heading:03d}.jpg
    Los tres campos identifican unívocamente la imagen y permiten trazarla
    de vuelta al hogar, la ola y el panorama de origen.

    Estructura de directorios:
        {output_dir}/fotos/ola_{ola}/

    Escenario (CONFIG["escenario"]):
        1 → solo la imagen frontal (1 imagen por panorama)
        2 → frontal + opuesta    (2 imágenes por panorama)
    """
    tareas    = []
    fotos_dir = Path(cfg["output_dir"]) / "fotos"
    escenario = cfg.get("escenario", 2)               # 1 o 2; por defecto 2 para compatibilidad
    n_sin_heading = 0

    for _, row in inv_ok.iterrows():
        consecutivo = str(row["consecutivo"])
        ola         = int(row["ola"])
        pano_id     = str(row["pano_id"])

        h_raw = row.get("heading")
        if h_raw is None or pd.isna(h_raw):           # heading faltante → no hay imagen posible
            n_sin_heading += 1
            continue
        heading_base = float(h_raw)

        # Determina los ángulos a descargar según el escenario configurado
        if escenario == 1:
            headings_a_descargar = [round(heading_base % 360, 1)]                      # solo frontal
        else:
            headings_a_descargar = [
                round(heading_base % 360, 1),                                           # frontal
                round((heading_base + 180) % 360, 1),                                  # opuesto
            ]

        for heading in headings_a_descargar:
            nombre  = f"{consecutivo}_{ola}_{pano_id}_{int(heading):03d}.jpg"
            ruta    = fotos_dir / f"ola_{ola}" / nombre

            tareas.append({
                # ── Identificadores ──────────────────────────────────────────
                "consecutivo":    consecutivo,
                "ola":            ola,
                "pano_id":        pano_id,
                # ── Metadatos del panorama (del inventario) ───────────────────
                "radio_m":        row.get("radio_m"),
                "distancia_m":    row.get("distancia_m"),
                "fecha_pano":     row.get("fecha"),
                "lat_pano":       row.get("lat_pano"),
                "lon_pano":       row.get("lon_pano"),
                # ── Parámetros de la imagen ───────────────────────────────────
                "heading":        heading,
                "pitch":          cfg["pitch"],
                "fov":            cfg["fov"],
                "size":           cfg["img_size"],
                # ── Rutas ────────────────────────────────────────────────────
                "nombre_archivo": nombre,
                "ruta_archivo":   str(ruta),
                # ── Estado (se actualiza durante la descarga) ─────────────────
                "exito":          None,               # True/False/None=pendiente
                "ya_descargada":  False,              # True si el archivo ya existía
                "omitida":        False,              # True si se salta por config
                "timestamp":      None,               # fecha/hora de la descarga
                "codigo_http":    None,               # código de respuesta HTTP
                "bytes":          None,               # tamaño del archivo en bytes
                "mensaje_error":  None,               # descripción del error si falló
            })

    if n_sin_heading > 0:
        log.warning(f"  {n_sin_heading:,} panoramas omitidos por heading faltante (status OK pero sin heading).")

    if not tareas:
        # Todos los panoramas carecen de heading. Causas comunes:
        #   · La columna 'heading' no existe en el inventario_panos.csv.
        #   · El inventario se generó con una versión anterior del script 01.
        # Solución: borra gsv/resultados_api_cache.csv y vuelve a ejecutar
        #   python 01_analisis_cobertura_gsv.py
        log.error("-" * 62)
        log.error("ERROR: El plan de descarga quedó vacío.")
        log.error(f"  Panoramas en inventario:        {len(inv_ok):,}")
        log.error(f"  Descartados por heading=NaN:    {n_sin_heading:,}")
        log.error("  La columna 'heading' en inventario_panos.csv tiene solo NaN.")
        log.error("  Solucion: borra gsv/resultados_api_cache.csv y ejecuta:")
        log.error("    python 01_analisis_cobertura_gsv.py")
        log.error("-" * 62)
        sys.exit(1)

    imgs_por_pano = escenario   # 1 o 2 según el escenario elegido
    plan = pd.DataFrame(tareas)
    log.info(f"Plan de descarga: {len(plan):,} imágenes "
             f"({len(inv_ok) - n_sin_heading:,} panoramas × {imgs_por_pano} ángulo(s), "
             f"Escenario {escenario})")
    return plan


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: CARGA DEL REGISTRO PREVIO (REANUDACIÓN)
# ──────────────────────────────────────────────────────────────────────────────

def cargar_registro(cfg: dict, plan: pd.DataFrame) -> pd.DataFrame:
    """
    Carga el registro de descargas anteriores y actualiza el plan.

    Para cada tarea en el plan determina si:
    · El archivo ya existe en disco y es JPEG válido → ya_descargada=True, omitir.
    · El registro dice exito=True → ya_descargada=True, omitir (verificando en disco).
    · El registro dice exito=False y reintentar_fallidas=False → omitida=True.
    · En todos los demás casos → pendiente de descarga.

    La verificación de existencia en disco es la fuente de verdad principal;
    el registro previo es secundario y se usa para controlar el comportamiento
    ante fallos cuando el archivo no existe en disco.
    """
    path_registro = Path(cfg["output_dir"]) / "registro_descargas.csv"

    # ── Verificación de archivos existentes en disco ──────────────────────────
    n_ya    = 0      # imágenes encontradas como ya descargadas
    n_omit  = 0      # imágenes omitidas por config

    # Carga el registro previo si existe (para controlar comportamiento de fallos)
    fallidas_previas = set()                           # claves de fallos previos
    if path_registro.exists() and not cfg["reintentar_fallidas"]:
        try:
            reg_prev = pd.read_csv(path_registro)
            # pd.read_csv carga True/False como strings de objeto, no como bool.
            # "False" == False es siempre False en Python → comparar como string.
            fallos   = reg_prev[reg_prev["exito"].astype(str).str.strip() == "False"]
            for _, f in fallos.iterrows():
                clave = (str(f.get("consecutivo", "")),
                         str(f.get("ola", "")),
                         str(f.get("pano_id", "")),
                         str(f.get("heading", "")))
                fallidas_previas.add(clave)
            log.info(f"Registro previo: {len(fallidas_previas):,} fallos cargados "
                     f"(reintentar_fallidas=False → se omitirán).")
        except Exception as e:
            log.warning(f"No se pudo leer el registro previo: {e}")

    # Actualiza el plan fila a fila
    for idx, row in plan.iterrows():
        ruta = Path(row["ruta_archivo"])

        # Regla 1: el archivo existe en disco y es un JPEG válido → ya descargada
        if ruta.exists() and ruta.stat().st_size >= _JPEG_MIN_BYTES:
            plan.at[idx, "ya_descargada"] = True
            plan.at[idx, "exito"]         = True
            plan.at[idx, "bytes"]         = ruta.stat().st_size
            n_ya += 1
            continue

        # Regla 2: fallo previo y no se reintenta → omitida
        if not cfg["reintentar_fallidas"]:
            clave = (str(row["consecutivo"]), str(row["ola"]),
                     str(row["pano_id"]),     str(row["heading"]))
            if clave in fallidas_previas:
                plan.at[idx, "omitida"] = True
                n_omit += 1

    n_pendientes = len(plan) - n_ya - n_omit          # pendientes reales

    log.info(f"  Ya descargadas (en disco): {n_ya:,}")
    log.info(f"  Omitidas (fallo previo):   {n_omit:,}")
    log.info(f"  Pendientes de descarga:    {n_pendientes:,}")

    return plan


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: DESCARGA DE UNA IMAGEN
# ──────────────────────────────────────────────────────────────────────────────

def _es_jpeg_valido(contenido: bytes) -> bool:
    """
    Verifica que el contenido descargado es un JPEG real.
    La API puede retornar imágenes placeholder o respuestas en HTML para
    pano_ids eliminados o inválidos. Se comprueban tamaño mínimo y magic bytes.
    """
    return len(contenido) >= _JPEG_MIN_BYTES and contenido[:2] == _JPEG_MAGIC


def descargar_imagen(tarea: dict, cfg: dict) -> dict:
    """
    Descarga una sola imagen usando el pano_id almacenado en la tarea.

    Usar pano_id (en lugar de coordenadas) garantiza:
    · Exactamente el panorama verificado por script 01.
    · Reproducibilidad: el mismo pano_id produce la misma imagen.
    · Sin consultas adicionales a la Metadata API.

    La función intenta hasta max_reintentos veces antes de marcar la tarea
    como fallida. Registra el código HTTP y el mensaje de error si ocurre un fallo.
    """
    ruta        = Path(tarea["ruta_archivo"])
    pano_id     = tarea["pano_id"]
    heading     = tarea["heading"]
    api_key     = cfg["api_key"]
    timeout_s   = cfg["timeout_s"]
    max_int     = cfg["max_reintentos"]
    pausa_s     = cfg["pausa_error_s"]

    params = {
        "size":    cfg["img_size"],    # resolución de la imagen
        "pano":    pano_id,            # ID del panorama (no coordenadas)
        "heading": heading,            # ángulo de la cámara en grados
        "pitch":   cfg["pitch"],       # ángulo vertical de la cámara
        "fov":     cfg["fov"],         # campo de visión en grados
        "key":     api_key,            # clave de autenticación
    }

    for intento in range(max_int):              # hasta max_reintentos intentos
        codigo_http    = None                   # se actualiza con el código HTTP real
        mensaje_error  = None

        try:
            resp        = requests.get(_GSV_IMG_URL, params=params, timeout=timeout_s)
            codigo_http = resp.status_code      # registra el código de respuesta

            if resp.status_code == 403:
                # 403 puede significar dos cosas distintas:
                # a) La Street View Static API no esta habilitada / sin facturacion.
                #    En ese caso TODOS los downloads fallaran con 403.
                #    Solucion: console.cloud.google.com → APIs → habilitar Street View Static API
                # b) El pano_id especifico fue eliminado o expiro en Google.
                #    En ese caso solo ALGUNOS downloads fallaran. Es normal y puede ignorarse.
                # Para distinguir: si el reporte final muestra >50% de fallos con 403, es caso (a).
                return {**tarea,
                        "exito": False, "codigo_http": 403,
                        "mensaje_error": "403 Forbidden — key sin habilitacion o pano_id expirado",
                        "timestamp": datetime.now().isoformat()}

            if resp.status_code == 429:         # rate limit superado
                time.sleep(10)                  # espera larga antes de reintentar
                continue

            resp.raise_for_status()             # lanza en otros errores HTTP 4xx/5xx
            contenido = resp.content

            if not _es_jpeg_valido(contenido):  # la API devolvió un placeholder o HTML
                mensaje_error = (
                    f"Contenido inválido: {len(contenido):,} bytes, "
                    f"inicio={contenido[:4].hex()}"
                )
                if intento < max_int - 1:
                    time.sleep(pausa_s)
                    continue
                return {**tarea,
                        "exito": False, "codigo_http": codigo_http,
                        "mensaje_error": mensaje_error,
                        "timestamp": datetime.now().isoformat()}

            # Guarda la imagen en disco
            ruta.parent.mkdir(parents=True, exist_ok=True)  # crea el directorio si no existe
            ruta.write_bytes(contenido)                      # escribe el archivo

            return {**tarea,
                    "exito":      True,
                    "codigo_http": codigo_http,
                    "bytes":      len(contenido),
                    "timestamp":  datetime.now().isoformat(),
                    "mensaje_error": None}

        except requests.exceptions.Timeout:
            mensaje_error = "Timeout"
        except requests.exceptions.HTTPError as e:
            mensaje_error = f"HTTP {codigo_http}: {e}"
        except requests.exceptions.RequestException as e:
            mensaje_error = f"Error de red: {e}"

        if intento < max_int - 1:               # si quedan reintentos
            time.sleep(pausa_s)                 # pausa antes del próximo intento

    # Agotó todos los reintentos
    return {**tarea,
            "exito": False, "codigo_http": codigo_http,
            "mensaje_error": mensaje_error or "Máximo de reintentos superado",
            "timestamp": datetime.now().isoformat()}


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5: DESCARGA EN PARALELO
# ──────────────────────────────────────────────────────────────────────────────

_lock_registro = threading.Lock()    # protege escrituras concurrentes al registro
_detener       = threading.Event()   # señal de Ctrl+C para detener limpiamente


def _manejador_sigint(_sig, _frame):
    """Captura Ctrl+C y pide al pool que termine las tareas activas antes de salir."""
    print("\n\n[Ctrl+C] Deteniendo — se esperan las tareas activas y se guarda el registro...")
    _detener.set()


def guardar_registro(plan: pd.DataFrame, cfg: dict):
    """Persiste el registro de descargas a disco (thread-safe)."""
    with _lock_registro:
        path = Path(cfg["output_dir"]) / "registro_descargas.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(path, index=False, encoding="utf-8")


def ejecutar_descargas(plan: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Descarga en paralelo todas las imágenes pendientes del plan.

    Usa ThreadPoolExecutor para ejecutar múltiples descargas simultáneas.
    Guarda el registro a disco cada _GUARDAR_CADA_N completadas para minimizar
    la pérdida de progreso ante una interrupción.

    Retorna el plan actualizado con el estado de cada descarga.
    """
    signal.signal(signal.SIGINT, _manejador_sigint)  # registra el manejador de Ctrl+C

    # Filtra solo las tareas pendientes (ni ya descargadas ni omitidas)
    mask_pend  = (plan["ya_descargada"] == False) & (plan["omitida"] == False)  # noqa: E712
    # dict(row) con iterrows() funciona porque pd.Series implementa keys().
    # NO usar itertuples(): devuelve namedtuples y dict(namedtuple) crashea con
    # ValueError al intentar usar cada valor como un par (clave, valor).
    pendientes = [dict(row) for _, row in plan[mask_pend].iterrows()]

    if not pendientes:
        log.info("No hay imágenes pendientes.")
        return plan

    n_pend   = len(pendientes)
    workers  = cfg["max_workers"]
    log.info(f"Iniciando descarga de {n_pend:,} imágenes ({workers} workers)...")
    log.info("Presiona Ctrl+C para detener y guardar el progreso.")

    completadas = exitos = fallos = 0

    # Índice rápido: (consecutivo, ola, pano_id, heading) → índice en plan.
    # Evita recorrer todo el DataFrame por cada imagen completada (O(1) vs O(n²)).
    idx_map: dict = {}
    for idx, row in plan[mask_pend].iterrows():
        clave = (str(row["consecutivo"]), str(row["ola"]),
                 str(row["pano_id"]),     float(row["heading"]))
        idx_map[clave] = idx

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futuros = {executor.submit(descargar_imagen, t, cfg): t for t in pendientes}

        with tqdm(total=n_pend, desc="Descargando", unit="img") as barra:
            for futuro in as_completed(futuros):
                if _detener.is_set():
                    for f in futuros:       # cancela futuros pendientes
                        f.cancel()
                    break

                try:
                    resultado = futuro.result()
                except Exception as exc:
                    # El worker lanzó una excepción inesperada; registra el fallo y continúa.
                    tarea_orig = futuros[futuro]
                    log.warning(f"Worker inesperado: {tarea_orig.get('nombre_archivo')} — {exc}")
                    resultado = {
                        **tarea_orig,
                        "exito":         False,
                        "codigo_http":   None,
                        "mensaje_error": f"ERROR_WORKER: {exc}",
                        "timestamp":     datetime.now().isoformat(),
                    }

                completadas += 1

                # Actualiza la fila usando el índice precalculado (O(1))
                clave = (str(resultado["consecutivo"]), str(resultado["ola"]),
                         str(resultado["pano_id"]),     float(resultado["heading"]))
                idx = idx_map.get(clave)
                if idx is not None:
                    for campo in ["exito", "timestamp", "codigo_http", "bytes", "mensaje_error"]:
                        plan.at[idx, campo] = resultado.get(campo)

                if resultado["exito"]:
                    exitos += 1
                else:
                    fallos += 1

                barra.set_postfix({"OK": exitos, "ERR": fallos}, refresh=False)
                barra.update(1)

                if completadas % _GUARDAR_CADA_N == 0:  # guardado incremental
                    guardar_registro(plan, cfg)

    guardar_registro(plan, cfg)    # guardado final
    log.info(f"Descarga completada: {exitos:,} exitosas, {fallos:,} fallidas.")
    return plan


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: RESUMEN PARA EL PANEL (Output 3)
# ──────────────────────────────────────────────────────────────────────────────

def construir_resumen_panel(plan: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Agrega el registro de descargas a nivel hogar × ola.

    Produce una tabla con una fila por (consecutivo, ola) que puede unirse
    al panel principal mediante un merge simple.

    Variables generadas:
        n_imagenes_programadas:   total de imágenes programadas para el hogar
        n_exitosas:               imágenes descargadas correctamente
        n_fallidas:               intentos que terminaron en error
        n_ya_descargadas:         imágenes que ya existían en disco
        n_omitidas:               imágenes omitidas por configuración
        n_panos_descargados:      panoramas únicos con al menos una imagen exitosa
        tiene_imagenes:           1 si al menos una imagen fue descargada; 0 si no
        fecha_descarga:           timestamp de la primera imagen exitosa
    """
    resumen_filas = []

    for (consecutivo, ola), grp in plan.groupby(["consecutivo", "ola"]):
        exitosas   = grp[grp["exito"] == True]          # noqa: E712
        n_ex       = len(exitosas)
        panos_ok   = exitosas["pano_id"].nunique()       # panoramas únicos con descarga OK
        fecha_prim = (exitosas["timestamp"].dropna().min()
                      if n_ex > 0 else None)             # fecha más temprana

        resumen_filas.append({
            "consecutivo":            consecutivo,
            "ola":                    ola,
            "n_imagenes_programadas": len(grp),
            "n_exitosas":             n_ex,
            "n_fallidas":             int((grp["exito"] == False).sum()),  # noqa: E712
            "n_ya_descargadas":       int(grp["ya_descargada"].sum()),
            "n_omitidas":             int(grp["omitida"].sum()),
            "n_panos_descargados":    panos_ok,
            "tiene_imagenes":         1 if n_ex > 0 else 0,
            "fecha_descarga":         fecha_prim,
        })

    resumen = pd.DataFrame(resumen_filas)
    path    = Path(cfg["output_dir"]) / "resumen_panel_gsv.csv"
    resumen.to_csv(path, index=False, encoding="utf-8")
    log.info(f"resumen_panel_gsv.csv: {len(resumen):,} filas → {path}")
    return resumen


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 7: REPORTE DESCRIPTIVO
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(plan: pd.DataFrame, t_inicio: float, cfg: dict) -> str:
    """
    Genera el reporte descriptivo completo del proceso de descarga.

    Incluye: totales, tasa de éxito, desglose por ola, errores más frecuentes
    y estadísticas de tiempo. Guarda el texto en reporte_descarga.txt.
    """
    sep    = "=" * 65
    t_fin  = time.time()
    t_tot  = t_fin - t_inicio
    lines  = [
        sep,
        "REPORTE DE DESCARGA — GOOGLE STREET VIEW FOTOS",
        f"Pipeline ELCA  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep, "",
    ]

    n_total   = len(plan)
    n_ex_tot  = int((plan["exito"] == True).sum())               # noqa: E712  todas exitosas (incl. ya en disco)
    n_fa      = int((plan["exito"] == False).sum())              # noqa: E712
    n_ya      = int(plan["ya_descargada"].sum())                  # ya existían en disco antes de esta sesión
    n_nuevas  = n_ex_tot - n_ya                                   # descargadas en esta sesión solamente
    n_om      = int(plan["omitida"].sum())                        # omitidas por config
    # n_ya ya está incluido en n_ex_tot → no restar dos veces
    n_pend    = n_total - n_ex_tot - n_fa - n_om                  # sin descargar (Ctrl+C o sesión incompleta)
    bytes_t   = plan.loc[plan["exito"] == True, "bytes"].sum()   # noqa: E712
    mb_t      = bytes_t / 1_048_576 if bytes_t > 0 else 0

    lines += [
        "── 1. RESUMEN GENERAL ────────────────────────────────────",
        f"  Imágenes programadas:              {n_total:,}",
        f"  Descargadas en esta sesión (OK):   {n_nuevas:,}",
        f"  Ya existían en disco:              {n_ya:,}",
        f"  Fallidas:                          {n_fa:,}",
        f"  Omitidas (fallo previo conservado):{n_om:,}",
        f"  Pendientes (interrupción):         {n_pend:,}",
        f"  Espacio en disco (esta sesión):    {mb_t:.1f} MB",
    ]

    # Tasa de éxito sobre las que se intentaron en esta sesión (excluye ya en disco)
    intentadas = n_nuevas + n_fa
    if intentadas > 0:
        pct_ok = 100 * n_nuevas / intentadas
        lines.append(f"  Tasa de éxito (sobre intentadas):  {pct_ok:.1f}%")

    # ── Tiempo ──────────────────────────────────────────────────────────────
    lines += [
        "", "── 2. TIEMPO ─────────────────────────────────────────────",
        f"  Duración total: {t_tot/60:.1f} min ({t_tot:.0f} s)",
    ]
    if intentadas > 0:
        lines.append(f"  Promedio por imagen: {t_tot/intentadas:.2f} s")

    # ── Desglose por ola ─────────────────────────────────────────────────────
    if "ola" in plan.columns:
        lines += ["", "── 3. DESGLOSE POR OLA ──────────────────────────────────"]
        for ola, grp in plan.groupby("ola"):
            n_g        = len(grp)
            n_ok       = int((grp["exito"] == True).sum())   # noqa: E712  exitosas totales (incl. ya en disco)
            n_ya_g     = int(grp["ya_descargada"].sum())
            n_nuevas_g = n_ok - n_ya_g                        # solo las descargadas en esta sesión
            # pct solo sobre n_ok (n_ya_g ya está incluido en n_ok → no sumar de nuevo)
            pct        = 100 * n_ok / n_g if n_g > 0 else 0
            lines.append(
                f"  Ola {ola}: {n_g:,} programadas / "
                f"{n_nuevas_g:,} nuevas / {n_ya_g:,} ya existían  ({pct:.1f}% exitosas)"
            )

    # ── Errores más frecuentes ────────────────────────────────────────────────
    fallidas = plan[(plan["exito"] == False) & plan["mensaje_error"].notna()]  # noqa: E712
    if len(fallidas) > 0:
        lines += [
            "", "── 4. ERRORES MÁS FRECUENTES ────────────────────────────",
            f"  Total con error: {len(fallidas):,}",
        ]
        top_err = fallidas["mensaje_error"].value_counts().head(5)
        for msg, cnt in top_err.items():
            lines.append(f"  {cnt:>6,}  {msg}")

        # Diagnóstico automático según el tipo de error dominante
        n_403   = int(fallidas["codigo_http"].eq(403).sum())
        n_tout  = int(fallidas["mensaje_error"].str.contains("Timeout", na=False).sum())
        pct_403 = 100 * n_403 / len(fallidas) if len(fallidas) > 0 else 0
        if pct_403 > 50:
            lines += [
                "",
                "  DIAGNOSTICO — Mayoria de fallos son 403:",
                "  Probable causa: Street View Static API no habilitada o sin facturacion.",
                "  Solucion: console.cloud.google.com → APIs → Street View Static API → Habilitar",
                "  Luego verifica que la facturacion este activa en el proyecto.",
            ]
        elif n_tout > len(fallidas) * 0.3:
            lines += [
                "",
                "  DIAGNOSTICO — Muchos Timeouts:",
                "  La conexion a internet es inestable o muy lenta para descargar imagenes.",
                "  Solucion: reduce CONFIG['max_workers'] a 3 o 5 para menos descargas simultaneas.",
                "  Al volver a ejecutar, el script retoma desde donde quedo (idempotente).",
            ]

    lines += ["", sep]
    reporte = "\n".join(lines)

    # Guarda en disco
    path = Path(cfg["output_dir"]) / "reporte_descarga.txt"
    path.write_text(reporte, encoding="utf-8")
    log.info(f"reporte_descarga.txt guardado.")
    return reporte


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Orquesta el pipeline completo de descarga de imágenes GSV.

    1. Carga y filtra el inventario de panoramas (script 01).
    2. Construye el plan de descarga (1 o 2 imágenes por panorama según CONFIG["escenario"]).
    3. Carga el registro previo y marca ya descargadas / omitidas.
    4. Ejecuta descargas en paralelo con registro incremental.
    5. Construye el resumen por hogar × ola (para merge con panel).
    6. Genera el reporte descriptivo.
    """
    log.info("── Script 02: Descarga de fotos GSV ─────────────────────────")
    log.info("Si ves ModuleNotFoundError, activa el entorno virtual primero:")
    log.info("  Mac/Linux : source .venv/bin/activate")
    log.info("  Windows   : .venv\\Scripts\\activate")
    log.info("  Luego     : pip install -r requirements.txt")
    t_inicio = time.time()

    if not CONFIG["api_key"]:
        log.error("-" * 62)
        log.error("ERROR: La API key de Google Street View no esta configurada.")
        log.error("OPCION 1 — define la variable de entorno antes de correr:")
        log.error("  Windows (cmd) : set GSV_API_KEY=tu_clave_aqui")
        log.error("  Windows (PS)  : $env:GSV_API_KEY='tu_clave_aqui'")
        log.error("  Mac/Linux     : export GSV_API_KEY='tu_clave_aqui'")
        log.error("OPCION 2 — edita CONFIG['api_key'] en este script con la clave directa.")
        log.error("-" * 62)
        sys.exit(1)

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Paso 1: inventario ────────────────────────────────────────────────────
    log.info("\n── Paso 1: Carga del inventario ─────────────────────────────")
    inv_ok = cargar_inventario(CONFIG)

    # ── Paso 2: plan de descarga ──────────────────────────────────────────────
    log.info("\n── Paso 2: Construcción del plan ────────────────────────────")
    plan = construir_plan(inv_ok, CONFIG)

    # ── Paso 3: registro previo ───────────────────────────────────────────────
    log.info("\n── Paso 3: Revisión de descargas anteriores ─────────────────")
    plan = cargar_registro(CONFIG, plan)

    n_pendientes = len(plan) - plan["ya_descargada"].sum() - plan["omitida"].sum()
    if n_pendientes == 0:
        log.info("Todas las imágenes ya estaban descargadas o no hay pendientes.")
    else:
        # ── Paso 4: descargar ─────────────────────────────────────────────────
        log.info(f"\n── Paso 4: Descarga ({n_pendientes:,} imágenes pendientes) ────")
        plan = ejecutar_descargas(plan, CONFIG)

    # ── Paso 5: resumen para el panel ────────────────────────────────────────
    log.info("\n── Paso 5: Resumen por hogar × ola ──────────────────────────")
    construir_resumen_panel(plan, CONFIG)

    # ── Paso 6: reporte ───────────────────────────────────────────────────────
    log.info("\n── Paso 6: Reporte descriptivo ──────────────────────────────")
    reporte = generar_reporte(plan, t_inicio, CONFIG)
    print("\n" + reporte)


if __name__ == "__main__":
    main()
