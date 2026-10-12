"""
01_analisis_cobertura_gsv.py
============================
Consulta la Google Street View Metadata API para construir un inventario
reproducible de panoramas candidatos y generar estadísticas descriptivas
de cobertura para el panel ELCA (olas 2010, 2013, 2016).

QUÉ HACE
    1. Carga el panel de coordenadas generado por el script 00.
    2. Separa hogares con coordenadas válidas (coord_valida_ola=1) de inválidos.
    3. Ejecuta consultas a la Metadata API con deduplicación y reanudación.
    4. Construye el inventario de panoramas (hogar × ola × radio).
    5. Construye el panel enriquecido con variables resumen de cobertura GSV.
    6. Genera un reporte descriptivo con estadísticas y análisis de costos.

NOTA METODOLÓGICA — LIMITACIÓN DE LA API
    La Metadata API devuelve UN SOLO panorama por consulta: el más cercano
    a la ubicación dada dentro del radio especificado. No retorna todos los
    panoramas disponibles en el área.

    Estrategia adoptada:
    · Se realizan consultas independientes para cada radio configurado
      (por defecto 50 m, 100 m, 200 m).
    · Cada consulta puede retornar el mismo pano_id que otra con radio menor
      (si ese panorama es el más cercano en ambos casos), o un pano_id distinto
      si el radio mayor alcanza uno que el menor no alcanzaba.
    · Las variables gsv_n_panos_{R}m son indicadores de PRESENCIA (0/1):
      «existe al menos un panorama dentro de R metros», no un recuento real
      de panoramas únicos. Para recuentos reales se requeriría la Maps
      JavaScript API o muestreo en grilla, ambos fuera del alcance de este script.
    · El inventario (Output 1) tiene una fila por (hogar × ola × radio),
      no por panorama único. Filas con el mismo pano_id en distintos radios
      corresponden al mismo panorama visto desde consultas con distinto radio.

    Esta limitación se documenta en el reporte de texto (Output 2).

OPCIONES DE DESCARGA — SCRIPT 02
    El output panel_enriquecido_gsv.csv contiene las columnas necesarias para
    descargar las imágenes en dos escenarios de volumen/costo distintos.
    Ambos escenarios usan el mismo pano_id; solo varía el ángulo de la cámara.

    Escenario 1 — 1 imagen por hogar (vista frontal hacia el hogar):
        columna:  gsv_heading
        uso:      descargar(pano_id=row["gsv_pano_id_cercano"],
                            heading=row["gsv_heading"])

    Escenario 2 — 2 imágenes por hogar (frontal + vista opuesta):
        columnas: gsv_heading  y  gsv_heading_opuesto
        uso:      descargar(pano_id=row["gsv_pano_id_cercano"],
                            heading=row["gsv_heading"])
                  descargar(pano_id=row["gsv_pano_id_cercano"],
                            heading=row["gsv_heading_opuesto"])

    En ambos casos filtrar primero por gsv_tiene_pano == 1.
    gsv_heading_opuesto = (gsv_heading + 180) % 360; es None cuando
    gsv_tiene_pano == 0, así que el filtro anterior los excluye automáticamente.

INPUTS
    coordenadas/panel_coordenadas.csv
    (output de 00_construir_panel_coordenadas.py)

OUTPUTS
    gsv/inventario_panos.csv       → hogar × ola × radio
    gsv/reporte_cobertura_gsv.txt  → estadísticas descriptivas y análisis de costos
    gsv/panel_enriquecido_gsv.csv  → hogar × ola (+ vars GSV y columnas de descarga)

CÓMO CORRER
    export GSV_API_KEY="tu_clave_aqui"
    python 01_analisis_cobertura_gsv.py

    O editar CONFIG["api_key"] directamente (no commitear al repositorio).
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import os        # os.environ.get: lee la API key desde variables de entorno
import sys       # sys.exit: termina el script en errores críticos
import math      # math.sin, cos, atan2, radians: cálculos geométricos (Haversine, heading)
import logging   # logging: imprime mensajes INFO, WARNING y ERROR con timestamp
import threading # threading.Lock: protege la lista de resultados en el modo paralelo

from pathlib import Path                                    # manejo de rutas multiplataforma
from concurrent.futures import ThreadPoolExecutor, as_completed  # paralelismo I/O-bound

# ── Librerías externas ─────────────────────────────────────────────────────────
# Si cualquiera de estos imports falla con ModuleNotFoundError, el entorno
# virtual no está activado o las dependencias no están instaladas. Solución:
#   Mac/Linux : source .venv/bin/activate
#   Windows   : .venv\Scripts\activate
#   Luego     : pip install pandas requests tqdm
#              (o: pip install -r requirements.txt si existe ese archivo)
try:
    import pandas as pd    # DataFrames, lectura de CSV, exportación de resultados
    import requests        # requests.get: llamadas HTTP a la Metadata API de GSV
    from tqdm import tqdm  # barra de progreso durante las consultas a la API
except ImportError as _e:
    print("=" * 62)
    print(f"ERROR DE IMPORTACION: {_e}")
    print("Activa el entorno virtual antes de correr este script:")
    print("  Mac/Linux : source .venv/bin/activate")
    print("  Windows   : .venv\\Scripts\\activate")
    print("Luego instala las dependencias:")
    print("  pip install pandas requests tqdm")
    print("  (o: pip install -r requirements.txt)")
    print("=" * 62)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script a nuevas rutas,
# radios de búsqueda o cuando cambie la clave de API.
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent   # carpeta GoogleStreetView/

CONFIG = {
    # ── Input ──────────────────────────────────────────────────────────────────
    # Panel generado por 00_construir_panel_coordenadas.py.
    # Se asume coord_valida_ola calculada y coordenadas ya depuradas.
    "input_panel": _HERE / "coordenadas" / "panel_coordenadas.csv",

    # ── API key ────────────────────────────────────────────────────────────────
    # Recomendado: variable de entorno GSV_API_KEY para no exponer la clave.
    # Alternativa: escribir la clave aquí, pero nunca commitear al repositorio.
    "api_key": os.environ.get("GSV_API_KEY", "PASTE_YOUR_API_KEY_HERE"),

    # ── Radios de búsqueda en metros ───────────────────────────────────────────
    # Se realiza una consulta independiente por cada radio.
    # La API devuelve el panorama más cercano dentro de cada radio.
    # Ver NOTA METODOLÓGICA: estos producen indicadores de presencia (0/1).
    "radios_m": [50, 100, 200, 400],

    # ── Columna de zona (urbano/rural) ─────────────────────────────────────────
    # Nombre exacto de la columna en el panel que identifica la zona.
    # Valores típicos: "Urbano" / "Rural" (o equivalentes según codificación).
    # Si la columna no existe en el panel, la desagregación por zona se omite.
    "col_zona": "zona",

    # ── Fuente de panoramas ────────────────────────────────────────────────────
    # "outdoor": solo exteriores (recomendado para entorno de hogares)
    # "default": incluye interiores y panoramas de Google Maps Business
    "source": "outdoor",

    # ── Workers paralelos ──────────────────────────────────────────────────────
    # Número de consultas simultáneas a la Metadata API.
    # Con 10 workers y ~150 ms de latencia → ~67 req/s (vs ~20 req/s secuencial).
    # Reducir a 5 si aparecen errores OVER_QUERY_LIMIT sostenidos.
    "n_workers": 10,

    # ── Salida ─────────────────────────────────────────────────────────────────
    "output_dir": _HERE / "gsv",
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,                                # nivel mínimo de mensajes a mostrar
    format="%(asctime)s [%(levelname)s] %(message)s",  # formato: timestamp + nivel + mensaje
    handlers=[logging.StreamHandler(sys.stdout)],      # envía todos los mensajes a la consola
    force=True,   # sobreescribe configuraciones previas del logger raíz (útil en IDEs)
)
log = logging.getLogger(__name__)

# URL del endpoint de Metadata (gratuito; no descarga imágenes)
_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# ──────────────────────────────────────────────────────────────────────────────
# VENTANAS TEMPORALES Y UTILIDADES DE ELEGIBILIDAD
# Una fotografía es elegible para una ola si su año de captura cae en la
# ventana temporal correspondiente. Estas constantes se usan en
# asignar_fotos_unicas() y en las secciones de estadísticas del reporte.
# ──────────────────────────────────────────────────────────────────────────────

VENTANAS_OLA: dict = {
    2010: (None, 2010),   # cualquier año ≤ 2010
    2013: (2011, 2013),   # años 2011, 2012, 2013
    2016: (2014, 2016),   # años 2014, 2015, 2016
}


def _anio_foto(fecha):
    """Extrae el año entero de una fecha 'YYYY-MM' o 'YYYY'. Retorna None si falla."""
    try:
        return int(str(fecha)[:4])
    except (ValueError, TypeError):
        return None


def _elegible_para_ola(anio, ola: int) -> bool:
    """True si el año de la foto cae dentro de la ventana temporal de la ola."""
    if anio is None or ola not in VENTANAS_OLA:
        return False
    min_anio, max_anio = VENTANAS_OLA[ola]
    if min_anio is not None and anio < min_anio:
        return False
    if max_anio is not None and anio > max_anio:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: CARGA DEL PANEL
# ──────────────────────────────────────────────────────────────────────────────

def cargar_panel(cfg: dict) -> tuple:
    """
    Lee el panel de coordenadas y separa válidos (coord_valida_ola==1) de inválidos.

    Retorna:
        (panel_valido, panel_invalido)
        panel_valido  → generan consultas a la API
        panel_invalido → se documentan en el inventario con status SIN_COORD_VALIDA
    """
    path = Path(cfg["input_panel"])
    if not path.exists():
        log.error("-" * 62)
        log.error("ERROR: Panel de coordenadas no encontrado.")
        log.error(f"Buscado en: {path.resolve()}")
        log.error("SOLUCION: Ejecuta primero el script anterior:")
        log.error("  python 00_construir_panel_coordenadas.py")
        log.error("Ese script genera: coordenadas/panel_coordenadas.csv")
        log.error("Si el archivo existe pero en otra carpeta, actualiza")
        log.error("CONFIG['input_panel'] en este script con la ruta correcta.")
        log.error("-" * 62)
        sys.exit(1)

    # Lee preservando ceros a la izquierda de los identificadores
    panel = pd.read_csv(
        path,
        dtype={"consecutivo": str, "llave": str, "llave_n16": str},
        low_memory=False,
    )
    log.info(f"Panel cargado: {len(panel):,} filas ({path.name})")

    if "coord_valida_ola" not in panel.columns:
        log.error("-" * 62)
        log.error("ERROR: El panel no tiene la columna 'coord_valida_ola'.")
        log.error("Columnas presentes: " + str(list(panel.columns)))
        log.error("Esto indica que el archivo fue generado por una version")
        log.error("antigua o incompleta de 00_construir_panel_coordenadas.py.")
        log.error("SOLUCION: Borra coordenadas/panel_coordenadas.csv y vuelve")
        log.error("a ejecutar 00_construir_panel_coordenadas.py desde cero.")
        log.error("-" * 62)
        sys.exit(1)

    mascara        = panel["coord_valida_ola"] == 1
    panel_valido   = panel[mascara].copy()
    panel_invalido = panel[~mascara].copy()

    log.info(f"  coord_valida_ola=1: {len(panel_valido):,} → generarán consultas")
    log.info(f"  Sin coord válida:   {len(panel_invalido):,} → sin consulta")
    return panel_valido, panel_invalido


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES GEOMÉTRICAS
# ──────────────────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos geográficos (fórmula de Haversine)."""
    R  = 6_371_000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def _heading(lat_pano: float, lon_pano: float,
             lat_hogar: float, lon_hogar: float) -> float:
    """
    Ángulo azimutal desde el panorama hacia el hogar en grados (0=Norte, 90=Este).
    Se usa como parámetro 'heading' en la descarga (script 02).
    heading + 180° da la vista opuesta (dos imágenes por panorama).
    """
    la1 = math.radians(lat_pano)
    la2 = math.radians(lat_hogar)
    dl  = math.radians(lon_hogar - lon_pano)
    x   = math.sin(dl) * math.cos(la2)
    y   = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dl)
    return round((math.degrees(math.atan2(x, y)) + 360) % 360, 1)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: CONSULTA INDIVIDUAL A LA METADATA API
# ──────────────────────────────────────────────────────────────────────────────

def consultar_metadata(lat: float, lon: float, radio: int,
                       api_key: str, source: str) -> dict:
    """
    Realiza una única consulta al endpoint de Metadata de GSV.

    La API devuelve el panorama MÁS CERCANO dentro del radio — nunca más de uno.

    Campos posibles en la respuesta JSON:
        status:   "OK" | "ZERO_RESULTS" | "NOT_FOUND" | "REQUEST_DENIED" | "UNKNOWN_ERROR"
        pano_id:  ID único del panorama (string de ~22 caracteres)
        date:     fecha aproximada "YYYY-MM" o "YYYY"
        location: {"lat": float, "lng": float} — coordenadas reales del panorama

    En caso de error de red retorna un dict con status "ERROR_*" en vez de
    lanzar una excepción, para no interrumpir el procesamiento de los demás hogares.
    """
    params = {
        "location": f"{lat},{lon}",
        "radius":   radio,
        "source":   source,
        "key":      api_key,
    }
    try:
        r = requests.get(_METADATA_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        # La API tardó más de 10 s. Ocurre en conexiones lentas o sobrecarga.
        # El resultado queda como ERROR_TIMEOUT en la caché; al retomar el
        # script lo reintentará automáticamente (no estará en caché como OK).
        return {"status": "ERROR_TIMEOUT"}
    except requests.exceptions.HTTPError:
        # HTTP 4xx/5xx. ERROR_HTTP_403 suele indicar clave inválida o sin billing.
        # ERROR_HTTP_429 indica rate limit — reduce CONFIG["n_workers"] a 5.
        return {"status": f"ERROR_HTTP_{r.status_code}"}
    except requests.exceptions.RequestException:
        # Sin conexión a internet u otro error de red.
        # Verifica conectividad: ping maps.googleapis.com
        return {"status": "ERROR_RED"}
    except ValueError:
        # json.JSONDecodeError (subclase de ValueError): la API devolvió HTML
        # en lugar de JSON. Ocurre en redes con portal cautivo (WiFi de campus)
        # o cuando un proxy intercepta el tráfico HTTPS.
        # Solución: verifica que la conexión sea directa (sin proxy corporativo)
        # y que puedas acceder a maps.googleapis.com desde el navegador.
        return {"status": "ERROR_JSON"}


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: EJECUCIÓN DE CONSULTAS CON DEDUPLICACIÓN Y REANUDACIÓN
# ──────────────────────────────────────────────────────────────────────────────

# Columnas que debe tener siempre el DataFrame de caché de la API.
# Garantiza que el merge en construir_inventario() no falle con KeyError
# cuando la caché está vacía (p.ej. primer arranque sin coordenadas válidas).
_CACHE_COLS = [
    "lat_decimal", "lon_decimal", "radio_m", "status",
    "pano_id", "distancia_m", "fecha", "lat_pano", "lon_pano", "heading",
]

def ejecutar_consultas(panel_valido: pd.DataFrame,
                       cfg: dict,
                       path_cache: Path) -> pd.DataFrame:
    """
    Consulta la Metadata API para todas las combinaciones (lat, lon, radio)
    del panel válido y retorna la caché de resultados.

    Estrategias de eficiencia:
    · DEDUPLICACIÓN: hogares con coordenadas idénticas (p.ej. sub-hogares del mismo
      edificio) comparten una sola consulta. La clave es (lat_decimal, lon_decimal, radio_m).
    · REANUDACIÓN: si el archivo caché ya existe, se cargan los resultados previos
      y se omiten las consultas ya realizadas. Permite retomar si el script fue interrumpido.
    · GUARDADO INCREMENTAL: la caché se guarda a disco cada 50 consultas nuevas para
      minimizar pérdidas si el script se interrumpe durante una sesión larga.

    La caché almacena una fila por (lat_decimal, lon_decimal, radio_m) — no por hogar.
    La unión con (consecutivo, ola) ocurre en construir_inventario().

    Retorna:
        DataFrame con columnas: lat_decimal, lon_decimal, radio_m, status,
        pano_id, distancia_m, fecha, lat_pano, lon_pano, heading.
    """
    radios    = cfg["radios_m"]
    api_key   = cfg["api_key"]
    source    = cfg["source"]
    n_workers = cfg["n_workers"]

    # ── Construir el conjunto completo de consultas (lat, lon, radio) ─────────
    coords_unicas = panel_valido[["lat_decimal", "lon_decimal"]].drop_duplicates()
    filas = []
    for _, row in coords_unicas.iterrows():
        for radio in radios:
            filas.append({
                "lat_decimal": row["lat_decimal"],
                "lon_decimal": row["lon_decimal"],
                "radio_m":     radio,
            })
    consultas = pd.DataFrame(filas)

    log.info(f"Consultas planificadas: {len(consultas):,} "
             f"({len(coords_unicas):,} coords × {len(radios)} radios)")

    # ── Reanudación: cargar caché existente y filtrar ya realizadas ───────────
    if path_cache.exists():
        cache_df = pd.read_csv(path_cache)
        log.info(f"Caché existente: {len(cache_df):,} consultas previas cargadas.")
        # Solo se consideran "hechas" las consultas con resultado DEFINITIVO.
        # Los errores transitorios se reintentan al retomar:
        #   · ERROR_TIMEOUT  → red lenta o servidor ocupado, reintentable.
        #   · ERROR_RED      → sin conexión, reintentable al recuperar internet.
        #   · ERROR_JSON     → proxy/portal cautivo, reintentable en red directa.
        #   · ERROR_HTTP_429 → rate limit, reintentable con menos workers.
        #   · OVER_QUERY_LIMIT → límite diario, reintentable al día siguiente.
        #   · ERROR_WORKER   → error inesperado en el worker, reintentable.
        # REQUEST_DENIED también se reintenta: si se corrige la clave de API
        # y se vuelve a correr, el script no ignorará esas filas.
        _ESTADOS_DEFINITIVOS = {"OK", "ZERO_RESULTS", "NOT_FOUND"}
        hechas = (
            cache_df[cache_df["status"].isin(_ESTADOS_DEFINITIVOS)]
            [["lat_decimal", "lon_decimal", "radio_m"]]
            .drop_duplicates()
        )
        pendientes = consultas.merge(
            hechas, on=["lat_decimal", "lon_decimal", "radio_m"],
            how="left", indicator=True,
        )
        consultas = (
            pendientes[pendientes["_merge"] == "left_only"]
            .drop(columns=["_merge"])
            .reset_index(drop=True)
        )
        log.info(f"Pendientes: {len(consultas):,} (de {len(filas):,} totales)")
    else:
        # Sin caché previa: inicializa con columnas explícitas para que el merge
        # en construir_inventario() no falle con KeyError si no hay consultas.
        cache_df = pd.DataFrame(columns=_CACHE_COLS)

    if len(consultas) == 0:
        log.info("Todas las consultas ya estaban en caché.")
        return cache_df

    # ── Función que procesa una sola fila (se ejecuta en cada worker) ─────────
    def _procesar(fila_dict: dict) -> dict:
        lat   = fila_dict["lat_decimal"]
        lon   = fila_dict["lon_decimal"]
        radio = int(fila_dict["radio_m"])

        resp   = consultar_metadata(lat, lon, radio, api_key, source)
        status = resp.get("status", "UNKNOWN")

        if status == "OK":
            pano_id  = resp.get("pano_id", "")
            fecha    = resp.get("date", "")
            loc      = resp.get("location", {})
            lat_pano = loc.get("lat")
            lon_pano = loc.get("lng")
            if lat_pano is not None and lon_pano is not None:
                dist = _haversine_m(lat, lon, lat_pano, lon_pano)
                hdg  = _heading(lat_pano, lon_pano, lat, lon)
            else:
                dist = hdg = None
        else:
            pano_id = fecha = lat_pano = lon_pano = dist = hdg = None

        return {
            "lat_decimal": lat,
            "lon_decimal": lon,
            "radio_m":     radio,
            "status":      status,
            "pano_id":     pano_id,
            "distancia_m": dist,
            "fecha":       fecha,
            "lat_pano":    lat_pano,
            "lon_pano":    lon_pano,
            "heading":     hdg,
        }

    # ── Ejecutar consultas en paralelo ────────────────────────────────────────
    # ThreadPoolExecutor es adecuado para I/O-bound (espera de red).
    # El Lock protege la lista `nuevas` y el guardado incremental.
    n_errores = 0
    nuevas    = []
    lock      = threading.Lock()
    filas_list = consultas.to_dict("records")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_procesar, f): f for f in filas_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Metadata API"):
            try:
                resultado = future.result()
            except Exception as exc:
                # Un error inesperado dentro del worker (no capturado por
                # consultar_metadata). Se registra como ERROR_WORKER y el
                # script continúa con las demás consultas.
                # Si esto ocurre con frecuencia, revisa el traceback completo
                # arriba en la consola para identificar la causa raíz.
                fila_orig = futures[future]
                log.warning(
                    f"Worker inesperado en "
                    f"({fila_orig['lat_decimal']},{fila_orig['lon_decimal']},"
                    f"{int(fila_orig['radio_m'])}m): {exc}"
                )
                resultado = {
                    "lat_decimal": fila_orig["lat_decimal"],
                    "lon_decimal": fila_orig["lon_decimal"],
                    "radio_m":     fila_orig["radio_m"],
                    "status":      "ERROR_WORKER",
                    "pano_id":     None, "distancia_m": None, "fecha":    None,
                    "lat_pano":    None, "lon_pano":    None, "heading":  None,
                }
            with lock:
                nuevas.append(resultado)
                if "ERROR" in str(resultado.get("status", "")):
                    n_errores += 1
                # Guardado incremental cada 50 resultados (minimiza pérdidas por interrupción)
                if len(nuevas) % 50 == 0:
                    parcial = pd.concat([cache_df, pd.DataFrame(nuevas)], ignore_index=True)
                    parcial.to_csv(path_cache, index=False)

    if n_errores > 0:
        log.warning(f"{n_errores:,} errores de red/API. Esas filas tienen status ERROR_*.")
        log.warning("Si la mayoria de consultas fallaron, verifica la conexion a internet.")
        log.warning("Los errores se guardan en la cache: puedes retomar sin perder progreso.")

    # Diagnóstico de statuses problemáticos de la API
    n_denied = sum(1 for r in nuevas if r.get("status") == "REQUEST_DENIED")
    n_limit  = sum(1 for r in nuevas if r.get("status") in ("OVER_DAILY_LIMIT", "OVER_QUERY_LIMIT"))
    if n_denied > 0:
        log.error("-" * 62)
        log.error(f"ALERTA: {n_denied} consultas devolvieron REQUEST_DENIED.")
        log.error("La API esta rechazando la clave. Causas posibles:")
        log.error("1. Street View Static API no habilitada en Google Cloud.")
        log.error("   Ve a: console.cloud.google.com → APIs → Street View Static API → Habilitar")
        log.error("2. Facturacion no activa en el proyecto de GCP.")
        log.error("   Ve a: console.cloud.google.com → Facturacion")
        log.error("3. La clave tiene restricciones de IP que bloquean esta maquina.")
        log.error("   Ve a: console.cloud.google.com → Credenciales → edita la clave.")
        log.error("-" * 62)
    if n_limit > 0:
        log.error("-" * 62)
        log.error(f"ALERTA: {n_limit} consultas devolvieron OVER_DAILY_LIMIT / OVER_QUERY_LIMIT.")
        log.error("Dos causas posibles:")
        log.error("  A) Limite diario de la API alcanzado.")
        log.error("     Solucion: espera 24 h y vuelve a correr — la cache reanuda")
        log.error("     desde donde quedo (no repite consultas ya exitosas).")
        log.error("  B) Rate limit por demasiados workers simultaneos.")
        log.error("     Solucion inmediata: edita CONFIG['n_workers'] = 5 (o menos)")
        log.error("     y vuelve a correr — la cache descartara los ERROR_ y")
        log.error("     reintentara solo esas consultas.")
        log.error("-" * 62)

    # ── Guardado final de la caché completa ───────────────────────────────────
    partes      = [cache_df, pd.DataFrame(nuevas)] if nuevas else [cache_df]
    cache_final = pd.concat(partes, ignore_index=True)

    # Deduplica: al reintentar errores, la misma (lat, lon, radio_m) puede tener
    # una fila antigua de error Y una nueva exitosa. Conserva solo la definitiva
    # para evitar filas dobles en el inventario y sobreconteo en el panel.
    _DEF = {"OK", "ZERO_RESULTS", "NOT_FOUND"}
    cache_final["_es_def"] = cache_final["status"].isin(_DEF).astype(int)
    cache_final = (
        cache_final
        .sort_values("_es_def", ascending=False)   # definitivos primero
        .drop_duplicates(subset=["lat_decimal", "lon_decimal", "radio_m"], keep="first")
        .drop(columns=["_es_def"])
        .reset_index(drop=True)
    )

    cache_final.to_csv(path_cache, index=False)
    log.info(f"Caché actualizada: {len(cache_final):,} filas → {path_cache.name}")

    return cache_final


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: INVENTARIO (hogar × ola × radio)
# ──────────────────────────────────────────────────────────────────────────────

def construir_inventario(panel_valido: pd.DataFrame,
                         panel_invalido: pd.DataFrame,
                         cache_api: pd.DataFrame,
                         cfg: dict) -> pd.DataFrame:
    """
    Une los resultados de la API con el panel para construir el inventario final.

    Unidad de análisis: hogar × ola × radio_m.
    · Cada hogar válido tiene R filas (una por radio configurado).
    · Los hogares sin coord_valida_ola=1 tienen R filas con status SIN_COORD_VALIDA.
    · El inventario registra tanto panoramas encontrados como ausencias (ZERO_RESULTS),
      lo que permite evitar re-consultas en futuras ejecuciones.

    La unión usa (lat_decimal, lon_decimal) como clave — sin radio_m —, lo que
    produce N_valido × R filas (cada hogar hereda resultados de todos los radios).
    Hogares con coordenadas idénticas comparten los mismos resultados de la API.
    """
    # ── Merge panel_valido × cache_api ───────────────────────────────────────
    # El merge en (lat_decimal, lon_decimal) produce R filas por hogar (una por radio)
    # porque cache_api tiene R filas por par de coordenadas.
    inv_valido = panel_valido.merge(
        cache_api,
        on=["lat_decimal", "lon_decimal"],
        how="left",
    )

    # ── Agrega hogares inválidos con status SIN_COORD_VALIDA ─────────────────
    partes_inv = []
    for radio in cfg["radios_m"]:
        tmp            = panel_invalido.copy()
        tmp["radio_m"] = radio
        tmp["status"]  = "SIN_COORD_VALIDA"
        partes_inv.append(tmp)

    if partes_inv:
        inv_invalido = pd.concat(partes_inv, ignore_index=True)
        inventario   = pd.concat([inv_valido, inv_invalido], ignore_index=True)
    else:
        inventario   = inv_valido

    cols_sort = [c for c in ["consecutivo", "ola", "radio_m"] if c in inventario.columns]
    if cols_sort:
        inventario = inventario.sort_values(cols_sort).reset_index(drop=True)

    log.info(f"Inventario construido: {len(inventario):,} filas "
             f"({len(panel_valido):,} válidos × {len(cfg['radios_m'])} radios"
             f" + {len(panel_invalido) * len(cfg['radios_m']):,} inválidos)")
    return inventario


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5: PANEL ENRIQUECIDO (hogar × ola)
# ──────────────────────────────────────────────────────────────────────────────

def construir_panel_enriquecido(panel: pd.DataFrame,
                                inventario: pd.DataFrame,
                                cfg: dict) -> pd.DataFrame:
    """
    Agrega variables resumen de GSV al panel original. Una fila por hogar × ola.

    Variables añadidas:
        gsv_n_panos_{R}m:     1 si se encontró panorama en el radio R; 0 si no.
                              PRESENCIA (0/1), no recuento. Ver NOTA METODOLÓGICA.
        gsv_tiene_pano:       1 si algún radio encontró panorama; 0 si ninguno.
        gsv_dist_cercano_m:   distancia al panorama más cercano (menor radio con OK).
        gsv_pano_id_cercano:  pano_id del panorama más cercano.
        gsv_fecha_cercano:    fecha del panorama más cercano.
        gsv_heading:          ángulo pano → hogar del más cercano (vista frontal, script 02).
        gsv_heading_opuesto:  (gsv_heading + 180) % 360 — vista opuesta (segunda imagen).
        gsv_status:           "OK" | "ZERO_RESULTS" | "ERROR_*" | "SIN_COORD_VALIDA".
    """
    radios = sorted(cfg["radios_m"])
    clave  = [c for c in ["consecutivo", "ola"] if c in inventario.columns]

    resumen_filas = []

    for vals, grp in inventario.groupby(clave):
        fila = dict(zip(clave, vals if isinstance(vals, tuple) else (vals,)))

        # ── Variables por radio ───────────────────────────────────────────────
        primer_ok = None
        for radio in radios:
            sub  = grp[grp["radio_m"] == radio]
            ok   = sub[sub["status"] == "OK"]
            tiene = 1 if len(ok) > 0 else 0
            fila[f"gsv_n_panos_{radio}m"] = tiene

            if tiene == 1 and primer_ok is None:
                primer_ok = ok.iloc[0]

        # ── Variables del panorama más cercano (primer radio con OK) ──────────
        # gsv_heading y gsv_heading_opuesto son los ángulos de cámara para
        # la descarga de imágenes en script 02 (ver sección OPCIONES DE DESCARGA):
        #   gsv_heading         → Escenario 1: 1 imagen/hogar (vista frontal)
        #   gsv_heading_opuesto → Escenario 2: +1 imagen/hogar (vista opuesta)
        if primer_ok is not None:
            hdg = primer_ok.get("heading")
            # pd.notna() en vez de "is not None": la caché puede tener NaN
            # (float) cuando heading no se pudo calcular, y NaN is not None
            # es True — causaría (NaN + 180) % 360 = NaN sin aviso.
            hdg = hdg if pd.notna(hdg) else None
            fila["gsv_tiene_pano"]        = 1
            fila["gsv_dist_cercano_m"]    = primer_ok.get("distancia_m")
            fila["gsv_pano_id_cercano"]   = primer_ok.get("pano_id")
            fila["gsv_fecha_cercano"]     = primer_ok.get("fecha")
            fila["gsv_heading"]           = hdg                                        # Escenario 1
            fila["gsv_heading_opuesto"]   = round((hdg + 180) % 360, 1) if hdg is not None else None  # Escenario 2
            fila["gsv_status"]            = "OK"
        else:
            fila["gsv_tiene_pano"]        = 0
            fila["gsv_dist_cercano_m"]    = None
            fila["gsv_pano_id_cercano"]   = None
            fila["gsv_fecha_cercano"]     = None
            fila["gsv_heading"]           = None   # sin panorama → no hay imagen para descargar
            fila["gsv_heading_opuesto"]   = None   # sin panorama → no hay imagen para descargar
            sub_mayor = grp[grp["radio_m"] == radios[-1]]
            fila["gsv_status"] = (
                sub_mayor["status"].iloc[0] if len(sub_mayor) > 0 else "UNKNOWN"
            )

        resumen_filas.append(fila)

    # Guarda si no hay grupos (inventario vacío — todas las coords inválidas).
    # pd.DataFrame([]) no tiene columnas y el merge fallaría con KeyError.
    if not resumen_filas:
        log.warning("construir_panel_enriquecido: inventario sin grupos — "
                    "todas las coordenadas son inválidas. Se retorna el panel sin vars GSV.")
        return panel

    resumen = pd.DataFrame(resumen_filas)
    panel_enriquecido = panel.merge(resumen, on=clave, how="left")
    return panel_enriquecido


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDAD DE FORMATO TABULAR
# ──────────────────────────────────────────────────────────────────────────────

def _tabla_txt(headers, rows, indent="  "):
    """
    Formatea una lista de filas como tabla de texto alineada a columnas.

    Parámetros:
        headers : lista de strings con los encabezados de columna.
        rows    : lista de listas; cada sublista es una fila de la tabla.
        indent  : string de sangría que se antepone a cada línea.

    Retorna:
        lista de strings (una línea por elemento) lista para extender `lines`.
    """
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep  = "  ".join("-" * w for w in widths)
    head = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    result = [indent + head, indent + sep]
    for row in rows:
        result.append(
            indent + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5.5: ASIGNACIÓN DE FOTOGRAFÍAS ÚNICAS
# ──────────────────────────────────────────────────────────────────────────────

def asignar_fotos_unicas(inventario: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa las reglas de unicidad de fotografías en dos dimensiones.

    Regla 2.1 — Unicidad entre radios:
        Para un mismo (hogar, ola, pano_id) la foto se asigna exclusivamente
        al radio más pequeño donde aparece. Los radios superiores no la cuentan.

    Regla 2.2 — Unicidad entre olas:
        Para un mismo (hogar, pano_id) la foto se asigna exclusivamente a la
        primera ola elegible según las ventanas temporales (VENTANAS_OLA).
        Las olas posteriores no la vuelven a contabilizar.

    Orden de prioridad (determinístico):
        1. Primera ola elegible según el año de captura de la foto.
        2. Dentro de esa ola, el radio más pequeño donde el pano_id aparece.

    Ventanas temporales:
        Ola 2010 : año foto ≤ 2010
        Ola 2013 : año foto 2011 – 2013
        Ola 2016 : año foto 2014 – 2016

    Parámetros:
        inventario : DataFrame hogar × ola × radio con columnas status, pano_id,
                     fecha, radio_m, ola. Debe tener 'consecutivo' o coordenadas.

    Retorna:
        DataFrame de asignaciones únicas. Columna 'foto_anio' con el año de captura.
        Cada pano_id aparece como máximo una vez por hogar en todo el panel.
    """
    id_col = "consecutivo" if "consecutivo" in inventario.columns else None

    # Filtrar a filas con panorama encontrado (OK) y pano_id no nulo
    df = inventario[
        (inventario["status"] == "OK") &
        inventario["pano_id"].notna()
    ].copy()

    if df.empty:
        return df

    # Extraer año de la foto y normalizar ola a entero para comparar con VENTANAS_OLA
    df["foto_anio"] = df["fecha"].apply(_anio_foto)
    df["_ola_int"]  = pd.to_numeric(df["ola"], errors="coerce")

    # Mantener solo filas cuyo año de foto es elegible para la ola del registro
    mascara = df.apply(
        lambda r: _elegible_para_ola(r["foto_anio"], r["_ola_int"]), axis=1
    )
    df = df[mascara].drop(columns=["_ola_int"])

    if df.empty:
        return df

    # Clave de identificación del hogar
    if id_col:
        claves_hogar = [id_col]
    else:
        # Fallback: coordenadas como proxy del hogar
        df["_hogar_key"] = (
            df["lat_decimal"].astype(str) + "_" + df["lon_decimal"].astype(str)
        )
        claves_hogar = ["_hogar_key"]

    # ── Regla 2.1: para (hogar, ola, pano_id) conservar el radio mínimo ──────
    df = (
        df.sort_values(claves_hogar + ["ola", "pano_id", "radio_m"])
        .drop_duplicates(subset=claves_hogar + ["ola", "pano_id"], keep="first")
    )

    # ── Regla 2.2: para (hogar, pano_id) conservar la ola mínima elegible ────
    df = (
        df.sort_values(claves_hogar + ["pano_id", "ola"])
        .drop_duplicates(subset=claves_hogar + ["pano_id"], keep="first")
    )

    return df.reset_index(drop=True)


def enriquecer_inventario(inventario: pd.DataFrame,
                          fotos_unicas: pd.DataFrame,
                          panel_valido: pd.DataFrame) -> pd.DataFrame:
    """
    Añade tres columnas al inventario para filtrar directamente las fotografías
    únicas de hogares con coordenadas válidas presentes en las tres olas.

    Columnas añadidas:
        foto_anio           : año de captura extraído de 'fecha' (int o NaN).
                              Facilita filtrar por ventana temporal sin cálculos
                              adicionales en Stata o en pandas.

        en_panel_balanceado : 1 si el hogar (consecutivo) tiene coord_valida_ola=1
                              en las tres olas (2010, 2013, 2016); 0 en caso contrario.
                              Denominador de todos los indicadores del panel balanceado.

        es_foto_unica       : 1 si esta fila (hogar × ola × radio_m × pano_id) es
                              la asignación canónica de ese pano_id para ese hogar:
                              primera ola elegible + radio mínimo (Reglas 2.1 y 2.2).
                              0 para todas las demás filas (duplicados entre radios
                              o entre olas, filas sin status OK, o sin pano_id).

    Selección del subconjunto de interés:
        inventario[
            (inventario["en_panel_balanceado"] == 1) &
            (inventario["es_foto_unica"] == 1)
        ]
    Ese filtro devuelve exactamente las fotografías únicas de los hogares
    balanceados, sin duplicaciones entre radios ni entre olas.
    """
    inv    = inventario.copy()
    col_id = "consecutivo"

    # ── foto_anio ─────────────────────────────────────────────────────────────
    if "fecha" in inv.columns:
        inv["foto_anio"] = inv["fecha"].apply(_anio_foto)
    else:
        inv["foto_anio"] = None

    # ── en_panel_balanceado ───────────────────────────────────────────────────
    # Un hogar está en el panel balanceado si tiene coord_valida_ola=1
    # en las tres olas objetivo: 2010, 2013 y 2016.
    OLAS_PANEL    = {2010, 2013, 2016}
    hogares_3olas = set()

    if col_id in panel_valido.columns and "ola" in panel_valido.columns:
        pv = panel_valido.copy()
        pv["_ola_int"] = pd.to_numeric(pv["ola"], errors="coerce")
        pv = pv[pv["_ola_int"].isin(OLAS_PANEL)]
        conteo        = pv.groupby(col_id)["_ola_int"].nunique()
        hogares_3olas = set(conteo[conteo == 3].index)

    if col_id in inv.columns:
        inv["en_panel_balanceado"] = inv[col_id].isin(hogares_3olas).astype(int)
    else:
        inv["en_panel_balanceado"] = 0

    # ── es_foto_unica ─────────────────────────────────────────────────────────
    # Marca con 1 la fila exacta que asignar_fotos_unicas() seleccionó como
    # asignación canónica: primera ola elegible y radio mínimo.
    # La clave de unión es (hogar, ola, radio_m, pano_id).
    inv["es_foto_unica"] = 0

    merge_cols = [c for c in [col_id, "ola", "radio_m", "pano_id"]
                  if c in inv.columns and c in fotos_unicas.columns]

    if merge_cols and not fotos_unicas.empty:
        fu_keys = fotos_unicas[merge_cols].drop_duplicates().copy()
        fu_keys["_unica"] = 1
        inv = inv.merge(fu_keys, on=merge_cols, how="left")
        inv["es_foto_unica"] = inv.pop("_unica").fillna(0).astype(int)

    return inv


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: REPORTE DESCRIPTIVO
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(panel_enriquecido: pd.DataFrame,
                    fotos_unicas: pd.DataFrame,
                    cfg: dict) -> str:
    """
    Genera el texto del reporte descriptivo de cobertura GSV.

    Secciones:
    1. Resumen general
    2. Cobertura por radio de búsqueda
    3. Cobertura por ola
    4. Distribución de distancia al panorama más cercano
    5. Nota metodológica sobre la limitación de la API
    6. Criterios de asignación de fotografías únicas  [NUEVO]
    7. Estadísticas descriptivas por ola y zona — Panel balanceado  [NUEVO]
       7.1 Cobertura de hogares según disponibilidad de fotografías
       7.2 Cobertura de fotografías únicas (distribución y temporal)
    8. Análisis de costos para descarga de imágenes (script 02)
    """
    sep   = "=" * 65
    pe    = panel_enriquecido
    lines = [sep, "REPORTE DE COBERTURA — GOOGLE STREET VIEW METADATA API",
             "Panel ELCA · Olas 2010, 2013, 2016", sep, ""]

    n_total   = len(pe)
    n_validas = int((pe["coord_valida_ola"] == 1).sum())
    n_inval   = n_total - n_validas

    # ── 1. Resumen general ────────────────────────────────────────────────────
    lines += [
        "── 1. RESUMEN GENERAL ────────────────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Cuántas observaciones del panel tienen coordenadas aptas para",
        "  consultar la API y, de ellas, cuántas obtuvieron al menos un",
        "  panorama en alguno de los radios configurados.",
        "",
        f"  Observaciones totales (hogar × ola): {n_total:,}",
        f"  Con coord_valida_ola=1 (consultadas): {n_validas:,}",
        f"  Sin coordenadas válidas (no consultadas): {n_inval:,}",
    ]

    if "gsv_tiene_pano" in pe.columns:
        val   = pe[pe["coord_valida_ola"] == 1]
        n_con = int((val["gsv_tiene_pano"] == 1).sum())
        n_sin = int((val["gsv_tiene_pano"] == 0).sum())
        pct_c = 100 * n_con / n_validas if n_validas > 0 else 0
        lines += [
            "",
            f"  Con panorama en algún radio: {n_con:,}  ({pct_c:.1f}% de válidas)",
            f"  Sin panorama en ningún radio: {n_sin:,}  ({100-pct_c:.1f}% de válidas)",
            "",
            "  INTERPRETACIÓN",
            "  El porcentaje de válidas con panorama indica la cobertura efectiva",
            "  de GSV para el panel. Una cobertura alta (>80%) permite usar las",
            "  imágenes como variable de entorno para la mayoría de las olas.",
            "  Los hogares sin panorama quedan excluidos del análisis visual.",
        ]

    # ── 2. Cobertura por radio ────────────────────────────────────────────────
    lines += [
        "", "── 2. COBERTURA POR RADIO DE BÚSQUEDA ──────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Para cada radio configurado, cuántos hogares válidos tienen al menos",
        "  un panorama dentro de esa distancia (indicador de presencia 0/1).",
        "  Ver Sección 5 (Nota metodológica) para la limitación de este indicador.",
        "",
        "  LÓGICA",
        "  La variable gsv_n_panos_{R}m vale 1 si la API devolvió status=OK",
        "  para ese radio, y 0 si devolvió ZERO_RESULTS (sin panorama en el radio).",
        "  Radios más grandes capturan panoramas más lejanos pero el panorama",
        "  retornado puede estar en una calle diferente a la del hogar.",
    ]
    for radio in sorted(cfg["radios_m"]):
        col = f"gsv_n_panos_{radio}m"
        if col in pe.columns:
            n_ok = int((pe[col] == 1).sum())
            pct  = 100 * n_ok / n_validas if n_validas > 0 else 0
            lines.append(f"  Radio {radio:>3}m: {n_ok:,} hogares ({pct:.1f}% de válidos)")

    # ── 3. Cobertura por ola ──────────────────────────────────────────────────
    if "ola" in pe.columns and "gsv_tiene_pano" in pe.columns:
        lines += [
            "", "── 3. COBERTURA POR OLA ─────────────────────────────────",
            "",
            "  QUÉ MIDE ESTA SECCIÓN",
            "  Desagregación de la cobertura por ola de la encuesta. Permite",
            "  identificar si la cobertura de GSV varía entre 2010, 2013 y 2016,",
            "  lo que podría reflejar expansión histórica de Street View en Colombia.",
            "",
            "  LÓGICA",
            "  Para cada ola se cuentan hogares válidos (coord_valida_ola=1) y,",
            "  de ellos, cuántos obtuvieron al menos un panorama en cualquier radio.",
            "  Un aumento de cobertura en olas más recientes es esperable dado que",
            "  Google amplió gradualmente su cobertura en Colombia desde ~2012.",
        ]
        for ola, grp in pe.groupby("ola"):
            n_g     = len(grp)
            n_g_val = int((grp["coord_valida_ola"] == 1).sum())
            n_g_ok  = int((grp["gsv_tiene_pano"] == 1).sum())
            pct_g   = 100 * n_g_ok / n_g_val if n_g_val > 0 else 0
            lines.append(
                f"  Ola {ola}: {n_g:,} obs / {n_g_val:,} válidas / "
                f"{n_g_ok:,} con pano ({pct_g:.1f}%)"
            )

    # ── 4. Distribución de distancia ──────────────────────────────────────────
    if "gsv_dist_cercano_m" in pe.columns:
        dist = pe["gsv_dist_cercano_m"].dropna()
        if len(dist) > 0:
            lines += [
                "", "── 4. DISTANCIA AL PANORAMA MÁS CERCANO ────────────────",
                "",
                "  QUÉ MIDE ESTA SECCIÓN",
                "  Distribución de la distancia (en metros) entre la coordenada",
                "  del hogar y el panorama más cercano encontrado (primer radio",
                "  con status=OK). Se calcula con la fórmula de Haversine.",
                "",
                "  INTERPRETACIÓN",
                "  Distancias cortas (<50 m) indican que el panorama está frente",
                "  al hogar o muy cerca, lo que maximiza la calidad del entorno",
                "  capturado. Distancias largas (>150 m) implican que la imagen",
                "  puede corresponder a un segmento de calle diferente al del hogar.",
                "  El percentil 90 señala el umbral por encima del cual una fracción",
                "  pequeña de hogares tiene panoramas muy alejados; conviene revisar",
                "  esos casos manualmente antes de incluirlos en el análisis.",
                "",
                f"  N con distancia disponible: {len(dist):,}",
                f"  Mínima:   {dist.min():.1f} m",
                f"  Mediana:  {dist.median():.1f} m",
                f"  Media:    {dist.mean():.1f} m",
                f"  P75:      {dist.quantile(0.75):.1f} m",
                f"  P90:      {dist.quantile(0.90):.1f} m",
                f"  Máxima:   {dist.max():.1f} m",
            ]

    # ── 5. Nota metodológica ──────────────────────────────────────────────────
    lines += [
        "", "── 5. NOTA METODOLÓGICA — LIMITACIÓN DE LA API ─────────",
        "  La Google Street View Metadata API devuelve UN SOLO panorama",
        "  por consulta (el más cercano dentro del radio dado). No es posible",
        "  recuperar todos los panoramas disponibles en un área con este endpoint.",
        "",
        "  Las variables gsv_n_panos_{R}m son indicadores de PRESENCIA (0/1):",
        "  · 1 = existe al menos un panorama dentro de R metros",
        "  · 0 = no se encontró ningún panorama dentro de R metros",
        "  No equivalen a un recuento de panoramas únicos.",
        "",
        "  El inventario (inventario_panos.csv) tiene una fila por",
        "  (hogar × ola × radio_m). Filas del mismo hogar con pano_id idéntico",
        "  en distintos radios corresponden al mismo panorama físico.",
        "",
        f"  Radios consultados: {sorted(cfg['radios_m'])} metros",
        f"  Fuente:             {cfg['source']}",
    ]

    # ── 6. Criterios de asignación de fotografías únicas ─────────────────────
    lines += [
        "", "── 6. CRITERIOS DE ASIGNACIÓN DE FOTOGRAFÍAS ÚNICAS ────",
        "",
        "  VENTANAS TEMPORALES",
        "  Cada fotografía se asigna exclusivamente a la primera ola para la cual",
        "  es elegible según su año de captura:",
        "    Ola 2010 : fotografías con año ≤ 2010",
        "    Ola 2013 : fotografías con año 2011 a 2013",
        "    Ola 2016 : fotografías con año 2014 a 2016",
        "",
        "  REGLAS DE UNICIDAD",
        "  2.1 Entre radios: cuando un mismo pano_id aparece en varios radios",
        "      para el mismo hogar y ola, se asigna al radio más pequeño.",
        "      Los radios superiores no lo contabilizan.",
        "  2.2 Entre olas: un pano_id se contabiliza una única vez por hogar",
        "      en todo el panel, en la primera ola elegible según su año.",
        "      Las olas posteriores no lo repiten.",
        "",
        "  PRIORIDAD DE ASIGNACIÓN (determinística)",
        "  1. Primera ola elegible según el año de captura de la foto.",
        "  2. Dentro de esa ola, el radio más pequeño donde el pano_id aparece.",
        "",
        "  RESULTADO: cada fotografía (pano_id) aparece como máximo una vez",
        "  por hogar en el conjunto completo de estadísticas del panel.",
        f"  Radios evaluados: {sorted(cfg['radios_m'])} metros",
    ]

    # ── 7. Estadísticas descriptivas por ola y zona — Panel balanceado ────────
    OLAS_PANEL = [2010, 2013, 2016]
    col_id   = "consecutivo"
    col_zona = cfg.get("col_zona", "zona")

    # Identificar hogares presentes en las 3 olas con coordenadas válidas
    hogares_3olas   = set()
    n_hogares_3olas = 0
    pe_bal          = pd.DataFrame()

    if col_id in pe.columns:
        val_bal = pe[pe["coord_valida_ola"] == 1].copy()
        val_bal["_ola_int"] = pd.to_numeric(val_bal["ola"], errors="coerce")
        val_bal = val_bal[val_bal["_ola_int"].isin(OLAS_PANEL)]
        conteo_olas   = val_bal.groupby(col_id)["_ola_int"].nunique()
        hogares_3olas = set(conteo_olas[conteo_olas == 3].index)
        n_hogares_3olas = len(hogares_3olas)
        if hogares_3olas:
            pe_bal = pe[pe[col_id].isin(hogares_3olas)].copy()
            pe_bal["_ola_int"] = pd.to_numeric(pe_bal["ola"], errors="coerce")

    tiene_zona = col_zona in pe.columns and bool(hogares_3olas)

    # Filtrar fotos_unicas al panel balanceado y a las olas objetivo
    fu_bal = pd.DataFrame()
    if not fotos_unicas.empty and col_id in fotos_unicas.columns and hogares_3olas:
        fu_bal = fotos_unicas[fotos_unicas[col_id].isin(hogares_3olas)].copy()
        fu_bal["_ola_int"] = pd.to_numeric(fu_bal["ola"], errors="coerce")
        fu_bal = fu_bal[fu_bal["_ola_int"].isin(OLAS_PANEL)]

    lines += [
        "", "── 7. ESTADÍSTICAS DESCRIPTIVAS POR OLA Y ZONA — PANEL BALANCEADO ─",
        "",
        "  Universo: hogares con coordenadas válidas (coord_valida_ola=1)",
        "  presentes en las tres olas del panel (2010, 2013, 2016).",
        "  Solo se contabilizan fotografías únicas según criterios de Sección 6.",
        "",
    ]

    if not hogares_3olas:
        lines.append(
            "  (Columna 'consecutivo' no encontrada o sin hogares en las 3 olas — "
            "sección omitida)"
        )
    else:
        lines.append(f"  Hogares en el panel balanceado: {n_hogares_3olas:,}")
        if not tiene_zona:
            lines.append(
                f"  (Columna de zona '{col_zona}' no encontrada — "
                f"desagregación por zona omitida)"
            )
        lines.append("")

        # ── 7.1: Cobertura de hogares por radio ──────────────────────────────
        lines += [
            "  7.1 COBERTURA DE HOGARES SEGÚN DISPONIBILIDAD DE FOTOGRAFÍAS",
            "",
            "  Número y porcentaje de hogares balanceados con al menos una",
            "  fotografía única por combinación ola × zona × radio.",
            "  Denominador: hogares balanceados en esa combinación ola × zona.",
            "",
        ]

        for radio in sorted(cfg["radios_m"]):
            lines.append(f"  Radio {radio}m:")

            if fu_bal.empty:
                lines.append("    (sin fotografías únicas disponibles)")
                lines.append("")
                continue

            fu_radio   = fu_bal[fu_bal["radio_m"] == radio]
            rows_tabla = []

            for ola in OLAS_PANEL:
                pe_ola = pe_bal[pe_bal["_ola_int"] == ola]
                fu_ola = fu_radio[fu_radio["_ola_int"] == ola]

                if tiene_zona:
                    zonas = sorted(pe_ola[col_zona].dropna().unique())
                    for zona in zonas:
                        pe_oz    = pe_ola[pe_ola[col_zona] == zona]
                        hog_oz   = set(pe_oz[col_id].unique())
                        n_denom  = len(hog_oz)
                        hog_foto = set(fu_ola[col_id].unique())
                        n_con    = len(hog_foto & hog_oz)
                        pct      = 100 * n_con / n_denom if n_denom > 0 else 0
                        rows_tabla.append(
                            [str(ola), str(zona), f"{n_con:,}", f"{pct:.1f}%",
                             f"{n_denom:,}"]
                        )
                else:
                    hog_ola  = set(pe_ola[col_id].unique())
                    n_denom  = len(hog_ola)
                    hog_foto = set(fu_ola[col_id].unique())
                    n_con    = len(hog_foto & hog_ola)
                    pct      = 100 * n_con / n_denom if n_denom > 0 else 0
                    rows_tabla.append(
                        [str(ola), f"{n_con:,}", f"{pct:.1f}%", f"{n_denom:,}"]
                    )

            if rows_tabla:
                hdrs = (["Ola", "Zona", "N con foto", "%", "N total"]
                        if tiene_zona else ["Ola", "N con foto", "%", "N total"])
                lines += _tabla_txt(hdrs, rows_tabla, indent="    ")
            lines.append("")

        # Cobertura acumulada en cualquier radio
        lines += [
            "  En cualquier radio (cobertura acumulada por ola × zona):",
            "  Denominador: hogares balanceados en esa combinación ola × zona.",
            "",
        ]
        rows_acum = []
        for ola in OLAS_PANEL:
            pe_ola = pe_bal[pe_bal["_ola_int"] == ola]
            fu_ola = (fu_bal[fu_bal["_ola_int"] == ola]
                      if not fu_bal.empty else pd.DataFrame())

            if tiene_zona:
                zonas = sorted(pe_ola[col_zona].dropna().unique())
                for zona in zonas:
                    pe_oz    = pe_ola[pe_ola[col_zona] == zona]
                    hog_oz   = set(pe_oz[col_id].unique())
                    n_denom  = len(hog_oz)
                    hog_foto = set(fu_ola[col_id].unique()) if not fu_ola.empty else set()
                    n_con    = len(hog_foto & hog_oz)
                    pct      = 100 * n_con / n_denom if n_denom > 0 else 0
                    rows_acum.append(
                        [str(ola), str(zona), f"{n_con:,}", f"{pct:.1f}%", f"{n_denom:,}"]
                    )
            else:
                hog_ola  = set(pe_ola[col_id].unique())
                n_denom  = len(hog_ola)
                hog_foto = set(fu_ola[col_id].unique()) if not fu_ola.empty else set()
                n_con    = len(hog_foto & hog_ola)
                pct      = 100 * n_con / n_denom if n_denom > 0 else 0
                rows_acum.append(
                    [str(ola), f"{n_con:,}", f"{pct:.1f}%", f"{n_denom:,}"]
                )

        if rows_acum:
            hdrs_a = (["Ola", "Zona", "N con foto (cualquier radio)", "%", "N total"]
                      if tiene_zona
                      else ["Ola", "N con foto (cualquier radio)", "%", "N total"])
            lines += _tabla_txt(hdrs_a, rows_acum, indent="    ")
        lines.append("")

        # ── 7.2: Cobertura de fotografías únicas ─────────────────────────────
        lines += [
            "  7.2 COBERTURA DE FOTOGRAFÍAS ÚNICAS",
            "",
            "  Denominador: total de fotografías únicas en el panel balanceado.",
            "",
        ]

        if fu_bal.empty:
            lines.append("    (sin fotografías únicas disponibles)")
            lines.append("")
        else:
            total_fotos_bal = len(fu_bal)
            lines.append(
                f"  Total fotografías únicas (panel balanceado): {total_fotos_bal:,}"
            )
            lines.append("")

            # Distribución por ola y zona
            lines.append("  Distribución por ola y zona:")
            lines.append("")
            rows_dist = []
            for ola in OLAS_PANEL:
                fu_ola = fu_bal[fu_bal["_ola_int"] == ola]
                if tiene_zona:
                    pe_ola = pe_bal[pe_bal["_ola_int"] == ola]
                    zonas  = sorted(pe_ola[col_zona].dropna().unique())
                    for zona in zonas:
                        pe_oz  = pe_ola[pe_ola[col_zona] == zona]
                        hog_oz = set(pe_oz[col_id].unique())
                        fu_oz  = fu_ola[fu_ola[col_id].isin(hog_oz)]
                        n      = len(fu_oz)
                        pct    = 100 * n / total_fotos_bal if total_fotos_bal > 0 else 0
                        rows_dist.append(
                            [str(ola), str(zona), f"{n:,}", f"{pct:.1f}%"]
                        )
                else:
                    n   = len(fu_ola)
                    pct = 100 * n / total_fotos_bal if total_fotos_bal > 0 else 0
                    rows_dist.append([str(ola), f"{n:,}", f"{pct:.1f}%"])

            if rows_dist:
                hdrs_d = (["Ola", "Zona", "N fotos únicas", "% del total"]
                          if tiene_zona else ["Ola", "N fotos únicas", "% del total"])
                lines += _tabla_txt(hdrs_d, rows_dist, indent="    ")
            lines.append("")

            # Distribución temporal de fotografías
            lines += [
                "  Distribución temporal de fotografías únicas:",
                "  Denominador: total de fotografías únicas en el panel balanceado.",
                "",
            ]
            if "foto_anio" in fu_bal.columns:
                anios = fu_bal["foto_anio"].dropna()
                if len(anios) > 0:
                    conteo_anio = anios.astype(int).value_counts().sort_index()
                    total_a     = len(anios)
                    acumulado   = 0.0
                    rows_temp   = []
                    for anio_val, n_val in conteo_anio.items():
                        pct_v      = 100 * n_val / total_a
                        acumulado += pct_v
                        rows_temp.append(
                            [str(anio_val), f"{n_val:,}",
                             f"{pct_v:.1f}%", f"{acumulado:.1f}%"]
                        )
                    lines += _tabla_txt(
                        ["Año", "N fotos únicas", "% del total", "% acumulado"],
                        rows_temp, indent="    "
                    )
            lines.append("")

    # ── 8. Análisis de costos para descarga de imágenes ───────────────────────
    # Precios vigentes Street View Static API (junio 2025):
    #   · Metadata API  → gratuita (sin costo por consulta).
    #   · Static API    → USD 7.00 por cada 1,000 imágenes descargadas.
    #   · Crédito mensual Google Maps Platform: USD 200 (≈ 28,571 imágenes gratis).
    PRECIO_POR_MIL    = 7.00    # USD por 1,000 requests a la Static API
    CREDITO_USD       = 200.00  # crédito mensual disponible
    IMAGENES_GRATIS   = int(CREDITO_USD / PRECIO_POR_MIL * 1_000)  # ≈ 28,571

    lines += [
        "", "── 8. ANÁLISIS DE COSTOS — DESCARGA DE IMÁGENES (SCRIPT 02) ─",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Estima el costo de descargar las imágenes de los panoramas encontrados",
        "  usando la Street View Static API (script 02), y determina si la",
        "  cantidad de fotos cabe dentro del crédito de USD 200 disponible.",
        "",
        "  ESTRUCTURA DE PRECIOS (Street View Static API, junio 2025)",
        "  · Metadata API (este script):  GRATUITA — no genera costo.",
        f"  · Static API (imágenes):       USD {PRECIO_POR_MIL:.2f} por cada 1,000 requests.",
        f"  · Crédito mensual Google Maps: USD {CREDITO_USD:.2f}",
        f"  · Imágenes gratis con el crédito: {IMAGENES_GRATIS:,} imágenes/mes.",
        "",
        "  LÓGICA DEL CÁLCULO",
        "  Por cada panorama encontrado (gsv_tiene_pano=1) el script 02 puede",
        "  descargar 1 o 2 imágenes por panorama:",
        "    · 1 imagen: vista frontal hacia el hogar (heading).",
        "    · 2 imágenes: vista frontal + vista opuesta (heading + 180°).",
        "  Cada imagen es una llamada independiente a la Static API.",
        "  El cálculo se presenta para ambos escenarios.",
    ]

    if "gsv_tiene_pano" in pe.columns:
        n_panos = int((pe["gsv_tiene_pano"] == 1).sum())

        for n_imgs, escenario in [(1, "1 imagen/panorama (solo frontal)"),
                                  (2, "2 imágenes/panorama (frontal + opuesta)")]:
            total_imgs   = n_panos * n_imgs
            imgs_pagas   = max(0, total_imgs - IMAGENES_GRATIS)
            costo_extra  = imgs_pagas / 1_000 * PRECIO_POR_MIL
            dentro_credito = total_imgs <= IMAGENES_GRATIS

            lines += [
                "",
                f"  Escenario {n_imgs}: {escenario}",
                f"    Panoramas únicos con foto disponible: {n_panos:,}",
                f"    Total de imágenes a descargar:        {total_imgs:,}",
                f"    Imágenes cubiertas por el crédito:    {min(total_imgs, IMAGENES_GRATIS):,}",
                f"    Imágenes fuera del crédito (pagas):   {imgs_pagas:,}",
            ]
            if dentro_credito:
                lines.append(
                    f"    Costo adicional estimado:             USD 0.00 "
                    f"— dentro del crédito de USD {CREDITO_USD:.0f} ✓"
                )
            else:
                lines.append(
                    f"    Costo adicional estimado:             USD {costo_extra:.2f} "
                    f"— SUPERA el crédito de USD {CREDITO_USD:.0f}"
                )

        lines += [
            "",
            "  INTERPRETACIÓN",
            "  Si el costo adicional es USD 0.00 en ambos escenarios, la descarga",
            "  completa de imágenes cabe dentro del crédito mensual gratuito de",
            f"  USD {CREDITO_USD:.0f} y no requiere ningún pago adicional.",
            "  Si algún escenario supera el crédito, se puede reducir el volumen",
            "  de descarga priorizando los panoramas más cercanos (menor",
            "  gsv_dist_cercano_m) o descargando solo la vista frontal.",
            "",
            "  NOTAS IMPORTANTES",
            "  · El crédito de USD 200 se reinicia cada mes calendario.",
            "  · La descarga puede distribuirse en varios meses para mantenerse",
            "    dentro del crédito gratuito y no incurrir en costos adicionales.",
            "  · Los precios pueden cambiar; verifica en:",
            "    developers.google.com/maps/documentation/streetview/usage-and-billing",
        ]
    else:
        lines.append("  (variable gsv_tiene_pano no disponible — ejecuta el pipeline completo)")

    lines += ["", sep]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Orquesta el pipeline completo de consulta a la Metadata API de GSV.

    1. Carga el panel (output de script 00).
    2. Ejecuta consultas con deduplicación y reanudación.
    3. Construye el inventario (hogar × ola × radio).
    4. Construye el panel enriquecido (hogar × ola + vars GSV).
    5. Genera el reporte descriptivo.
    6. Exporta los tres outputs.
    """
    log.info("=" * 62)
    log.info("  01_analisis_cobertura_gsv.py")
    log.info("=" * 62)

    if not CONFIG["api_key"]:
        log.error("-" * 62)
        log.error("ERROR: La API key de Google Street View no esta configurada.")
        log.error("OPCION 1 — define la variable de entorno antes de correr:")
        log.error("  Windows (cmd) : set GSV_API_KEY=tu_clave_aqui")
        log.error("  Windows (PS)  : $env:GSV_API_KEY='tu_clave_aqui'")
        log.error("  Mac/Linux     : export GSV_API_KEY='tu_clave_aqui'")
        log.error("OPCION 2 — edita CONFIG['api_key'] en este script con la clave directa.")
        log.error("La clave debe empezar con 'AIza...' y tener Street View Static API habilitada.")
        log.error("-" * 62)
        sys.exit(1)

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Paso 1: carga del panel ───────────────────────────────────────────
        log.info("\n── Paso 1: Carga del panel ──────────────────────────────────")
        panel_valido, panel_invalido = cargar_panel(CONFIG)
        panel_completo = pd.concat([panel_valido, panel_invalido], ignore_index=True)

        # ── Paso 2: consultas a la API ────────────────────────────────────────
        log.info("\n── Paso 2: Consultas a la Metadata API ──────────────────────")
        log.info("  Endpoint gratuito — no descarga imágenes.")
        n_consultas = (
            len(panel_valido[["lat_decimal", "lon_decimal"]].drop_duplicates())
            * len(CONFIG["radios_m"])
        )
        log.info(f"  Consultas únicas estimadas: {n_consultas:,}")
        # Estimado conservador: ~150 ms de latencia por request, n_workers en paralelo
        t_min = n_consultas * 0.15 / CONFIG["n_workers"] / 60
        log.info(f"  Workers paralelos: {CONFIG['n_workers']}  — tiempo estimado: ~{t_min:.0f} min")

        path_cache = out_dir / "resultados_api_cache.csv"
        cache_api  = ejecutar_consultas(panel_valido, CONFIG, path_cache)

        # ── Paso 3: inventario (sin exportar aún) ────────────────────────────
        log.info("\n── Paso 3: Inventario (hogar × ola × radio) ─────────────────")
        inventario = construir_inventario(panel_valido, panel_invalido, cache_api, CONFIG)
        log.info(f"  Inventario construido: {len(inventario):,} filas (se exporta en Paso 4.5)")

        # ── Paso 4: panel enriquecido ─────────────────────────────────────────
        log.info("\n── Paso 4: Panel enriquecido (hogar × ola) ──────────────────")
        panel_enriquecido = construir_panel_enriquecido(panel_completo, inventario, CONFIG)
        path_panel        = out_dir / "panel_enriquecido_gsv.csv"
        panel_enriquecido.to_csv(path_panel, index=False)
        log.info(f"  panel_enriquecido_gsv.csv: {len(panel_enriquecido):,} filas")

        # ── Paso 4.5: fotografías únicas + enriquecimiento del inventario ──────
        # asignar_fotos_unicas aplica las reglas 2.1 y 2.2:
        #   · Regla 2.1 (entre radios): un pano_id → radio mínimo por (hogar, ola).
        #   · Regla 2.2 (entre olas):   un pano_id → primera ola elegible por hogar.
        # enriquecer_inventario añade foto_anio, en_panel_balanceado y es_foto_unica
        # al inventario para que el filtro sea directo en Stata o pandas.
        log.info("\n── Paso 4.5: Fotografías únicas + enriquecimiento del inventario ─")
        fotos_unicas = asignar_fotos_unicas(inventario)
        inventario   = enriquecer_inventario(inventario, fotos_unicas, panel_valido)

        path_inv = out_dir / "inventario_panos.csv"
        inventario.to_csv(path_inv, index=False)
        n_unicas_bal = int(
            ((inventario["es_foto_unica"] == 1) &
             (inventario["en_panel_balanceado"] == 1)).sum()
        )
        log.info(f"  inventario_panos.csv: {len(inventario):,} filas")
        log.info(f"    · es_foto_unica=1 & en_panel_balanceado=1: {n_unicas_bal:,} filas")
        log.info(
            "    · Filtro para Stata: keep if es_foto_unica==1 & en_panel_balanceado==1"
        )

        path_fu = out_dir / "fotos_unicas.csv"
        fotos_unicas.to_csv(path_fu, index=False)
        log.info(
            f"  fotos_unicas.csv: {len(fotos_unicas):,} asignaciones únicas "
            f"(pano_id × hogar, sin duplicados entre radios ni olas)"
        )

        # ── Paso 5: reporte ───────────────────────────────────────────────────
        log.info("\n── Paso 5: Reporte descriptivo ──────────────────────────────")
        reporte  = generar_reporte(panel_enriquecido, fotos_unicas, CONFIG)
        path_rep = out_dir / "reporte_cobertura_gsv.txt"
        path_rep.write_text(reporte, encoding="utf-8")
        log.info(f"  reporte_cobertura_gsv.txt guardado")

        print("\n" + reporte)

    except KeyboardInterrupt:
        # El usuario presionó Ctrl+C. Los workers activos terminan su consulta
        # actual antes de detenerse (el ThreadPoolExecutor hace shutdown limpio).
        # El guardado incremental cada 50 resultados preserva el progreso.
        log.warning("-" * 62)
        log.warning("Script interrumpido por el usuario (Ctrl+C).")
        log.warning("El progreso hasta el ultimo guardado incremental")
        log.warning(f"está en: {out_dir / 'resultados_api_cache.csv'}")
        log.warning("Para retomar: vuelve a correr el script exactamente igual.")
        log.warning("Reanudará desde donde quedó — no repite consultas OK.")
        log.warning("-" * 62)
        sys.exit(0)


if __name__ == "__main__":
    main()
