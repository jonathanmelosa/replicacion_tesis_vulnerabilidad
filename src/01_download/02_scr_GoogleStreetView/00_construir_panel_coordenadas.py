"""
00_construir_panel_coordenadas.py
=================================
Construye la base de panel de hogares con información de georreferenciación para las tres olas de la ELCA (2010, 2013, 2016).

QUÉ HACE
    1. Carga y combina los archivos rural + urbano de cada ola.
    2. Agrega indicador de hogar dividido (es_split).
    3. Detecta automáticamente las variables de coordenadas disponibles.
    4. Construye latitud y longitud en formato decimal cuando es posible.
    5. Valida la calidad de las coordenadas:
         - Olas 2010 y 2013: usa la variable coordenadas_obs ya revisada.
         - Ola 2016: verifica pertenencia al territorio colombiano.
    6. Construye el panel en formato largo (una fila por hogar × ola).
    7. Verifica los identificadores del panel y reporta hogares divididos.
    8. Calcula cambio_residencia_ola usando la clave correcta por transición:
         - 2010 → 2013: enlace por consecutivo
         - 2013 → 2016: enlace por llave
    9. Exporta el panel como CSV y genera un reporte de calidad.

ESTRUCTURA DE IDENTIFICADORES
    2010: consecutivo  (6 dígitos, único por hogar en la ola base)
    2013: llave = consecutivo + hogar (zero-padded a 2 dígitos)
          hogar=1 → hogar no dividido; hogar>1 → sub-hogar producto de una división
    2016: llave_n16 = llave + 2 dígitos adicionales
          últimos 2 dígitos '01' → sin nueva división en 2016; '02'+ → nueva división

OUTPUTS
    coordenadas/panel_coordenadas.csv
    coordenadas/reporte_calidad_coordenadas.txt

CÓMO CORRER
    python 00_construir_panel_coordenadas.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import math       # módulo de matemáticas: usado para sin, cos, atan2, radians en Haversine
import logging    # módulo de registro: imprime mensajes INFO, WARNING y ERROR con timestamp
import sys        # módulo de sistema: usado solo para sys.exit() en errores críticos
from pathlib import Path         # clase Path: manejo de rutas de forma robusta y multiplataforma

# ── Librerías externas ─────────────────────────────────────────────────────────
import numpy as np   # numpy: np.nan y operaciones numéricas vectorizadas
import pandas as pd  # pandas: DataFrames, lectura de archivos, exportación de resultados

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script a nuevas rutas
# o cuando los nombres de variables cambien en futuras versiones de los datos.
# ──────────────────────────────────────────────────────────────────────────────

# Ruta al directorio de datos crudos: __file__ es la ruta de este script;
# .resolve() la convierte en ruta absoluta; .parent sube un nivel hasta
# GoogleStreetView/; luego se desciende a raw/ donde están los datos ELCA.
_DATA_RAW = Path(__file__).resolve().parent / "raw"

CONFIG = {
    # ── Archivos de entrada (rural y urbano por ola) ──────────────────────────
    # Si uno de los dos archivos no existe, el script continúa con el otro.
    "archivos": {
        2010: {
            "rural":  _DATA_RAW / "elca_2010" / "RHogar-csv.tab",   # hogares rurales 2010
            "urbano": _DATA_RAW / "elca_2010" / "UHogar-csv.tab",   # hogares urbanos 2010
        },
        2013: {
            "rural":  _DATA_RAW / "elca_2013" / "RHogar-csv.tab",   # hogares rurales 2013
            "urbano": _DATA_RAW / "elca_2013" / "UHogar-csv.tab",   # hogares urbanos 2013
        },
        2016: {
            "rural":  _DATA_RAW / "elca_2016" / "RHogar-csv.tab",   # hogares rurales 2016
            "urbano": _DATA_RAW / "elca_2016" / "UHogar-csv.tab",   # hogares urbanos 2016
        },
    },

    # ── Identificadores por ola ───────────────────────────────────────────────
    # Columnas que se incluyen como identificadores en el panel.
    # Solo se agregan las que existen realmente en los archivos de cada ola.
    "ids_por_ola": {
        2010: ["consecutivo"],                                       # único ID disponible en 2010
        2013: ["consecutivo", "hogar", "llave"],                     # llave = consecutivo + hogar
        2016: ["consecutivo", "hogar_n16", "llave", "llave_n16"],   # en 2016 el sub-hogar se llama hogar_n16
    },

    # ── Clave de enlace entre olas para cambio_residencia_ola ────────────────
    # Define qué columna se usa para emparejar cada ola con la anterior.
    # En 2010 no existe llave, por eso la transición 2010→2013 usa consecutivo.
    "clave_enlace": {
        (2010, 2013): "consecutivo",  # todos los splits de 2013 heredan coords de 2010
        (2013, 2016): "llave",        # llave es el ID común entre 2013 y 2016
    },

    # ── Variables contextuales a conservar en el panel ────────────────────────
    # Todos los candidatos posibles. El script incluye solo los que existen en cada ola.
    "vars_contexto_candidatos": [
        "zona",         # zona corriente (Rural/Urbano) disponible en 2010 y 2013
        "zona_2010",    # zona según la ola 2010, disponible en 2013 y 2016
        "zona_2016",    # zona según la ola 2016, disponible en 2016
        "region",       # región geográfica en 2010 y 2013
        "RegionLb",     # etiqueta de región disponible en 2013 y 2016
        "region_2016",  # región según la ola 2016, disponible en 2016
        "dpto",         # código de departamento 
        "mpio",         # código de municipio 
        "t_hogar",      # tipo de hogar (disponible en 2010 y 2016 urbano)
        "t_personas",   # número de personas en el hogar (disponible en todas las olas)
    ],

    # ── Variables de coordenadas decimales por ola ───────────────────────────
    # Olas 2010 y 2013: variables decimales directas confirmadas.
    # Ola 2016: no existen variables decimales directas; se construyen desde GMS
    #           (grados-minutos-segundos) mediante candidatos_dms_lat/lon abajo.
    "candidatos_lat": {
        2010: ["coord_latitud"],   # confirmado
        2013: ["coord_latitud"],   # confirmado
        2016: [],                  # no existe variable decimal directa en 2016
    },
    "candidatos_lon": {
        2010: ["coord_longitud"],  # confirmado
        2013: ["coord_longitud"],  # confirmado
        2016: [],                  # no existe variable decimal directa en 2016
    },

    # ── Componentes GMS para construcción de coordenadas en ola 2016 ─────────
    # Nombres confirmados en los archivos de 2016.
    # La conversión usa: decimal = grados + minutos/60 + segundos/3600
    # Convención de signo:
    #   Latitud  — el signo se toma de coorlat_gra (positivo = norte, negativo = sur).
    #   Longitud — siempre negativa en Colombia (oeste de Greenwich); se niega el resultado.
    "candidatos_dms_lat": {
        2016: {
            "gra": "coorlat_gra",
            "min": "coorlat_min",
            "seg": "coorlat_seg",
        },
    },
    "candidatos_dms_lon": {
        2016: {
            "gra": "coorlon_gra",
            "min": "coorlon_min",
            "seg": "coorlon_seg",
        },
    },

    # ── Bounding box de Colombia para validación mínima de ola 2016 ───────────
    "colombia_bbox": {
        "lat_min": -4.2,   # latitud mínima del territorio colombiano (sur, Amazonas)
        "lat_max": 12.5,   # latitud máxima del territorio colombiano (norte, La Guajira)
        "lon_min": -79.0,  # longitud mínima (costa Pacífica)
        "lon_max": -66.8,  # longitud máxima (frontera con Venezuela)
    },

    # ── Tolerancia espacial para detectar cambio de residencia ───────────────
    # Distancia en metros bajo la cual se considera que el hogar no se mudó.
    # 50 m es suficiente para distinguir domicilios distintos en áreas urbanas.
    "tolerancia_residencia_m": 50,

    # ── Carpeta de salida ─────────────────────────────────────────────────────
    # Se crea junto al script, en GoogleStreetView/coordenadas/
    "output_dir": Path(__file__).resolve().parent / "coordenadas",
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,                               # nivel mínimo de mensajes a mostrar
    format="%(asctime)s [%(levelname)s] %(message)s", # formato: timestamp + nivel + mensaje
    handlers=[logging.StreamHandler(sys.stdout)],     # envía todos los mensajes a la consola
    force=True,   # sobreescribe configuraciones previas del logger raíz (útil en IDEs)
)
log = logging.getLogger(__name__)  # logger específico de este módulo


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

def cargar_ola(ano: int, cfg: dict) -> pd.DataFrame:
    """
    Lee y combina los archivos rural y urbano de una ola. Soporta archivos: .csv .tab .dta (Stata) Agrega:
    - zona_archivo: "rural" o "urbano"
    - ola: año de la encuesta

    Retorna un DataFrame vacío si ningún archivo existe.
    """
    archivos = cfg["archivos"][ano] # diccionario con rutas de archivos para esta ola: {"rural": ruta, "urbano": ruta}
    partes = [] # lista que acumulará los DataFrames de cada archivo (rural y urbano) para esta ola

    for zona_archivo, ruta in archivos.items(): # itera sobre "rural" y "urbano" con su respectiva ruta
        ruta = Path(ruta) # asegura que la ruta sea un objeto Path para manejo robusto de archivos

        if not ruta.exists():
            log.warning(f"  Archivo no encontrado: {ruta.resolve()}")
            log.warning(f"  Revisa CONFIG['archivos'][{ano}]['{zona_archivo}'] en este script.")
            log.warning(f"  Estructura esperada dentro de raw/:")
            log.warning(f"    raw/elca_{ano}/RHogar-csv.tab")
            log.warning(f"    raw/elca_{ano}/UHogar-csv.tab")
            log.warning(f"  Si los archivos tienen otro nombre, actualiza CONFIG['archivos'].")
            continue

        # -------------------------------
        # Leer según la extensión
        # -------------------------------
        if ruta.suffix.lower() == ".dta":
            try:
                df = pd.read_stata(
                    ruta,
                    convert_categoricals=False,
                )
                df = df.astype(str)
            except Exception as e:
                log.warning(f"  Error al leer {ruta.name}: {e}")
                log.warning(f"  Si fue guardado con Stata 14+, instala pyreadstat:")
                log.warning(f"    pip install pyreadstat")
                log.warning(f"  Luego pandas lo usara automaticamente para archivos .dta nuevos.")
                log.warning(f"  Verifica tambien que el archivo no este corrompido.")
                continue

        elif ruta.suffix.lower() in [".csv", ".tab"]:
            try:
                df = pd.read_csv(
                    ruta,
                    sep="\t" if ruta.suffix.lower() == ".tab" else ",",
                    dtype=str,
                    encoding="utf-8",
                    on_bad_lines="skip",
                )
            except UnicodeDecodeError:
                # Error de codificacion: comun cuando el archivo viene de Windows o Stata.
                # Se reintenta con latin-1, que cubre caracteres hispanicos (tildes, enie).
                log.warning(f"  El archivo {ruta.name} no es UTF-8.")
                log.warning(f"  Reintentando con latin-1 (codificacion comun en Windows/Stata)...")
                try:
                    df = pd.read_csv(
                        ruta,
                        sep="\t" if ruta.suffix.lower() == ".tab" else ",",
                        dtype=str,
                        encoding="latin-1",
                        on_bad_lines="skip",
                    )
                    log.info(f"  Leido con latin-1: {len(df):,} registros — OK")
                except Exception as e2:
                    log.warning(f"  Fallo tambien con latin-1: {e2}")
                    log.warning(f"  Verifica que el archivo no este corrompido.")
                    continue

        else:
            log.warning(f"  Formato no soportado: '{ruta.suffix}' en {ruta.name}")
            log.warning(f"  Formatos aceptados: .csv  .tab  .dta (Stata)")
            log.warning(f"  Renombra el archivo o actualiza CONFIG['archivos'].")
            continue

        df["zona_archivo"] = zona_archivo # agrega columna que indica si el hogar viene del archivo rural o urbano
        partes.append(df) # agrega el DataFrame de esta parte (rural o urbano) a la lista de partes para esta ola

        log.info(
            f"  {zona_archivo.upper()} {ano}: {len(df):,} registros — {ruta.name}"
        )

    if not partes: #    si no se pudo cargar ningún archivo para esta ola, retorna un DataFrame vacío y registra un error
        log.error(f"Sin archivos disponibles para ola {ano}.")
        return pd.DataFrame()

    df_ola = pd.concat(partes, ignore_index=True) # concatena los DataFrames de rural y urbano para esta ola, ignorando los índices originales para crear un nuevo índice secuencial
    df_ola["ola"] = ano # agrega columna que indica el año de la ola para cada registro

    return df_ola # retorna el DataFrame combinado de esta ola con zona_archivo y ola incorporadas

# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: INDICADOR DE HOGAR DIVIDIDO
# ──────────────────────────────────────────────────────────────────────────────

def agregar_es_split(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    """
    Agrega es_split: indica si el hogar es producto de una división del hogar original.
    Lógica por ola:
    - 2010: siempre 0. Es la ola base; todos los hogares son originales.
    - 2013: se basa en la variable 'hogar'.
        hogar == '1' → hogar principal, no dividido       → es_split = 0
        hogar != '1' → sub-hogar producto de una división → es_split = 1
        hogar ausente → no se puede determinar            → es_split = NA
    - 2016: misma lógica pero con la variable 'hogar_n16' (renombrada en la ola 2016).
        hogar_n16 == '1' → hogar principal → es_split = 0
        hogar_n16 != '1' → sub-hogar       → es_split = 1
        hogar_n16 ausente → es_split = NA
    """
    if ano == 2010:                    # en 2010 no existe 'hogar' y no hay divisiones
        df["es_split"] = pd.array([0] * len(df), dtype="Int8")  # Int8 para que pd.concat no upcastee
        return df                      # retorna sin más procesamiento

    # En 2016 la variable de sub-hogar se llama hogar_n16, no hogar
    col_hogar = "hogar_n16" if ano == 2016 else "hogar"

    if col_hogar not in df.columns:
        log.warning(f"  Ola {ano}: columna '{col_hogar}' no encontrada. es_split quedará como NA.")
        df["es_split"] = pd.NA
        return df

    hogar_str = df[col_hogar].astype(str).str.strip()  # normaliza a string y elimina espacios

    df["es_split"] = pd.array(        # usa pandas Int8 (entero nullable) que admite NA
        [pd.NA] * len(df), dtype="Int8"
    )
    df.loc[hogar_str == "1",   "es_split"] = 0   # hogar principal sin división
    df.loc[hogar_str == "nan", "es_split"] = pd.NA  # NaN original → indeterminado
    df.loc[(hogar_str != "1") & (hogar_str != "nan"), "es_split"] = 1  # hogar dividido

    n_splits = int((df["es_split"] == 1).sum())   # cuenta hogares divididos para el log
    log.info(f"  es_split: {n_splits:,} hogares divididos de {len(df):,} totales en ola {ano}")

    return df   # retorna el DataFrame con es_split incorporada


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3 (→ 4): CONSTRUCCIÓN DE COORDENADAS DECIMALES
# ──────────────────────────────────────────────────────────────────────────────

def _gms_a_decimal(gra: pd.Series, min_: pd.Series, seg: pd.Series) -> pd.Series:
    """
    Convierte grados-minutos-segundos a grados decimales.
    El signo del resultado se toma de gra (positivo = norte/este, negativo = sur/oeste).
    Valores no convertibles quedan como NaN.
    """
    gra_n = pd.to_numeric(gra.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    min_n = pd.to_numeric(min_.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    seg_n = pd.to_numeric(seg.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    signo    = np.where(gra_n < 0, -1, 1)
    decimal  = np.abs(gra_n) + min_n / 60 + seg_n / 3600
    return pd.Series(signo * decimal, index=gra.index)


def construir_coords_decimales(df: pd.DataFrame, ano: int, cfg: dict) -> pd.DataFrame:
    """
    Extrae lat_decimal y lon_decimal de las variables consolidadas de la ELCA.

    La ELCA provee variables ya consolidadas en grados decimales:
      - Ola 2010: coord_latitud / coord_longitud
      - Ola 2013: coord_latitud / coord_longitud
      - Ola 2016: candidatos definidos en cfg["candidatos_lat/lon"][2016]
    Estas variables se usan directamente, sin transformación de signo.
    La convención estándar — y la que exige la API de Google Street View — es:
      - Latitud  POSITIVA al norte del ecuador  (la gran mayoría de Colombia)
      - Latitud  NEGATIVA al sur del ecuador    (extremo amazónico, hasta ≈ -4.2°)
      - Longitud NEGATIVA al oeste de Greenwich (toda Colombia, entre -79° y -67°)
    Las variables consolidadas de la ELCA ya respetan esta convención.
    Si ninguna variable decimal está disponible, lat_decimal y lon_decimal quedan como NaN.
    """
    cols = df.columns.tolist()  # lista de columnas presentes en el DataFrame

    # ── Latitud decimal ───────────────────────────────────────────────────────
    lat_col = next(                        # busca el primer candidato que exista en el archivo
        (c for c in cfg["candidatos_lat"][ano] if c in cols),
        None,                              # None si ningún candidato de latitud existe
    )
    if lat_col:                            # variable decimal de latitud encontrada
        df["lat_decimal"] = pd.to_numeric( # convierte a float numérico
            df[lat_col].astype(str).str.replace(",", ".", regex=False),  # normaliza separador decimal
            errors="coerce",               # valores no convertibles → NaN
        )
        log.info(f"  Latitud decimal: variable '{lat_col}' usada directamente.")
    elif ano in cfg.get("candidatos_dms_lat", {}):   # fallback GMS para ola 2016
        dms = cfg["candidatos_dms_lat"][ano]
        falta = [v for v in dms.values() if v not in cols]
        if falta:
            df["lat_decimal"] = np.nan
            log.warning(f"  Ola {ano}: componentes GMS de latitud no encontrados: {falta}")
            log.warning(f"  EFECTO: lat_decimal = NaN para todos los hogares de ola {ano}.")
        else:
            df["lat_decimal"] = _gms_a_decimal(df[dms["gra"]], df[dms["min"]], df[dms["seg"]])
            n_ok = df["lat_decimal"].notna().sum()
            log.info(f"  Latitud decimal construida desde GMS ({dms['gra']}, {dms['min']}, {dms['seg']}): {n_ok:,} valores.")
    else:
        df["lat_decimal"] = np.nan
        log.warning(f"  Ola {ano}: latitud no encontrada. Candidatos buscados: {cfg['candidatos_lat'][ano]}")
        similares_lat = [c for c in cols if "lat" in c.lower() or "coord" in c.lower()]
        if similares_lat:
            log.warning(f"  Columnas similares en el archivo: {similares_lat}")
            log.warning(f"  Si alguna es la correcta, actualiza CONFIG['candidatos_lat'][{ano}].")
        else:
            log.warning(f"  No se encontro ninguna columna de latitud. Revisa el archivo manualmente.")
        log.warning(f"  EFECTO: lat_decimal = NaN para todos los hogares de ola {ano}.")

    # ── Longitud decimal ──────────────────────────────────────────────────────
    lon_col = next(
        (c for c in cfg["candidatos_lon"][ano] if c in cols),
        None,
    )
    if lon_col:
        df["lon_decimal"] = pd.to_numeric(
            df[lon_col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        log.info(f"  Longitud decimal: variable '{lon_col}' usada directamente.")
    elif ano in cfg.get("candidatos_dms_lon", {}):   # fallback GMS para ola 2016
        dms = cfg["candidatos_dms_lon"][ano]
        falta = [v for v in dms.values() if v not in cols]
        if falta:
            df["lon_decimal"] = np.nan
            log.warning(f"  Ola {ano}: componentes GMS de longitud no encontrados: {falta}")
            log.warning(f"  EFECTO: lon_decimal = NaN para todos los hogares de ola {ano}.")
        else:
            # Colombia está íntegramente al oeste de Greenwich → longitud siempre negativa
            df["lon_decimal"] = -_gms_a_decimal(df[dms["gra"]], df[dms["min"]], df[dms["seg"]])
            n_ok = df["lon_decimal"].notna().sum()
            log.info(f"  Longitud decimal construida desde GMS ({dms['gra']}, {dms['min']}, {dms['seg']}): {n_ok:,} valores (negadas).")
    else:
        df["lon_decimal"] = np.nan
        log.warning(f"  Ola {ano}: longitud no encontrada. Candidatos buscados: {cfg['candidatos_lon'][ano]}")
        similares_lon = [c for c in cols if "lon" in c.lower() or "coord" in c.lower()]
        if similares_lon:
            log.warning(f"  Columnas similares en el archivo: {similares_lon}")
            log.warning(f"  Si alguna es la correcta, actualiza CONFIG['candidatos_lon'][{ano}].")
        else:
            log.warning(f"  No se encontro ninguna columna de longitud. Revisa el archivo manualmente.")
        log.warning(f"  EFECTO: lon_decimal = NaN para todos los hogares de ola {ano}.")

    return df   # retorna el DataFrame con lat_decimal y lon_decimal incorporadas

# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: VALIDACIÓN DE COORDENADAS
# ──────────────────────────────────────────────────────────────────────────────

def _en_colombia(lat: float, lon: float, bbox: dict) -> bool: # función auxiliar para verificar si un punto está dentro del bounding box de Colombia
    """Retorna True si las coordenadas están dentro del bounding box de Colombia."""
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]    # dentro del rango norte-sur
        and bbox["lon_min"] <= lon <= bbox["lon_max"]  # dentro del rango este-oeste
    )

def validar_coords(df: pd.DataFrame, ano: int, cfg: dict) -> pd.DataFrame:
    """
    Agrega coord_valida_ola según la estrategia por ola.

    2010 y 2013 — usa coordenadas_obs (revisión previa ya realizada):
        1  → coordenada válida (coordenadas_obs == 0)
        0  → coordenada no válida (coordenadas_obs == 1 o 2)
        NA → coordenadas_obs no disponible todavía

    2016 — validación mínima con bounding box:
        1  → dentro del territorio colombiano
        0  → fuera del territorio colombiano
        NA → lat_decimal o lon_decimal ausente
    """
    if ano in (2010, 2013):                         # olas con revisión previa disponible

        if "coordenadas_obs" in df.columns:         # la variable de revisión existe
            obs = pd.to_numeric(df["coordenadas_obs"], errors="coerce")  # convierte a número
            df["coord_valida_ola"] = (obs == 0).astype("Int8")  # 1 si obs==0, 0 si no
            df.loc[pd.isna(obs), "coord_valida_ola"] = pd.NA     # obs faltante → NA (no 0)
        else:
            log.warning(
                f"  coordenadas_obs no disponible para ola {ano}. "
                "coord_valida_ola quedara como NA."
            )
            log.warning(f"  EFECTO: el script 01 NO consultara la API GSV para esta ola.")
            log.warning(f"  Busca en los datos la variable de revision de coordenadas.")
            log.warning(f"  Si se llama diferente a 'coordenadas_obs', actualiza")
            log.warning(f"  la funcion validar_coords() con el nombre correcto.")
            df["coord_valida_ola"] = pd.NA

    elif ano == 2016:                               # validación con bounding box

        bbox = cfg["colombia_bbox"]                 # límites del territorio colombiano
        tiene_coords = df["lat_decimal"].notna() & df["lon_decimal"].notna()  # ambas disponibles

        en_col = tiene_coords & df.apply(           # verifica pertenencia a Colombia fila a fila
            lambda r: (
                _en_colombia(r["lat_decimal"], r["lon_decimal"], bbox)
                if pd.notna(r["lat_decimal"]) and pd.notna(r["lon_decimal"])
                else False                          # faltante no puede estar en Colombia
            ),
            axis=1,
        )

        df["coord_valida_ola"] = en_col.astype("Int8")       # 1 en Colombia, 0 fuera
        df.loc[~tiene_coords, "coord_valida_ola"] = pd.NA    # NA donde faltan coordenadas

    return df   # retorna el DataFrame con coord_valida_ola incorporada

# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 7: CONSTRUCCIÓN DEL PANEL LARGO
# ──────────────────────────────────────────────────────────────────────────────

def construir_panel(ola_dfs: dict, cfg: dict) -> pd.DataFrame:
    """
    Concatena las olas en formato largo: una fila por hogar × ola.

    Columnas incluidas (solo las que existen en cada ola):
    - Identificadores: consecutivo, hogar, llave, llave_n16
    - ola, zona_archivo
    - Variables contextuales: zona, zona_2010, zona_2016, region, RegionLb,
      region_2016, dpto, mpio, t_hogar, t_personas
    - Variables de coordenadas originales detectadas
    - Columnas derivadas: es_split, lat_decimal, lon_decimal, coord_valida_ola
    """
    partes = []  # lista que acumulará el slice de columnas de cada ola

    for ano in sorted(ola_dfs):                     # procesa cada ola cronológicamente
        df = ola_dfs[ano].copy()                    # copia defensiva para no alterar el original

        # Identificadores disponibles en esta ola (solo los que existen en el DataFrame)
        ids = [c for c in cfg["ids_por_ola"][ano] if c in df.columns]

        # Variables contextuales presentes en este DataFrame
        contexto = [c for c in cfg["vars_contexto_candidatos"] if c in df.columns]

        # Columnas de coordenadas originales de la ELCA presentes en el DataFrame
        # (variables decimales directas + componentes GMS de 2016 + coordenadas_obs)
        vars_coords_elca = (
            cfg["candidatos_lat"][ano]
            + cfg["candidatos_lon"][ano]
            + list(cfg.get("candidatos_dms_lat", {}).get(ano, {}).values())
            + list(cfg.get("candidatos_dms_lon", {}).get(ano, {}).values())
            + ["coordenadas_obs"]
        )
        coords_orig = [c for c in dict.fromkeys(vars_coords_elca) if c in df.columns]

        # Columnas derivadas generadas por los módulos anteriores
        derivadas = [c for c in ["es_split", "lat_decimal", "lon_decimal", "coord_valida_ola"]
                     if c in df.columns]

        # Columna que indica el sub-archivo de origen (rural o urbano)
        zona_arch = ["zona_archivo"] if "zona_archivo" in df.columns else []

        # Combina todas las listas de columnas; dict.fromkeys elimina duplicados y preserva orden
        columnas = list(dict.fromkeys(
            ids + ["ola"] + zona_arch + contexto + coords_orig + derivadas
        ))

        partes.append(df[columnas])                 # agrega el slice de esta ola a la lista

    # Concatena todas las olas; sort=False preserva el orden original de las columnas
    panel = pd.concat(partes, ignore_index=True, sort=False)
    log.info(f"Panel construido: {len(panel):,} obs × {panel.shape[1]} columnas")

    return panel   # retorna el panel en formato largo


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 8: VERIFICACIÓN DE IDENTIFICADORES
# ──────────────────────────────────────────────────────────────────────────────

def verificar_identificadores(panel: pd.DataFrame) -> dict:
    """
    Verifica la integridad de los identificadores del panel y cuantifica los hogares divididos.
    Identificador único por ola:
    - 2010: consecutivo
    - 2013: llave (= consecutivo + hogar, cero-relleno a 2 dígitos)
    - 2016: llave_n16 (= llave + 2 dígitos de subdivisión en 2016)
    Los duplicados de consecutivo en 2013 y 2016 son ESPERADOS: representan
    hogares que se dividieron. El error real sería un duplicado del ID único
    de cada ola (llave en 2013, llave_n16 en 2016).

    Retorna un diccionario con:
    - total_consecutivos_unicos:   hogares originales únicos de 2010
    - consecutivos_por_n_olas:     cobertura de panel por n° de olas
    - splits_por_ola:              hogares divididos (es_split=1) por ola
    - duplicados_id_unico_por_ola: duplicados del ID único real por ola
    """
    resultado = {}   # diccionario que acumulará todos los resultados

    # ── Cobertura del panel por consecutivo (hogar original 2010) ────────────
    resultado["total_consecutivos_unicos"] = int(panel["consecutivo"].nunique())

    # Para cada consecutivo, cuenta en cuántas olas distintas aparece
    olas_por_consec = panel.groupby("consecutivo")["ola"].nunique()  # serie: consec → n_olas
    resultado["consecutivos_por_n_olas"] = (
        olas_por_consec.value_counts()    # cuántos consecutivos tienen 1, 2 y 3 olas
        .sort_index()                     # ordena ascendente (1, 2, 3)
        .to_dict()                        # convierte a {n_olas: n_consecutivos}
    )

    # ── Hogares divididos por ola ─────────────────────────────────────────────
    splits_por_ola = {}                   # sub-diccionario {ola: n_splits}
    if "es_split" in panel.columns:       # solo si es_split fue calculada
        for ola, grp in panel.groupby("ola"):
            n_split = int((grp["es_split"] == 1).sum())  # cuenta es_split == 1 en esta ola
            splits_por_ola[int(ola)] = n_split
            if n_split > 0:              # informa y explica que es esperado
                log.info(
                    f"  Ola {ola}: {n_split:,} hogares divididos (es_split=1) "
                    "— duplicados de consecutivo esperados"
                )
            else:
                log.info(f"  Ola {ola}: sin hogares divididos")
    resultado["splits_por_ola"] = splits_por_ola

    # ── Duplicados del ID único real por ola ──────────────────────────────────
    # Los duplicados de consecutivo son esperados en 2013/2016 (splits).
    # El error real es duplicar el identificador propio de cada ola.
    id_por_ola = {                        # identificador único por ola
        2010: "consecutivo",             # en 2010 no existe llave
        2013: "llave",                   # en 2013 el ID único es llave
        2016: "llave_n16",               # en 2016 el ID único es llave_n16
    }
    dup_por_ola = {}                      # sub-diccionario {ola: n_duplicados}
    for ola, grp in panel.groupby("ola"): # itera por ola para verificar duplicados del ID único de esa ola
        id_col = id_por_ola.get(int(ola), "consecutivo")  # obtiene el ID único de esta ola
        if id_col not in grp.columns:    # la columna de ID no está disponible
            log.warning(f"  Ola {ola}: '{id_col}' no disponible — no se verifica duplicados.") # no se puede verificar duplicados sin la columna de ID único
            dup_por_ola[int(ola)] = None # asigna None para indicar que no se pudo verificar esta ola
            continue
        n_dup = int(grp[id_col].dropna().duplicated().sum())  # cuenta IDs repetidos no-nulos
        dup_por_ola[int(ola)] = n_dup # almacena el número de duplicados para esta ola en el resultado
        if n_dup > 0:                    # esto sí es un problema real
            log.warning(f"  Ola {ola}: {n_dup:,} duplicados de '{id_col}' — REVISAR") # duplicados del ID único indican un error en la construcción del panel (IDs que deberían ser únicos se repiten)
        else:
            log.info(f"  Ola {ola}: '{id_col}' sin duplicados — OK") # ningún duplicado del ID único es un buen indicador de integridad en la construcción del panel
    resultado["duplicados_id_unico_por_ola"] = dup_por_ola

    return resultado   # retorna el diccionario completo


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 9: CAMBIO DE RESIDENCIA
# ──────────────────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula la distancia en metros entre dos puntos geográficos usando la fórmula
    de Haversine, que considera la curvatura de la Tierra.
    """
    R     = 6_371_000                          # radio medio de la Tierra en metros
    phi1  = math.radians(lat1)                 # latitud del punto 1 en radianes
    phi2  = math.radians(lat2)                 # latitud del punto 2 en radianes
    dphi  = math.radians(lat2 - lat1)          # diferencia de latitudes en radianes
    dlam  = math.radians(lon2 - lon1)          # diferencia de longitudes en radianes
    a = (                                      # semiverseno al cuadrado (término intermedio)
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))  # distancia en metros


def _calcular_transicion( # función auxiliar para calcular cambio_residencia_ola entre dos olas específicas
    panel: pd.DataFrame,
    ola_ant: int,
    ola_act: int,
    clave: str,
    tolerancia: float,
) -> pd.DataFrame:
    """
    Calcula cambio_residencia_ola para los hogares de ola_act respecto a ola_ant.

    Estrategia:
    1. Toma las coordenadas de ola_ant indexadas por `clave` como referencia.
       Si existen duplicados de la clave en ola_ant (no deberían), conserva el primero.
    2. Hace un merge izquierdo desde ola_act sobre ola_ant usando `clave`.
       - Hogares sin match (nuevos en ola_act o sin llave en 2016) → NA.
       - Hogares divididos en 2013 comparten la misma referencia de 2010.
       - Hogares divididos en 2016 comparten la misma referencia de 2013.
    3. Calcula Haversine donde ambas coordenadas (actual y referencia) existen.
    4. Asigna el resultado al panel usando el índice original (no el del merge).

    Retorna el panel modificado con cambio_residencia_ola actualizado para ola_act.
    """
    # Coordenadas de referencia de la ola anterior, indexadas por la clave de enlace
    coords_ref = ( # DataFrame con columnas: clave, _lat_ref, _lon_ref
        panel[panel["ola"] == ola_ant]   # filtra filas de la ola anterior
        [[clave, "lat_decimal", "lon_decimal"]]  # conserva solo clave y coordenadas
        .dropna(subset=[clave])          # descarta filas sin valor en la clave
        .drop_duplicates(subset=[clave], keep="first")  # una fila por valor de clave
        .rename(columns={               # renombra para distinguirlas de las columnas actuales
            "lat_decimal": "_lat_ref",
            "lon_decimal": "_lon_ref",
        })
    )

    if coords_ref.empty:               # la ola anterior no tiene coordenadas disponibles
        log.info(f"  Transición {ola_ant}→{ola_act}: sin coordenadas de referencia. Se omite.")
        return panel                   # retorna el panel sin modificar esta transición

    # Filas de la ola actual con su índice original preservado
    mask_act = panel["ola"] == ola_act  # máscara booleana de las filas de ola_act
    filas_act = panel.loc[mask_act, [clave, "lat_decimal", "lon_decimal"]].copy()
    filas_act["_orig_idx"] = filas_act.index  # guarda el índice original antes del merge

    # Merge izquierdo: cada fila de ola_act obtiene las coordenadas de su referencia en ola_ant
    # how="left" garantiza que las filas sin match (nuevas en ola_act) queden con NaN
    filas_merged = filas_act.merge(coords_ref, on=clave, how="left")

    # Máscara de filas donde tanto la coordenada actual como la referencia están disponibles
    tiene_ambas = (
        filas_merged["lat_decimal"].notna()  # latitud actual disponible
        & filas_merged["lon_decimal"].notna()  # longitud actual disponible
        & filas_merged["_lat_ref"].notna()    # latitud de referencia disponible
        & filas_merged["_lon_ref"].notna()    # longitud de referencia disponible
    )

    # Inicializa la columna de cambio con NA para todas las filas de esta transición
    filas_merged["_cambio"] = pd.array([pd.NA] * len(filas_merged), dtype="Int8")

    if tiene_ambas.any():              # solo calcula si hay al menos una comparación posible
        distancias = filas_merged[tiene_ambas].apply(  # Haversine fila por fila
            lambda r: _haversine_m( #   calcula la distancia entre la coordenada de referencia y la actual
                r["_lat_ref"],   r["_lon_ref"],    # coordenadas de la ola anterior
                r["lat_decimal"], r["lon_decimal"], # coordenadas de la ola actual
            ),
            axis=1,
        )
        # Asigna 1 si la distancia supera la tolerancia, 0 si no la supera
        filas_merged.loc[tiene_ambas, "_cambio"] = (
            (distancias >= tolerancia).astype("Int8").values
        )

    # Asigna los resultados de vuelta al panel usando el índice original preservado
    # _orig_idx contiene los índices del panel original; _cambio los valores calculados
    panel.loc[filas_merged["_orig_idx"].values, "cambio_residencia_ola"] = (
        filas_merged["_cambio"].values
    )

    log.info( # loguea cuántas comparaciones se realizaron para esta transición
        f"  Transición {ola_ant}→{ola_act} (clave: '{clave}'): " #  indica la transición y la clave usada
        f"{tiene_ambas.sum():,} comparaciones realizadas." #  indica cuántas filas tenían ambas coordenadas disponibles para comparar
    )

    return panel   # retorna el panel con cambio_residencia_ola actualizado para ola_act


def calcular_cambio_residencia(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Orquesta el cálculo de cambio_residencia_ola para cada transición entre olas.
    Usa la clave de enlace correcta según la transición:
    - 2010 → 2013: clave = consecutivo
        Los hogares divididos en 2013 comparten la misma referencia 2010
        (la del hogar original); todos parten desde la misma ubicación.
    - 2013 → 2016: clave = llave
        Los hogares que se dividen nuevamente en 2016 comparten la referencia
        2013 de su llave (la del hogar que se subdividió).

    Valores de cambio_residencia_ola:
        NA — primera ola del hogar (sin referencia anterior), o
             coordenada faltante en ola_actual o en la referencia
        0  — distancia < tolerancia (sin cambio de residencia)
        1  — distancia >= tolerancia (cambio de residencia detectado)
    """
    if "lat_decimal" not in panel.columns or "lon_decimal" not in panel.columns: # verifica que las coordenadas decimales estén disponibles
        log.warning("  lat_decimal / lon_decimal no disponibles — " # Sin coordenadas decimales no es posible calcular distancias entre olas
                    "cambio_residencia_ola se omite.")
        return panel   # retorna sin modificar

    tolerancia = cfg["tolerancia_residencia_m"]   # umbral en metros
    clave_enlace = cfg["clave_enlace"]             # {(ola_ant, ola_act): clave}

    panel = panel.copy()                          # copia para no modificar el original
    panel["cambio_residencia_ola"] = pd.array([pd.NA] * len(panel), dtype="Int8")  # Int8 evita mezcla de dtypes al asignar via .loc

    olas = sorted(panel["ola"].dropna().unique())  # olas disponibles en orden cronológico

    for i in range(len(olas) - 1):                 # itera sobre cada par de olas consecutivas
        ola_ant = int(olas[i])                     # ola anterior (fuente de referencia)
        ola_act = int(olas[i + 1])                 # ola actual (recibe el indicador)

        clave = clave_enlace.get((ola_ant, ola_act))  # obtiene la clave de enlace
        if clave is None:                          # transición no definida en CONFIG
            log.warning(
                f"  Transición {ola_ant}→{ola_act}: "
                "clave de enlace no definida en CONFIG. Se omite."
            )
            continue                               # pasa a la siguiente transición

        if clave not in panel.columns:             # la columna de clave no existe en el panel
            log.warning(
                f"  Clave '{clave}' no encontrada en el panel. "
                f"Transición {ola_ant}→{ola_act} omitida."
            )
            continue

        # Delega el cálculo de esta transición a la función auxiliar
        panel = _calcular_transicion(panel, ola_ant, ola_act, clave, tolerancia)

    return panel   # retorna el panel con cambio_residencia_ola calculada para todas las transiciones


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 10: REPORTE DE CALIDAD
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(panel: pd.DataFrame, info_ids: dict, cfg: dict) -> str:
    """
    Genera el texto del reporte de calidad del panel y las coordenadas.

    Secciones:
    1. Calidad del panel (cobertura, hogares, divisiones, duplicados)
    2. Calidad de coordenadas (variables detectadas, disponibilidad, validez)
    3. Cambio de residencia (distribución de cambio_residencia_ola)
    """
    sep   = "=" * 65
    lines = [
        sep,
        "REPORTE DE CALIDAD — PANEL COORDENADAS ELCA",
        sep,
        "",
        "── 1. CALIDAD DEL PANEL ──────────────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Describe la estructura del panel longitudinal: cuántos hogares",
        "  se identificaron en la ola base (2010), cuántos se lograron",
        "  seguir en olas posteriores y cuántos se dividieron en sub-hogares.",
        "  No involucra coordenadas; es una verificación de integridad de los",
        "  identificadores.",
        "  Fuente: archivos RHogar-csv.tab y UHogar-csv.tab de cada ola.",
        "",
        f"  Consecutivos únicos (hogares originales 2010): "
        f"{info_ids['total_consecutivos_unicos']:,}",
        "",
        "  EXPLICACIÓN",
        "  'Consecutivo' es el identificador del hogar en 2010 (ola base).",
        "  Este número es el total de hogares originales encuestados en 2010",
        "  y sirve como referencia máxima posible de hogares seguidos en el panel.",
        "",
        "  Cobertura del panel (por consecutivo):",
        "  LÓGICA: para cada consecutivo se cuenta en cuántas olas distintas",
        "  aparece (1, 2 ó 3). Un consecutivo puede tener varias filas por",
        "  divisiones de hogar, pero aquí solo importa si aparece o no en cada ola.",
    ]

    total_consec = info_ids["total_consecutivos_unicos"]
    for n_olas, n_consec in sorted(info_ids["consecutivos_por_n_olas"].items()):
        pct = 100 * n_consec / total_consec if total_consec else 0
        lines.append(f"    En {n_olas} ola(s): {n_consec:,} consecutivos ({pct:.1f} %)")

    lines += [
        "",
        "  INTERPRETACIÓN",
        "  - Hogares en 3 olas: núcleo del panel equilibrado (seguimiento completo).",
        "  - Hogares en 2 olas: abandonaron el panel en algún punto o tienen",
        "    problemas de vinculación de identificadores entre olas.",
        "  - Hogares en 1 ola:  mayor pérdida; encuestados una única vez por",
        "    abandono, fallecimiento o problema de identificación.",
        "",
        "  Observaciones y hogares únicos por ola:",
        "  LÓGICA: cada columna mide algo distinto.",
        "  - 'obs': filas totales en esa ola (incluye sub-hogares por divisiones).",
        "  - 'consecutivos': hogares originales de 2010 presentes en la ola.",
        "    Un consecutivo puede tener más de una fila si el hogar se dividió.",
        "  - 'ID únicos (llave / llave_n16)': identificador propio de cada ola,",
        "    que sí distingue sub-hogares. Por construcción es igual a 'obs'.",
        "    ID por ola → 2010: consecutivo | 2013: llave | 2016: llave_n16.",
        "  INTERPRETACIÓN: si obs > consecutivos únicos en 2013 o 2016, hay",
        "  hogares divididos (consecutivos que aparecen más de una vez).",
        "  En 2010 las tres columnas son iguales porque no existe división.",
    ]

    for ola, grp in panel.groupby("ola"):
        n_consec = grp["consecutivo"].nunique()
        id_ola = {2010: "consecutivo", 2013: "llave", 2016: "llave_n16"}.get(int(ola))
        if id_ola and id_ola in grp.columns:
            n_unicos = grp[id_ola].nunique()
            lines.append(
                f"    {ola}: {len(grp):,} obs / {n_consec:,} consecutivos / "
                f"{n_unicos:,} {id_ola} únicos"
            )
        else:
            lines.append(f"    {ola}: {len(grp):,} obs / {n_consec:,} consecutivos únicos")

    lines += [
        "",
        "  Hogares divididos (es_split=1) por ola:",
        "  LÓGICA: un hogar se 'divide' cuando miembros del hogar original",
        "  forman un nuevo hogar independiente entre olas. La ELCA lo rastrea",
        "  con la variable 'hogar' (2013) y 'hogar_n16' (2016):",
        "    hogar / hogar_n16 == 1  →  hogar principal  →  es_split = 0",
        "    hogar / hogar_n16 != 1  →  sub-hogar dividido →  es_split = 1",
        "  En 2010 no existe esta variable (es la ola base, sin divisiones).",
        "  INTERPRETACIÓN: los sub-hogares (es_split=1) se tratan como",
        "  observaciones independientes pero comparten la coordenada de",
        "  referencia de la ola anterior (el domicilio del que se separaron).",
    ]

    for ola, n_split in sorted(info_ids.get("splits_por_ola", {}).items()):
        lines.append(f"    {ola}: {n_split:,} hogares divididos")

    lines += [
        "",
        "  Duplicados del ID único por ola (error real si > 0):",
        "  LÓGICA: cada ola tiene su propio identificador que debe ser único",
        "  por fila → 2010: consecutivo | 2013: llave | 2016: llave_n16.",
        "  Los duplicados de 'consecutivo' en 2013/2016 son ESPERADOS (splits).",
        "  Un duplicado del ID único de la ola SÍ es un error de construcción.",
    ]

    id_por_ola = {2010: "consecutivo", 2013: "llave", 2016: "llave_n16"}
    for ola, n_dup in sorted(info_ids["duplicados_id_unico_por_ola"].items()):
        id_col = id_por_ola.get(int(ola), "?")
        if n_dup is None:
            lines.append(f"    {ola} ({id_col}): no verificado — columna no disponible")
        elif n_dup > 0:
            lines.append(f"    {ola} ({id_col}): {n_dup:,} — REVISAR")
        else:
            lines.append(f"    {ola} ({id_col}): ninguno — OK")

    lines += [
        "",
        "── 2. CALIDAD DE COORDENADAS ─────────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Verifica si las variables de coordenadas configuradas en el script",
        "  existen en los archivos de cada ola y cuántos hogares tienen",
        "  coordenadas válidas para consultar la API de Google Street View.",
        "  Fuente: variables de lat/lon de RHogar-csv.tab y UHogar-csv.tab.",
        "",
        "  Variables de coordenadas por ola:",
        "  LÓGICA: el script busca en cada archivo las columnas cuyos nombres",
        "  aparecen aquí. Si la columna no existe, lat_decimal y lon_decimal",
        "  quedan como NaN para toda la ola.",
        "  - 2010 y 2013: variables decimales directas.",
        "  - 2016: se construyen desde componentes grados-minutos-segundos (GMS)",
        "    usando la fórmula: decimal = grados + minutos/60 + segundos/3600.",
        "    La longitud se niega porque Colombia está íntegramente al oeste",
        "    de Greenwich (longitudes negativas entre -79° y -67°).",
    ]

    for ano in sorted(cfg["candidatos_lat"]):
        lat_list = cfg["candidatos_lat"][ano]
        lon_list = cfg["candidatos_lon"][ano]
        dms_lat  = cfg.get("candidatos_dms_lat", {}).get(ano, {})
        dms_lon  = cfg.get("candidatos_dms_lon", {}).get(ano, {})
        if lat_list:
            lines.append(f"    Ola {ano}: lat={lat_list[0]!r}  lon={lon_list[0]!r}")
        elif dms_lat:
            lines.append(
                f"    Ola {ano}: construida desde GMS → "
                f"lat({dms_lat['gra']}, {dms_lat['min']}, {dms_lat['seg']})  "
                f"lon({dms_lon['gra']}, {dms_lon['min']}, {dms_lon['seg']})"
            )
        else:
            lines.append(f"    Ola {ano}: sin variables configuradas")

    lines += [
        "",
        "  Disponibilidad y validez de coordenadas por ola:",
        "  LÓGICA de coord_valida_ola:",
        "  - 2010 y 2013: usa 'coordenadas_obs' (revisión manual previa en ELCA).",
        "    coordenadas_obs == 0  →  coord_valida_ola = 1  (válida)",
        "    coordenadas_obs == 1 o 2  →  coord_valida_ola = 0  (inválida)",
        "    Si coordenadas_obs no existe  →  coord_valida_ola = NA",
        "  - 2016: valida con bounding box del territorio colombiano",
        "    (lat entre -4.2° y 12.5°; lon entre -79.0° y -66.8°).",
        "    Dentro del bbox  →  coord_valida_ola = 1",
        "    Fuera del bbox   →  coord_valida_ola = 0",
        "    Coordenada ausente  →  coord_valida_ola = NA",
        "  INTERPRETACIÓN: solo los hogares con coord_valida_ola = 1 serán",
        "  consultados en la API de Google Street View.",
        "  Si el porcentaje de coordenadas decimales es 0 %, los nombres de",
        "  las variables en el archivo no coinciden con los configurados.",
    ]

    for ola, grp in panel.groupby("ola"):
        n = len(grp)
        if "lat_decimal" in grp.columns:
            n_lat = grp["lat_decimal"].notna().sum()
            pct   = 100 * n_lat / n if n else 0
            lines.append(
                f"  Ola {ola} — con coordenadas decimales: {n_lat:,}/{n:,} ({pct:.1f} %)"
            )
        if "coord_valida_ola" in grp.columns:
            dist = grp["coord_valida_ola"].value_counts(dropna=False).to_dict()
            lines.append(f"            coord_valida_ola: {dist}")

    lines += [
        "",
        "── 3. CAMBIO DE RESIDENCIA ───────────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Indica si cada hogar cambió de domicilio entre olas consecutivas,",
        "  comparando las coordenadas del hogar en la ola anterior y la actual.",
        "  Fuente: lat_decimal y lon_decimal construidas en la sección 2.",
        "",
        "  LÓGICA",
        "  Se calcula la distancia en metros entre las coordenadas del hogar",
        "  en la ola anterior y en la ola actual usando la fórmula de Haversine",
        "  (considera la curvatura de la Tierra). Si la distancia supera la",
        "  tolerancia, el hogar se considera que cambió de residencia.",
        f"  Tolerancia: {cfg['tolerancia_residencia_m']} m — umbral bajo el cual se",
        "  asume mismo domicilio; 50 m distingue casas distintas incluso en",
        "  zonas urbanas densas.",
        "  Clave de enlace: consecutivo (2010→2013); llave (2013→2016).",
        "  Define qué columna empareja cada hogar con su versión anterior.",
        "  Los sub-hogares divididos (es_split=1) heredan la coordenada de",
        "  referencia del hogar original del que se separaron.",
        "",
        "  Valores posibles de cambio_residencia_ola:",
        "    0  =  distancia < tolerancia → mismo domicilio",
        "    1  =  distancia >= tolerancia → cambió de residencia",
        "    NA =  sin referencia anterior (primera ola del hogar)",
        "          o coordenada faltante en alguna de las dos olas",
        "",
        "  Distribución:",
    ]

    if "cambio_residencia_ola" in panel.columns:
        n_total = len(panel)
        dist    = panel["cambio_residencia_ola"].value_counts(dropna=False)
        etiquetas = {
            0:     "sin cambio de residencia    (0)",
            1:     "con cambio de residencia    (1)",
            pd.NA: "sin dato / primera ola      (NA)",
        }
        for val, cnt in dist.items():
            etiq = etiquetas.get(val, str(val))
            lines.append(f"    {etiq}: {cnt:,} ({100*cnt/n_total:.1f} %)")
        lines += [
            "",
            "  INTERPRETACIÓN: si el 100 % es NA, las coordenadas no estaban",
            "  disponibles en ninguna ola (ver sección 2). Una vez corregidos",
            "  los nombres de las variables, esta sección mostrará valores 0 y 1",
            "  para hogares de 2013 y 2016, y NA solo para los de 2010 (sin",
            "  ola anterior de referencia) y hogares sin coordenada en alguna ola.",
        ]
    else:
        lines.append("    (variable no calculada — coordenadas no disponibles)")

    lines += ["", sep]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  00_construir_panel_coordenadas.py")
    log.info("  Si ves ModuleNotFoundError, el entorno virtual no esta")
    log.info("  activado o las dependencias no estan instaladas.")
    log.info("  Solucion:")
    log.info("    Mac/Linux : source .venv/bin/activate")
    log.info("    Windows   : .venv\\Scripts\\activate")
    log.info("    Luego     : pip install -r requirements.txt")
    log.info("=" * 62)
    cfg     = CONFIG                              # referencia al diccionario de configuración
    out_dir = Path(cfg["output_dir"])             # ruta de la carpeta de salida como Path
    out_dir.mkdir(parents=True, exist_ok=True)    # crea la carpeta y las intermedias si no existen

    # ── Paso 1: carga de datos ────────────────────────────────────────────────
    ola_dfs = {}    # {año: DataFrame} que se llenará con los datos de cada ola
    for ano in sorted(cfg["archivos"]):
        log.info(f"\n── Ola {ano} ─────────────────────────────────────────")
        df = cargar_ola(ano, cfg)               # combina rural + urbano
        if df.empty:                            # ningún archivo disponible para esta ola
            log.warning(f"Ola {ano} omitida: sin datos disponibles.")
            continue
        df = agregar_es_split(df, ano)          # agrega indicador de hogar dividido
        ola_dfs[ano] = df                       # almacena el DataFrame en el diccionario

    if not ola_dfs:
        log.error("-" * 62)
        log.error("ERROR: No se cargo ninguna ola de datos.")
        log.error(f"Carpeta de datos crudos configurada: {_DATA_RAW.resolve()}")
        if not _DATA_RAW.exists():
            log.error("ESA CARPETA NO EXISTE en este equipo.")
            log.error("Crea la carpeta y copia los datos ELCA con esta estructura:")
            log.error("  raw/elca_2010/RHogar-csv.tab  y  UHogar-csv.tab")
            log.error("  raw/elca_2013/RHogar-csv.tab  y  UHogar-csv.tab")
            log.error("  raw/elca_2016/RHogar-csv.tab  y  UHogar-csv.tab")
        else:
            log.error("La carpeta raw/ existe pero ningun archivo se cargo.")
            log.error("Lee los avisos WARNING anteriores para ver que archivo fallo.")
            log.error("Verifica que los nombres coincidan con CONFIG['archivos'].")
        log.error("-" * 62)
        sys.exit(1)

    # ── Paso 2: coordenadas — conversión y validación ────────────────────────
    for ano, df in ola_dfs.items():
        log.info(f"\n── Coordenadas ola {ano} ────────────────────────────")
        df = construir_coords_decimales(df, ano, cfg)      # agrega lat_decimal, lon_decimal
        df = validar_coords(df, ano, cfg)                  # agrega coord_valida_ola
        ola_dfs[ano] = df                                  # actualiza el DataFrame

    # ── Paso 3: panel largo ───────────────────────────────────────────────────
    log.info("\n── Construyendo panel largo ──────────────────────────────")
    panel = construir_panel(ola_dfs, cfg)       # concatena olas en formato largo

    # ── Paso 4: verificación de identificadores ───────────────────────────────
    log.info("\n── Verificando identificadores ───────────────────────────")
    info_ids = verificar_identificadores(panel)  # comprueba unicidad y reporta splits

    # ── Paso 5: cambio de residencia ──────────────────────────────────────────
    log.info("\n── Calculando cambio de residencia ───────────────────────")
    panel = calcular_cambio_residencia(panel, cfg)  # agrega cambio_residencia_ola

    # ── Paso 6: exportación ───────────────────────────────────────────────────
    log.info("\n── Exportando resultados ─────────────────────────────────")

    csv_path = out_dir / "panel_coordenadas.csv"            # ruta del CSV de salida
    panel.to_csv(csv_path, index=False, encoding="utf-8")   # guarda sin índice de fila
    log.info(f"  CSV exportado: {csv_path}")

    reporte = generar_reporte(panel, info_ids, cfg)  # construye el texto
    txt_path = out_dir / "reporte_calidad_coordenadas.txt"         # ruta del TXT de salida
    txt_path.write_text(reporte, encoding="utf-8")                 # escribe en disco
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)   # imprime el reporte en consola para revisión inmediata

if __name__ == "__main__":
    main()   # ejecuta el pipeline solo cuando el script se corre directamente