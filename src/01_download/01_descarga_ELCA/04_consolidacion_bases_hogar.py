"""
Encuesta Longitudinal Colombiana (ELCA)
Consolidación de las bases de datos de hogares – olas 2010, 2013 y 2016.

Autor: Jonathan Melo Sarta
Fecha: Junio 2026

Contexto
--------
Los módulos RHogar-csv.tab y UHogar-csv.tab recogen información a nivel del
hogar: características de la vivienda, acceso a servicios públicos, tenencia,
activos del hogar (electrodomésticos, bienes durables), créditos y deudas,
redes de apoyo (transferencias, ayudas), afiliación a programas sociales y
percepción de riesgo ante desastres naturales.  Son las bases más amplias de
la ELCA en cuanto a número de columnas (hasta 522 por ola/zona).

Lógica general
--------------
Las seis bases (U y R × 2010, 2013, 2016) ya vienen en formato ANCHO: una
fila por hogar.  El script se limita a:
  1. Leer cada archivo fuente.
  2. Aplicar correcciones de codificación (caracteres '???' y '??').
  3. Concatenar las seis bases en un único dataset longitudinal;
     las columnas ausentes en alguna ola/zona quedan como NaN.

Unidad de análisis
------------------
Sub-hogar × ola × zona.  Los hogares que se dividen entre olas quedan como
filas separadas, identificadas por su propia llave/llave_n16.
En 2010 no hay información de divisiones; la unidad es simplemente hogar.

Identificadores incluidos
-------------------------
- 2010 : ola, zona, consecutivo
- 2013 : ola, zona, zona_2010, consecutivo, hogar, llave
- 2016 : ola, zona_2010, zona_2016, consecutivo, hogar, llave, hogar_n16, llave_n16

Nota sobre las columnas 'zona' y 'region' en 2016
---------------------------------------------------
En 2010 y 2013 existen las columnas 'zona' ('Urbano'/'Rural') y 'region'.
En 2016 esas columnas NO existen en el archivo fuente; sus equivalentes son
'zona_2016' y 'region_2016'.  Este script NO unifica esas columnas: se
concatenan tal cual vienen, por lo que 'zona' y 'region' quedan en NaN para
todas las filas de 2016 (mientras que 'zona_2016'/'region_2016' solo están
pobladas en esa ola).  La unificación de 'zona' y 'region' para que
representen siempre el valor vigente en cada ola se realiza en
src/03_clean/01_limpieza_base_hogar.py, que parte de este
output (hogar_elca_longitudinal.parquet) y genera la base
hogar_elca_longitudinal_clean.parquet.  La columna 'zona_2010' indica la
zona de origen del hogar al momento de la primera ola y está presente en
2013 y 2016.

Codificación y correcciones
-----------------------------
Los archivos contienen '???' (y en algunos casos '??') donde deberían
aparecer caracteres especiales del español (á, é, í, ó, ú, ñ, ü, etc.).
Este problema afecta especialmente:
  - Variables categóricas estructuradas: region, material_pisos,
    material_paredes, tenencia_vivienda, obtencion_agua, servicio_sanitario,
    eliminan_basura, energia_cocinan, dcto_vivienda, tipo_deuda_*, religion*.
  - Indicadores binarios Sí/No: columnas sp_*, act_*, uay_*, retpag_*
    en 2016 U usan 'S???' en lugar de 'Sí'; las columnas uay_* en 2013 R
    y 2016 R usan 'S??' en lugar de 'Sí'.
  - Campos de texto libre (conquien_cual_*, aquien_cual_*, etc.) pueden
    retener artefactos '???' residuales para respuestas poco frecuentes no
    cubiertas por el diccionario de correcciones.

Diferencias de columnas entre olas y zonas
--------------------------------------------
El número de columnas varía significativamente entre olas y zonas porque
cada instrumento incorporó, eliminó o renombró módulos:

  2010 U: 451 cols, 5 275 filas  – No tiene columnas de crédito con plazo
           explícito (pacto_meses_*) ni módulo de seguros detallado.
  2010 R: 384 cols, 4 578 filas  – Incluye módulo de choques internos
           (choquec_*, hizo*c_*) ausente en 2010 U; carece de módulo de
           seguros y de algunos identificadores de región.
  2013 U: 468 cols, 4 910 filas  – Agrega módulo de seguros, identificador
           de llave y módulo detallado de créditos (pacto_meses_*).
  2013 R: 522 cols, 4 351 rows   – Incluye módulo agrícola (act_otrosing,
           afectaciones por desastres vivien_*/servag_*/alcant_*) ausente en U.
  2016 U: 396 cols, 4 860 filas  – Renombra destino1_* → destino2016_*,
           agrega llave_n16 / hogar_n16 y módulo de tarjetas (hogar_tarjeta).
  2016 R: 390 cols, 3 958 filas  – Similar a 2016 U sin módulo de tarjetas.

  Las columnas ausentes en una ola/zona quedan como NaN al concatenar.

Dimensiones del output esperado
---------------------------------
  6 bases concatenadas → 27 932 filas × 827 columnas (unión de todas las olas y zonas)
9 853 hogares únicos por consecutivo
"""

import os

from pathlib import Path

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = str(PROJECT_ROOT / "data" / "interim" / "raw")
OUTPUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal.parquet")

# ─── Identificadores por ola ─────────────────────────────────────────────────
# Misma jerarquía que en los demás módulos de la ELCA:
#   2010 → consecutivo
#   2013 → llave = consecutivo + hogar
#   2016 → llave_n16 (escisiones adicionales dentro de 2016)

IDS_POR_OLA = {
    2010: {"key": "consecutivo",  "links": []},
    2013: {"key": "llave",        "links": ["consecutivo", "hogar"]},
    2016: {"key": "llave_n16",    "links": ["consecutivo", "llave", "hogar_n16"]},
}

# ─── Correcciones de codificación ────────────────────────────────────────────
# Se aplican sobre TODAS las columnas de texto mediante reemplazo de subcadenas
# (str.replace con regex=False).
#
# ORDEN IMPORTANTE: 'S???' (3 signos) debe preceder a 'S??' (2 signos)
# para que el reemplazo del patrón más largo se aplique primero y no deje
# caracteres sueltos.  La misma lógica aplica para cualquier par de patrones
# donde uno es prefijo del otro.
#
# Las correcciones están organizadas temáticamente para facilitar el
# mantenimiento.  Los campos de texto libre poco frecuentes (conquien_cual_*,
# aquien_cual_*, uay_otros_cual, etc.) pueden retener '???' residuales para
# respuestas individuales no cubiertas por este diccionario.

CORRECCIONES_HOGAR = {

    # ── Indicadores binarios Sí/No ────────────────────────────────────────────
    # 2016 U: columnas sp_*, act_*, retpag_*, seguro_*, obrero, hogar_tarjeta,
    #         credito_*, tratan_agua, religion, equip_*, ayu_*, eay_*
    "S???":                             "Sí",
    # 2013 R y 2016 U/R: columnas uay_* (uso del auxilio)
    "S??":                              "Sí",

    # ── Región geográfica (region, RegionLb, region_2016) ─────────────────────
    "Bogot???":                         "Bogotá",
    "Atl???ntica":                      "Atlántica",
    "Pac???fica":                        "Pacífica",

    # ── Tipo de vivienda ──────────────────────────────────────────────────────
    "vag???n":                          "vagón",
    "embarcaci???n":                    "embarcación",
    "ind???gena":                       "indígena",

    # ── Materiales de construcción (material_pisos, material_paredes) ─────────
    "m???rmol":                         "mármol",
    "parqu???":                         "parqué",
    "tabl???n":                         "tablón",
    "ca???a":                           "caña",
    "cart???n":                         "cartón",
    "pl???sticos":                      "plásticos",

    # ── Agua, saneamiento y basura ────────────────────────────────────────────
    # obtencion_agua, eliminan_basura, servicio_sanitario
    "R???o":                            "Río",
    "r???o":                            "río",
    "ca???o":                           "caño",
    "bald???o":                         "baldío",
    "p???blico":                        "público",
    "p???blica":                        "pública",
    "P???blicos":                       "Públicos",
    "P???blica":                        "Pública",
    "s???ptico":                        "séptico",
    "jag???ey":                         "jagüey",

    # ── Alimentación y cocina ─────────────────────────────────────────────────
    # sitio_alimentos, energia_cocinan
    "s???lo":                           "sólo",
    "tambi???n":                        "también",
    "Le???a":                           "Leña",
    "le???a":                           "leña",
    "Carb???n":                         "Carbón",
    "carb???n":                         "carbón",
    "petr???leo":                       "petróleo",

    # ── Tenencia y documentos de vivienda ─────────────────────────────────────
    # tenencia_vivienda, dcto_vivienda, compro_vivcuando
    "est???n":                          "están",
    "Instrumentos P???blicos":          "Instrumentos Públicos",
    "Adquiri???":                       "Adquirió",
    "Compr???":                         "Compró",
    "construy???":                      "construyó",

    # ── Religión (religion1, religion2, religion1_2013, religion2_2013) ────────
    "Cat???lica":                       "Católica",
    "Evang???lica":                     "Evangélica",
    "evang???lico":                     "evangélico",
    "Jehov???":                         "Jehová",
    "agn???stico":                      "agnóstico",

    # ── Créditos y deudas ─────────────────────────────────────────────────────
    # tipo_deuda_*, aquien_deben_*, con_quien_*, destino*_*, pacto_meses_*,
    # no_credito_sf*, rechazo_credito
    "Cr???dito":                        "Crédito",
    "cr???dito":                        "crédito",
    "dep???sito":                       "depósito",
    "inversi???n":                      "inversión",
    "Inversi???n":                      "Inversión",
    "Educaci???n":                      "Educación",
    "educaci???n":                      "educación",
    "Recreaci???n":                     "Recreación",
    "celebraciones":                    "celebraciones",   # ya correcto
    "mercanc???a":                      "mercancía",
    "Ganader???a":                      "Ganadería",
    "garant???a":                       "garantía",
    "Garant???a":                       "Garantía",
    "veh???culo":                       "vehículo",
    "Veh???culo":                       "Vehículo",
    "veh???culos":                      "vehículos",
    "electrodom???sticos":              "electrodomésticos",
    "aprobar???an":                     "aprobarían",
    "tr???mite":                        "trámite",
    "est??? en":                        "está en",
    "cat???logo":                       "catálogo",
    "compensaci???n":                   "compensación",
    "empe???o":                         "empeño",
    "agr???colas":                      "agrícolas",
    "???cu???l?":                        "¿cuál?",
    "endeudado":                        "endeudado",       # ya correcto

    # ── Seguros ───────────────────────────────────────────────────────────────
    # no_seguros, no_seguros_cual
    "d???nde":                          "dónde",
    "D???nde":                          "Dónde",
    "conf???a":                         "confía",
    "inter???s":                        "interés",

    # ── Desastres naturales y afectaciones ────────────────────────────────────
    # vivien_*, servag_*, alcant_*, otro_desastre_cual
    "afectaci???n":                     "afectación",
    "el???ctrico":                      "eléctrico",
    "rayo":                             "rayo",            # ya correcto

    # ── Programas sociales y ayudas ───────────────────────────────────────────
    # recibio_ayu_alim, envio_ayu_alim, otra_financiacion_cual
    "extrajudicial":                    "extrajudicial",   # ya correcto
    "restituci???n":                    "restitución",
    "alcald???a":                       "alcaldía",

    # ── Otras actividades e ingresos ─────────────────────────────────────────
    # act_otrosing_cual, retpag_otro_cual
    "Telefon???a":                      "Telefonía",
    "???rboles":                        "árboles",

    # ── Personas y relaciones familiares ─────────────────────────────────────
    # conquien_cual_*, aquien_cual_*, rechazo_cred_cual, uay_otros_cual
    "Se???or":                          "Señor",
    "se???or":                          "señor",
    "Se???ora":                         "Señora",
    "se???ora":                         "señora",
    "Ni???o":                           "Niño",
    "ni???o":                           "niño",
    "Ni???a":                           "Niña",
    "ni???a":                           "niña",
    "Due???o":                          "Dueño",
    "due???o":                          "dueño",
    "Cu???ada":                         "Cuñada",
    "cu???ada":                         "cuñada",
    "Cu???ado":                         "Cuñado",
    "cu???ado":                         "cuñado",
    "Se???or":                          "Señor",
    "cosm???ticos":                     "cosméticos",
    "Tr???nsito":                       "Tránsito",
    "tr???nsito":                       "tránsito",
}


# ─── Normalización de region_2016 ────────────────────────────────────────────
#
# En 2016, la columna region_2016 viene con un prefijo numérico ('1. ', '2. ',
# etc.) y con nombres que difieren de los usados en region/RegionLb de 2010 y
# 2013.  Sin normalización, 'Atlántica' (2010/2013) y '1. Atlántica' (2016)
# quedan como categorías distintas aunque representan lo mismo.
#
# Adicionalmente, 2016 U tiene 'Atlántica Media' codificado con '??' (2 signos)
# en lugar de '???' → '6. Atl??ntica Media' no es corregido por CORRECCIONES_HOGAR
# que solo tiene el patrón de 3 signos.  El mapa lo maneja directamente con
# la clave '6. Atl??ntica Media'.
#
# Se usa Series.replace() (mapa valor → valor, no subcadena) para no afectar
# otras columnas ni otros valores que no sean exactamente los esperados.

REGION_2016_MAPA = {
    # 2016 U y R (después de corrección '???')
    "1. Atlántica":         "Atlántica",
    "2. Oriental":          "Oriental",
    "3. Central":           "Central",
    "4. Pacífica":          "Pacífica",
    "5. Bogotá":            "Bogotá",           # solo 2016 U
    "6. Atlántica Media":   "Atlántica Media",  # 2016 U (??? → á corregido)
    "6. Atl??ntica Media":  "Atlántica Media",  # 2016 U (2 signos, sin corrección previa)
    "6. Atlantica Media":   "Atlántica Media",  # 2016 R (sin tilde en fuente)
    "7. Cundi_boyacense":   "Cundi-Boyacense",  # guion bajo → guion, capitalización
    "8. Eje cafetero":      "Eje Cafetero",     # capitalización
    "9. Centro Oriente":    "Centro-Oriente",   # falta guion
}


# ─── Normalización de region y RegionLb en 2013 U ────────────────────────────
#
# En 2013 U la columna 'region' trae dos valores que difieren de la forma
# canónica usada en 2010 y 2013 R:
#   - 'Atlantica-Media'  → sin tilde y con guion en lugar de espacio
#   - 'Orinoquia�Amazonas' → el separador original (guion o raya) fue
#     reemplazado por � (U+FFFD, carácter de reemplazo Unicode) al leer
#     el archivo con la codificación equivocada
#
# En 2013 U la columna 'RegionLb' usa el mismo carácter � en lugar de
# las vocales con tilde ('á', 'ó', 'í'), también producto de la mala lectura
# de la codificación original (probablemente Windows-1252):
#   - 'Atl�ntica'       → cada � reemplaza exactamente una 'á'
#   - 'Atl�ntica Media' → ídem
#   - 'Bogot�'          → reemplaza la 'á' final
#   - 'Pac�fica'        → reemplaza la 'í'
#
# CORRECCIONES_HOGAR no puede cubrir estos casos porque sus patrones son
# subcadenas '???' (ASCII 0x3F × 3) mientras que � (U+FFFD) es un
# carácter Unicode distinto y no es capturado por str.replace('???', ...).
#
# Se usa Series.replace() (mapa valor → valor) para no afectar celdas que
# contengan � en un contexto distinto.

REGION_2013U_MAPA = {
    "Atlantica-Media":          "Atlántica Media",   # sin tilde, guion → espacio
    "Orinoquia�Amazonas":  "Orinoquía-Amazonas",  # � = separador original
}

REGIONLB_2013U_MAPA = {
    "Atl�ntica":           "Atlántica",
    "Atl�ntica Media":     "Atlántica Media",
    "Bogot�":              "Bogotá",
    "Pac�fica":            "Pacífica",
}


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def corregir_columnas(df: pd.DataFrame, correcciones: dict) -> pd.DataFrame:
    """
    Aplica reemplazos exactos de subcadena en todas las columnas de texto del
    DataFrame.  Devuelve una copia con las correcciones aplicadas.

    El orden de los pares en 'correcciones' importa cuando un patrón es
    prefijo de otro (p. ej. 'S???' debe ir antes que 'S??').
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        for patron, correcto in correcciones.items():
            df[col] = df[col].str.replace(patron, correcto, regex=False)
    return df


# ─── Procesamiento por ola ────────────────────────────────────────────────────

def procesar_2010(tipo: str) -> pd.DataFrame:
    """
    Lee UHogar o RHogar de 2010 y corrige la codificación.

    Características de 2010:
    - Solo 'consecutivo' como identificador de hogar (sin llave ni hogar_n16).
    - La columna 'zona' indica 'Urbano' o 'Rural'.
    - UHogar tiene 451 columnas; RHogar tiene 384 columnas.
    - RHogar incluye columnas de choques internos (choquec_*, hizo*c_*)
      que no existen en UHogar.
    - Ambas zonas presentan '???' en variables categóricas estructuradas y
      en algunos campos de texto libre.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2010", f"{tipo}Hogar-csv.tab")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = corregir_columnas(df, CORRECCIONES_HOGAR)

    print(f"    {zona} 2010: {df.shape[0]:>5} filas × {df.shape[1]:>3} columnas")
    return df


def procesar_2013(tipo: str) -> pd.DataFrame:
    """
    Lee UHogar o RHogar de 2013 y corrige la codificación.

    Características de 2013:
    - Agrega 'hogar' y 'llave' (llave = consecutivo + hogar) como
      identificadores para sub-hogares resultantes de divisiones.
    - Incluye 'zona_2010' (zona original en primera ola) y 'zona' (ola actual).
    - UHogar tiene 468 columnas; RHogar tiene 522 columnas.
    - RHogar incluye módulo de afectaciones por desastres (vivien_*/servag_*
      /alcant_*) y actividades agrícolas adicionales.
    - 2013 U presenta caracteres U+FFFD (�) en region (2 valores) y en
      RegionLb (4 valores): producto de una codificación original distinta
      (probable Windows-1252) que no es capturada por CORRECCIONES_HOGAR.
      Se corrige con REGION_2013U_MAPA y REGIONLB_2013U_MAPA.
    - 2013 R presenta '???' en region, RegionLb, variables categóricas
      estructuradas y en varios campos de créditos y programas sociales.
    - Las columnas uay_* (uso del auxilio) en 2013 R presentan 'S??' (2 signos)
      en lugar de 'Sí'.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2013", f"{tipo}Hogar-csv.tab")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = corregir_columnas(df, CORRECCIONES_HOGAR)

    # Correcciones adicionales para 2013 U: U+FFFD en region y RegionLb
    if tipo == "U":
        if "region" in df.columns:
            df["region"] = df["region"].replace(REGION_2013U_MAPA)
        if "RegionLb" in df.columns:
            df["RegionLb"] = df["RegionLb"].replace(REGIONLB_2013U_MAPA)

    print(f"    {zona} 2013: {df.shape[0]:>5} filas × {df.shape[1]:>3} columnas")
    return df


def procesar_2016(tipo: str) -> pd.DataFrame:
    """
    Lee UHogar o RHogar de 2016 y corrige la codificación.

    Características de 2016:
    - Agrega 'hogar_n16' y 'llave_n16' como identificadores de sub-hogar
      para escisiones adicionales ocurridas dentro de esta ola.
    - El archivo fuente NO tiene columnas 'zona' ni 'region': usa
      'zona_2010'/'zona_2016' y 'RegionLb'/'region_2016' en su lugar.  Esta
      función NO las unifica (ver nota en el docstring del módulo); 'zona'
      y 'region' quedan en NaN para 2016 hasta la limpieza posterior en
      src/03_clean.
    - Los destinos de crédito se renombran a 'destino2016_*' (en 2013 eran
      'destino2013_*'; en 2010 eran 'destino1_*', 'destino2_*').
    - UHogar (396 cols) presenta '???' en region/RegionLb, variables
      categóricas y en indicadores binarios Sí/No ('S???').
    - RHogar (390 cols) presenta '???' en las mismas estructuras que UHogar.
    - Las columnas uay_* en ambas zonas usan 'S??' (2 signos) para 'Sí'.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2016", f"{tipo}Hogar-csv.tab")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = corregir_columnas(df, CORRECCIONES_HOGAR)

    # Normalizar region_2016: quitar prefijo numérico y uniformizar nombres
    # para que sean comparables con region/RegionLb de 2010 y 2013.
    if "region_2016" in df.columns:
        df["region_2016"] = df["region_2016"].replace(REGION_2016_MAPA)

    print(f"    {zona} 2016: {df.shape[0]:>5} filas × {df.shape[1]:>3} columnas")
    return df



# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Lee, corrige y concatena las seis bases de hogares (U y R × 3 olas).

    La concatenación usa la unión de columnas (outer join implícito de
    pd.concat); las columnas ausentes en una ola/zona quedan como NaN.

    Ordenamiento de columnas del output
    ------------------------------------
    1. Identificadores (de más reciente a más antiguo):
       llave_n16, llave, consecutivo, consecutivo_c, hogar_n16, hogar,
       ola, zona, zona_2010, zona_2016, region, RegionLb, region_2016.
    2. Resto de columnas en el orden original de aparición (no reordenadas),
       para preservar la agrupación temática de los instrumentos originales.

    Retorna el DataFrame final.
    """
    frames = []

    for tipo in ["U", "R"]:
        print(f"\n  Procesando {tipo}Hogar...")

        print("    Leyendo 2010...")
        frames.append(procesar_2010(tipo))

        print("    Leyendo 2013...")
        frames.append(procesar_2013(tipo))

        print("    Leyendo 2016...")
        frames.append(procesar_2016(tipo))

    # Concatenar: la unión de columnas llena con NaN las celdas ausentes.
    # .copy() desfragmenta el DataFrame tras el concat para evitar advertencias
    # de rendimiento al asignar columnas derivadas.
    base_final = pd.concat(frames, axis=0, ignore_index=True).copy()

    # Reordenar columnas: identificadores primero, luego el resto.
    # Se filtran los ids que efectivamente existen en el DataFrame resultante.
    all_id_cols = [
        "llave_n16", "llave", "consecutivo", "consecutivo_c",
        "hogar_n16", "hogar",
        "ola", "zona", "zona_2010", "zona_2016",
        "region", "RegionLb", "region_2016",
    ]
    id_cols      = [c for c in all_id_cols if c in base_final.columns]
    otras_cols   = [c for c in base_final.columns if c not in id_cols]
    base_final   = base_final[id_cols + otras_cols]

    # Normalizar columnas de tipo object con valores mixtos (str + float) que
    # surgen cuando la misma columna es numérica en una ola y de texto en otra.
    # pyarrow rechaza arrays object con mezcla de tipos; se unifican como str
    # preservando NaN (que parquet serializa como null).
    for col in base_final.select_dtypes(include="object").columns:
        no_nulos = base_final[col].dropna()
        if len(no_nulos) > 0 and not no_nulos.apply(lambda x: isinstance(x, str)).all():
            base_final[col] = base_final[col].where(
                base_final[col].isna(),
                base_final[col].astype(str),
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    base_final.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\nBase final guardada en : {OUTPUT_PATH}")
    print(f"Dimensiones            : {base_final.shape[0]:>6} filas × {base_final.shape[1]:>4} columnas")
    print(f"Olas presentes         : {sorted(base_final['ola'].unique())}")
    print(f"Hogares únicos         : {base_final['consecutivo'].nunique():>6}")
    print()
    print("  Filas por ola y zona:")
    for ola_val in sorted(base_final["ola"].unique()):
        sub = base_final[base_final["ola"] == ola_val]
        # Detectar la columna de zona disponible según la ola
        zona_col = "zona" if "zona" in sub.columns and sub["zona"].notna().any() else "zona_2016"
        if zona_col in sub.columns:
            for zona_val, n in sub[zona_col].value_counts().sort_index().items():
                print(f"    Ola {ola_val} – {zona_val:<10}: {n:>5} filas")
        else:
            print(f"    Ola {ola_val}: {len(sub):>5} filas")

    return base_final


if __name__ == "__main__":
    base_final = main()
