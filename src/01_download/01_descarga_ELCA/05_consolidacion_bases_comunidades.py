"""
Encuesta Longitudinal Colombiana (ELCA)
Consolidación de las bases de datos de comunidades – olas 2010, 2013 y 2016.

Autor: Jonathan Melo Sarta
Fecha: Junio 2026

Contexto
--------
Los módulos UComunidades-csv.tab y RComunidades-csv.tab recogen información
a nivel de COMUNIDAD (barrio en la muestra urbana, vereda en la rural),
no a nivel de hogar individual.  Cada fila es una comunidad encuestada en
una ola determinada.  El cuestionario lo responden los líderes locales
(hasta 6 por comunidad) y registra: equipamiento e infraestructura del
barrio/vereda, obras y proyectos prioritarios, problemas de convivencia y
seguridad, presencia de grupos armados, conflictos de tierras (solo rural),
afectaciones por desastres naturales, y organización social.

Unidad de análisis
------------------
Comunidad × ola: una fila por cada comunidad (identificada por consecutivo_c)
en cada ola en que fue encuestada.  Los hogares del panel se vinculan a sus
comunidades mediante la columna consecutivo_c, que es común a este módulo y
a los archivos de hogar (UHogar / RHogar).

A diferencia del resto de módulos, Comunidades NO tiene identificadores de
sub-hogar (no existen columnas consecutivo, llave ni llave_n16) porque la
encuesta se aplica a la comunidad, no al hogar ni al individuo.

Identificadores presentes
-------------------------
- 2010 U : ola, zona, region, consecutivo_c                (132 columnas)
- 2010 R : ola, zona, region, consecutivo_c                (235 columnas)
- 2013 U : ola, zona, consecutivo_c                        (271 columnas)
- 2013 R : ola, zona, consecutivo_c                        (408 columnas)
- 2016 U : ola, zona_2016, consecutivo_c                   (280 columnas)
- 2016 R : ola, zona_2016, consecutivo_c                   (424 columnas)

Nota sobre la columna 'zona'
-----------------------------
En 2010 y 2013 la columna se llama 'zona' (valores 'Urbano'/'Rural' en 2010,
'Urbana'/'Rural' en 2013).  En 2016 pasa a llamarse 'zona_2016'; no existe
una columna 'zona' de referencia anterior como en Hogar.  Al concatenar,
'zona' quedará NaN para las filas de 2016.

Diferencias estructurales clave entre olas y zonas
----------------------------------------------------
Las diferencias más importantes (además del número de columnas) son:

  2010 U  : El archivo CSV tiene encabezado; 132 columnas.  No incluye
             módulos de conflictos de tierras (exclusivo de R) ni de
             afectaciones por desastres (añadido en 2013).
  2010 R  : El archivo CSV NO tiene encabezado; los nombres de columna
             se recuperan del archivo RComunidades.tab que sí los incluye.
             235 columnas.  Incluye módulo de conflictos de tierra
             (cod_conf_*, anos_resolver_*, resolvio_con_*) y módulo de
             actividades agrícolas (fincas_hoy, venden_prod_agr, btol_*).
  2013 U  : 271 columnas.  Añade módulo de afectaciones por desastres
             (inu_*, ava_*, desb_*, desl_*, ven_*, otrd_*) y módulo de
             ayuda humanitaria (clfq_*, rec_*).  Agrega 'educ_lider*'.
             Los valores de inu_*/ ava_* y similares tienen espacios
             iniciales (' Destru???da Totalmente') que se eliminan al
             aplicar str.strip() tras las correcciones de codificación.
  2013 R  : 408 columnas.  Incluye módulo de conflictos de tierra,
             desastres, y variables de mercado agropecuario (venden_prod_agr,
             n_hogares_*).
  2016 U  : 280 columnas.  Añade org_* (organizaciones sociales),
             fb_* (si el hogar tiene familia en esa organización) y
             prt_* (tipo de protesta en que participa la comunidad).
             Renombra 'zona' a 'zona_2016'.  Columnas binarias usan 'S???'
             (3 signos) en vez de 'Si'.
  2016 R  : 424 columnas.  Igual a 2016 U más variables agrícolas adicionales
             (pp_*, trab_*, hay_minerilegal) y módulo de desastres ampliado
             (sequias, vendavales).

Codificación y correcciones
-----------------------------
Los archivos presentan '???' y '??' donde deberían aparecer caracteres
especiales del español.

  - CORRECCIONES_COMUNES (3 signos '???'): cubren todos los archivos.
    Incluyen 'S???' → 'Sí' (columnas binarias de 2016 U/R, y en 2013 R
    columnas armadoseran_*, amenazas, etc.).

  - CORRECCIONES_2010R (2 signos '??'): exclusivas del archivo
    RComunidades-csv.tab de 2010, cuya codificación fuente difiere de los
    demás.  Aplican sobre obra_prioritaria*, proy_prioritario*, campos
    libres de texto y nombres de obras.

    IMPORTANTE DE ORDEN: '???' → 'Sí' se aplica ANTES que '??' → 'Sí'
    para evitar que la corrección de 2 signos deje el tercer '?' suelto
    en un valor que debía recibir la corrección de 3 signos.

  - Espacios iniciales: en 2013 U/R algunos valores de los módulos de
    desastres tienen un espacio inicial (p. ej. ' Destru???da Totalmente').
    Se eliminan con str.strip() tras aplicar las correcciones de subcadena.

Decisiones de implementación
------------------------------
Las siguientes decisiones se tomaron tras inspeccionar los seis archivos
fuente antes de escribir el script.  Se documentan aquí para que un lector
futuro entienda el por qué de cada elección sin necesidad de re-inspeccionar
los archivos originales.

  1. Unidad de análisis → consecutivo_c × ola (comunidad × ola)
     El módulo Comunidades carece de los identificadores de hogar que usan
     los demás módulos (consecutivo, llave, llave_n16).  El único
     identificador presente en las seis bases es 'consecutivo_c', que es el
     código de la comunidad (barrio urbano o vereda rural).  Por eso la
     unidad de análisis es la comunidad, no el sub-hogar dividido.
     Los hogares del panel se vinculan a sus comunidades mediante esta misma
     columna consecutivo_c en los archivos UHogar/RHogar.

  2. Header de 2010 RComunidades → nombres extraídos de RComunidades.tab
     El archivo RComunidades-csv.tab de 2010 NO tiene fila de encabezado:
     la primera línea del fichero ya contiene datos.  Si se lee con
     header=0 (predeterminado), la primera fila de datos se convierte
     erróneamente en nombres de columna y se pierde un registro.
     El archivo compañero RComunidades.tab SÍ tiene encabezado (aunque sus
     datos están en formato codificado, no etiquetado).  Ambos archivos
     tienen exactamente 235 columnas, por lo que los nombres se pueden
     asignar posicionalmente: se lee solo la primera fila de .tab con
     nrows=0 para obtener la lista de nombres y se pasa como 'names='
     al leer el -csv.tab con header=None.
     Los demás cinco archivos (UComunidades 2010, U/R 2013, U/R 2016)
     tienen encabezado y se leen con header=0 sin ningún ajuste.

  3. Correcciones '???' → CORRECCIONES_COMUNES, aplica a las 6 bases
     Los 5 archivos con encabezado presentan la codificación '???' (3 signos)
     para los caracteres especiales del español.  Se usa un diccionario único
     (CORRECCIONES_COMUNES) aplicado sobre todas las columnas de texto
     mediante str.replace(regex=False), lo que garantiza consistencia entre
     olas y zonas.  UComunidades 2010 es el único archivo sin ningún
     problema de codificación; las correcciones no alteran sus datos porque
     ningún patrón del diccionario aparece en sus valores.

  4. Correcciones '??' → CORRECCIONES_2010R, exclusivas de 2010 R
     El archivo RComunidades-csv.tab de 2010 es el único de los seis que usa
     '??' (2 signos) en lugar de '???' (3 signos) para codificar caracteres
     especiales.  Los patrones afectados son distintos de los de '???': por
     ejemplo, 'Construcci??n' (2 signos) vs 'Construcci???n' (3 signos).
     Por eso se necesita un segundo diccionario (CORRECCIONES_2010R) que se
     aplica únicamente a 2010 R, siempre DESPUÉS de CORRECCIONES_COMUNES.
     Si se aplicase CORRECCIONES_2010R primero (o a todos los archivos),
     la corrección 'S??' → 'Sí' convertiría 'S???' → 'Sí?' en archivos que
     aún no han sido procesados por la corrección de 3 signos.

  5. Espacios iniciales → str.strip() dentro de corregir_columnas()
     En 2013 U/R, algunos valores de los módulos de desastres (inu_*, ava_*,
     desb_*, desl_*, ven_*, otrd_*) tienen un espacio inicial no imprimible
     (p. ej. ' Destru???da Totalmente' con espacio antes de 'D').  Esto
     provoca que, tras corregir '???', el valor quede como ' Destruida
     Totalmente' (con espacio), que es distinto de 'Destruida Totalmente'
     (sin espacio) y genera valores duplicados en la misma columna para el
     mismo concepto.  Se añade df[col].str.strip() al final de cada columna
     dentro de corregir_columnas(), lo que elimina esos espacios sin afectar
     a los demás archivos.

  6. Tipos mixtos str + float → normalización antes de guardar parquet
     Algunas columnas aparecen como numéricas (float) en una ola y como
     texto (object) en otra.  Al concatenar, pandas las retiene como object
     pero con valores mezclados (strings y floats en la misma serie).
     pyarrow no puede serializar este tipo de array y lanza ArrowTypeError.
     Se detectan y convierten a string puro preservando NaN → null (parquet)
     antes de llamar a to_parquet().  El mismo patrón se usa en el script
     de consolidación de Hogar (04_consolidacion_bases_hogar.py).

Vinculación con el módulo de Hogar
------------------------------------
El módulo de Comunidades no contiene identificadores de hogar; la relación
con los hogares del panel se establece a través de la columna consecutivo_c,
que es común a este archivo y a hogar_elca_longitudinal.parquet.

La relación es de uno a muchos: una comunidad (consecutivo_c) agrupa N
hogares.  Los ~9 853 hogares únicos del panel están distribuidos en las
~780 comunidades del estudio, por eso este archivo tiene 2 330 filas (≈ 780
comunidades × 3 olas) y no 9 853.

Para enriquecer cada hogar con las características de su comunidad:

    hogar = pd.read_parquet("data/processed/hogar_elca_longitudinal.parquet")
    com   = pd.read_parquet("data/processed/comunidades_elca_longitudinal.parquet")

    merged = hogar.merge(
        com,
        on=["consecutivo_c", "ola"],
        how="left",           # conserva todos los hogares aunque falte dato de comunidad
        suffixes=("_hog", "_com"),
    )

El join es por consecutivo_c Y ola porque las características de una
comunidad (obras, seguridad, desastres) cambian entre olas; unir solo por
consecutivo_c mezclaría información de olas distintas.

El how="left" protege los hogares cuya comunidad tiene ola=NaN en 2010
(consecutivo_c 16407 y 22113): esas filas no encontrarían match con un
inner join y se perderían del análisis de hogar.

Para verificar la integridad de la llave antes del merge:

    en_hogar = set(hogar["consecutivo_c"].dropna().unique())
    en_com   = set(com["consecutivo_c"].dropna().unique())
    print("Solo en hogar:", en_hogar - en_com)   # hogares sin dato de comunidad
    print("Solo en com:  ", en_com - en_hogar)   # comunidades sin hogares en el panel

Calidad de datos en el output: NaN en la columna 'ola'
-------------------------------------------------------
El archivo UComunidades-csv.tab de 2010 contiene 2 filas con NaN en la
columna 'ola' (comunidades consecutivo_c 16407 y 22113).  Este es un
problema preexistente en el archivo fuente, no introducido por el script.
El resto de campos de esas filas (zona, region, consecutivo_c y variables
temáticas) están completos y son válidos.  El script los conserva tal cual
para no perder información; el analista debe tenerlos en cuenta al filtrar
por ola.

Dimensiones del output
-----------------------
  2 330 filas × 558 columnas (unión de todas las olas y zonas).

  Desglose por ola y zona:
    Ola 1 – Urbano :  568 filas ×  132 columnas (2 de ellas con ola = NaN)
    Ola 1 – Rural  :  224 filas ×  235 columnas
    Ola 2 – Urbana :  547 filas ×  271 columnas
    Ola 2 – Rural  :  231 filas ×  408 columnas
    Ola 3 – Urbana :  526 filas ×  280 columnas
    Ola 3 – Rural  :  234 filas ×  424 columnas
"""

import os

from pathlib import Path

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = str(PROJECT_ROOT / "data" / "interim" / "raw")
OUTPUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "comunidades_elca_longitudinal.parquet")

# ─── Correcciones de codificación comunes (3 signos '???') ───────────────────
# Se aplican a TODOS los archivos mediante reemplazo de subcadenas
# (str.replace con regex=False), de más largo a más corto para evitar
# reemplazos parciales encadenados.
#
# La corrección 'S???' → 'Sí' va siempre ANTES de cualquier corrección
# que use '??' (2 signos), ya que 'S???' contiene 'S??' como prefijo y
# un reemplazo previo de 2 signos convertiría 'S???' → 'Sí?' (incorrecto).

CORRECCIONES_COMUNES = {

    # ── Indicadores binarios Sí/No ────────────────────────────────────────────
    # 2016 U: columnas hog_icbf, guarderia_icbf, preescolar, esc_primaria,
    #         col_secundaria, rest_escolares, puesto_salud, jac, org_*, fb_*,
    #         grarmados_*, amenazas, atentados, etc.
    # 2013 R / 2016 R: columnas armadoseran_*, btol_* (R 2010), hay_*.
    "S???":                             "Sí",

    # ── Región geográfica ─────────────────────────────────────────────────────
    "Atl???ntica":                      "Atlántica",
    "Pac???fica":                        "Pacífica",
    "Bogot???":                         "Bogotá",

    # ── Líderes comunitarios ──────────────────────────────────────────────────
    # cargo_lider* : 'Político'
    # educ_lider*  : 'Técnico', 'técnica' (en valores de proyectos)
    # acude_solucion, solucion_nopenal, resolvio_con_* : 'líderes'/'líder'
    "Pol???tico":                       "Político",
    "T???cnico":                        "Técnico",
    "t???cnica":                        "técnica",
    "L???deres":                        "Líderes",
    "l???der":                          "líder",

    # ── Infraestructura y servicios ───────────────────────────────────────────
    # enfermo_grave: 'clínica'
    # barren_calles: 'días'
    # vias_barrio  : 'Vías peatonales'
    # acceso, medio_transporte: 'público/a'
    "cl???nica":                        "clínica",
    "p???blico":                        "público",
    "p???blica":                        "pública",
    "P???blicos":                       "Públicos",
    "V???as":                           "Vías",
    "v???as":                           "vías",
    "d???as":                           "días",
    "r???o":                            "río",
    "ca???o":                           "caño",
    "Ca???o":                           "Caño",

    # ── Desastres naturales y afectaciones ────────────────────────────────────
    # Módulos inu_*, ava_*, desb_*, desl_*, ven_*, otrd_* (2013 U/R, 2016 U/R)
    # Valores: 'No se destruyó', 'No tenía', 'Destruida Totalmente',
    #          'Destruida parcialmente', 'Aún no han recibido' (clfq_*)
    # NOTA: las cadenas con espacio inicial (' Destru???da ...') son manejadas
    # correctamente por str.replace (subcadena) y el espacio se elimina luego
    # con str.strip() en corregir_columnas().
    "No se destruy???":                 "No se destruyó",
    "No ten???a":                       "No tenía",
    "Destru???da Totalmente":           "Destruida Totalmente",
    "Destru???da parcialmente":         "Destruida parcialmente",
    "A???n no han recibido":            "Aún no han recibido",

    # ── Obras y proyectos prioritarios ────────────────────────────────────────
    # obra_prioritaria* y proy_prioritario* en 2013 U/R y 2016 U/R
    # Los valores usan 'Construcción', 'remodelación', 'creación', 'Campañas',
    # etc. En 2016 U los valores cambian de formato respecto a 2013
    # (p. ej. 'Ampliación o creación de guarderías').
    "Construcci???n":                   "Construcción",
    "remodelaci???n":                   "remodelación",
    "creaci???n":                       "creación",
    "Creaci???n":                       "Creación",
    "guarder???as":                     "guarderías",
    "vacunaci???n":                     "vacunación",
    "agr???cola":                       "agrícola",
    "Ampliaci???n":                     "Ampliación",
    "Campa???as":                       "Campañas",
    "campa???as":                       "campañas",
    "Campa???a":                        "Campaña",
    "campa???a":                        "campaña",
    "m???s":                            "más",
    "M???s":                            "Más",
    "Ni???ez":                          "Niñez",
    "ni???ez":                          "niñez",
    "Ni???o":                           "Niño",
    "ni???o":                           "niño",
    "Ni???a":                           "Niña",
    "ni???a":                           "niña",

    # ── Conflictos de tierra (R) ──────────────────────────────────────────────
    # cod_conf_*, resolvio_con_*, anos_resolver_* en 2010/2013/2016 R
    "titulaci???n":                     "titulación",
    "Invasi???n":                       "Invasión",
    "invasi???n":                       "invasión",
    "Expulsi???n":                      "Expulsión",
    "expulsi???n":                      "expulsión",
    "usurpaci???n":                     "usurpación",
    "Reclamaci???n":                    "Reclamación",
    "reclamaci???n":                    "reclamación",
    "v???ctimas":                       "víctimas",
    "restituci???n":                    "restitución",
    "cr???ditos":                       "créditos",
    "due???o":                          "dueño",
    "Due???o":                          "Dueño",
    "ra???z":                           "raíz",
    "Peque???os":                       "Pequeños",
    "peque???os":                       "pequeños",
    "peque???as":                       "pequeñas",
    "a???os":                           "años",
    "aparcer???a":                      "aparcería",
    "intervenci???n":                   "intervención",
    "Intervenci???n":                   "Intervención",
    # 'Nunca se resolvió': el fragmento 'resolvi???' cubre también
    # 'Un año o menos – Nunca se resolvió' y 'Varios años – se resolvió'
    "resolvi???":                       "resolvió",
    "legaliz???":                       "legalizó",
    # 'Entre ellos?' tiene '???' como cierre de oración abierta (2013/2016 R)
    "Entre ellos???":                   "Entre ellos",

    # ── Seguridad y grupos armados ────────────────────────────────────────────
    # razon_seguridad, hecho_seguridad, armadoseran_*, acude_solucion
    "comit???s":                        "comités",
    "comit???":                         "comité",
    "Dos o m???s":                      "Dos o más",

    # ── Mercado de tierras (R) ────────────────────────────────────────────────
    # fincas_hoy: 'Más pequeñas que hace 10 años' / 'Más grandes que hace 10 años'
    # inform_compran: 'Directamente al dueño', 'A intermediarios de finca raíz'
    "finca ra???z":                     "finca raíz",

    # ── Barrio y legalidad (U) ────────────────────────────────────────────────
    # barrio_legal: 'Tuvo origen ilegal, pero ya se legalizó'
    # (ya cubierto por el fragmento 'legaliz???')
}

# ─── Correcciones adicionales para 2010 RComunidades (2 signos '??') ─────────
# El archivo RComunidades-csv.tab de 2010 usa '??' (2 signos de interrogación)
# donde los demás archivos usan '???' (3 signos).  Esto refleja una codificación
# fuente distinta para ese archivo en particular.
#
# Estas correcciones se aplican DESPUÉS de CORRECCIONES_COMUNES para evitar
# interferencias entre los patrones de 2 y 3 signos.
#
# Afectan principalmente: obra_prioritaria*, proy_prioritario* y campos de
# texto libre (problemas_otro_cual, camp_otro_cual, proy_*_cual).

CORRECCIONES_2010R = {
    # La cadena 'S???' ya fue corregida a 'Sí' por CORRECCIONES_COMUNES.
    # Solo queda el patrón de 2 signos para este archivo.
    "S??":               "Sí",     # venden_mas10, tierras_arriend10, btol_*

    # Obras prioritarias (letras A., B., C. ... como prefijo):
    # 'Construcci??n y mejoramiento de v??as'
    # 'Construcci??n o remodelaci??n de escuelas y colegios'
    # 'Creaci??n de fuentes de empleo'
    # 'Programas de construcci??n y mejoramiento de la vivienda'
    # 'Presencia  de la Polic??a'
    "Construcci??n":     "Construcción",
    "remodelaci??n":     "remodelación",
    "Creaci??n":         "Creación",
    "creaci??n":         "creación",
    "Polic??a":          "Policía",

    # Proyectos prioritarios y campos de texto libre
    "campa??a":          "campaña",
    "Campa??a":          "Campaña",
    "a??os":             "años",
    "ni??os":            "niños",
    "Ni??os":            "Niños",
    "da??a":             "daña",

    # Otros fragmentos comunes con '??'
    "m??s":              "más",
    "p??blico":          "público",
    "p??blica":          "pública",
    "v??as":             "vías",
    "due??o":            "dueño",
    "ra??z":             "raíz",
    "peque??as":         "pequeñas",
    "peque??os":         "pequeños",
    "ca??o":             "caño",
    "r??o":              "río",
    "comit??":           "comité",
    "Pol??tico":         "Político",
    "t??cnico":          "técnico",
}


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def corregir_columnas(df: pd.DataFrame, correcciones: dict) -> pd.DataFrame:
    """
    Aplica reemplazos exactos de subcadena en todas las columnas de texto del
    DataFrame y elimina espacios iniciales/finales residuales.

    El recorrido es en orden de inserción del diccionario.  Para evitar que
    una corrección de subcadena corta cree efectos secundarios sobre una
    cadena larga, se pasan primero los patrones más largos (práctica aplicada
    en la construcción de CORRECCIONES_COMUNES).

    El str.strip() final resuelve espacios iniciales que aparecen en 2013 U/R
    en los módulos de desastres (p. ej. ' Destru???da Totalmente').
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        for patron, correcto in correcciones.items():
            df[col] = df[col].str.replace(patron, correcto, regex=False)
        df[col] = df[col].str.strip()
    return df


# ─── Procesamiento por ola ────────────────────────────────────────────────────

def procesar_2010(tipo: str) -> pd.DataFrame:
    """
    Lee UComunidades o RComunidades de 2010 y corrige la codificación.

    Diferencia crítica entre U y R en esta ola
    -------------------------------------------
    UComunidades-csv.tab tiene una fila de encabezado (header=0).
    RComunidades-csv.tab NO tiene encabezado: la primera fila del archivo
    ya es un registro de datos.  Los nombres de columna se recuperan del
    archivo compañero RComunidades.tab, que sí incluye el encabezado pero
    almacena los datos en formato codificado (enteros), no en formato
    etiquetado.  Ambos archivos tienen exactamente 235 columnas, por lo que
    la asignación posicional de nombres es válida.

    Correcciones de codificación aplicadas
    ---------------------------------------
    - CORRECCIONES_COMUNES ('???', 3 signos): aplica a U y R.
    - CORRECCIONES_2010R ('??', 2 signos): aplica solo a R (obra_prioritaria*,
      proy_prioritario*, campos de texto libre).

    Columnas de identificación disponibles en 2010: ola, zona, region,
    consecutivo_c.  No existen llave ni hogar_n16.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path_csv = os.path.join(DATA_ROOT, "elca_2010", f"{tipo}Comunidades-csv.tab")

    if tipo == "R":
        # El CSV carece de encabezado; se leen los nombres desde el .tab
        path_tab = os.path.join(DATA_ROOT, "elca_2010", f"{tipo}Comunidades.tab")
        cols = pd.read_csv(path_tab, sep="\t", nrows=0).columns.tolist()
        df = pd.read_csv(path_csv, sep="\t", header=None, names=cols, low_memory=False)
    else:
        df = pd.read_csv(path_csv, sep="\t", header=0, low_memory=False)

    # Correcciones comunes (3 signos '???')
    df = corregir_columnas(df, CORRECCIONES_COMUNES)

    # Correcciones adicionales exclusivas de 2010 R (2 signos '??')
    if tipo == "R":
        df = corregir_columnas(df, CORRECCIONES_2010R)

    print(f"    {zona} 2010: {df.shape[0]:>4} filas × {df.shape[1]:>3} columnas")
    return df


def procesar_2013(tipo: str) -> pd.DataFrame:
    """
    Lee UComunidades o RComunidades de 2013 y corrige la codificación.

    Diferencias respecto a 2010
    ----------------------------
    - Ambas zonas tienen encabezado en el CSV.
    - Añade 'educ_lider*' (nivel educativo de cada líder encuestado).
    - Agrega módulos de afectaciones por desastres naturales (inu_*, ava_*,
      desb_*, desl_*, ven_*, otrd_*) con valores 'Destruida Totalmente',
      'Destruida parcialmente', 'No se destruyó', 'No tenía'.
    - Añade módulo de ayuda humanitaria (clfq_*, rec_*) con valor
      'Aún no han recibido'.
    - 2013 R incluye conflictos de tierra (cod_conf_*, anos_resolver_*,
      resolvio_con_*) y variables de mercado agropecuario.
    - Ambas zonas presentan '???' (3 signos) en variables categóricas.
      No se usan los patrones de '??' (2 signos) propios de 2010 R.
    - Algunos valores de los módulos de desastres tienen un espacio inicial
      (p. ej. ' Destru???da Totalmente'); str.strip() lo elimina en
      corregir_columnas() tras los reemplazos de subcadena.

    Columnas de identificación en 2013: ola, zona, consecutivo_c.
    La columna 'region' no está presente en 2013 (solo en 2010 y 2016 con
    distinto nombre: 'region_2016').
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2013", f"{tipo}Comunidades-csv.tab")

    df = pd.read_csv(path, sep="\t", header=0, low_memory=False)
    df = corregir_columnas(df, CORRECCIONES_COMUNES)

    print(f"    {zona} 2013: {df.shape[0]:>4} filas × {df.shape[1]:>3} columnas")
    return df


def procesar_2016(tipo: str) -> pd.DataFrame:
    """
    Lee UComunidades o RComunidades de 2016 y corrige la codificación.

    Diferencias respecto a 2013
    ----------------------------
    - La columna de zona pasa a llamarse 'zona_2016' (valores 'Urbana'/'Rural').
      No existe columna 'zona_2010' (a diferencia del módulo de Hogar).
    - Columnas binarias Sí/No usan 'S???' (3 signos) en lugar de 'Si':
      hog_icbf, guarderia_icbf, preescolar, esc_primaria, col_secundaria,
      rest_escolares, puesto_salud, jac, org_*, fb_*, grarmados_*, amenazas,
      rec_*, prt_*, etc.
    - Añade módulo de organizaciones sociales (org_*, fb_*).
    - Añade módulo de protestas ciudadanas (prt_*).
    - 2016 U no tiene columna 'region'; 2016 R tampoco.
    - 2016 R amplía el módulo de desastres con 'sequias' y 'vendavales'.

    Correcciones aplicadas: solo CORRECCIONES_COMUNES ('???', 3 signos).
    No se usan los patrones de '??' (2 signos) propios de 2010 R.

    Columnas de identificación en 2016: ola, zona_2016, consecutivo_c.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2016", f"{tipo}Comunidades-csv.tab")

    df = pd.read_csv(path, sep="\t", header=0, low_memory=False)
    df = corregir_columnas(df, CORRECCIONES_COMUNES)

    print(f"    {zona} 2016: {df.shape[0]:>4} filas × {df.shape[1]:>3} columnas")
    return df


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Lee, corrige y concatena las seis bases de comunidades (U y R × 3 olas).

    Ordenamiento de columnas del output
    ------------------------------------
    1. Identificadores: consecutivo_c, ola, zona, zona_2016, region.
    2. Resto de columnas en el orden original de aparición, que preserva la
       agrupación temática de los instrumentos originales de la encuesta.

    Tratamiento de columnas con tipos mixtos
    ----------------------------------------
    Al concatenar archivos de distintas olas, algunas columnas que en un
    archivo son numéricas (float) y en otro son de texto (object) resultan en
    columnas de tipo object con valores mixtos.  pyarrow no puede serializar
    este tipo de array.  Se unifican como string preservando NaN (→ null en
    parquet) antes de guardar.

    Retorna el DataFrame final.
    """
    frames = []

    for tipo in ["U", "R"]:
        print(f"\n  Procesando {tipo}Comunidades...")

        print("    Leyendo 2010...")
        frames.append(procesar_2010(tipo))

        print("    Leyendo 2013...")
        frames.append(procesar_2013(tipo))

        print("    Leyendo 2016...")
        frames.append(procesar_2016(tipo))

    # Concatenar: la unión de columnas llena con NaN las celdas ausentes.
    base_final = pd.concat(frames, axis=0, ignore_index=True).copy()

    # Reordenar columnas: identificadores primero, luego el resto.
    all_id_cols = ["consecutivo_c", "ola", "zona", "zona_2016", "region"]
    id_cols    = [c for c in all_id_cols if c in base_final.columns]
    otras_cols = [c for c in base_final.columns if c not in id_cols]
    base_final = base_final[id_cols + otras_cols]

    # Normalizar columnas object con tipos mixtos (str + float) antes de
    # serializar a parquet: pyarrow rechaza arrays con mezcla de tipos.
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
    print(f"Dimensiones            : {base_final.shape[0]:>5} filas × {base_final.shape[1]:>4} columnas")
    print(f"Olas presentes         : {sorted(base_final['ola'].unique())}")
    print(f"consecutivo_c únicos   : {base_final['consecutivo_c'].nunique():>5}")
    print()
    print("  Filas por ola y zona:")
    for ola_val in sorted(base_final["ola"].unique()):
        sub = base_final[base_final["ola"] == ola_val]
        zona_col = (
            "zona"      if "zona" in sub.columns and sub["zona"].notna().any()
            else "zona_2016"
        )
        if zona_col in sub.columns:
            for zona_val, n in sub[zona_col].value_counts().sort_index().items():
                print(f"    Ola {ola_val} – {zona_val:<10}: {n:>4} filas")
        else:
            print(f"    Ola {ola_val}: {len(sub):>4} filas")

    return base_final


if __name__ == "__main__":
    base_final = main()
