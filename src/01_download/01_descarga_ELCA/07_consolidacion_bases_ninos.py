"""
Encuesta Longitudinal Colombiana (ELCA)
Consolidación de las bases de datos de niños – olas 2010, 2013 y 2016.

Autor: Jonathan Melo Sarta
Fecha: Junio 2026

Contexto
--------
Los módulos de niños recogen información sobre los menores presentes en
los hogares del panel.  El cuestionario lo responde la madre u otro
cuidador principal e incluye: cuidado (quién lo cuida, dónde, costo),
asistencia a establecimiento educativo, actividades en el hogar y trabajo
infantil, lactancia y alimentación (frutas, verduras, carnes, leche),
salud (enfermedades recientes, vacunas), y medidas antropométricas
(peso, talla, perímetro cefálico).

El módulo de niños es uno de los más ricos para el análisis del desarrollo
infantil temprano y fue evolucionando entre olas tanto en el rango de
edad cubierto como en los temas incluidos.

Cambio de nombre de archivo y rango de edad por ola
----------------------------------------------------
El nombre del módulo cambia en cada ola porque también cambia el rango
de edad del cuestionario aplicado:

  2010 : UNinosde0a9 / RNinosde0a9   → niños de 0 a 9 años
  2013 : UNinos0a13  / RNinos0a13    → niños de 0 a 13 años
  2016 : UNinos6a16  / RNinos6a16    → niños de 6 a 16 años

Esto implica que el output consolidado NO es directamente comparable en
toda su extensión entre olas:
  - Niños de 0-5 años en 2010 y 2013 no tienen equivalente en 2016.
  - Niños de 10-13 años en 2013 no fueron encuestados en 2010.
  - Niños de 14-16 años solo aparecen en 2016.
  - El rango 6-9 años es el único presente en las tres olas.

El script consolida los seis archivos tal como están (sin filtrar por
edad), reflejando fielmente lo que se encuestó en cada ola.  Las columnas
exclusivas de un subconjunto de olas quedan con NaN en el resto.

Unidad de análisis
------------------
Niño × ola: una fila por cada menor en cada ola en que fue encuestado.

Identificadores por ola y zona
-------------------------------
  2010 U/R : consecutivo, orden, llave_ID_lb
             (sin llave, hogar, llaveper)
  2013 U/R : consecutivo, hogar, orden, llave, llaveper, llave_ID_lb
  2016 U/R : consecutivo, hogar, hogar_n16, orden, llave, llave_n16,
             llaveper, llaveper_n16, llave_ID_lb

El identificador de niño más granular disponible es:
  - 2010 : consecutivo + orden
  - 2013 : llaveper (= llave + orden)
  - 2016 : llaveper_n16 (= llave_n16 + hogar_n16 + orden)

La columna llave_ID_lb vincula al niño con su registro en Personas si
era miembro original del panel; tiene NaN para niños nacidos después de
2010 o para los rangos de edad no cubiertos en ola 1.

Nota sobre las columnas 'ola' y 'zona'
---------------------------------------
Ninguno de los seis archivos fuente incluye columna 'ola' ni 'zona'.
El script las agrega al leer cada archivo:
  - zona : 'Urbano' (UNinos*) o 'Rural' (RNinos*)
  - ola  : 1 (2010), 2 (2013), 3 (2016)

Vinculación con el módulo de Personas
--------------------------------------
Los niños del panel se vinculan con su registro en el módulo de Personas
mediante la columna llave_ID_lb (para niños originales del panel) o,
para todos los niños, mediante la combinación:
    2013 : llaveper  + ola
    2016 : llaveper_n16 + ola

    ninos   = pd.read_parquet("data/processed/ninos_elca_longitudinal.parquet")
    personas = pd.read_parquet("data/processed/personas_elca_longitudinal.parquet")

    merged = ninos.merge(
        personas,
        on=["llaveper", "ola"],   # o llaveper_n16 para 2016
        how="left",
        suffixes=("_nin", "_per"),
    )

Diferencias estructurales entre olas
--------------------------------------
  2010 (113 cols) : Cuestionario básico de cuidado, educación inicial,
                    actividades cotidianas, lactancia y nutrición básica.
                    Módulo de medidas antropométricas incluido.
  2013 (348 cols) : Ampliación muy significativa: añade módulo de salud
                    neonatal (cuidado del recién nacido, asesoría médica),
                    lactancia extendida (razones de abandono), vacunas,
                    trabajo infantil ampliado (descrip_oficio, riesgo),
                    módulo de desarrollo cognitivo (prueba TVIP), y
                    razon_no_asiste ampliada.  Es la ola con más columnas.
  2016 (145/147 cols): Cuestionario más reducido que 2013 pero con rango
                    de edad más amplio.  Mantiene: cuidado, educación,
                    nutrición y medidas.  Elimina módulo neonatal,
                    lactancia y prueba TVIP.  Añade resultado_m mejorado.

Diferencia de columnas entre U y R en 2016
-------------------------------------------
UNinos6a16-csv.tab tiene 145 columnas y RNinos6a16-csv.tab tiene 147.
Las 2 columnas extra de R son ano_edad_padre y ano_edad_madre, que
registran si la edad del padre/madre se proporcionó como año de
nacimiento o como edad directa (presentes en 2010 R, 2013 R y 2016 R,
ausentes en las versiones U).

Codificación y correcciones
-----------------------------
El patrón de corrupción de codificación es inusual en este módulo porque
varía dentro del mismo archivo en 2010:

  - 2010 U/R  : 'La mayoría de columnas categóricas usan '???' (3 signos)
                 para los caracteres especiales del español (á, é, í, ó,
                 ú, ñ, ü).  Sin embargo, un subconjunto de columnas usa
                 '??' (2 signos): resultado_m, pecho, razon_no_asiste_cual,
                 cuidado_prefiere_cual.  Es el único módulo del panel donde
                 aparecen dos patrones de codificación dentro del mismo
                 archivo.  Se aplican primero las correcciones de 3 signos
                 (CORRECCIONES_COMUNES) y luego las de 2 signos
                 (CORRECCIONES_2010_DOS_SIGNOS) para evitar colisiones.

  - 2013 U    : Prácticamente sin corrupción.  Solo la columna
                 cuidado_prefiere tiene el valor 'Otro, cu?l?' con signo
                 único '?'.  El resto de columnas categóricas están limpias.

  - 2013 R    : Usa '???' (3 signos) en columnas categóricas, igual que
                 los demás módulos.  Adicionalmente, ' ??? ' (con espacios)
                 representa el guion '–' en valores como '2 ??? 3 veces a
                 la semana' y 'Bienestar Familiar ??? ICBF'.

  - 2016 U/R  : Usa '???' (3 signos).  Los valores de 'cuál' en opciones
                 abiertas (razon_noverduras, razon_nocarnes, lugar_alimenta)
                 aparecen como 'Otra, ???cu???l?' → 'Otra, ¿cuál?'.
                 2016 U no tiene encabezado en el -csv.tab; los nombres de
                 columna se recuperan de UNinos6a16.tab.

Decisiones de implementación
------------------------------
  1. Rango de edad → consolidar tal como está
     El rango de edad cubierto cambia entre olas (0-9 / 0-13 / 6-16).
     El script consolida los seis archivos sin filtrar por edad; las
     columnas exclusivas de una ola quedan NaN en las demás.  Si el
     análisis requiere comparar el mismo rango entre olas (p. ej. 6-9
     años, el único rango común), filtrar por edad_nino después de leer
     el parquet.

  2. Ola y zona → columnas agregadas por el script
     Ningún archivo tiene ola ni zona.  El script agrega ola=1/2/3 y
     zona='Urbano'/'Rural' al inicio de cada DataFrame antes de concatenar.

  3. Header de 2016 UNinos6a16-csv.tab → recuperado de UNinos6a16.tab
     El archivo UNinos6a16-csv.tab de 2016 Urbano NO tiene fila de
     encabezado: la primera línea ya es un dato.  El compañero
     UNinos6a16.tab sí tiene encabezado y sus 145 columnas coinciden
     exactamente con las 145 del -csv.tab (verificado).  Los nombres se
     extraen del .tab y se asignan posicionalmente.  Los otros cinco
     archivos tienen encabezado y se leen con header=0.

  4. Correcciones 2010: dos diccionarios, dos pasadas
     Los archivos de 2010 tienen '???' (3 signos) en la mayoría de
     columnas Y '??' (2 signos) en resultado_m, pecho, razon_no_asiste_cual
     y cuidado_prefiere_cual.  Se aplica CORRECCIONES_COMUNES primero
     (cubre las columnas de 3 signos sin afectar las de 2 signos) y
     CORRECCIONES_2010_DOS_SIGNOS después (corrige las columnas de 2
     signos que quedaron intactas en la primera pasada).

  5. Correcciones 2013 U: mínimas
     Solo cuidado_prefiere tiene 'Otro, cu?l?' (signo único '?').  Se
     corrige con una entrada puntual en CORRECCIONES_2013U.  El resto
     de columnas está limpio.

  6. Correcciones ??? → CORRECCIONES_COMUNES, aplica a los 6 archivos
     Un único diccionario cubre los patrones de 3 signos presentes en
     2010, 2013 R y 2016 U/R.  Para 2013 U y para las columnas de 2 signos
     en 2010, las correcciones de este diccionario no tienen efecto (no
     hay matches de 3 signos en esas columnas).

  7. Tipos mixtos str + float → normalización antes de parquet
     Mismo patrón que en Hogar (04), Comunidades (05) y Personas (06).

Dimensiones del output
-----------------------
  25 636 filas × 433 columnas (unión de las seis olas y zonas).

  Desglose por ola y zona (+ zona y ola agregadas por el script):
    Ola 1 – Urbano :   4 282 niños ×  115 columnas  (0-9 años, edad en meses)
    Ola 1 – Rural  :   4 411 niños ×  115 columnas  (0-9 años, edad en meses)
    Ola 2 – Urbano :   5 130 niños ×  350 columnas  (0-13 años)
    Ola 2 – Rural  :   5 254 niños ×  350 columnas  (0-13 años)
    Ola 3 – Urbano :   3 409 niños ×  147 columnas  (5-16 años)
    Ola 3 – Rural  :   3 150 niños ×  149 columnas  (5-16 años)

  Nota: en 2010 la edad está en meses (columna edad_mm); en 2013 y 2016
  está en años (columna edad_nino) con los meses complementarios en
  edad_meses_nino.
"""

import os

from pathlib import Path

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = str(PROJECT_ROOT / "data" / "interim" / "raw")
OUTPUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "ninos_elca_longitudinal.parquet")

# Nombre de archivo de niños por ola (cambia entre olas)
NOMBRE_ARCHIVO = {
    2010: "Ninosde0a9",
    2013: "Ninos0a13",
    2016: "Ninos6a16",
}

# ─── Correcciones comunes (3 signos '???') ────────────────────────────────────
#
# Aplican a TODOS los archivos (2010 U/R, 2013 U/R, 2016 U/R) porque el
# patrón de 3 signos está presente en todos excepto 2013 U.
# En 2013 U las correcciones no tienen efecto (no hay matches).
#
# Orden: de subcadena más específica a más general para evitar que una
# corrección corta deje restos de '?' en valores que debían ser corregidos
# por una corrección más larga.
#
# Nota sobre ' ??? ': en 2013 R y 2016, la secuencia ' ??? ' (espacio +
# 3 signos + espacio) representa el guion '–' (en dash), por ejemplo en
# '2 ??? 3 veces a la semana' o 'Bienestar Familiar ??? ICBF'.  Se
# corrige antes que otros patrones de 3 signos porque contiene ' ' que
# la hace inequívoca.

CORRECCIONES_COMUNES = {

    # ── Guion largo (en dash) representado como ' ??? ' ───────────────────────
    # tipo_hogar 2013 R: 'Hogar comunitario del Bienestar Familiar ??? ICBF'
    # freq_lee / freq_conversa 2013 R y 2016: '2 ??? 3 veces a la semana'
    " ??? ":                        " – ",

    # ── Indicador binario Sí/No ───────────────────────────────────────────────
    # Presente en: lavar, planchar, cocinar, limpieza, cuidar_ninos,
    # cuidar_enfermos, atender_huerta, traer_agua, recoger_lena, mandados,
    # otro_oficio, carne_vacunas, madre_biologica, salud_gestacion, diarrea,
    # fiebre, tos, etc.
    "S???":                         "Sí",

    # ── Tiempo / frecuencias ─────────────────────────────────────────────────
    # freq_lee, freq_conversa, freq_juegadentro, freq_juegafuera, freq_vetv,
    # freq_ensena, lee_libros, conversa, juega_encasa, ven_television, ensena:
    # 'Todos los d???as', 'Cada 15 d???as'
    # come_frutas, come_verduras: 'Todos los d???as, m???s de una vez al d???a'
    "d???as":                       "días",
    "d???a":                        "día",
    "m???s":                        "más",
    "M???s":                        "Más",
    "a???os":                       "años",
    "a???o":                        "año",
    "A???o":                        "Año",
    "a???n":                        "aún",
    "est???n":                      "están",
    "todav???a":                    "todavía",

    # ── Resultado de la medición ──────────────────────────────────────────────
    # resultado_m en 2013 R y 2016 U/R: 'Ni???o(a) ausente',
    # 'Ni???o con discapacidad severa'
    "Ni???o(a)":                    "Niño(a)",
    "Ni???o":                       "Niño",
    "Ni???a":                       "Niña",
    "NI???O":                       "NIÑO",
    "NI???A":                       "NIÑA",
    "NI???OS":                      "NIÑOS",

    # ── Parentesco del informante y cuidadores ────────────────────────────────
    # parent_inform en 2013 R y 2016: 'T???o(a)'
    # quien_cuida en 2010/2013: 'Su t???o(a)', 'Un t???o(a)'
    # parentesco_cuidador 2010: 'T???o(a)'
    "T???o(a)":                     "Tío(a)",
    "t???o(a)":                     "tío(a)",

    # ── Fallecimiento (padre, madre) ──────────────────────────────────────────
    # padre_vive, madre_vive: 'Ya falleci???'
    "falleci???":                   "falleció",

    # ── Educación de padres y cuidadores ─────────────────────────────────────
    # educ_padre, educ_madre, pcuida_niveledu, nivel_edu_cuidador:
    # 'B???sica primaria (1 a 5)', 'B???sica secundaria (6 a 13)',
    # 'T???cnico con t???tulo', 'Universidad con t???tulo',
    # 'Uno o m???s a???os de t???cnica o tecnol???gica'
    "B???sica":                     "Básica",
    "T???cnico":                    "Técnico",
    "T???cnica":                    "Técnica",
    "t???cnica":                    "técnica",
    "Tecnol???gica":                "Tecnológica",
    "t???tulo":                     "título",

    # ── Situación laboral de padres ───────────────────────────────────────────
    # trabajo_padre, trabajo_madre:
    # 'Jornalero o pe???n', 'Empleado dom???stico',
    # 'Nunca ha trabajado o nunca trabaj???',
    # 'Trabajador de su propia finca que ten???a... en arriendo o aparcer???a',
    # 'Trabajador familiar sin remuneraci???n', 'Patr???n o empleador',
    # 'Polic???a, militar o afines'
    "pe???n":                       "peón",
    "Pe???n":                       "Peón",
    "dom???stico":                  "doméstico",
    "dom???stica":                  "doméstica",
    "trabaj???":                    "trabajó",
    "ten???a":                      "tenía",
    "aparcer???a":                  "aparcería",
    "remuneraci???n":               "remuneración",
    "Patr???n":                     "Patrón",
    "Polic???a":                    "Policía",

    # ── Tipo de establecimiento educativo ─────────────────────────────────────
    # tipo_hogar (preescolar): 'Guarder???a, jard???n o preescolar oficial',
    #   'Hogar comunitario del Bienestar Familiar – ICBF' (??? ya corregido)
    # cuidado_prefiere: iguales valores
    "Guarder???a":                  "Guardería",
    "guarder???a":                  "guardería",
    "Jard???n":                     "Jardín",
    "jard???n":                     "jardín",

    # ── Razones de no asistencia escolar ─────────────────────────────────────
    # razon_no_asiste: 'Prefiere que no asista todav???a o considera que no
    #   est??? en edad de asistir', 'No encontr??? cupo', 'Otra raz???n'
    "est???":                       "está",
    "Est???":                       "Está",
    "encontr???":                   "encontró",
    "raz???n":                      "razón",

    # ── Pago por cuidado ──────────────────────────────────────────────────────
    # paga_cuida: 'No se pag??? nada'
    "pag???":                       "pagó",

    # ── Lactancia (2013 R) ────────────────────────────────────────────────────
    # pecho en 2013 R: 'S???, a???n le est???n dando'  (cubierto por S???, a???n, est???n)
    # rz_dejo_lactar: 'Deb???a trabajar y no pudo seguir con la lactancia',
    #   'Por malas experiencias f???sicas -dolor, enfermedad-',
    #   'Empez??? a planificar'
    # dejo_lactar_cual: '10. No le baj???', 'Le inici???', 'Consider???'
    "Deb???a":                      "Debía",
    "f???sicas":                    "físicas",
    "Empez???":                     "Empezó",
    "baj???":                       "bajó",
    "inici???":                     "inició",
    "Consider???":                  "Consideró",

    # ── Asesoría médica neonatal (2013 R) ────────────────────────────────────
    # quien_cuidado, quien_alimenta, quien_masaje, quien_salud:
    # 'De personal m???dico en la cl???nica donde dio a luz',
    # 'Del pediatra que atiende/???a al reci???n nacido'
    "m???dico":                     "médico",
    "cl???nica":                    "clínica",
    "atiende/???a":                 "atiende/ía",
    "reci???n":                     "recién",

    # ── Nutrición ─────────────────────────────────────────────────────────────
    # come_frutas, come_verduras, razon_nofrutas, razon_noverduras:
    # 'Al ni???o(a) no le gustan las frutas / verduras'
    # 'En el barrio/ciudad donde reside es dif???cil conseguir'
    # 'Los l???cteos y sus derivados no hacen parte de la dieta familiar'
    # 'Le produce malestar en la salud del ni???o(a)'
    "dif???cil":                    "difícil",
    "l???cteos":                    "lácteos",

    # ── Etiquetas de opción abierta '¿cuál?' en 2016 ─────────────────────────
    # razon_noverduras, razon_nocarnes, razon_noleche, lugar_alimenta:
    # 'Otra, ???cu???l?' (el '???' inicial representa '¿' y el central 'á')
    "???cu???l?":                  "¿cuál?",

    # ── Trabajo y oficios infantiles ──────────────────────────────────────────
    # oficios_hogar, otro_oficio: 'S???' (ya cubierto)
    # otro_oficio_cual, descrip_oficio (2013 R, 2016 R):
    # '1. Cuidar y orde???ar el ganado', '4. Traer le???a',
    # 'l. Traer le???a', 'm. Cuidar y orde???ar el ganado'
    # 'ORDE???AR', 'TRAER LE???A', 'CUIDAR NI???OS'
    "orde???ar":                    "ordeñar",
    "Orde???ar":                    "Ordeñar",
    "ORDE???AR":                    "ORDEÑAR",
    "le???a":                       "leña",
    "Le???a":                       "Leña",
    "LE???A":                       "LEÑA",

    # ── Lugar de alimentación (2016) ──────────────────────────────────────────
    # lugar_alimenta: 'En el hogar del ni???o/a con su familia'
    # (Ni???o ya cubierto más arriba; ni???o/a cubierto por t???o)
    # en realidad 'ni???o/a' → 'niño/a': el fragmento 'Ni???o' cubre la N mayúscula
    # y 'ni???o' la n minúscula
    "ni???o/a":                     "niño/a",
    "ni???o(a)":                    "niño(a)",
}

# ─── Correcciones adicionales 2010: patrones '??' (2 signos) ─────────────────
#
# En 2010 U y R, un subconjunto de columnas usa '??' (2 signos) en lugar
# de '???' (3 signos):
#   - resultado_m : 'Ni??o(a) / Madre(Padre) rehus??',
#                   'Ni??o(a) con discapacidad severa', 'Ni??o(a) ausente'
#   - pecho       : 'Si, y a??n le est??n dando'
#   - razon_no_asiste_cual : 'LLORA MUCHO Y NO SE AMA??A',
#                            'NO SE AMA??O', 'NO SE AMA??A EN EL JARDIN'
#   - cuidado_prefiere_cual: 'ninguno es muy peque??ita'
#
# Se aplican DESPUÉS de CORRECCIONES_COMUNES porque los patrones de 3
# signos ya fueron corregidos y no pueden interferir con los de 2 signos.
# No se aplican a 2013 ni a 2016 (en esos archivos '??' no aparece).

CORRECCIONES_2010_DOS_SIGNOS = {
    # resultado_m (2 signos para ñ y ó)
    "Ni??o":         "Niño",     # 'Ni??o(a) / Madre(Padre) rehus??'
    "NI??O":         "NIÑO",
    "NI??A":         "NIÑA",
    "NI??OS":        "NIÑOS",
    "rehus??":       "rehusó",   # 'Madre(Padre) rehus??'
    # pecho (2 signos para ú y á)
    "a??n":          "aún",      # 'Si, y a??n le est??n dando'
    "est??n":        "están",
    # razon_no_asiste_cual y cuidado_prefiere_cual (texto libre)
    "AMA??A":        "AMAÑA",    # 'NO SE AMA??A EN EL JARDIN'
    "AMA??O":        "AMAÑÓ",    # 'NO SE AMA??O'
    "peque??ita":    "pequeñita",
}

# ─── Correcciones 2013 U: etiqueta de opción abierta (signo único '?') ───────
#
# En 2013 U la única columna con corrupción es cuidado_prefiere, cuyo
# valor 'Otro, cu?l?' tiene '?' (un solo signo) en lugar de '¿' y 'á'.
# El resto del archivo está limpio; no se necesita corrección adicional.

CORRECCIONES_2013U = {
    "cu?l?": "¿cuál?",   # 'Otro, cu?l?' → 'Otro, ¿cuál?'
}


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def corregir_columnas(df: pd.DataFrame, correcciones: dict) -> pd.DataFrame:
    """
    Aplica reemplazos exactos de subcadena en todas las columnas de texto
    del DataFrame y elimina espacios iniciales/finales residuales.
    Devuelve una copia con las correcciones aplicadas.
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
    Lee UNinosde0a9 o RNinosde0a9 de 2010 y aplica correcciones.

    Ambos archivos tienen encabezado (header=0).

    Correcciones aplicadas:
    1. CORRECCIONES_COMUNES ('???', 3 signos): cubre la mayoría de columnas
       categóricas (educación padres, trabajo padres, tipo_hogar, frecuencias
       de actividades, razones de no asistencia, etc.).
    2. CORRECCIONES_2010_DOS_SIGNOS ('??', 2 signos): cubre las columnas
       resultado_m, pecho, razon_no_asiste_cual y cuidado_prefiere_cual, que
       usan 2 signos en lugar de 3 para la misma corrupción.

    Identificadores en 2010: consecutivo, orden, llave_ID_lb.
    No existen llave, hogar, llaveper ni sus equivalentes de ola 3.
    Rango de edad cubierto: 0-9 años (edad_mm = meses).
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    fname = f"{tipo}{NOMBRE_ARCHIVO[2010]}-csv.tab"
    path  = os.path.join(DATA_ROOT, "elca_2010", fname)

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.insert(0, "ola",  1)
    df.insert(0, "zona", zona)

    # Correcciones ??? (3 signos) — mayoría de columnas
    df = corregir_columnas(df, CORRECCIONES_COMUNES)
    # Correcciones ?? (2 signos) — columnas específicas de 2010
    df = corregir_columnas(df, CORRECCIONES_2010_DOS_SIGNOS)

    print(f"    {zona} 2010: {df.shape[0]:>5} filas × {df.shape[1]:>4} columnas")
    return df


def procesar_2013(tipo: str) -> pd.DataFrame:
    """
    Lee UNinos0a13 o RNinos0a13 de 2013 y aplica correcciones.

    Ambos archivos tienen encabezado (header=0).

    Correcciones aplicadas:
    - U: CORRECCIONES_2013U ('?', signo único) — solo cuidado_prefiere
         ('Otro, cu?l?' → 'Otro, ¿cuál?').  El resto del archivo está
         limpio; las correcciones de 3 signos no tienen efecto en U.
    - R: CORRECCIONES_COMUNES ('???', 3 signos) — cubre educ_padre/madre,
         trabajo_padre/madre, tipo_hogar ('Hogar comunitario ... – ICBF'),
         frecuencias ('2 – 3 veces a la semana'), razones de lactancia,
         módulo neonatal, nutrición y trabajo infantil.

    Nuevas columnas respecto a 2010: lactancia (rz_dejo_lactar,
    dejo_lactar_cual), módulo neonatal (quien_cuidado, quien_alimenta,
    quien_masaje, quien_salud), vacunas (carne_vacunas), desarrollo
    cognitivo (prueba TVIP: resultado_tvip, riesgo), trabajo infantil
    ampliado (trabajo, descrip_oficio, otro_oficio_cual).
    Rango de edad cubierto: 0-13 años.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    fname = f"{tipo}{NOMBRE_ARCHIVO[2013]}-csv.tab"
    path  = os.path.join(DATA_ROOT, "elca_2013", fname)

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.insert(0, "ola",  2)
    df.insert(0, "zona", zona)

    if tipo == "U":
        # 2013 U: solo la etiqueta 'cuál' en una columna
        df = corregir_columnas(df, CORRECCIONES_2013U)
    else:
        # 2013 R: patrones ??? amplios
        df = corregir_columnas(df, CORRECCIONES_COMUNES)

    print(f"    {zona} 2013: {df.shape[0]:>5} filas × {df.shape[1]:>4} columnas")
    return df


def procesar_2016(tipo: str) -> pd.DataFrame:
    """
    Lee UNinos6a16 o RNinos6a16 de 2016 y aplica correcciones.

    Diferencia crítica entre U y R en esta ola
    -------------------------------------------
    UNinos6a16-csv.tab (Urbano) NO tiene fila de encabezado: la primera
    línea del archivo es ya un registro de datos.  El archivo compañero
    UNinos6a16.tab sí tiene el encabezado y sus 145 columnas coinciden
    posicionalmente con las del -csv.tab (verificado).  Los nombres se
    leen del .tab y se asignan con header=None + names=.
    RNinos6a16-csv.tab (Rural) tiene encabezado normal (header=0).

    Correcciones aplicadas:
    CORRECCIONES_COMUNES ('???', 3 signos): cubre todos los campos
    categóricos de 2016 U y R, incluyendo el patrón '???cu???l?' para
    los valores de opción abierta '¿cuál?' en razon_noverduras, etc.

    Nuevas columnas respecto a 2013: resultado_m mejorado, llave_n16,
    hogar_n16, llaveper_n16.  Se eliminan: módulo neonatal, lactancia,
    prueba TVIP.
    2016 R tiene 2 columnas más que 2016 U (ano_edad_padre, ano_edad_madre).
    Rango de edad cubierto: 6-16 años.
    """
    zona  = "Urbano" if tipo == "U" else "Rural"
    fname = f"{tipo}{NOMBRE_ARCHIVO[2016]}-csv.tab"
    path  = os.path.join(DATA_ROOT, "elca_2016", fname)

    if tipo == "U":
        # El CSV carece de encabezado; los nombres se leen del .tab compañero
        tab_path = os.path.join(
            DATA_ROOT, "elca_2016", f"{tipo}{NOMBRE_ARCHIVO[2016]}.tab"
        )
        cols = pd.read_csv(tab_path, sep="\t", nrows=0).columns.tolist()
        df   = pd.read_csv(path, sep="\t", header=None, names=cols, low_memory=False)
    else:
        df = pd.read_csv(path, sep="\t", header=0, low_memory=False)

    df.insert(0, "ola",  3)
    df.insert(0, "zona", zona)

    df = corregir_columnas(df, CORRECCIONES_COMUNES)

    print(f"    {zona} 2016: {df.shape[0]:>5} filas × {df.shape[1]:>4} columnas")
    return df


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Lee, corrige y concatena las seis bases de niños (U y R × 3 olas).

    Ordenamiento de columnas del output
    ------------------------------------
    1. zona, ola  : variables administrativas agregadas por el script
    2. Identificadores del niño (del más granular al más general):
         llaveper_n16, llaveper, llave_ID_lb,
         consecutivo, hogar, hogar_n16, orden, llave, llave_n16
    3. Demografía del niño: edad_nino, edad_meses_nino (o edad_mm)
    4. Resto de columnas en el orden original de aparición.

    Tratamiento de columnas con tipos mixtos
    -----------------------------------------
    Al concatenar seis archivos con esquemas distintos, algunas columnas
    que en un archivo son numéricas (float) y en otro son de texto (object)
    resultan en object con valores mixtos.  pyarrow rechaza estos arrays.
    Se unifican como string preservando NaN antes de llamar a to_parquet().
    Mismo patrón que en Hogar (04), Comunidades (05) y Personas (06).

    Retorna el DataFrame final.
    """
    frames = []

    for tipo in ["U", "R"]:
        print(f"\n  Procesando {tipo}Ninos...")

        print("    Leyendo 2010...")
        frames.append(procesar_2010(tipo))

        print("    Leyendo 2013...")
        frames.append(procesar_2013(tipo))

        print("    Leyendo 2016...")
        frames.append(procesar_2016(tipo))

    # Concatenar: la unión de columnas llena con NaN las celdas ausentes.
    base_final = pd.concat(frames, axis=0, ignore_index=True).copy()

    # Reordenar columnas: zona/ola e identificadores primero, luego el resto.
    all_id_cols = [
        "zona", "ola",
        "llaveper_n16", "llaveper", "llave_ID_lb",
        "consecutivo", "hogar", "hogar_n16", "orden", "llave", "llave_n16",
        "edad_nino", "edad_meses_nino", "edad_mm", "edad_ames",
    ]
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
    print()
    print("  Filas por ola y zona:")
    for ola_val in sorted(base_final["ola"].unique()):
        for zona_val in ["Urbano", "Rural"]:
            n = ((base_final["ola"] == ola_val) & (base_final["zona"] == zona_val)).sum()
            col_edad = "edad_nino" if "edad_nino" in base_final.columns else "edad_mm"
            sub = base_final[
                (base_final["ola"] == ola_val) & (base_final["zona"] == zona_val)
            ]
            if col_edad in sub.columns and sub[col_edad].notna().any():
                edad_min = int(sub[col_edad].min())
                edad_max = int(sub[col_edad].max())
                print(f"    Ola {int(ola_val)} – {zona_val:<6}: {n:>5} niños, edad {edad_min}-{edad_max} años")
            else:
                print(f"    Ola {int(ola_val)} – {zona_val:<6}: {n:>5} niños")

    return base_final


if __name__ == "__main__":
    base_final = main()
