"""
Encuesta Longitudinal Colombiana (ELCA)
Consolidación de las bases de datos de personas – olas 2010, 2013 y 2016.

Autor: Jonathan Melo Sarta
Fecha: Junio 2026

Contexto
--------
Los módulos UPersonas-csv.tab y RPersonas-csv.tab recogen información
a nivel de PERSONA (un registro por individuo presente en el hogar al
momento de la encuesta).  El cuestionario cubre: características
demográficas y de parentesco, salud (enfermedades, discapacidades,
consultas, hospitalización, nutrición, actividad física), educación
(nivel alcanzado, asistencia escolar, razones de deserción), mercado
laboral (actividad principal, ingresos, tipo de contrato, pensiones),
organizaciones sociales y participación política, y ahorros e ingresos
no laborales.

Unidad de análisis
------------------
Persona × ola: una fila por cada individuo en cada ola en que fue
encuestado.  El identificador de persona varía por ola:

  - 2010 : no existe un identificador de persona construido; se usa la
           combinación consecutivo + orden (= persona dentro del hogar).
           La columna llave_ID_lb = consecutivo (base 2010) + orden y
           sirve como referencia longitudinal para los miembros originales.
  - 2013 : llaveper = llave + orden (concatenación de 10 dígitos).
           llave_ID_lb vincula al miembro con su registro de 2010 si era
           miembro original; NaN para personas que se incorporaron después.
  - 2016 : llaveper_n16 = llave_n16 + hogar_n16 + orden (12 dígitos).
           llave_ID_lb cumple el mismo rol de vínculo longitudinal.

La columna llave_ID_lb presente en los seis archivos es la llave de
seguimiento longitudinal de los miembros originales del panel.  Tiene
NaN para personas que se unieron al hogar después de 2010 (nuevos
miembros, nacidos, cónyuges que llegaron, etc.).

Identificadores presentes por ola y zona
-----------------------------------------
  2010 U (455 cols): ola, consecutivo, orden, llave_ID_lb
  2010 R (415 cols): ola, consecutivo, orden, llave_ID_lb
  2013 U (529 cols): ola, consecutivo, hogar, orden, llave, llaveper, llave_ID_lb
  2013 R (723 cols): ola, consecutivo, hogar, orden, llave, llaveper, llave_ID_lb
  2016 U (609 cols): ola, consecutivo, hogar, hogar_n16, orden, llave,
                     llave_n16, llaveper, llaveper_n16, llave_ID_lb
  2016 R (810 cols): ola, consecutivo, hogar, hogar_n16, orden, llave,
                     llave_n16, llaveper, llaveper_n16, llave_ID_lb

Nota sobre las columnas 'ola' y 'zona'
---------------------------------------
Columna 'ola': solo el archivo de 2010 la incluye (valor 1).  Los archivos
de 2013 y 2016 no la tienen; el script la agrega con valor 2 y 3
respectivamente.

Columna 'zona': ninguno de los seis archivos fuente la incluye.  El script
la agrega asignando 'Urbano' (UPersonas) o 'Rural' (RPersonas).

Ambas columnas se insertan en las posiciones 0 y 1 del DataFrame antes de
cualquier otra operación, para que queden siempre como primeras columnas.

Vinculación con los módulos de Hogar y Comunidades
----------------------------------------------------
Las personas del panel se vinculan con su hogar y su comunidad mediante
la jerarquía de identificadores:

  Comunidades ←[consecutivo_c]→ Hogar ←[llave / llave_n16]→ Personas

Para cruzar personas con su comunidad o con variables del hogar:

    personas = pd.read_parquet("data/processed/personas_elca_longitudinal.parquet")
    hogar    = pd.read_parquet("data/processed/hogar_elca_longitudinal.parquet")

    # Join persona → hogar (por sub-hogar y ola)
    ph = personas.merge(
        hogar,
        on=["llave", "ola"],          # 2013: llave; 2016: llave_n16
        how="left",
        suffixes=("_per", "_hog"),
    )

Para 2016, el join debe hacerse por llave_n16 + hogar_n16 + ola porque los
sub-hogares se identifican con las llaves de ola 3.

Diferencias estructurales entre olas y zonas
---------------------------------------------
Las diferencias más importantes (además del número de columnas) son:

  2010 U/R : No existe identificador compuesto de persona (llaveper).
             Módulos: parentesco, demografía básica, salud, educación,
             mercado laboral rural/urbano básico, migración.
             No incluye módulos de organizaciones ni de discapacidad
             expandida que aparecen en 2016.
  2013 U/R : Añade novedad_perso (novedades en el seguimiento del panel:
             nuevos miembros, nacimientos, fallecimientos).  Agrega módulo
             de embarazo expandido, actividad física, consumo de tabaco y
             alcohol, organizaciones sociales.
             2013 R es notablemente más ancha (723 cols vs 529 U) por el
             módulo de mercado laboral rural ampliado.
  2016 U/R : Añade llave_n16, hogar_n16, llaveper_n16 para rastrear
             divisiones de hogar en ola 3.  Incorpora módulo de uso del
             tiempo (actividades diarias), banca móvil y acceso a internet,
             y expande el módulo de organizaciones sociales.
             2016 R es la más ancha del panel (810 cols) con los módulos
             de mercado laboral rural más completos.

Codificación y correcciones
-----------------------------
  - 2010 U/R : Sin problemas de codificación.
  - 2013 U/R : Corrupción de signo único '?' para caracteres especiales
               del español (á, é, í, ó, ú, ñ, ¿), pero solo en un
               subconjunto pequeño de columnas: etiquetas de categoría
               abierta ('Otra, ¿cuál?', 'Otro. ¿Cuál?') y texto libre de
               ocupaciones (descrip_oficio).  Los campos categóricos
               analíticamente importantes (parentesco, sexo, estado_civil,
               etnia, nivel_educ, etc.) están limpios en 2013.
               Se corrigen solo las etiquetas de categoría fija; el texto
               libre se deja tal cual (requeriría un diccionario de miles
               de palabras).
  - 2016 U/R : Corrupción '???' (3 signos) en la mayoría de columnas
               categóricas.  Adicionalmente, la columna novedad_perso usa
               '??' (2 signos) en lugar de '???' —la misma dualidad de
               codificación observada en RComunidades 2010.

Decisiones de implementación
------------------------------
  1. Ola y zona → columnas agregadas por el script
     Los archivos de 2013 y 2016 no tienen columna 'ola'; solo 2010 la
     incluye (valor 1).  El script agrega ola=2 en 2013 y ola=3 en 2016.
     Ninguno de los seis archivos tiene columna 'zona'; el script agrega
     zona='Urbano' o 'Rural' según el prefijo del archivo.  Ambas columnas
     se insertan como primeras columnas del DataFrame.

  2. Identificador de persona → llaveper / llaveper_n16
     Se eligió como clave granular de persona la combinación llaveper
     (2013) y llaveper_n16 (2016), que rastrean a la persona dentro del
     sub-hogar dividido.  En 2010 no existe este identificador; la clave
     es consecutivo + orden.  llave_ID_lb está disponible en las tres
     olas pero tiene NaN para personas que se incorporaron después de
     2010 (nuevos miembros, nacidos), así que no puede usarse como clave
     única en el output consolidado.

  3. Correcciones 2013 → solo etiquetas de categoría fija
     La corrupción '?' de 2013 afecta principalmente texto libre
     (descrip_oficio, nombres de partidos políticos, etc.) y un par de
     etiquetas de opción abierta ('Otra, ¿cuál?', 'Otro. ¿Cuál?').
     Los campos categóricos clave (parentesco, sexo, estado_civil, etnia,
     nivel_educ, actividad_ppal, ocupacion) están limpios.  Se corrigen
     las etiquetas fijas de 'cuál' para uniformidad, y se deja el texto
     libre como está.

  4. Correcciones 2016 ??? → CORRECCIONES_2016, aplica a U y R
     Los dos archivos de 2016 presentan '???' (3 signos) en columnas
     categóricas: parentesco, estado_civil, etnia, nivel_educ,
     todas las variables binarias Sí/No, y las descripciones de
     actividades laborales y de salud.  Se usa un diccionario único
     aplicado globalmente a las columnas de texto.

  5. Correcciones 2016 ?? → CORRECCIONES_NOVEDAD_PERSO, solo novedad_perso
     La columna novedad_perso en 2016 U y R usa '??' (2 signos) en lugar
     de '???' para codificar las ó (past-tense) y ó (accent): 'Llegó',
     'Nació', 'casó', 'organizó', 'integró', 'separó', 'enviudó',
     'adopción'.  La corrección de '???' no puede cubrir estos valores
     porque los patrones son distintos.  Se aplica CORRECCIONES_2016
     primero (sobre todo el DataFrame) y luego CORRECCIONES_NOVEDAD_PERSO
     (solo sobre la columna novedad_perso) para evitar que '??' colisione
     con '???' en otras columnas.

  6. Tipos mixtos str + float → normalización antes de parquet
     Mismo patrón que en Hogar (04) y Comunidades (05): al concatenar
     seis archivos con esquemas distintos, algunas columnas quedan como
     object con valores mixtos (str + float).  Se detectan y convierten
     a string puro preservando NaN antes de llamar a to_parquet().

Calidad de datos: NaN en llave y hogar en 2016
-----------------------------------------------
En 2016, hay personas para las que llave y hogar son NaN (ver ejemplo en
la inspección de datos: consecutivo=111001, orden=6,7,8 tienen llave=NaN).
Esto ocurre cuando la persona fue detectada en 2016 pero su hogar de
referencia no tiene un sub-hogar asignado en ola 2.  Estas personas
tienen llave_n16 y llaveper_n16 válidos y pueden ser identificadas, pero
el join con la base de hogar por llave no funcionará para ellas.

Dimensiones del output
-----------------------
  118 824 filas × 1 359 columnas (unión de las seis olas y zonas).

  Desglose por ola y zona:
    Ola 1 – Urbano :  22 179 personas × 456 columnas (incluye zona+ola agregadas)
    Ola 1 – Rural  :  21 019 personas × 416 columnas
    Ola 2 – Urbano :  20 574 personas × 531 columnas
    Ola 2 – Rural  :  19 339 personas × 725 columnas
    Ola 3 – Urbano :  19 298 personas × 611 columnas
    Ola 3 – Rural  :  16 415 personas × 812 columnas

  Nota: ola y zona (+1 col cada una vs. los archivos fuente) se agregan
  en procesar_*(). llave_ID_lb únicos = 43 198 (miembros panel original).
"""

import os

from pathlib import Path

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = str(PROJECT_ROOT / "data" / "interim" / "raw")
OUTPUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal.parquet")

# ─── Correcciones 2013: etiquetas de categoría abierta (signo único '?') ─────
#
# Solo afectan a un puñado de columnas donde la etiqueta de opción
# "Otra, ¿cuál?" y similares quedaron con '?' en lugar de '¿' y 'á'.
# Los demás campos categóricos de 2013 están limpios.
# El texto libre (descrip_oficio, nombres de partidos) se deja como está.

CORRECCIONES_2013 = {
    "?cu?l?":  "¿cuál?",   # rzn_dejoestudiar, simpatiza_parpol: 'Otra, ?cu?l?'
    "Cu?l?":   "¿Cuál?",   # ocupacion*: 'Otro. Cu?l?'
}

# ─── Correcciones 2016: patrones '???' (3 signos) ────────────────────────────
#
# Se aplican a TODOS los archivos de 2016 (U y R) porque ambos presentan
# la misma codificación de 3 signos.  Las entradas están ordenadas de
# fragmento más largo a más corto para evitar sustituciones parciales
# (p. ej. 'adopci???n' antes de 'ci???n', aunque aquí no hay ambigüedad
# directa porque los patrones son distintos).
#
# 'S???' → 'Sí' cubre cientos de columnas binarias (embarazada, cotizando,
# afiliacion, ceguera, estudia, ahorra, org_*, asist_*, etc.) y debe
# preceder a cualquier corrección de 2 signos que pudiera aplicarse.

CORRECCIONES_2016 = {

    # ── Indicador binario Sí/No ───────────────────────────────────────────────
    # Aplica a prácticamente todas las columnas booleanas del cuestionario.
    "S???":                         "Sí",

    # ── Parentesco ────────────────────────────────────────────────────────────
    # parentesco: 'C???nyuge o compa???era(o)', 'Nieto(a)del jefe del hogar
    # o de su c???nyuge', 'Bisnieto del jefe del hogar o de su c???nyuge',
    # 'T???o(a)', 'Servicio dom???stico, cuidandero y sus parientes'
    "compa???era(o)":               "compañera(o)",
    "C???nyuge":                    "Cónyuge",
    "c???nyuge":                    "cónyuge",
    "T???o(a)":                     "Tío(a)",
    "dom???stico":                  "doméstico",
    "dom???stica":                  "doméstica",

    # ── Estado civil ─────────────────────────────────────────────────────────
    # estado_civil: 'En uni???n libre'
    "uni???n libre":                "unión libre",

    # ── Etnia ─────────────────────────────────────────────────────────────────
    # etnia: 'Ind???gena', 'Raizal del archipi???lago'
    "Ind???gena":                   "Indígena",
    "ind???gena":                   "indígena",
    "archipi???lago":               "archipiélago",
    "resguardo ind???":             "resguardo indí",   # 'ind???gena' ya cubierto arriba

    # ── Salud: afiliación ─────────────────────────────────────────────────────
    # afilia_porque: 'Pertenece a un resguardo ind???gena', 'Est??? afiliado
    # a un r???gimen especial (Fuerzas Armadas, Polic???a Nacional, ...)'
    # regimen: 'Especial (Fuerzas Armadas, Ecopetrol, Universidades p???blicas)'
    # no_afiliado: 'No, en los ???LTIMOS 3 A???OS siempre ha tenido cubrimiento'
    "Polic???a":                    "Policía",
    "r???gimen":                    "régimen",
    "p???blicas":                   "públicas",
    "p???blicos":                   "públicos",
    "p???blica":                    "pública",
    "p???blico":                    "público",
    "???LTIMOS":                    "ÚLTIMOS",
    "A???OS":                       "AÑOS",

    # ── Salud: consultas y hospitalización ───────────────────────────────────
    # tratar_problema: 'Acudi??? a un hospital, cl???nica...'
    # noprof_problema: 'Demora en la asignaci???n de citas', 'No conf???a',
    #                  'Demora en la atenci???n', 'Muchos tr???mites'
    # ultima_hosp: 'Parto por ces???rea', 'Cirug???a', 'Accidente de tr???nsito'
    "Acudi???":                     "Acudió",
    "cl???nica":                    "clínica",
    "m???dico":                     "médico",
    "instituci???n":                "institución",
    "Instituci???n":                "Institución",
    "recet???":                     "recetó",
    "ces???rea":                    "cesárea",
    "Cirug???a":                    "Cirugía",
    "cirug???a":                    "cirugía",
    "tr???nsito":                   "tránsito",
    "asignaci???n":                 "asignación",
    "conf???a":                     "confía",
    "atenci???n":                   "atención",
    "tr???mites":                   "trámites",
    "Home???pata":                  "Homeópata",
    "acupunturista":                "acupunturista",   # sin acento; no cambia

    # ── Salud: condiciones crónicas ───────────────────────────────────────────
    # hipertenso: 'S???, pero s???lo en el embarazo'
    # diabetes: mismo valor
    "s???lo":                       "sólo",

    # ── Salud: actividad física y nutrición ───────────────────────────────────
    # come_frutas, come_verduras: 'Una vez al d???a, todos los d???as',
    #                             'Todos los d???as, m???s de una vez al d???a'
    # paquete_freq, fritos_freq: 'Una vez al d???a', 'Dos veces al d???a'
    # camina_tiempo, moderada_tiempo: 'Minutos por d???a', 'Horas por d???a'
    # ha_fumado: 'fumo_ultimavez': 'M???s de 10 a???os'
    "d???a":                        "día",
    "d???as":                       "días",
    "M???s":                        "Más",
    "m???s":                        "más",
    "a???o":                        "año",
    "a???os":                       "años",

    # ── Fallecimiento ─────────────────────────────────────────────────────────
    # padre_vive, madre_vive: 'Ya falleci???', 'S???'
    "falleci???":                   "falleció",

    # ── Educación: nivel ─────────────────────────────────────────────────────
    # nivel_educ: 'T???cnico sin t???tulo', 'Tecnol???gico con t???tulo',
    #             'B???sica secundaria y media (6 a 13)', 'Posgrado con t???tulo'
    # nivel_educ_cursa: 'Tecnol???gico', 'B???sica primaria (1 a 5)'
    # pcuida_niveledu: 'Uno o m???s a???os de t???cnica o tecnol???gica'
    "T???cnico":                    "Técnico",
    "T???cnica":                    "Técnica",
    "t???cnico":                    "técnico",
    "t???cnica":                    "técnica",
    "Tecnol???gico":                "Tecnológico",
    "Tecnol???gica":                "Tecnológica",
    "B???sica":                     "Básica",
    "t???tulo":                     "título",

    # ── Educación: asistencia y jornada ──────────────────────────────────────
    # jornada_esc: 'Ma???ana', 'Formaci???n a distancia'
    # tipo_hogar (preescolar): 'Jard???n o preescolar oficial',
    #                          'Modalidad familiar o ???mbito familiar del ICBF'
    "Ma???ana":                     "Mañana",
    "Formaci???n":                  "Formación",
    "formaci???n":                  "formación",
    "Jard???n":                     "Jardín",
    "jard???n":                     "jardín",
    "???mbito":                     "Ámbito",

    # ── Educación: razones de deserción ──────────────────────────────────────
    # razon_noestudia, rzn_dejoestudiar:
    #   'Termin??? su ciclo educativo', 'No quer???a estudiar m???s',
    #   'Porque tuvo hijos, por embarazo o porque se cas???',
    #   'Otra raz???n', 'Necesita educaci???n especial'
    # rzn_dejoestudiar: 'Otra raz???n: ???Cu???l?'
    "Termin???":                    "Terminó",
    "quer???a":                     "quería",
    "cas???":                       "casó",
    "raz???n":                      "razón",
    "Raz???n":                      "Razón",
    "educaci???n":                  "educación",
    "Educaci???n":                  "Educación",
    "???Cu???l?":                   "¿Cuál?",
    "No exist???a":                 "No existía",
    "No ten???an":                  "No tenían",
    "No hab???a":                   "No había",
    "Cambi???":                     "Cambió",

    # ── Actividad principal y mercado laboral ─────────────────────────────────
    # actividad_ppal:
    #   'Trabaj??? en forma remunerada...', 'Trabaj??? por lo menos UNA HORA
    #    y busc??? trabajo', 'No trabaj??? pero ten???a un empleo',
    #   'Trabaj??? como ayudante familiar sin que le pagaran'
    # descripcion_ciiu: 'Educaci???n', 'Construcci???n', 'reparaci???n de
    #   veh???culos', 'Administraci???n p???blica', 'Distribuci???n de agua;
    #   evacuaci???n y tratamiento', 'Actividades art???sticas',
    #   'cient???ficas y t???cnicas'
    # ocupacion: 'Patr???n o empleador', 'Jornalero o pe???n',
    #            'remuneraci???n', 'Trabajador por d???as'
    # tipo_contrato: 'Contrato escrito a t???rmino fijo / indefinido'
    "Trabaj???":                    "Trabajó",
    "trabaj???":                    "trabajó",
    "busc???":                      "buscó",
    "ten???a":                      "tenía",
    "Construcci???n":               "Construcción",
    "construcci???n":               "construcción",
    "reparaci???n":                 "reparación",
    "veh???culos":                  "vehículos",
    "Administraci???n":             "Administración",
    "administraci???n":             "administración",
    "evacuaci???n":                 "evacuación",
    "art???sticas":                 "artísticas",
    "cient???ficas":                "científicas",
    "t???cnicas":                   "técnicas",
    "Patr???n":                     "Patrón",
    "pe???n":                       "peón",
    "Pe???n":                       "Peón",
    "remuneraci???n":               "remuneración",
    "t???rmino":                    "término",

    # ── Búsqueda de empleo e inactividad ─────────────────────────────────────
    # medio_consiguio: 'No necesit??? o no recurri??? a ning???n medio',
    #                  'A trav???s del SENA'
    # t_busco_trab, t_trabajo_ult: 'Hace 3 a???os o m???s', 'Entre 2 y m???s'
    # razon_dejo_bus: 'Porque tuvo hijos, por embarazo o porque se cas???',
    #                 'Se cans??? de buscar trabajo', 'No encontr??? trabajo'
    "necesit???":                   "necesitó",
    "recurri???":                   "recurrió",
    "ning???n":                     "ningún",
    "trav???s":                     "través",
    "cans???":                      "cansó",
    "encontr???":                   "encontró",
    "profesi???n":                  "profesión",

    # ── Pensión y trayectoria laboral ─────────────────────────────────────────
    # rzn_nocotiza: 'Porque ya est??? pensionado',
    #               'Porque est??? esperando cumplir la edad para pensionarse'
    # razon_dejo_trab: 'Se pension??? o jubil???', 'Cierre o reestructuraci???n'
    "est???":                       "está",
    "Est???":                       "Está",
    "pension???":                   "pensionó",
    "jubil???":                     "jubiló",
    "reestructuraci???n":           "reestructuración",

    # ── Ahorros y finanzas ────────────────────────────────────────────────────
    # rzn_nosist_finan: 'No conf???a en el sistema financiero',
    #   'Se necesitan muchos tr???mites', 'No sabe c???mo hacerlo',
    #   'Cree que la entidad financiera se negar???a a abrirle una cuenta',
    #   'Lo intent??? pero la entidad financiera se neg??? a abrirle cuenta'
    "c???mo":                       "cómo",
    "negar???a":                    "negaría",
    "intent???":                    "intentó",
    "neg???":                       "negó",
    "Distribuci???n":               "Distribución",

    # ── Uso del tiempo (2016 R) ───────────────────────────────────────────────
    # actividades con '???': 'Cuidado de ni???os', 'ba???arse', 'Ocio y
    # recreaci???n', 'Tr???mites para producci???n', 'cr???ditos', 'pr???stamos'
    "ni???os":                      "niños",
    "ba???arse":                    "bañarse",
    "recreaci???n":                 "recreación",
    "producci???n":                 "producción",
    "cr???ditos":                   "créditos",
    "pr???stamos":                  "préstamos",

    # ── Actividad laboral rural (2016 R) ──────────────────────────────────────
    # actividad_ppal en R: 'Trabaj??? como ayudante familiar sin que le
    #   pagaran por lo menos UNA hora', 'Trabaj??? por lo menos UNA hora
    #   en una actividad que le gener??? alg???n ingreso'
    # ocupacion en R: 'Jornalero o pe???n en otras fincas'
    # 'Trabajador de su propia finca (...aparcer???a...)'
    "gener???":                     "generó",
    "aparcer???a":                  "aparcería",

    # ── Partido político (2016 R) ─────────────────────────────────────────────
    # simpatiza_parpol: 'Partido de Integraci???n Nacional (PIN)'
    "Integraci???n":                "Integración",

    # ── Independencia / motivación laboral (2016 R) ───────────────────────────
    # razon_tiene_negocio, razon_dejo_bus: 'Por independencia econ???mica',
    #   'Porque pagan mejor o es m???s rentable'
    "econ???mica":                  "económica",
    "rentable":                     "rentable",   # sin acento; no cambia

    # ── Noprof_serv_cual (texto libre de baja cardinalidad en U) ─────────────
    # 'NO TENIAN QUIEN LA ACOMPA???ARA', 'LOS MEDICAMENTOS LE HACEN DA???O'
    "ACOMPA???ARA":                 "ACOMPAÑARA",
    "DA???O":                       "DAÑO",
}

# ─── Correcciones 2016: patrones '??' (2 signos) en novedad_perso ────────────
#
# La columna novedad_perso en 2016 U y R usa '??' (2 signos) en lugar de
# '???' para los mismos caracteres especiales.  Esto refleja una codificación
# fuente distinta para ese campo específico.
#
# Se aplican DESPUÉS de CORRECCIONES_2016 porque los patrones de 2 signos
# son subcadenas de los de 3 signos: si se aplican primero, 'Lleg???' quedaría
# como 'Llegó?' en lugar de 'Llegó???' → 'Llegó'.  Como novedad_perso en
# 2016 usa solo '??' y nunca '???', el orden inverso es seguro.

CORRECCIONES_NOVEDAD_PERSO = {
    "adopci??n":        "adopción",   # 'Lleg?? por proceso de adopci??n...'
    "Lleg??":           "Llegó",      # prácticamente todos los valores nuevos
    "organiz??":        "organizó",   # 'se cas??/organiz?? con un miembro'
    "integr??":         "integró",    # 'Lleg?? con un pariente que se integr??'
    "Naci??":           "Nació",
    "separ??":          "separó",     # 'se separ??/enviud??'
    "enviud??":         "enviudó",
    "cas??":            "casó",       # 'se cas??/organiz??'
}


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def corregir_columnas(df: pd.DataFrame, correcciones: dict) -> pd.DataFrame:
    """
    Aplica reemplazos exactos de subcadena en todas las columnas de texto
    del DataFrame y elimina espacios iniciales/finales residuales.

    El recorrido respeta el orden de inserción del diccionario; los patrones
    más largos deben ir primero para evitar efectos secundarios de fragmentos
    más cortos (p. ej. 'S???' antes de 'S??' si ambos estuviesen en el mismo
    diccionario).
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
    Lee UPersonas o RPersonas de 2010 y agrega la columna zona.

    No hay problemas de codificación en ninguno de los dos archivos de 2010.

    Identificadores disponibles: ola, consecutivo, orden, llave_ID_lb.
    No existen llave, hogar, llaveper ni sus equivalentes de ola 3.

    La columna zona no está en el archivo fuente; se agrega como 'Urbano'
    o 'Rural' según el prefijo del archivo (U o R).
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2010", f"{tipo}Personas-csv.tab")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # 2010 ya tiene columna 'ola' (valor 1); se agrega 'zona' que no existe
    df.insert(0, "zona", zona)

    print(f"    {zona} 2010: {df.shape[0]:>6} filas × {df.shape[1]:>4} columnas")
    return df


def procesar_2013(tipo: str) -> pd.DataFrame:
    """
    Lee UPersonas o RPersonas de 2013 y aplica correcciones mínimas.

    Correcciones aplicadas (CORRECCIONES_2013):
    - 'Otra, ¿cuál? __________________' en rzn_dejoestudiar:
      el patrón '?cu?l?' (signo único) se reemplaza por '¿cuál?'.
    - 'Otro. ¿Cuál?' en columnas ocupacion*:
      el patrón 'Cu?l?' se reemplaza por '¿Cuál?'.
    Los demás campos categóricos (parentesco, sexo, estado_civil, etnia,
    nivel_educ, actividad_ppal, ocupacion, etc.) están limpios en 2013 y
    no requieren corrección.  El texto libre de descrip_oficio y nombres
    de partidos políticos se deja sin modificar.

    Nuevas columnas respecto a 2010: hogar, llave, llaveper, novedad_perso,
    embarazada, organizaciones sociales.
    2013 R es significativamente más ancha que 2013 U (723 vs 529 columnas)
    por el módulo de mercado laboral rural ampliado.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2013", f"{tipo}Personas-csv.tab")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # 2013 no tiene columna 'ola'; se agregan 'ola' y 'zona' manualmente
    df.insert(0, "ola",  2)
    df.insert(0, "zona", zona)
    df = corregir_columnas(df, CORRECCIONES_2013)

    print(f"    {zona} 2013: {df.shape[0]:>6} filas × {df.shape[1]:>4} columnas")
    return df


def procesar_2016(tipo: str) -> pd.DataFrame:
    """
    Lee UPersonas o RPersonas de 2016 y aplica correcciones de codificación.

    Correcciones aplicadas:
    1. CORRECCIONES_2016 ('???', 3 signos): se aplican a todas las columnas
       de texto.  Cubren parentesco, estado_civil, etnia, nivel_educ,
       todas las variables binarias Sí/No, salud, educación, mercado laboral,
       organizaciones sociales y actividad principal.
    2. CORRECCIONES_NOVEDAD_PERSO ('??', 2 signos): se aplican solo sobre
       la columna novedad_perso, que usa 2 signos en lugar de 3 para las
       mismas ó, ó y ó del español.  Se aplican después de CORRECCIONES_2016
       para evitar colisiones entre patrones de distinta longitud.

    Nuevas columnas respecto a 2013: llave_n16, hogar_n16, llaveper_n16,
    uso del tiempo (actividades diarias), banca móvil, acceso a internet.
    2016 R es la más ancha del panel (810 columnas).

    Calidad de datos en llave y hogar:
    En 2016, algunas personas tienen llave=NaN y hogar=NaN.  Esto ocurre
    cuando la persona fue detectada en la ola 3 pero su hogar de referencia
    no tiene sub-hogar asignado en ola 2.  Tienen llave_n16 y llaveper_n16
    válidos.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2016", f"{tipo}Personas-csv.tab")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # 2016 no tiene columna 'ola'; se agregan 'ola' y 'zona' manualmente
    df.insert(0, "ola",  3)
    df.insert(0, "zona", zona)

    # Correcciones ??? (3 signos) — aplica a todo el DataFrame
    df = corregir_columnas(df, CORRECCIONES_2016)

    # Correcciones ?? (2 signos) — exclusivas de la columna novedad_perso
    if "novedad_perso" in df.columns:
        for patron, correcto in CORRECCIONES_NOVEDAD_PERSO.items():
            df["novedad_perso"] = df["novedad_perso"].str.replace(
                patron, correcto, regex=False
            )

    print(f"    {zona} 2016: {df.shape[0]:>6} filas × {df.shape[1]:>4} columnas")
    return df


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Lee, corrige y concatena las seis bases de personas (U y R × 3 olas).

    Ordenamiento de columnas del output
    ------------------------------------
    1. zona        : 'Urbano' / 'Rural' (agregada por el script)
    2. Identificadores por ola, del más granular al más general:
         llaveper_n16, llaveper, llave_ID_lb, ola, consecutivo, hogar,
         hogar_n16, orden, llave, llave_n16
    3. Resto de columnas en el orden original de aparición.

    Tratamiento de columnas con tipos mixtos
    -----------------------------------------
    Al concatenar archivos de distintas olas, algunas columnas que en un
    archivo son numéricas (float) y en otro son de texto (object) resultan
    en columnas de tipo object con valores mixtos.  pyarrow no puede
    serializar este tipo de array (ArrowTypeError).  Se unifican como
    string preservando NaN (→ null en parquet) antes de guardar.
    Mismo patrón que en Hogar (04) y Comunidades (05).

    Retorna el DataFrame final.
    """
    frames = []

    for tipo in ["U", "R"]:
        print(f"\n  Procesando {tipo}Personas...")

        print("    Leyendo 2010...")
        frames.append(procesar_2010(tipo))

        print("    Leyendo 2013...")
        frames.append(procesar_2013(tipo))

        print("    Leyendo 2016...")
        frames.append(procesar_2016(tipo))

    # Concatenar: la unión de columnas llena con NaN las celdas ausentes.
    base_final = pd.concat(frames, axis=0, ignore_index=True).copy()

    # Reordenar columnas: zona e identificadores primero, luego el resto.
    all_id_cols = [
        "zona", "llaveper_n16", "llaveper", "llave_ID_lb",
        "ola", "consecutivo", "hogar", "hogar_n16", "orden",
        "llave", "llave_n16",
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
    print(f"Dimensiones            : {base_final.shape[0]:>6} filas × {base_final.shape[1]:>4} columnas")
    print(f"Olas presentes         : {sorted(base_final['ola'].unique())}")
    print(f"llave_ID_lb únicos     : {base_final['llave_ID_lb'].nunique():>6}  (miembros panel original)")
    print()
    print("  Filas por ola y zona:")
    olas = sorted(base_final["ola"].dropna().unique())
    for ola_val in olas:
        for zona_val in ["Urbano", "Rural"]:
            n = ((base_final["ola"] == ola_val) & (base_final["zona"] == zona_val)).sum()
            print(f"    Ola {int(ola_val)} – {zona_val:<6}: {n:>6} personas")
    nan_ola = base_final["ola"].isna().sum()
    if nan_ola:
        print(f"    Ola NaN (calidad datos): {nan_ola:>6} personas — ola=NaN en archivo fuente")

    return base_final


if __name__ == "__main__":
    base_final = main()
