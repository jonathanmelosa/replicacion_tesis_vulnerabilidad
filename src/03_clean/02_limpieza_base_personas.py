"""
Encuesta Longitudinal Colombiana (ELCA)
Limpieza de la base de personas (olas 2010, 2013 y 2016).

Contexto
--------
Este script parte del output de
src/01_download/01_descarga_ELCA/06_consolidacion_bases_personas.py
(data/processed/personas_elca_longitudinal.parquet), que documentaba la
corrupción de codificación de 2013 como acotada a "texto libre y un par de
etiquetas de opción abierta" ("Otra, ¿cuál?"). Al construir las variables de
composición del hogar para el modelo benchmark de prediccion de pobreza
 se
encontró que esa descripción era incompleta.

Problema identificado
-----------------------
El caracter de reemplazo Unicode "�" (U+FFFD) reemplaza vocales acentuadas
(á, é, í, ó, ú) y otros caracteres especiales del español en **535 columnas**
de personas_elca_longitudinal.parquet -- 180 en ola 1 (2010) y 435 en ola 2
(2013); ola 3 (2016) no tiene ningun caso. Esto incluye variables
categoricas analiticamente relevantes que el docstring de 06_... describia
como limpias en 2013: `parentesco`, `etnia`, `afiliacion`, entre otras.

Se verifico contra los bytes crudos de UPersonas-csv.tab 2013 que la corrupcion YA esta en el archivo fuente que
distribuye la ELCA -- no es un problema de como se lee el archivo (una
codificacion equivocada seria corregible re-leyendo con el encoding
correcto), es una perdida de informacion irreversible a nivel de caracter:
el byte original que representaba la vocal acentuada fue reemplazado por
el caracter de reemplazo UTF-8 (EF BF BD) antes de que este pipeline
tuviera acceso al archivo. La unica forma de recuperar el valor correcto es
por contexto: comparar contra la misma categoria aparecida sin corrupcion
en otra fila/ola, o contra el diccionario de la encuesta.

Alcance de esta limpieza (deliberadamente acotado, ver "Lo que NO se
corrige" mas abajo)
---------------------------------------------------------------------------
De las 535 columnas afectadas, 484 tienen vocabulario cerrado (<=25
categorias distintas) y 51 son de alta cardinalidad (texto libre:
descripciones de oficio, nombres de programas sociales, etc.). Este script:

  1. Corrige automaticamente, dentro de cada columna de vocabulario
     cerrado, los valores corruptos que tienen EXACTAMENTE UN candidato
     limpio compatible en la misma columna (en cualquier ola) --
     construyendo un patron donde cada "�" es un comodin de un caracter y
     buscando que valor limpio de esa misma columna calza exactamente. Es
     automatico y reproducible (no depende de una lista escrita a mano),
     pero deliberadamente conservador: si hay 0 o mas de 1 candidato, NO
     se corrige (se deja documentado, ver mas abajo).
  2. Corrige la ambiguedad mas frecuente por separado: cuando el unico
     comodin resuelve a {"Si", "Sí"} (con o sin tilde), se usa "Sí" -- la
     grafia que usa la ELCA de forma consistente en el resto del
     cuestionario (ver SI_TOKENS en build_ingreso_hogar.py).
  3. Corrige a mano, verificado contra ortografia estandar del español,
     las variables priorizadas para el modelo benchmark que la correccion
     automatica no pudo resolver (ningun candidato limpio en la misma
     columna). Ver CORRECCIONES_MANUALES_PRIORITARIAS -- esta lista crece
     a medida que se van necesitando mas variables del modulo Personas
     (ver seccion "Este es un script vivo" mas abajo); no es una lista
     cerrada de una sola vez.

Resultado: las variables en CORRECCIONES_MANUALES_PRIORITARIAS quedan
100% libres de "�" al terminar este script (se valida con assert por
columna). El resto de columnas de vocabulario cerrado quedan mayormente
corregidas (ver reporte impreso al ejecutar); las columnas de texto libre
NO se tocan en absoluto.

Lo que NO se corrige (documentado, no oculto)
------------------------------------------------
De los 956 valores corruptos en columnas de vocabulario cerrado, 322 se
resuelven automaticamente (puntos 1-2) y ~28 mas con la lista manual
(punto 3). Los ~600 restantes, repartidos en ~270 columnas que HOY no son
candidatas para el benchmark, no tienen ningun valor limpio equivalente en
ningun lado del dataset para inferir la correccion -- resolverlos
requeriria consultar, columna por columna, los diccionarios PDF especificos
de cada ola. Se deja como tarea pendiente, a resolver cuando (si) esas
columnas se necesiten; el reporte completo de que quedo sin resolver se
guarda en docs/variable_audit/personas_corrupcion_residual.csv para que la
proxima vez que se necesite una de esas columnas no haya que re-descubrir
el problema desde cero. Las 51 columnas de texto libre (>25 categorias)
tampoco se tocan -- mismo criterio que el proyecto ya aplica en otros
modulos (ARMONIZACION_ARTICULOS, CORRECCIONES_2013 de
06_consolidacion_bases_personas.py): no vale la pena un diccionario de
miles de palabras para campos que no se usan como variable categorica.

Este es un script "vivo": si en el futuro se necesita una columna hoy
pendiente, agregar su correccion en CORRECCIONES_MANUALES_PRIORITARIAS (o
investigar por que la correccion automatica no la resolvio) en vez de
trabajar alrededor del problema en el script de features que la use.
"""

from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
DOC_PATH = PROJECT_ROOT / "docs" / "variable_audit" / "personas_corrupcion_residual.csv"

CARDINALIDAD_MAXIMA_CERRADA = 25
MARCADOR = "�"  # "�"

# ─── Correcciones manuales: variables prioritarias sin match automatico ──────
# Verificadas contra ortografia estandar del español. `ocupacion`/`parentesco`
# tienen demasiadas categorias (>25) para entrar al barrido automatico por
# vocabulario cerrado, aunque su cardinalidad real es manejable a mano.

CORRECCIONES_MANUALES_PRIORITARIAS = {
    "estaba_sss": {
        # mismo patron que "sindicato"/"cotiza_fp": encontrado al comparar
        # descripciones de pregunta contra los diccionarios PDF para buscar
        # renombrados entre olas (`estaba_sss` <-> `segsoc_salud`).
        "S�": "Sí",
    },
    "estaba_fp": {
        # mismo patron, encontrado junto con "estaba_sss" al armonizar
        # `estaba_fp` (ola 1) con `afiliacion_fp` (ola 2/3).
        "S�": "Sí",
    },
    "cotiza_fp": {
        # mismo patron que "sindicato": categoria exclusiva de ola 1 sin
        # candidato limpio en la misma columna contra el cual hacer match
        # automatico. Encontrado junto con "sindicato" al armonizar
        # `cotiza_fp` (ola 1) con `cotizando` (ola 2/3) en
        # build_ahorro_capital_social_hogar.py.
        "Si est� cotizando y recibe pensi�n": "Si está cotizando y recibe pensión",
    },
    "sindicato": {
        # cardinalidad cerrada (2) pero SIN candidato limpio en la misma
        # columna: en ola 1, TODAS las respuestas "Sí" quedaron corruptas
        # como "S�" (175 casos) -- la correccion automatica no tiene con
        # que comparar dentro de la columna y lo deja en el residual
        # (ver personas_corrupcion_residual.csv, fila "sindicato"). Unico
        # candidato compatible con el vocabulario Sí/No/No informa de la
        # pregunta: "Sí". Encontrado al verificar renombrados entre olas
        # (sindicato/org_sindicato) para build_ahorro_capital_social_hogar.py.
        "S�": "Sí",
    },
    "parentesco": {
        # cardinalidad (26) > CARDINALIDAD_MAXIMA_CERRADA: nunca pasa por la
        # correccion automatica, se listan aqui las 5 corruptas completas.
        "C�nyuge o compa�era(o)": "Cónyuge o compañera(o)",
        "Servicio dom�stico, cuidandero y sus parientes": "Servicio doméstico, cuidandero y sus parientes",
        "T�o(a)": "Tío(a)",
        "Nieto(a)del jefe del hogar o de su c�nyuge": "Nieto(a)del jefe del hogar o de su cónyuge",
        "Bisnieto del jefe del hogar o de su c�nyuge": "Bisnieto del jefe del hogar o de su cónyuge",
    },
    "ocupacion": {
        # cardinalidad (31) > CARDINALIDAD_MAXIMA_CERRADA: nunca pasa por la
        # correccion automatica, se listan aqui las 11 corruptas completas.
        "Asalariado de empresa particular con contrato a t�rmino indefinido":
            "Asalariado de empresa particular con contrato a término indefinido",
        "Asalariado de empresa particular con contrato a t�rmino fijo":
            "Asalariado de empresa particular con contrato a término fijo",
        "Asalariado del Gobierno con contrato a t�rmino indefinido":
            "Asalariado del Gobierno con contrato a término indefinido",
        "Asalariado del Gobierno con contrato a t�rmino fijo":
            "Asalariado del Gobierno con contrato a término fijo",
        "Trabajador de su propia finca (en arriendo o aparcer�a)":
            "Trabajador de su propia finca (en arriendo o aparcería)",
        "Trabajador sin remuneraci�n": "Trabajador sin remuneración",
        "Empleado dom�stico": "Empleado doméstico",
        "Jornalero o pe�n": "Jornalero o peón",
        "Patr�n o empleador": "Patrón o empleador",
        "Trabajador familiar sin remuneraci�n": "Trabajador familiar sin remuneración",
        "Jornalero o trabajador por d�as": "Jornalero o trabajador por días",
    },
    "etnia": {
        "Raizal de archipi�lago": "Raizal de archipiélago",
    },
    "nivel_educ": {
        "B�sica secundaria (6 a 13)": "Básica secundaria (6 a 13)",
    },
    "trabajo_padre": {
        "Jornalero o pe�n": "Jornalero o peón",
        "Trabajador de su propia finca o de una finca que ten�a o tiene en arriendo o aparcer�a":
            "Trabajador de su propia finca o de una finca que tenía o tiene en arriendo o aparcería",
        "Nunca ha trabajado o nunca trabaj�": "Nunca ha trabajado o nunca trabajó",
        "Empleado dom�stico": "Empleado doméstico",
        "Patr�n o empleador": "Patrón o empleador",
        "Trabajador familiar sin remuneraci�n": "Trabajador familiar sin remuneración",
    },
    "trabajo_madre": {
        "Nunca ha trabajado o nunca trabaj�": "Nunca ha trabajado o nunca trabajó",
        "Empleado dom�stico": "Empleado doméstico",
        "Trabajador de su propia finca o de una finca que ten�a o tiene en arriendo o aparcer�a":
            "Trabajador de su propia finca o de una finca que tenía o tiene en arriendo o aparcería",
        "Jornalero o pe�n": "Jornalero o peón",
        "Trabajador familiar sin remuneraci�n": "Trabajador familiar sin remuneración",
        "Patr�n o empleador": "Patrón o empleador",
    },
    "nivel_educ_cursa": {
        "B�sica secundaria(6 a 13)": "Básica secundaria(6 a 13)",
    },
    # ─── Resto del pool candidato a benchmark (ola1+ola2), verificado tras
    # el cruce completo contra docs/variable_audit/personas_hogar_construccion.csv ───
    "ocupacion2": {
        "Trabajador familiar sin remuneraci�n": "Trabajador familiar sin remuneración",
        "Jornalero o pe�n": "Jornalero o peón",
        "Trabajador de su propia finca o de una finca que tiene en arriendo o aparcer�a":
            "Trabajador de su propia finca o de una finca que tiene en arriendo o aparcería",
        # truncado en la fuente original; se conserva el truncamiento
        "Trabajador de su propia finca (propia, en arriendo o aparcer�a, etc. Independien":
            "Trabajador de su propia finca (propia, en arriendo o aparcería, etc. Independien",
        "Otro. Cu�l?": "Otro. Cuál?",
    },
    "registro_mercantil": {
        "Si lo tiene pero no lo renov� este a�o": "Si lo tiene pero no lo renovó este año",
        "Si lo tiene y lo renov� este a�o": "Si lo tiene y lo renovó este año",
        "S� lo tiene pero no lo renov� este a�o": "Sí lo tiene pero no lo renovó este año",
        "S� lo tiene y lo renov� este a�o": "Sí lo tiene y lo renovó este año",
    },
    "medio_consiguio": {
        "No necesit� o no recurri� a ning�n medio": "No necesitó o no recurrió a ningún medio",
        # "directaemente" es un typo de la fuente original, se conserva
        "El empleador lo contact� directaemente": "El empleador lo contactó directaemente",
        "El empleador lo contact� directamente": "El empleador lo contactó directamente",
    },
    "t_busco_trab": {
        "Hace m�s de 5 a�os": "Hace más de 5 años",
        "Entre 2 y menos de 5 a�os": "Entre 2 y menos de 5 años",
    },
    "razon_tiene_negocio": {
        "Por tradici�n familiar": "Por tradición familiar",
    },
    "padre_vive": {"Falleci�": "Falleció"},
    "madre_vive": {"Falleci�": "Falleció"},
    "ultima_hosp": {"Accidente de tr�nsito": "Accidente de tránsito"},
    "recibio_beca": {"No recibi� ninguno": "No recibió ninguno"},
    # Columnas Sí/No con filtro previo (solo se pregunta si aplica): la unica
    # categoria presente termina siendo "S�" sin ningun "No"/"Sí" limpio en la
    # misma columna para que la correccion automatica lo intente resolver.
    "ahorro_futuro": {"S�": "Sí"},
    "ahorro_educ": {"S�": "Sí"},
    "ahorro_casa": {"S�": "Sí"},
    "ahorro_carro": {"S�": "Sí"},
    "ahorro_otros_act": {"S�": "Sí"},
    "ahorro_recre": {"S�": "Sí"},
    "ahorro_montar": {"S�": "Sí"},
    "ahorro_otro": {"S�": "Sí"},
    "beca_otro": {"S�": "Sí"},
    "beca_misma_ins": {"S�": "Sí"},
    "beca_emp_pri": {"S�": "Sí"},
    "beca_emp_pub": {"S�": "Sí"},
    "beca_accionsocial": {"S�": "Sí"},
    "beca_prg_gob": {"S�": "Sí"},
    "beca_cajacom": {"S�": "Sí"},
    # vr_ganancia/vr_salario son numericas con un token categorico ("No
    # recibió") mezclado, mismo patron que ZERO_TOKENS en build_ingreso_hogar.py.
    "vr_ganancia": {"No recibi�": "No recibió"},
    "vr_salario": {"No recibi�": "No recibió"},
    # n_empleados: categoria top-coded, no texto libre real.
    "n_empleados": {"50 personas y m�s": "50 personas y más"},
    "razon_noestudia": {
        "Debe encargarse de labores dom�sticas y/o del cuidado de los ni�os, ancianos o discapacitados":
            "Debe encargarse de labores domésticas y/o del cuidado de los niños, ancianos o discapacitados",
        "No quiere estudiar m�s": "No quiere estudiar más",
        "Termin� su ciclo educativo": "Terminó su ciclo educativo",
        "Necesita educaci�n especial": "Necesita educación especial",
        "Porque tuvo hijos, por embarazo o se cas�": "Porque tuvo hijos, por embarazo o se casó",
    },
    "razon_dejo_trab": {
        "Decidi� no trabajar m�s": "Decidió no trabajar más",
        "Cumpli� el ciclo en ese trabajo": "Cumplió el ciclo en ese trabajo",
        "Se pension� / jubil�": "Se pensionó / jubiló",
        "Otra raz�n": "Otra razón",
        "Cierre o reestructuraci�n de la empresa": "Cierre o reestructuración de la empresa",
        "Despido o declaraci�n de insubsistencia": "Despido o declaración de insubsistencia",
        "Le sali� un trabajo mejor": "Le salió un trabajo mejor",
        "Porque tuvo hijos,  por embarazo o porque se cas�": "Porque tuvo hijos,  por embarazo o porque se casó",
        "Se pension� o jubil�": "Se pensionó o jubiló",
    },
    "actividad_ppal": {
        "Trabaj� en forma remunerada por lo menos una hora":
            "Trabajó en forma remunerada por lo menos una hora",
        "No trabaj� pero ten�a un empleo o trabajo de por lo menos una hora":
            "No trabajó pero tenía un empleo o trabajo de por lo menos una hora",
        "Trabaj� como ayudante familiar sin remuneraci�n por lo menos una hora":
            "Trabajó como ayudante familiar sin remuneración por lo menos una hora",
        "Trabaj� por lo menos una hora y busc� trabajo":
            "Trabajó por lo menos una hora y buscó trabajo",
        # Truncado en la fuente original (corta en "gener"); se conserva el
        # truncamiento, solo se corrige el caracter de reemplazo.
        "Trabaj� en forma remunerada por lo menos UNA hora  en una actividad que le gener":
            "Trabajó en forma remunerada por lo menos UNA hora  en una actividad que le generó",
        "Trabaj� por lo menos UNA hora  en una actividad que le gener� alg�n ingreso":
            "Trabajó por lo menos UNA hora  en una actividad que le generó algún ingreso",
    },
}

# ─── Correcciones verificadas contra los diccionarios PDF de la encuesta ─────
# A diferencia de CORRECCIONES_MANUALES_PRIORITARIAS (verificadas a mano por
# ortografia), estas se extrajeron programaticamente de los 15 diccionarios
# disponibles (Personas + Hogar + RActivos_hogar, olas 2010/2013/2016, ambas
# zonas -- data/interim/raw/elca_*/{R,U}{Personas,Hogar,Activos_hogar}.pdf
# via pdftotext), que documentan las etiquetas de cada categoria en texto NO
# corrupto -- una fuente independiente del archivo .tab exportado (se
# amplio de los 4 diccionarios de Personas 2010/2013 a los 15 disponibles
# tras una segunda ronda de revision). Para cada
# valor corrupto se construyo un patron con limites de palabra (cada '�' es
# un comodin de 1 caracter) y se busco una coincidencia UNICA en el texto
# completo de los 15 diccionarios; solo se acepta si hay exactamente un
# candidato Y pasa la validacion de longitud/posicion (ver
# validar_correccion()): la cadena
# limpia debe tener EXACTAMENTE la misma longitud que la corrupta, y todo
# caracter que no sea '�' en el original debe coincidir caracter por
# caracter con el resultado. Un bug de anclaje de regex en la primera
# version de este metodo (que no exigia coincidencia de longitud) genero
# varias correcciones falsas, detectadas en revision manual antes de
# aplicarse.
CORRECCIONES_DICCIONARIO_PDF = {
    'bebe_freq': {
        'No consumiste alcohol los �ltimos 12 meses': 'No consumiste alcohol los últimos 12 meses',
    },
    'bustrab_fh': {
        'S�, en todos los meses': 'Sí, en todos los meses',
        'S�, en algunos meses': 'Sí, en algunos meses',
    },
    'cargo_cr': {
        'S�, en algunos meses': 'Sí, en algunos meses',
        'S�, en todos los meses': 'Sí, en todos los meses',
    },
    'celular_vecinos': {
        'La mayor�a': 'La mayoría',
    },
    'cotiza_fp': {
        'Si est� cotizando, pero todav�a no es pensionado': 'Si está cotizando, pero todavía no es pensionado',
        'No, porque ya est� pensionado': 'No, porque ya está pensionado',
        'No cotiza porque est� esperando cumplir la edad para pensionarse': 'No cotiza porque está esperando cumplir la edad para pensionarse',
    },
    'cr_ganaba_in': {
        'M�s del salario m�nimo': 'Más del salario mínimo',
        'El salario m�nimo': 'El salario mínimo',
        'Menos del salario m�nimo': 'Menos del salario mínimo',
    },
    'crees_vivir': {
        'A�os': 'Años',
    },
    'cuidado_personal': {
        'Es incapaz de ba�arse o vestirse': 'Es incapaz de bañarse o vestirse',
    },
    'dejoestudio': {
        'a�os': 'años',
    },
    'descrip_activ3': {
        'Actividades art�sticas, de entretenimiento y recreaci�n': 'Actividades artísticas, de entretenimiento y recreación',
        'Agricultura, ganader�a, caza, silvicultura y pesca': 'Agricultura, ganadería, caza, silvicultura y pesca',
    },
    'descrip_activ4': {
        'Construcci�n': 'Construcción',
        'Actividades de atenci�n de la salud humana y de asistencia social': 'Actividades de atención de la salud humana y de asistencia social',
        'Agricultura, ganader�a, caza, silvicultura y pesca': 'Agricultura, ganadería, caza, silvicultura y pesca',
    },
    'descrip_activ5': {
        'Agricultura, ganader�a, caza, silvicultura y pesca': 'Agricultura, ganadería, caza, silvicultura y pesca',
    },
    'descrip_activ6': {
        'Agricultura, ganader�a, caza, silvicultura y pesca': 'Agricultura, ganadería, caza, silvicultura y pesca',
    },
    'diria_que_eleccion': {
        'Vota en la mayor�a de las elecciones': 'Vota en la mayoría de las elecciones',
    },
    'diria_que_partido': {
        'Vota por el mismo partido en la mayor�a de las elecciones': 'Vota por el mismo partido en la mayoría de las elecciones',
    },
    'educ_madre': {
        'Algunos a�os de primaria': 'Algunos años de primaria',
        'Algunos a�os de secundaria': 'Algunos años de secundaria',
        'Universidad con t�tulo': 'Universidad con título',
        'Universidad sin t�tulo': 'Universidad sin título',
        'Uno o m�s a�os de t�cnica o tecnol�gica': 'Uno o más años de técnica o tecnológica',
    },
    'educ_padre': {
        'Algunos a�os de primaria': 'Algunos años de primaria',
        'Universidad con t�tulo': 'Universidad con título',
        'Algunos a�os de secundaria': 'Algunos años de secundaria',
        'Universidad sin t�tulo': 'Universidad sin título',
        'Uno o m�s a�os de t�cnica o tecnol�gica': 'Uno o más años de técnica o tecnológica',
    },
    'pcuida_niveledu': {
        'Uno o m�s a�os de t�cnica o tecnol�gica': 'Uno o más años de técnica o tecnológica',
    },
    'fuente_amigos': {
        'Us�': 'Usó',
        'No us�': 'No usó',
    },
    'fuente_diarios': {
        'No us�': 'No usó',
        'Us�': 'Usó',
    },
    'fuente_internet': {
        'No us�': 'No usó',
        'Us�': 'Usó',
    },
    'fuente_libros': {
        'No us�': 'No usó',
        'Us�': 'Usó',
    },
    'fuente_radio': {
        'No us�': 'No usó',
        'Us�': 'Usó',
    },
    'fuente_revistas': {
        'No us�': 'No usó',
        'Us�': 'Usó',
    },
    'fuente_tv': {
        'Us�': 'Usó',
        'No us�': 'No usó',
    },
    'fundo_negocio': {
        '�l(Ella) y otros familiares': 'El(Ella) y otros familiares',
        'Lo hered�': 'Lo heredó',
    },
    'informante': {
        'Encuestado id�neo': 'Encuestado idóneo',
    },
    'lugar_nacimiento': {
        'En otro pa�s': 'En otro país',
    },
    'lugar_vivia': {
        'En otro pa�s': 'En otro país',
    },
    'lugar_vivia5': {
        'En otro pa�s': 'En otro país',
    },
    'motivo_mig_1': {
        'Regres� al hogar': 'Regresó al hogar',
    },
    'motivo_mig_2': {
        'Regres� al hogar': 'Regresó al hogar',
    },
    'motivo_mig_3': {
        'Regres� al hogar': 'Regresó al hogar',
    },
    'nivel_educ_2010': {
        'T�cnico sin t�tulo': 'Técnico sin título',
        'Tecnol�gico sin t�tulo': 'Tecnológico sin título',
        'Universitario con t�tulo': 'Universitario con título',
        'Universitario sin t�tulo': 'Universitario sin título',
        'T�cnico con t�tulo': 'Técnico con título',
        'Posgrado con t�tulo': 'Posgrado con título',
        'Tecnol�gico con t�tulo': 'Tecnológico con título',
    },
    'noprof_acci': {
        'No tiene EPS o seguro m�dico': 'No tiene EPS o seguro médico',
        'No conf�a en los m�dicos': 'No confía en los médicos',
        'Muchos tr�mites para la cita': 'Muchos trámites para la cita',
        'El centro de atenci�n queda lejos': 'El centro de atención queda lejos',
    },
    'noprof_enfe': {
        'No tiene EPS o seguro m�dico': 'No tiene EPS o seguro médico',
        'Muchos tr�mites para la cita': 'Muchos trámites para la cita',
        'El centro de atenci�n queda lejos': 'El centro de atención queda lejos',
        'No conf�a en los m�dicos': 'No confía en los médicos',
    },
    'noprof_odon': {
        'El centro de atenci�n queda lejos': 'El centro de atención queda lejos',
        'Muchos tr�mites para la cita': 'Muchos trámites para la cita',
        'No conf�a en los m�dicos': 'No confía en los médicos',
        'No tiene EPS o seguro m�dico': 'No tiene EPS o seguro médico',
    },
    'noprof_problema': {
        'No sab�a que ten�a derecho': 'No sabía que tenía derecho',
    },
    'noson_hogar': {
        'Por independencia econ�mica': 'Por independencia económica',
        'Porque pagan mejor o es m�s rentable': 'Porque pagan mejor o es más rentable',
    },
    'noson_hogar2': {
        'Por independencia econ�mica': 'Por independencia económica',
    },
    'ocupacion_pt': {
        'Empleado dom�stico': 'Empleado doméstico',
        'Trabajador familiar sin remuneraci�n': 'Trabajador familiar sin remuneración',
        'Jornalero o pe�n': 'Jornalero o peón',
        'Patr�n o empleador': 'Patrón o empleador',
    },
    'ocupacion_tenia': {
        'Empleado dom�stico': 'Empleado doméstico',
        'Trabajador familiar sin remuneraci�n': 'Trabajador familiar sin remuneración',
        'Jornalero o pe�n': 'Jornalero o peón',
        'Patr�n o empleador': 'Patrón o empleador',
    },
    'pais_nac': {
        'Espa�a': 'España',
        'Per�': 'Perú',
        'Rep�blica Dominicana': 'República Dominicana',
    },
    'pais_vivia': {
        'Per�': 'Perú',
        'Espa�a': 'España',
        'Rep�blica Dominicana': 'República Dominicana',
    },
    'pais_vivia5': {
        'Espa�a': 'España',
    },
    'parent_inform': {
        'Servicio dom�stico, cuidandero y sus parientes': 'Servicio doméstico, cuidandero y sus parientes',
    },
    'pariente': {
        'Servicio dom�stico, cuidandero y sus parientes': 'Servicio doméstico, cuidandero y sus parientes',
    },
    'prestamo_vecino': {
        'La mayor�a': 'La mayoría',
    },
    'quien_cuida': {
        'Una ni�era': 'Una niñera',
    },
    'quisiera_vivir': {
        'A�os': 'Años',
    },
    'razon_jornalero': {
        'Mejores pagos o m�s rentabilidad': 'Mejores pagos o más rentabilidad',
        'Independencia econ�mica': 'Independencia económica',
    },
    'razon_llego': {
        'Regres� al hogar': 'Regresó al hogar',
    },
    'razon_negocio1': {
        'Por tradici�n familiar': 'Por tradición familiar',
    },
    'razon_negocio2': {
        'Por tradici�n familiar': 'Por tradición familiar',
    },
    'razon_negocio3': {
        'Por tradici�n familiar': 'Por tradición familiar',
    },
    'razon_negocio4': {
        'Por tradici�n familiar': 'Por tradición familiar',
    },
    'razon_noacepto': {
        'Ubicaci�n geogr�fica inadecuada': 'Ubicación geográfica inadecuada',
    },
    'razon_noahorra': {
        'Est� pagando una deuda': 'Está pagando una deuda',
    },
    'razon_novivia': {
        'Otra raz�n': 'Otra razón',
        'Se fue de la casa o abandon� el hogar': 'Se fue de la casa o abandonó el hogar',
        'Ambos hab�an fallecido': 'Ambos habían fallecido',
    },
    'razon_nsfinan': {
        'Hay que hacer muchos tr�mites': 'Hay que hacer muchos trámites',
        'El dinero no est� disponible inmediatamente': 'El dinero no está disponible inmediatamente',
        'No conf�a en el sistema financiero': 'No confía en el sistema financiero',
    },
    'razon_retiro_in': {
        'Cierre o reestructuraci�n de la empresa': 'Cierre o reestructuración de la empresa',
        'Le sali� un trabajo mejor': 'Le salió un trabajo mejor',
        'Decidi� no trabajar m�s': 'Decidió no trabajar más',
        'Despido o declaraci�n de insubsistencia': 'Despido o declaración de insubsistencia',
        'Cumpli� el ciclo en ese trabajo': 'Cumplió el ciclo en ese trabajo',
        'Otra raz�n': 'Otra razón',
    },
    'razon_retiro_pt': {
        'Decidi� no trabajar m�s': 'Decidió no trabajar más',
        'Le sali� un trabajo mejor': 'Le salió un trabajo mejor',
        'Despido o declaraci�n de insubsistencia': 'Despido o declaración de insubsistencia',
        'Otra raz�n': 'Otra razón',
        'Cierre o reestructuraci�n de la empresa': 'Cierre o reestructuración de la empresa',
        'Cumpli� el ciclo en ese trabajo': 'Cumplió el ciclo en ese trabajo',
    },
    'recibia_pt': {
        'El salario m�nimo': 'El salario mínimo',
        'Menos del salario m�nimo': 'Menos del salario mínimo',
    },
    'recibio_ganancia': {
        'No recibi�': 'No recibió',
    },
    'rzn_dejoestudiar': {
        'Deb�a encargarse de labores dom�sticas y/o del cuidado de los ni�os, ancianos o': 'Debía encargarse de labores domésticas y/o del cuidado de los niños, ancianos o',
    },
    'sss_porque': {
        'Lo tiene afiliado(a) una persona de este u otro hogar con la que no tiene v�nculo laboral': 'Lo tiene afiliado(a) una persona de este u otro hogar con la que no tiene vínculo laboral',
    },
    't_dejo_trabajar': {
        'Menos de 1 a�o': 'Menos de 1 año',
        'Hace m�s de 5 a�os': 'Hace más de 5 años',
        'Entre 1 y menos de 2 a�os': 'Entre 1 y menos de 2 años',
        'Entre 2 y menos de 5 a�os': 'Entre 2 y menos de 5 años',
    },
    'tam_empresa': {
        '50 personas y m�s': '50 personas y más',
    },
    'tamano_pt': {
        '50 personas y m�s': '50 personas y más',
    },
    'tamano_tenia': {
        '50 personas y m�s': '50 personas y más',
    },
    'tipo_contrato3': {
        'Contrato escrito a t�rmino fijo': 'Contrato escrito a término fijo',
    },
    'tratar_acci': {
        'Acudi� a un hospital, cl�nica, centro de salud u otra instituci�n de salud': 'Acudió a un hospital, clínica, centro de salud u otra institución de salud',
        'Consult� a un tegua, curandero, yerbatero, comadrona': 'Consultó a un tegua, curandero, yerbatero, comadrona',
        'Acudi� al boticario, farmaceuta, droguista': 'Acudió al boticario, farmaceuta, droguista',
        'Acudi� a un m�dico general, especialista particular u odont�logo': 'Acudió a un médico general, especialista particular u odontólogo',
        'Us� remedios caseros': 'Usó remedios caseros',
        'Acudi� a un profesional de medicina alternativa': 'Acudió a un profesional de medicina alternativa',
    },
    'tratar_ciru': {
        'Acudi� a un hospital, cl�nica, centro de salud u otra instituci�n de salud': 'Acudió a un hospital, clínica, centro de salud u otra institución de salud',
        'Acudi� a un m�dico general, especialista particular u odont�logo': 'Acudió a un médico general, especialista particular u odontólogo',
        'Us� remedios caseros': 'Usó remedios caseros',
    },
    'tratar_enfe': {
        'Acudi� a un hospital, cl�nica, centro de salud u otra instituci�n de salud': 'Acudió a un hospital, clínica, centro de salud u otra institución de salud',
        'Acudi� a un m�dico general, especialista particular u odont�logo': 'Acudió a un médico general, especialista particular u odontólogo',
        'Us� remedios caseros': 'Usó remedios caseros',
        'Acudi� al boticario, farmaceuta, droguista': 'Acudió al boticario, farmaceuta, droguista',
        'Acudi� a un profesional de medicina alternativa': 'Acudió a un profesional de medicina alternativa',
        'Consult� a un tegua, curandero, yerbatero, comadrona': 'Consultó a un tegua, curandero, yerbatero, comadrona',
    },
    'tratar_odon': {
        'Acudi� a un m�dico general, especialista particular u odont�logo': 'Acudió a un médico general, especialista particular u odontólogo',
        'Us� remedios caseros': 'Usó remedios caseros',
        'Acudi� a un hospital, cl�nica, centro de salud u otra instituci�n de salud': 'Acudió a un hospital, clínica, centro de salud u otra institución de salud',
        'Acudi� al boticario, farmaceuta, droguista': 'Acudió al boticario, farmaceuta, droguista',
        'Acudi� a un profesional de medicina alternativa': 'Acudió a un profesional de medicina alternativa',
        'Consult� a un tegua, curandero, yerbatero, comadrona': 'Consultó a un tegua, curandero, yerbatero, comadrona',
    },
    'tratar_problema': {
        'Acudi� a un m�dico general, especialista particular u odont�logo': 'Acudió a un médico general, especialista particular u odontólogo',
        'Us� remedios caseros': 'Usó remedios caseros',
        'Consult� a un tegua, curandero, yerbatero, comadrona': 'Consultó a un tegua, curandero, yerbatero, comadrona',
    },
    'vivia_con': {
        'S�lo con la madre': 'Sólo con la madre',
        'S�lo con el padre': 'Sólo con el padre',
    },
    'vr_ganancia4': {
        'No recibi�': 'No recibió',
    },
    'vr_ganancia5': {
        'No recibi�': 'No recibió',
    },
    'vr_ganancia6': {
        'No recibi�': 'No recibió',
    },
}


# ─── Funciones de limpieza ────────────────────────────────────────────────────

def identificar_columnas_afectadas(df: pd.DataFrame) -> tuple[list, list]:
    """
    Separa las columnas de texto con al menos un valor corrupto en
    (vocabulario_cerrado, texto_libre), segun CARDINALIDAD_MAXIMA_CERRADA.
    La cardinalidad se mide sobre los valores tal cual vienen (antes de
    corregir), que es representativa del numero de categorias reales de la
    pregunta salvo en casos de texto libre.
    """
    afectadas = []
    for c in df.select_dtypes(include="object").columns:
        if df[c].astype(str).str.contains(MARCADOR, regex=False, na=False).any():
            afectadas.append(c)

    cardinalidad = df[afectadas].nunique()
    cerradas = cardinalidad[cardinalidad <= CARDINALIDAD_MAXIMA_CERRADA].index.tolist()
    abiertas = cardinalidad[cardinalidad > CARDINALIDAD_MAXIMA_CERRADA].index.tolist()
    return cerradas, abiertas


def _patron_desde_corrupto(valor_corrupto: str) -> re.Pattern:
    """Construye un regex donde cada '�' es un comodin de exactamente 1 caracter."""
    partes = valor_corrupto.split(MARCADOR)
    return re.compile("^" + ".".join(re.escape(p) for p in partes) + "$")


def corregir_vocabulario_cerrado_automatico(
    df: pd.DataFrame, columnas: list
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Para cada columna de vocabulario cerrado, corrige los valores corruptos
    que tienen exactamente un candidato limpio compatible en esa MISMA
    columna (cualquier ola). Si el unico candidato compatible cae en
    {"Si", "Sí"}, se usa "Sí" (grafia estandar de la ELCA). Los valores sin
    match unico NO se tocan.

    Retorna (df_corregido, mapa_aplicado, residual_sin_resolver), donde
    mapa_aplicado y residual_sin_resolver son {columna: {corrupto: limpio}}
    y {columna: [corruptos]} respectivamente, para el reporte.
    """
    df = df.copy()
    mapa_aplicado: dict = {}
    residual: dict = {}

    for c in columnas:
        serie = df[c].astype(str)
        mask_corrupto = serie.str.contains(MARCADOR, regex=False, na=False)
        if not mask_corrupto.any():
            continue
        limpios = set(serie[~mask_corrupto & (serie != "nan")].unique())
        corruptos = serie[mask_corrupto].unique()

        mapa_col = {}
        for corr in corruptos:
            candidatos = [l for l in limpios if _patron_desde_corrupto(corr).match(l)]
            if len(candidatos) == 1:
                mapa_col[corr] = candidatos[0]
            elif set(candidatos) <= {"Si", "Sí"} and candidatos:
                mapa_col[corr] = "Sí"
            else:
                residual.setdefault(c, []).append(corr)

        if mapa_col:
            df[c] = df[c].replace(mapa_col)
            mapa_aplicado[c] = mapa_col

    return df, mapa_aplicado, residual


def aplicar_correcciones_diccionario_pdf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica CORRECCIONES_DICCIONARIO_PDF. A diferencia de
    aplicar_correcciones_manuales_prioritarias, NO se valida con assert que
    la columna quede en 0 -- estas correcciones cubren solo los valores que
    tuvieron una coincidencia unica en el diccionario, algunas columnas
    pueden conservar otros valores corruptos sin resolver.
    """
    df = df.copy()
    for col, mapa in CORRECCIONES_DICCIONARIO_PDF.items():
        df[col] = df[col].replace(mapa)
    return df


def aplicar_correcciones_manuales_prioritarias(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica CORRECCIONES_MANUALES_PRIORITARIAS y valida que no quede '�'."""
    df = df.copy()
    for col, mapa in CORRECCIONES_MANUALES_PRIORITARIAS.items():
        df[col] = df[col].replace(mapa)
        residual = df[col].astype(str).str.contains(MARCADOR, regex=False, na=False)
        assert not residual.any(), (
            f"'{col}' sigue con {residual.sum()} valores corruptos tras aplicar "
            "CORRECCIONES_MANUALES_PRIORITARIAS -- faltan casos por agregar al mapa."
        )
    return df


def documentar_residual(df_final: pd.DataFrame, columnas_texto_libre: list) -> None:
    """
    Guarda un CSV con todas las columnas que quedan con corrupcion "�" sin
    resolver EN EL RESULTADO FINAL (tras las 4 capas de correccion:
    automatica, diccionario PDF, familia, manual), para no tener que
    redescubrir el problema si se necesitan despues. Se calcula sobre el
    dataframe ya corregido, no sobre listas intermedias, para que el reporte
    sea siempre exacto sin importar cuantas capas de correccion se agreguen.
    """
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    for col in df_final.select_dtypes(include="object").columns:
        serie = df_final[col].astype(str)
        mask = serie.str.contains(MARCADOR, regex=False, na=False)
        n = mask.sum()
        if n == 0:
            continue
        tipo = "texto_libre_no_tocado" if col in columnas_texto_libre else "vocabulario_cerrado_sin_match"
        filas.append({
            "columna": col,
            "tipo": tipo,
            "n_valores_residuales": n,
            "ejemplo": serie[mask].iloc[0] if tipo == "vocabulario_cerrado_sin_match" else "",
        })
    pd.DataFrame(filas).sort_values("n_valores_residuales", ascending=False).to_csv(
        DOC_PATH, index=False
    )


def main() -> None:
    print(f"Leyendo: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Dimensiones de entrada: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")

    cerradas, abiertas = identificar_columnas_afectadas(df)
    print(f"\nColumnas afectadas por '�': {len(cerradas) + len(abiertas)}")
    print(f"  Vocabulario cerrado (<= {CARDINALIDAD_MAXIMA_CERRADA} categorias): {len(cerradas)}")
    print(f"  Texto libre (sin tocar): {len(abiertas)}")

    df, mapa_aplicado, residual = corregir_vocabulario_cerrado_automatico(df, cerradas)
    n_aplicado = sum(len(v) for v in mapa_aplicado.values())
    n_residual = sum(len(v) for v in residual.values())
    print(f"\nCorreccion automatica (match unico + regla Si/Sí):")
    print(f"  Valores corregidos: {n_aplicado} en {len(mapa_aplicado)} columnas")
    print(f"  Valores sin resolver automaticamente: {n_residual} en {len(residual)} columnas")

    df = aplicar_correcciones_diccionario_pdf(df)
    n_dicc = sum(len(v) for v in CORRECCIONES_DICCIONARIO_PDF.values())
    print(f"\nCorreccion via diccionarios PDF (fuente independiente del .tab corrupto):")
    print(f"  Valores corregidos: {n_dicc} en {len(CORRECCIONES_DICCIONARIO_PDF)} columnas")

    df = aplicar_correcciones_manuales_prioritarias(df)
    n_manual = sum(len(v) for v in CORRECCIONES_MANUALES_PRIORITARIAS.values())
    print(f"\nCorreccion manual de variables prioritarias: {n_manual} valores en "
          f"{len(CORRECCIONES_MANUALES_PRIORITARIAS)} columnas ({', '.join(CORRECCIONES_MANUALES_PRIORITARIAS)})")
    print(f"  Validado: 0 valores '�' restantes en esas {len(CORRECCIONES_MANUALES_PRIORITARIAS)} columnas.")

    documentar_residual(df, abiertas)
    print(f"\nBacklog de columnas no resueltas (calculado sobre el resultado final) en: {DOC_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nBase limpia guardada en: {OUTPUT_PATH}")
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")


if __name__ == "__main__":
    main()
