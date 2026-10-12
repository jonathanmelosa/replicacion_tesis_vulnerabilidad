"""
Construccion de covariables de contexto comunitario a partir del modulo de
Comunidades (ELCA 2010, 2013, 2016), para el modelo benchmark de prediccion
de transicion a la pobreza. Primer bloque del modulo de Comunidades -- distinto de
Personas: la unidad de analisis original YA es la comunidad (una entrevista
a lideres comunitarios por comunidad-ola), no requiere agregacion desde
individuos. Cada hogar hereda las variables de SU comunidad via
`consecutivo_c` (identificador de comunidad = `consecutivo` del hogar sin el
ultimo digito, verificado directamente contra `UHogar-csv.tab`: hogares
111001/111002/111003 -> consecutivo_c 11001, hogar 111004 -> 11002, etc.).

Auditoria completa (mismo nivel de rigor que Personas)
------------------------------------------------------------------------
Se aplico el mismo proceso de 4 pasos usado en Personas:
1. **Limpieza de corrupcion U+FFFD**: `03_limpieza_base_comunidades.py`
   (26 columnas afectadas de 558, mucho mas acotado que Personas -- 22
   cerradas resueltas automaticamente + 1 correccion manual para `region`,
   4 de texto libre sin tocar). Ver ese script para el detalle.
2. **Clasificacion completa de las 558 columnas**
   (`docs/variable_audit/comunidades_construccion.csv`, mismo criterio
   >1% de cobertura por ola que Personas): 212 `CANDIDATO_BENCHMARK`
   (presente ola1 Y ola2), 207 `EXCLUIDA_NO_EN_OLA1`, 40
   `EXCLUIDA_NO_EN_OLA2`, 52 `EXCLUIDA_SOLO_OLA3`, 39
   `EXCLUIDA_CASI_VACIA`, 5 `IDENTIFICADOR`, 3 `EXCLUIDA_OTRO_PATRON`
   (suma verificada: 558).
3. **Busqueda de renombrados entre olas** (nombre relajado + diccionarios
   PDF de `elca_{2010,2013}/{U,R}Comunidades.pdf`, mismo metodo que
   Personas): sin hallazgos nuevos. Los unicos pares con similitud alta
   son (a) bloques de items numerados con distinto N por ola (ej.
   `sexo_lider5/6` sin equivalente porque ola 2 solo pregunta hasta el
   lider 4; `cod_conf_1..4` vs `cod_conf_7..12`, mismo patron para
   conflictos) -- NO son renombrados, son preguntas repetidas con
   distinto limite de repeticion por ola; y (b) `grarmados_2001..2010`
   vs `grarmados_2011..2013` -- diseño de VENTANA MOVIL de años (cada ola
   pregunta por los grupos armados presentes en años especificos, sin
   solapamiento entre olas), tampoco es un renombrado sino un cambio de
   referencia temporal por diseño.
4. **Aplicacion del umbral >=10% por tema** sobre las 212 candidatas para
   decidir que construir, con exclusiones documentadas por tema (no solo
   "no me alcanzo el tiempo") -- ver mas abajo.

**Temas construidos** (ver "Variables construidas"): seguridad/violencia,
contaminacion/riesgo ambiental, desplazamiento, organizaciones sociales,
capital social/resolucion de conflictos, inversion reciente en
infraestructura, servicios de primera infancia, espacios publicos,
intensidad de conflicto armado, legalidad del barrio, acceso a agua.

**Temas EXCLUIDOS deliberadamente, con razon documentada**:
  - Demografia de lideres comunitarios (`cargo_lider*`, `sexo_lider*`,
    `anos_vive_lider*`, `n_lideres`): metadato del informante (quien
    respondio la encuesta), no una caracteristica de la comunidad --
    mismo criterio que exlcuyo `informante` en Personas.
  - `hecho_seguridad`/`razon_seguridad`/`*_cual` (texto libre o
    sub-pregunta filtrada de baja cobertura, 25%-32%): ya se captura la
    percepcion de inseguridad con `seguridad`, estas son elaboraciones de
    menor valor marginal.
  - Cluster de acceso a mercado agricola rural (~25 columnas: `acceso`,
    `facil_venda_terreno`, `medio_transporte`, `quienes_compran`,
    `inf_lacteos/mataderos/plazas_mercado`, `pp_*`, `vr_jornal_*`, etc.,
    cobertura 26%-31% que coincide con la fraccion RURAL de la muestra):
    submodulo exclusivo de comunidades rurales sobre comercializacion
    agricola. Fuera de alcance de este bloque de contexto general; podria
    justificar un bloque separado "Comunidades rural extendido" si se
    decide relevante para el benchmark.
  - Cluster de tipos de trabajo rural (`trab_pesca/agrgan/cultili/
    mineria/expmadera/artesania/comercio/industria/construccion/
    servicios/gr_armados`, misma cobertura ~23%-28% rural-only) y
    calendario climatico mensual (`btol_ene`...`btol_dic`): mismo
    submodulo rural, misma razon de exclusion.
  - Cluster de proveedores de salud alternativos (`medico_nopuesto`,
    `odontologo_nopuesto`, `enfermera_nopuesto`, `promotor_salud`,
    `comadrona`, `curandero`, `primeros_auxilios`, ~26%-30% cobertura
    rural-only): mismo submodulo rural.
  - Cluster de campañas de salud/agricolas (`camp_vacunacion`,
    `camp_brigadas_salud`, `camp_reciclaje`, etc.): salta de 26% (ola 1)
    a 100% (ola 2/3) en varias -- mismo patron de posible problema de
    comparabilidad poblacional ya visto en discapacidad (Personas, Bloque
    3), sin investigar a fondo por alcance; se deja fuera hasta poder
    confirmar si es un cambio real de cobertura del programa o un
    artefacto del diseño muestral entre olas.
  - Extension del problema ambiental (`contaminacion_agua`,
    `contaminacion_aire`, `destruccion_medioamb`, `explotacion_
    recnatural`, `hambrunas`, `epidemias`, `distr_alucinogenos`,
    `uso_fungicidas`, `problemas_otro`): cobertura mas baja (26%-30%,
    tambien rural-only) y tematicamente redundante con
    `problema_contaminacion_comunidad` (65%-77% cobertura) ya construido
    con una bateria de mayor cobertura -- combinarlas diluiria la señal.
  - `grarmados_2010` (1 columna suelta del patron de ventana movil
    documentado arriba): sin equivalente comparable en otras olas.
  - `barrio_legal_ano`, `veces_serv_publico`, `frecuencia_problema1..5`,
    `t_hogares`, `vivir_otro_lugar`: sub-preguntas de seguimiento de baja
    cobertura (10%-16%) sobre temas ya cubiertos por variables de mayor
    cobertura en este bloque.

Cobertura del join hogar->comunidad: dos causas distintas de no-match
-------------------------------------------------------------------------
El `consecutivo_c` del HOGAR (no solo el de Comunidades) tambien trae, en
ola 2 y 3, un valor sentinela "8888888" (4.468 filas, un codigo estandar de
"sin dato" en encuestas) y algunos codigos malformados de 8-10 digitos que
coinciden EXACTAMENTE con las 25 comunidades urbanas de 2016 con
`consecutivo_c` de 8-10 digitos ya detectadas en el archivo crudo de
Comunidades (`UComunidades-csv.tab`, ej. 8110011099) -- confirma que es un
problema real del dato fuente de ELCA en ambos lados (hogar y comunidad),
no un bug de esta consolidacion.

Filtrando esos IDs invalidos (>=1.000.000), queda un segundo problema
independiente: en ola 1, 80 de los ~792 codigos de comunidad que
referencian los hogares (8.1% de las filas, 803 hogares) simplemente NO
aparecen en `comunidades_elca_longitudinal.parquet` -- comunidades cuyo
hogar fue encuestado pero cuyo lider comunitario no (o cuya encuesta de
comunidad no se incluyo en el archivo consolidado). En ola 2/3 este
segundo problema es menor (2.9%/3.6% de no-match tras filtrar IDs
invalidos). Verificado que es un problema real de cobertura del
cuestionario de Comunidades, no un error de calculo del ID (`consecutivo_c`
= `consecutivo` del hogar sin el ultimo digito, confirmado exacto contra
`UHogar-csv.tab`). Se deja como NaN, sin imputar.

`homicidios`: escalas distintas entre zona URBANA (Sí/No) y RURAL (Nunca/
Algunas veces/Frecuentemente) -- verificado por value_counts cruzado con
zona antes de construir (mismo principio que las verificaciones de
comparabilidad poblacional en Personas). Se arma un indicador binario
armonizado: Urbano Sí -> 1, Rural "Algunas veces"/"Frecuentemente" -> 1,
resto -> 0.

`seguridad`: escala ordinal de 4 niveles con variantes de genero (Muy
seguro/segura, Relativamente seguro/segura, Inseguro/Insegura, Muy
inseguro/insegura) -- se normaliza el genero y se mapea a escala 1 (muy
seguro) a 4 (muy inseguro), orientada hacia percepcion de INSEGURIDAD para
alinear el signo con vulnerabilidad.

`barrio_legal` tenia una categoria con corrupcion U+FFFD ("...ya se
legaliz�", 103 casos) -- resuelta por la correccion automatica de
`03_limpieza_base_comunidades.py` (tenia candidato limpio en la misma
columna), no requirio intervencion manual.

Variables construidas (nivel hogar, heredadas de la comunidad)
----------------------------------------------------------------
  percepcion_inseguridad_comunidad: escala 1-4, 4 = percepcion de mayor
                                inseguridad.
  problema_homicidios_comunidad: 1 si la comunidad reporta homicidios
                                (armonizado urbano/rural), 0 si no.
  n_problemas_convivencia_comunidad: conteo (0-7) de problemas de convivencia
                                presentes: atracos_robos, pandillas,
                                drogas_alucinogenas, alcohol_publicos,
                                prostibulos, duermen_calles,
                                invasion_publico.
  problema_contaminacion_comunidad: 1 si la comunidad reporta CUALQUIERA de
                                contaminacion_bas, contaminacion_medioamb,
                                aguas_negras.
  riesgo_inundacion_comunidad: 1 si la comunidad reporta inundaciones.
  acceso_agua_comunidad: 1 si los hogares de la comunidad pueden
                                consumir el agua sin problema
                                (puede_consumir_agua).
  hay_desplazados_comunidad: 1 si la comunidad reporta hogares
                                desplazados.
  n_desplazados_comunidad: numero de personas desplazadas
                                reportadas (NaN si `hay_desplazados`=No, ya
                                que la pregunta no aplica).
  n_organizaciones_comunidad: conteo (0-14) de tipos de organizacion
                                social presentes en la comunidad (jac,
                                org_caridad/comunitaria/religiosas/etnica/
                                cultural/educativa/medio_amb/otra,
                                aso_vig_seguridad, sindicato, mvto_politico,
                                junta_edificio, part_promov_estado).
  tiene_puesto_salud_comunidad: 1 si la comunidad tiene puesto de salud.
  tiene_escuela_primaria_comunidad: 1 si la comunidad tiene escuela primaria.
  tiene_colegio_secundaria_comunidad: 1 si la comunidad tiene colegio de
                                secundaria.
  tiene_transporte_publico_comunidad: 1 si la comunidad tiene transporte
                                publico.
  barrio_legal_comunidad: 1 si el barrio/vereda es legal (desde
                                el comienzo o ya legalizado), 0 si permanece
                                ilegal.
  solidaridad_comunidad: escala 0 (no se ayudan) a 2 (se ayudan
                                mucho) -- capital social/cohesion vecinal.
  acude_justicia_formal_comunidad: 1 si la comunidad reporta que ante un
                                conflicto se acude PRINCIPALMENTE a la
                                justicia formal (vs. lideres comunales,
                                religiosos, grupos armados u otro).
  cortes_agua_comunidad: 1 si la comunidad reporta cortes de
                                agua frecuentes.
  n_obras_infraestructura_reciente_comunidad: conteo (0-6) de tipos de obra
                                de infraestructura realizada en los ultimos
                                2 años (educacion, salud, acueducto/
                                alcantarillado, entretenimiento, vias,
                                otro) -- señal de inversion publica
                                reciente, distinta de `tiene_*_comunidad`
                                (presencia actual).
  n_servicios_primera_infancia_comunidad: conteo (0-4) de servicios de
                                primera infancia presentes (hogar
                                comunitario ICBF, guarderia ICBF,
                                preescolar, restaurante escolar).
  n_espacios_publicos_comunidad: conteo (0-4) de espacios/servicios
                                publicos presentes (canchas deportivas,
                                parques publicos, salon comunal, puesto de
                                policia).
  n_acciones_conflicto_armado_comunidad: conteo (0-11) de acciones de
                                coercion/conflicto armado reportadas
                                (grupos armados imponen reglas, exigen
                                dinero, sometieron a la poblacion,
                                atentados, amenazas, reclutamiento de
                                jovenes, desalojos, secuestros, amenazas
                                de grupos armados, otras acciones) --
                                escalas armonizadas a binario donde
                                mezclaban Sí/No con Nunca/Algunas veces/
                                Frecuentemente/Todo el tiempo (mismo
                                patron de `homicidios`, verificado antes
                                de construir).

Caveats observados en la validacion (no bloquean, quedan como observacion)
----------------------------------------------------------------------------
`n_organizaciones_comunidad` cae de 4.25 (ola 1) a 3.74 (ola 2) a 2.10 (ola
3) entre las comunidades que SI respondieron (la cobertura de las 14
preguntas de organizacion es estable entre olas, 65%-100%, asi que la caida
no es un artefacto de denominador) -- podria reflejar una caida real de
capital social organizativo o un cambio de redaccion no detectado; no se
investigo mas a fondo. `n_desplazados_comunidad` cae de una media de 195
personas (ola 1) a ~70-73 (olas 2/3) -- diferencia de escala grande que
podria reflejar un cambio en el periodo de referencia de la pregunta
("alguna vez" vs. "en el ultimo año"); no verificado contra el diccionario,
queda como caveat para revisar si esta variable se usa en un analisis mas
fino.

Output: data/processed/comunidades_hogar_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMUNIDADES_PATH = PROJECT_ROOT / "data" / "processed" / "comunidades_elca_longitudinal_clean.parquet"
HOGAR_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "comunidades_hogar_elca_longitudinal.parquet"

COLUMNAS_PROBLEMAS_CONVIVENCIA = [
    "atracos_robos", "pandillas", "drogas_alucinogenas", "alcohol_publicos",
    "prostibulos", "duermen_calles", "invasion_publico",
]
COLUMNAS_CONTAMINACION = ["contaminacion_bas", "contaminacion_medioamb", "aguas_negras"]
COLUMNAS_ORGANIZACION = [
    "jac", "org_caridad", "org_comunitaria", "org_religiosas", "org_etnica",
    "org_cultural", "org_educativa", "org_medio_amb", "org_otra",
    "aso_vig_seguridad", "sindicato", "mvto_politico", "junta_edificio",
    "part_promov_estado",
]
COLUMNAS_INFRAESTRUCTURA_RECIENTE = [
    "inf_educacion", "inf_salud", "inf_acue_alcan", "inf_entretenimiento",
    "inf_vias", "inf_otro",
]
COLUMNAS_PRIMERA_INFANCIA = ["hog_icbf", "guarderia_icbf", "preescolar", "rest_escolares"]
COLUMNAS_ESPACIOS_PUBLICOS = ["canchas_deportivas", "parques_publicos", "salon_comunal", "puesto_policia"]
# Sí/No directo en la fuente
COLUMNAS_CONFLICTO_SI_NO = [
    "imponen_reglas", "exigen_dinero", "sometieron_gra", "atentados",
    "amenazas", "recluta_jovenes_gra", "otras_acciones", "otra_acc_conflic",
]
# escalas mixtas Sí/No + Nunca/Algunas veces/Frecuentemente/Todo el tiempo
# dentro de la MISMA columna (mismo patron que `homicidios`) -- se arman
# aparte y se agregan al set de conflicto ya armonizadas a binario.
COLUMNAS_CONFLICTO_ESCALA_MIXTA = ["desalojos", "secuestros", "amenazas_gra"]
CONFLICTO_ESCALA_MIXTA_SI = {"Si", "Sí", "Algunas veces", "Frecuentemente", "Todo el tiempo"}
CONFLICTO_ESCALA_MIXTA_NO = {"No", "Nunca"}

SEGURIDAD_MAPA = {
    "Muy seguro": 1, "Muy segura": 1,
    "Relativamente seguro": 2, "Relativamente segura": 2,
    "Inseguro": 3, "Insegura": 3,
    "Muy inseguro": 4, "Muy insegura": 4,
}

BARRIO_LEGAL_SI = {"Legal desde el comienzo", "Tuvo origen ilegal, pero ya se legalizó"}
BARRIO_LEGAL_NO = {"Permanece ilegal"}

SOLIDARIDAD_MAPA = {"No se ayudan": 0, "Se ayudan poco": 1, "Se ayudan mucho": 2}


def normalizar_si_no(serie: pd.Series) -> pd.Series:
    s = serie.astype(str).str.strip().replace({"Si": "Sí"})
    return s.where(s.isin(["Sí", "No"]))


def cargar_comunidades() -> pd.DataFrame:
    c = pd.read_parquet(COMUNIDADES_PATH)
    c["consecutivo_c"] = pd.to_numeric(c["consecutivo_c"], errors="coerce")
    c["ola"] = pd.to_numeric(c["ola"], errors="coerce").astype("Int64")

    for col in COLUMNAS_PROBLEMAS_CONVIVENCIA + COLUMNAS_CONTAMINACION + COLUMNAS_ORGANIZACION:
        c[col] = normalizar_si_no(c[col])
    c["inundaciones"] = normalizar_si_no(c["inundaciones"])
    c["puede_consumir_agua"] = normalizar_si_no(c["puede_consumir_agua"])
    c["hay_desplazados"] = normalizar_si_no(c["hay_desplazados"])
    c["puesto_salud"] = normalizar_si_no(c["puesto_salud"])
    c["esc_primaria"] = normalizar_si_no(c["esc_primaria"])
    c["col_secundaria"] = normalizar_si_no(c["col_secundaria"])
    c["transp_publico"] = normalizar_si_no(c["transp_publico"])

    c["n_desplazados"] = pd.to_numeric(c["n_desplazados"], errors="coerce")
    c.loc[c["hay_desplazados"] != "Sí", "n_desplazados"] = np.nan

    seguridad_norm = c["seguridad"].astype(str).str.strip()
    c["percepcion_inseguridad_comunidad"] = seguridad_norm.map(SEGURIDAD_MAPA)

    homicidios_bin = pd.Series(np.nan, index=c.index, dtype=object)
    es_urbano = c["zona"].isin(["Urbano", "Urbana"])
    hom = c["homicidios"].astype(str).str.strip()
    homicidios_bin[es_urbano & hom.isin(["Si", "Sí"])] = "Sí"
    homicidios_bin[es_urbano & (hom == "No")] = "No"
    homicidios_bin[~es_urbano & hom.isin(["Algunas veces", "Frecuentemente"])] = "Sí"
    homicidios_bin[~es_urbano & (hom == "Nunca")] = "No"
    c["problema_homicidios_comunidad"] = homicidios_bin

    c["barrio_legal_comunidad"] = np.where(
        c["barrio_legal"].isin(BARRIO_LEGAL_SI), 1.0,
        np.where(c["barrio_legal"].isin(BARRIO_LEGAL_NO), 0.0, np.nan),
    )

    es_si = c[COLUMNAS_PROBLEMAS_CONVIVENCIA].eq("Sí")
    tiene_dato = c[COLUMNAS_PROBLEMAS_CONVIVENCIA].notna()
    c["n_problemas_convivencia_comunidad"] = np.where(
        tiene_dato.any(axis=1), es_si.sum(axis=1), np.nan
    )

    es_si_contam = c[COLUMNAS_CONTAMINACION].eq("Sí")
    tiene_dato_contam = c[COLUMNAS_CONTAMINACION].notna()
    c["problema_contaminacion_comunidad"] = np.where(
        tiene_dato_contam.any(axis=1), es_si_contam.any(axis=1).astype(float), np.nan
    )

    es_si_org = c[COLUMNAS_ORGANIZACION].eq("Sí")
    tiene_dato_org = c[COLUMNAS_ORGANIZACION].notna()
    c["n_organizaciones_comunidad"] = np.where(
        tiene_dato_org.any(axis=1), es_si_org.sum(axis=1), np.nan
    )

    c["solidaridad_comunidad"] = c["solidaridad"].astype(str).str.strip().map(SOLIDARIDAD_MAPA)

    c["acude_solucion"] = c["acude_solucion"].replace({"Otro.  ???Cu???l?": "Otro"})
    c["acude_justicia_formal_comunidad"] = np.where(
        c["acude_solucion"].notna(), (c["acude_solucion"] == "La justicia").astype(float), np.nan
    )

    c["cortes_agua"] = normalizar_si_no(c["cortes_agua"])
    c["cortes_agua_comunidad"] = (c["cortes_agua"] == "Sí").astype(float)
    c.loc[c["cortes_agua"].isna(), "cortes_agua_comunidad"] = np.nan

    for col in COLUMNAS_INFRAESTRUCTURA_RECIENTE:
        c[col] = normalizar_si_no(c[col])
    es_si_infra = c[COLUMNAS_INFRAESTRUCTURA_RECIENTE].eq("Sí")
    tiene_dato_infra = c[COLUMNAS_INFRAESTRUCTURA_RECIENTE].notna()
    c["n_obras_infraestructura_reciente_comunidad"] = np.where(
        tiene_dato_infra.any(axis=1), es_si_infra.sum(axis=1), np.nan
    )

    for col in COLUMNAS_PRIMERA_INFANCIA:
        c[col] = normalizar_si_no(c[col])
    es_si_infancia = c[COLUMNAS_PRIMERA_INFANCIA].eq("Sí")
    tiene_dato_infancia = c[COLUMNAS_PRIMERA_INFANCIA].notna()
    c["n_servicios_primera_infancia_comunidad"] = np.where(
        tiene_dato_infancia.any(axis=1), es_si_infancia.sum(axis=1), np.nan
    )

    for col in COLUMNAS_ESPACIOS_PUBLICOS:
        c[col] = normalizar_si_no(c[col])
    es_si_espacios = c[COLUMNAS_ESPACIOS_PUBLICOS].eq("Sí")
    tiene_dato_espacios = c[COLUMNAS_ESPACIOS_PUBLICOS].notna()
    c["n_espacios_publicos_comunidad"] = np.where(
        tiene_dato_espacios.any(axis=1), es_si_espacios.sum(axis=1), np.nan
    )

    for col in COLUMNAS_CONFLICTO_SI_NO:
        c[col] = normalizar_si_no(c[col])
    for col in COLUMNAS_CONFLICTO_ESCALA_MIXTA:
        s = c[col].astype(str).str.strip()
        armonizada = pd.Series(np.nan, index=c.index, dtype=object)
        armonizada[s.isin(CONFLICTO_ESCALA_MIXTA_SI)] = "Sí"
        armonizada[s.isin(CONFLICTO_ESCALA_MIXTA_NO)] = "No"
        c[col] = armonizada

    columnas_conflicto = COLUMNAS_CONFLICTO_SI_NO + COLUMNAS_CONFLICTO_ESCALA_MIXTA
    es_si_conflicto = c[columnas_conflicto].eq("Sí")
    tiene_dato_conflicto = c[columnas_conflicto].notna()
    c["n_acciones_conflicto_armado_comunidad"] = np.where(
        tiene_dato_conflicto.any(axis=1), es_si_conflicto.sum(axis=1), np.nan
    )

    return c


def construir_variables_comunidad(c: pd.DataFrame) -> pd.DataFrame:
    resultado = c[[
        "consecutivo_c", "ola",
        "percepcion_inseguridad_comunidad",
        "n_problemas_convivencia_comunidad", "problema_contaminacion_comunidad",
        "n_organizaciones_comunidad", "barrio_legal_comunidad",
    ]].copy()
    resultado["problema_homicidios_comunidad"] = (c["problema_homicidios_comunidad"] == "Sí").astype(float)
    resultado.loc[c["problema_homicidios_comunidad"].isna(), "problema_homicidios_comunidad"] = np.nan
    resultado["riesgo_inundacion_comunidad"] = (c["inundaciones"] == "Sí").astype(float)
    resultado.loc[c["inundaciones"].isna(), "riesgo_inundacion_comunidad"] = np.nan
    resultado["acceso_agua_comunidad"] = (c["puede_consumir_agua"] == "Sí").astype(float)
    resultado.loc[c["puede_consumir_agua"].isna(), "acceso_agua_comunidad"] = np.nan
    resultado["hay_desplazados_comunidad"] = (c["hay_desplazados"] == "Sí").astype(float)
    resultado.loc[c["hay_desplazados"].isna(), "hay_desplazados_comunidad"] = np.nan
    resultado["n_desplazados_comunidad"] = c["n_desplazados"]
    resultado["tiene_puesto_salud_comunidad"] = (c["puesto_salud"] == "Sí").astype(float)
    resultado.loc[c["puesto_salud"].isna(), "tiene_puesto_salud_comunidad"] = np.nan
    resultado["tiene_escuela_primaria_comunidad"] = (c["esc_primaria"] == "Sí").astype(float)
    resultado.loc[c["esc_primaria"].isna(), "tiene_escuela_primaria_comunidad"] = np.nan
    resultado["tiene_colegio_secundaria_comunidad"] = (c["col_secundaria"] == "Sí").astype(float)
    resultado.loc[c["col_secundaria"].isna(), "tiene_colegio_secundaria_comunidad"] = np.nan
    resultado["tiene_transporte_publico_comunidad"] = (c["transp_publico"] == "Sí").astype(float)
    resultado.loc[c["transp_publico"].isna(), "tiene_transporte_publico_comunidad"] = np.nan

    for col in [
        "solidaridad_comunidad", "acude_justicia_formal_comunidad", "cortes_agua_comunidad",
        "n_obras_infraestructura_reciente_comunidad", "n_servicios_primera_infancia_comunidad",
        "n_espacios_publicos_comunidad", "n_acciones_conflicto_armado_comunidad",
    ]:
        resultado[col] = c[col]

    if resultado.duplicated(subset=["consecutivo_c", "ola"]).any():
        raise ValueError("consecutivo_c+ola no es unico en Comunidades: revisar supuesto de join.")
    return resultado


def main() -> None:
    comunidades = cargar_comunidades()
    comunidad_vars = construir_variables_comunidad(comunidades)

    hogar = pd.read_parquet(HOGAR_PATH, columns=["consecutivo", "consecutivo_c", "ola", "zona", "llave", "llave_n16"])
    hogar["consecutivo_c"] = pd.to_numeric(hogar["consecutivo_c"], errors="coerce")
    hogar["ola"] = pd.to_numeric(hogar["ola"], errors="coerce").astype("Int64")

    salida = hogar.merge(comunidad_vars, on=["consecutivo_c", "ola"], how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print("Hogares sin comunidad emparejada por ola (total, incluye IDs sentinela/malformados):")
    print(salida.groupby("ola")["percepcion_inseguridad_comunidad"].apply(lambda s: s.isna().mean()))
    print()
    print("De esos, tasa de no-match SOLO en hogares con consecutivo_c valido (<1,000,000):")
    id_valido = salida["consecutivo_c"] < 1_000_000
    print(
        salida[id_valido]
        .groupby("ola")["percepcion_inseguridad_comunidad"]
        .apply(lambda s: s.isna().mean())
    )
    print()
    print(salida.groupby("ola")[
        ["percepcion_inseguridad_comunidad", "problema_homicidios_comunidad",
         "n_problemas_convivencia_comunidad", "problema_contaminacion_comunidad",
         "n_organizaciones_comunidad", "hay_desplazados_comunidad",
         "tiene_puesto_salud_comunidad", "barrio_legal_comunidad",
         "solidaridad_comunidad", "acude_justicia_formal_comunidad",
         "cortes_agua_comunidad", "n_obras_infraestructura_reciente_comunidad",
         "n_servicios_primera_infancia_comunidad", "n_espacios_publicos_comunidad",
         "n_acciones_conflicto_armado_comunidad"]
    ].mean())


if __name__ == "__main__":
    main()
