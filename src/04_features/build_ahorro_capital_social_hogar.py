"""
Construccion de covariables de ahorro y participacion en organizaciones
sociales (capital social) a partir del modulo de Personas (ELCA 2010, 2013,
2016), para el modelo benchmark de prediccion de transicion a la pobreza
. Cuarto
bloque del inventario de 139 candidatas (los anteriores: composicion del
hogar, educacion/ocupacion, salud/discapacidad).

Ambos grupos de variables (`ahorra`, `org_*`) se preguntan al mismo
subconjunto de personas -- cobertura ~39% en ambas olas, edades 13-94 (ola
1) / 15-97 (ola 2), consistente entre olas.

Ahorro: `ahorra` tiene mas categorias que Sí/No
---------------------------------------------------
`ahorra` no es un Sí/No simple: incluye "No, no recibe ingresos" y "No
recibe ingresos" (variantes de redaccion del mismo motivo entre olas) ademas
de "Sí"/"Si" (sin tilde) y "No informa". Para el indicador binario, ambas
variantes de "no recibe ingresos" se tratan como "No" (el hogar no esta
ahorrando, independientemente del motivo declarado).

Las 8 sub-variables de MOTIVO de ahorro (`ahorro_futuro`, `ahorro_educ`,
`ahorro_casa`, `ahorro_carro`, `ahorro_otros_act`, `ahorro_recre`,
`ahorro_montar`, `ahorro_otro`) tienen solo 5.7% de cobertura --
esperable, son sub-preguntas filtradas SOLO para quienes ya respondieron
"Sí" a `ahorra` (un subconjunto ya pequeño). Se EXCLUYEN de este bloque
por ser demasiado dispersas para aportar señal util a nivel de hogar (la
gran mayoria de hogares quedaria en NaN); documentado como decision de
alcance, no un descarte accidental.

Organizaciones sociales: 11 tipos de participacion
-------------------------------------------------------
`org_jac/caridad/comunitaria/religiosa/iestado/etnica/culdep/educ/mamb/otra`
(Junta de Accion Comunal, caridad, comunitaria, religiosa, instancia del
estado, etnica, cultural o deportiva, educativa, medio ambiente, otra) --
10 preguntas Sí/No independientes sobre pertenencia a cada tipo de
organizacion. `org_ninguna` (solo ola 1) se EXCLUYE por no existir en
ambas olas del benchmark (ver Eje 1 de comparabilidad, metodologia del
benchmark); `org_otra_cual` es texto libre (motivo "otra, ¿cual?"),
tampoco se usa.

`org_agremia` (afiliacion gremial/profesional, solo ola 2/3) se sigue
EXCLUYENDO -- no tiene equivalente verificado en ola 1.

**Correccion (2026-08-09): `sindicato` SI tiene equivalente en ola 1.**
Un chequeo de columnas "desaparecidas" entre olas encontro que
`sindicato` (ola 1, cobertura 39.2%, mismo rango de edad y misma
estructura Sí/No/No informa) es la MISMA pregunta que `org_sindicato`
(ola 2/3, cobertura 39.5%/39.1%) bajo un nombre distinto -- exactamente
el mismo patron de re-nombrado ya visto en `n_empleados` (Bloque 5). La
clasificacion automatica inicial las trato como "no presente en ambas
olas" y las excluyo por error. Se arma la columna `sindicato_armonizada`
(usa `sindicato` en ola 1, `org_sindicato` en ola 2/3) y se agrega como
undecimo tipo de organizacion a `COLUMNAS_ORGANIZACION`.

**Correccion (2026-08-09): `cotiza_fp` (ola 1) SI tiene equivalente
Sí/No en ola 2/3.** Al relajar el umbral de similitud de nombres para
buscar mas renombrados, aparecio `cotiza_fp` (ola 1) emparejado con
`cotizando` (ola 2/3) por coincidencia de cobertura (39.2% vs
39.5%/39.1%). El nombre es enganoso: `cotiza_fp` en ola 1 NO es una
sub-pregunta filtrada (a diferencia de `cotiza_cual`/`afilia_cual` en
ola 2/3, que SI son texto libre casi vacio y NO son la misma variable --
verificado y descartado). `cotiza_fp` es una pregunta COMBINADA que
cubre todo el universo de respuesta con 9 categorias, mezclando el
estado (cotiza o no) con el motivo si no cotiza (ej. "No cotiza porque
no tiene dinero", "Si está cotizando, pero todavía no es pensionado").
Se colapsa a binario: "Sí" si el texto empieza por "Si est" (cotizando,
con o sin pensión ya reconocida), "No" en cualquier otro caso (incluye
"ya está pensionado", que no es cotizacion activa). El binario resultante
da 15.7% de "Sí" en ola 1, contra 16.4% en `cotizando` de ola 2 --
magnitud consistente, confirma que es la misma pregunta. Mismo rango de
edad (12-15+) y misma poblacion filtrada que `ahorra`/`org_*`.

**Correccion (2026-08-09): comparacion directa contra los diccionarios
PDF encuentra 2 renombrados mas.** Se fue un paso mas alla
del cruce por nombre/cobertura y comparar el TEXTO DE LA PREGUNTA de las
columnas excluidas por presencia de ola contra los diccionarios oficiales
de la encuesta (`data/interim/raw/elca_{2010,2013}/{U,R}Personas.pdf`,
extraidos con `pdftotext -layout` y parseados a pares
variable->descripcion). Se calculo similitud de Jaccard sobre las
palabras de la descripcion (sin tildes, sin stopwords) entre las ~230
columnas de ola 1 y ~260 de ola 2/3 con descripcion recuperable. De ~35
pares con Jaccard>=0.3, la mayoria son preguntas relacionadas pero
distintas (ej. `medio_busco`/`medio_bus_trabajo`, ambas <10% cobertura,
sin efecto practico). Dos SI son la misma pregunta, confirmado por
coincidencia de cobertura y edad:
  - `estaba_sss` (ola 1, 14.9%) = `segsoc_salud` (ola 2/3, 14.4%/14.8%)
    -- afiliacion a seguridad social en SALUD ligada al trabajo (edad
    13-71 en ola 1, 17-88 en ola 2/3 -- filtro mas angosto que
    `afiliacion` del Bloque 3, que es la afiliacion general sin filtro de
    tipo de vinculo laboral; NO es redundante, es informacion
    complementaria sobre el tipo de vinculo).
  - `estaba_fp` (ola 1, 14.9%) = `afiliacion_fp` (ola 2/3, 14.4%/14.8%)
    -- afiliacion (no necesariamente cotizacion activa) a fondo de
    PENSIONES, distinta de `cotiza_pension` (cotizacion activa) ya
    construida arriba: se puede estar afiliado sin cotizar activamente.
  Ambas tenian corrupcion U+FFFD residual sin resolver ("S�", mismo
  patron que `sindicato`/`cotiza_fp`: sin candidato limpio en la misma
  columna), agregadas a `CORRECCIONES_MANUALES_PRIORITARIAS`.

Variables construidas
-------------------------
Nivel jefe de hogar (directo, sin agregar):
  ahorra_jefe: 1 si el jefe ahorra (Sí), 0 si no
                                (incluye "no recibe ingresos"), NaN si no
                                informa o no aplica.
  participa_organizacion_jefe: 1 si el jefe participa en CUALQUIERA de
                                los 11 tipos de organizacion.
  n_tipos_organizacion_jefe: conteo de tipos de organizacion en los que
                                participa el jefe (0-11) -- intensidad de
                                participacion civica, no solo presencia/
                                ausencia.
  cotiza_pension_jefe: 1 si el jefe cotiza actualmente a fondo de
                                pensiones, 0 si no (incluye pensionado sin
                                cotizacion activa), NaN si no aplica.
  afiliado_pension_jefe: 1 si el jefe esta afiliado a fondo de
                                pensiones (con o sin cotizacion activa).
  afiliado_salud_laboral_jefe: 1 si el jefe esta afiliado a seguridad
                                social en salud a traves de un vinculo
                                laboral.

Nivel hogar:
  tasa_ahorro_hogar: proporcion de adultos del hogar que ahorran.
  pct_hogar_participa_organizacion: proporcion de adultos que participan
                                en al menos un tipo de organizacion.
  tasa_cotizacion_pension_hogar: proporcion de adultos que cotizan
                                actualmente a fondo de pensiones --
                                indicador de formalidad laboral.
  tasa_afiliacion_pension_hogar: proporcion de adultos afiliados a fondo
                                de pensiones (con o sin cotizacion activa).
  tasa_afiliacion_salud_laboral_hogar: proporcion de adultos afiliados a
                                salud por vinculo laboral.

Output: data/processed/ahorro_capital_social_hogar_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ahorro_capital_social_hogar_elca_longitudinal.parquet"

JEFE_TOKENS = {"jefe de hogar", "jefe(a)"}
EDAD_MIN_TRABAJAR = 15  # mismo corte que build_educacion_ocupacion_hogar.py

COLUMNAS_ORGANIZACION = [
    "org_jac", "org_caridad", "org_comunitaria", "org_religiosa",
    "org_iestado", "org_etnica", "org_culdep", "org_educ", "org_mamb", "org_otra",
    "sindicato_armonizada",
]

NO_AHORRA_TOKENS = {"no", "no, no recibe ingresos", "no recibe ingresos"}


def normalizar_espacios(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalizar_cotiza_fp(serie: pd.Series) -> pd.Series:
    """Colapsa las 9 categorias de `cotiza_fp` (ola 1) a Sí/No de cotizacion activa."""
    s = normalizar_espacios(serie)
    resultado = pd.Series(np.nan, index=serie.index, dtype=object)
    tiene_dato = s != "None"
    resultado[tiene_dato] = "No"
    resultado[s.str.startswith("Si est", na=False)] = "Sí"
    return resultado


def normalizar_ahorra(serie: pd.Series) -> pd.Series:
    """'Si'/'Sí' -> 'Sí'; variantes de 'no recibe ingresos' -> 'No'; resto -> NaN."""
    s = normalizar_espacios(serie)
    s_lower = s.str.lower()
    resultado = pd.Series(np.nan, index=serie.index, dtype=object)
    resultado[s_lower.isin({"si", "sí"})] = "Sí"
    resultado[s_lower.isin(NO_AHORRA_TOKENS)] = "No"
    return resultado


def normalizar_si_no(serie: pd.Series) -> pd.Series:
    s = normalizar_espacios(serie).replace({"Si": "Sí"})
    return s.where(s.isin(["Sí", "No"]))


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def cargar_personas() -> pd.DataFrame:
    personas = pd.read_parquet(PERSONAS_PATH)
    personas["llave_c"] = _llave_compuesta(personas)
    personas["edad"] = pd.to_numeric(personas["edad"], errors="coerce")

    personas["ahorra"] = normalizar_ahorra(personas["ahorra"])

    personas["sindicato_armonizada"] = personas["sindicato"].where(
        personas["ola"] == 1, personas["org_sindicato"]
    )

    for col in COLUMNAS_ORGANIZACION:
        personas[col] = normalizar_si_no(personas[col])

    cotiza_fp_bin = normalizar_cotiza_fp(personas["cotiza_fp"])
    cotizando_bin = normalizar_si_no(personas["cotizando"])
    personas["cotiza_pension"] = cotiza_fp_bin.where(personas["ola"] == 1, cotizando_bin)

    estaba_fp_bin = normalizar_si_no(personas["estaba_fp"])
    afiliacion_fp_bin = normalizar_si_no(personas["afiliacion_fp"])
    personas["afiliado_pension"] = estaba_fp_bin.where(personas["ola"] == 1, afiliacion_fp_bin)

    estaba_sss_bin = normalizar_si_no(personas["estaba_sss"])
    segsoc_salud_bin = normalizar_si_no(personas["segsoc_salud"])
    personas["afiliado_salud_laboral"] = estaba_sss_bin.where(personas["ola"] == 1, segsoc_salud_bin)

    es_si = personas[COLUMNAS_ORGANIZACION].eq("Sí")
    tiene_dato = personas[COLUMNAS_ORGANIZACION].notna()
    personas["n_tipos_organizacion"] = es_si.sum(axis=1)
    personas["participa_organizacion"] = np.where(
        tiene_dato.any(axis=1), (personas["n_tipos_organizacion"] > 0).astype(float), np.nan
    )
    return personas


def construir_variables_jefe(personas: pd.DataFrame) -> pd.DataFrame:
    es_jefe = personas["parentesco"].astype(str).str.strip().str.lower().isin(JEFE_TOKENS)
    jefes = personas[es_jefe].set_index("llave_c")
    if jefes.index.duplicated().any():
        raise ValueError("Mas de un jefe de hogar por sub-hogar: revisar supuesto de unicidad.")

    resultado = pd.DataFrame(index=jefes.index)
    resultado["ahorra_jefe"] = (jefes["ahorra"] == "Sí").astype(float)
    resultado.loc[jefes["ahorra"].isna(), "ahorra_jefe"] = np.nan
    resultado["participa_organizacion_jefe"] = jefes["participa_organizacion"]
    resultado["n_tipos_organizacion_jefe"] = jefes["n_tipos_organizacion"].where(
        jefes["participa_organizacion"].notna()
    )
    resultado["cotiza_pension_jefe"] = (jefes["cotiza_pension"] == "Sí").astype(float)
    resultado.loc[jefes["cotiza_pension"].isna(), "cotiza_pension_jefe"] = np.nan
    resultado["afiliado_pension_jefe"] = (jefes["afiliado_pension"] == "Sí").astype(float)
    resultado.loc[jefes["afiliado_pension"].isna(), "afiliado_pension_jefe"] = np.nan
    resultado["afiliado_salud_laboral_jefe"] = (jefes["afiliado_salud_laboral"] == "Sí").astype(float)
    resultado.loc[jefes["afiliado_salud_laboral"].isna(), "afiliado_salud_laboral_jefe"] = np.nan
    return resultado


def construir_variables_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """Vectorizado: precomputar indicadores 0/1 por persona, un solo groupby().agg()."""
    es_adulto = personas["edad"] >= EDAD_MIN_TRABAJAR

    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]
    ind["ahorra_si"] = ((personas["ahorra"] == "Sí") & es_adulto).astype(float)
    ind["ahorra_valido"] = (personas["ahorra"].notna() & es_adulto).astype(float)
    ind["participa_si"] = ((personas["participa_organizacion"] == 1) & es_adulto).astype(float)
    ind["participa_valido"] = (personas["participa_organizacion"].notna() & es_adulto).astype(float)
    ind["cotiza_si"] = ((personas["cotiza_pension"] == "Sí") & es_adulto).astype(float)
    ind["cotiza_valido"] = (personas["cotiza_pension"].notna() & es_adulto).astype(float)
    ind["afil_pension_si"] = ((personas["afiliado_pension"] == "Sí") & es_adulto).astype(float)
    ind["afil_pension_valido"] = (personas["afiliado_pension"].notna() & es_adulto).astype(float)
    ind["afil_salud_si"] = ((personas["afiliado_salud_laboral"] == "Sí") & es_adulto).astype(float)
    ind["afil_salud_valido"] = (personas["afiliado_salud_laboral"].notna() & es_adulto).astype(float)

    agg = ind.groupby("llave_c").agg(
        n_ahorra=("ahorra_si", "sum"),
        n_ahorra_validos=("ahorra_valido", "sum"),
        n_participa=("participa_si", "sum"),
        n_participa_validos=("participa_valido", "sum"),
        n_cotiza=("cotiza_si", "sum"),
        n_cotiza_validos=("cotiza_valido", "sum"),
        n_afil_pension=("afil_pension_si", "sum"),
        n_afil_pension_validos=("afil_pension_valido", "sum"),
        n_afil_salud=("afil_salud_si", "sum"),
        n_afil_salud_validos=("afil_salud_valido", "sum"),
    )

    resultado = pd.DataFrame(index=agg.index)
    resultado["tasa_ahorro_hogar"] = agg["n_ahorra"] / agg["n_ahorra_validos"]
    resultado.loc[agg["n_ahorra_validos"] == 0, "tasa_ahorro_hogar"] = np.nan
    resultado["pct_hogar_participa_organizacion"] = agg["n_participa"] / agg["n_participa_validos"]
    resultado.loc[agg["n_participa_validos"] == 0, "pct_hogar_participa_organizacion"] = np.nan
    resultado["tasa_cotizacion_pension_hogar"] = agg["n_cotiza"] / agg["n_cotiza_validos"]
    resultado.loc[agg["n_cotiza_validos"] == 0, "tasa_cotizacion_pension_hogar"] = np.nan
    resultado["tasa_afiliacion_pension_hogar"] = agg["n_afil_pension"] / agg["n_afil_pension_validos"]
    resultado.loc[agg["n_afil_pension_validos"] == 0, "tasa_afiliacion_pension_hogar"] = np.nan
    resultado["tasa_afiliacion_salud_laboral_hogar"] = agg["n_afil_salud"] / agg["n_afil_salud_validos"]
    resultado.loc[agg["n_afil_salud_validos"] == 0, "tasa_afiliacion_salud_laboral_hogar"] = np.nan
    return resultado


def main() -> None:
    personas = cargar_personas()

    jefe = construir_variables_jefe(personas)
    hogar = construir_variables_hogar(personas)
    salida = jefe.join(hogar, how="outer")
    salida = salida.reset_index().rename(columns={"llave_c": "llave_compuesta"})

    ids = personas.drop_duplicates("llave_c")[
        ["llave_c", "ola", "zona", "consecutivo", "llave", "llave_n16"]
    ].rename(columns={"llave_c": "llave_compuesta"})
    salida = ids.merge(salida, on="llave_compuesta", how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print()
    print(salida.groupby("ola")[
        ["ahorra_jefe", "participa_organizacion_jefe", "n_tipos_organizacion_jefe",
         "tasa_ahorro_hogar", "pct_hogar_participa_organizacion",
         "cotiza_pension_jefe", "tasa_cotizacion_pension_hogar",
         "afiliado_pension_jefe", "tasa_afiliacion_pension_hogar",
         "afiliado_salud_laboral_jefe", "tasa_afiliacion_salud_laboral_hogar"]
    ].mean())


if __name__ == "__main__":
    main()
