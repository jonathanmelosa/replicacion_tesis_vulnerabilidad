"""
Construccion de covariables de educacion y mercado laboral a partir del
modulo de Personas (ELCA 2010, 2013, 2016), para el modelo benchmark de
prediccion de transicion a la pobreza. Segundo bloque del inventario de 139
candidatas (el primero fue composicion del hogar, build_personas_hogar.py).

Parte de personas_elca_longitudinal_clean.parquet. Antes de construir
cualquier variable de este bloque se encontraron y corrigieron dos
problemas de calidad de datos NO relacionados con la corrupcion U+FFFD ya
documentada:

1) Duplicacion de categorias por doble espacio interno
   ------------------------------------------------------
   `actividad_ppal` (y potencialmente otras variables de texto largo)
   tiene la MISMA categoria contada como dos categorias distintas por un
   doble espacio en el texto fuente, ej. en ola 2:
     "No trabajó pero tenía un empleo o trabajo por el que recibe ingresos" (173 casos, un espacio)
     "No trabajó pero tenía un empleo o trabajo por el que recibe ingresos" (118 casos, doble espacio)
   Se normaliza colapsando espacios multiples a uno solo
   (`normalizar_espacios()`) antes de cualquier mapeo categorico en este
   script.

2) "Sí"/"Si" sin unificar en variables binarias
   -----------------------------------------------
   `lee_escribe` y `estudia` tienen "Sí" y "Si" (sin tilde) como dos
   categorias separadas para la misma respuesta, ademas de "No" y
   "No informa". Se normaliza a {"Sí", "No"} (NaN para "No informa").

Nivel educativo: escala ordinal
-----------------------------------
`nivel_educ` tiene dos etiquetas para el mismo nivel entre olas:
"Básica secundaria (6 a 13)" y "Básica secundaria y media (6 a 13)" --
ambas cubren los mismos grados (6 a 13), se tratan como el mismo nivel.
Escala ordinal adoptada (0=sin estudios, 9=posgrado completo), NO oficial
del DANE sino una construccion propia para este proyecto -- ver
ESCALA_NIVEL_EDUC:
  0 Ninguno | 1 Preescolar | 2 Primaria | 3 Secundaria |
  4 Tecnica/tecnologica sin titulo | 5 Tecnica/tecnologica con titulo |
  6 Universitaria sin titulo | 7 Universitaria con titulo |
  8 Posgrado sin titulo | 9 Posgrado con titulo

Ocupacion: regrupamiento a 7 categorias
-------------------------------------------
`ocupacion` mezcla
niveles de detalle distintos entre olas (ej. "Asalariado de empresa
particular" sin desagregar tipo de contrato en unas filas, desagregado en
"...con contrato a termino indefinido/fijo" en otras). Se regrupa a 7
categorias que preservan el gradiente de informalidad relevante para
vulnerabilidad (ver CATEGORIA_OCUPACIONAL_MAPA): Patrón o empleador >
Asalariado > Cuenta propia / independiente > Jornalero > Empleado
doméstico > Sin remuneración > Otro.

Actividad principal: definicion de "ocupado"
------------------------------------------------
`actividad_ppal` tiene redaccion distinta por ola (ver auditoria de
build_ingreso_hogar.py para el mismo patron en otras variables) pero el
concepto subyacente es comparable. Se define `ocupado`=1 para cualquier
categoria que implique haber trabajado al menos 1 hora (incluye
trabajadores familiares sin remuneracion, criterio OIT/DANE estandar) o
tener un empleo del que esta temporalmente ausente; 0 para "Ninguna de
las anteriores" y "Es incapacitado(a) permanente para trabajar"; NaN para
"No informa". LIMITACION: la categoria residual "Ninguna de las
anteriores" mezcla personas inactivas (estudiantes, hogar, jubilados) con
posibles desocupados buscando trabajo -- esta variable NO permite separar
desocupado de inactivo de forma confiable, por lo que este script
construye unicamente el indicador binario ocupado/no-ocupado, no una
tasa de desempleo. Documentado como limitacion, no se fuerza una
distincion que los datos no permiten hacer con confianza.

Correccion (2026-08-09): `jornalero`/`fue_jornalero` SI tiene equivalente
entre olas
-------------------------------------------------------------------------
Un chequeo de columnas "desaparecidas" entre olas encontro que
`fue_jornalero` (ola 1, cobertura 18.3%, edad 13+) es la misma pregunta
que `jornalero` (ola 2/3, cobertura 19.0%/18.3%, edad 15+) bajo un nombre
distinto -- mismo patron de re-nombrado ya visto en `n_empleados`.
Pregunta distinta de `ocupacion`/`categoria_ocupacional_jefe`: esta
captura si la persona ejercio como jornalero en ALGUN momento del
periodo de referencia, sea o no su ocupacion principal, por lo que no es
enteramente redundante con la categoria ocupacional principal ya
construida. Se arma la columna `jornalero_armonizado` (usa
`fue_jornalero` en ola 1, `jornalero` en ola 2/3).

Variables construidas
-------------------------
Nivel jefe de hogar (directo, sin agregar):
  nivel_educ_jefe, nivel_educ_ordinal_jefe, ocupado_jefe,
  categoria_ocupacional_jefe, horas_trabajo_jefe, jornalero_jefe

Nivel hogar (agregado sobre todos los miembros):
  nivel_educ_max_hogar: maximo de nivel_educ_ordinal en el hogar.
  pct_adultos_alfabetizados: % de personas 15+ con lee_escribe="Sí".
  tasa_asistencia_escolar: % de personas 6-17 anos con estudia="Sí"
                              (rango de edad escolar tipico).
  tasa_ocupacion_hogar: % de personas 15+ con
                              ocupado=1.
  pct_adultos_fue_jornalero: % de personas 15+ que ejercieron como
                              jornalero en el periodo de referencia.

Output: data/processed/educacion_ocupacion_hogar_elca_longitudinal.parquet
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "educacion_ocupacion_hogar_elca_longitudinal.parquet"

JEFE_TOKENS = {"jefe de hogar", "jefe(a)"}
EDAD_MIN_TRABAJAR = 15  # estandar DANE/GEIH

ESCALA_NIVEL_EDUC = {
    "Ninguno": 0,
    "Preescolar": 1,
    "Básica primaria (1 a 5)": 2,
    "Básica secundaria (6 a 13)": 3,
    "Básica secundaria y media (6 a 13)": 3,
    "Técnico sin título": 4,
    "Tecnológico sin título": 4,
    "Técnico con título": 5,
    "Tecnológico con título": 5,
    "Universitario sin título": 6,
    "Universitario con título": 7,
    "Posgrado sin título": 8,
    "Posgrado con título": 9,
}

CATEGORIA_OCUPACIONAL_MAPA = {
    "Patrón o empleador": "Patrón o empleador",
    "Asalariado de empresa particular": "Asalariado",
    "Asalariado de empresa particular con contrato a término indefinido": "Asalariado",
    "Asalariado de empresa particular con contrato a término fijo": "Asalariado",
    "Obrero o empleado de empresa particular": "Asalariado",
    "Asalariado del gobierno": "Asalariado",
    "Asalariado del Gobierno": "Asalariado",
    "Asalariado del Gobierno con contrato a término indefinido": "Asalariado",
    "Asalariado del Gobierno con contrato a término fijo": "Asalariado",
    "Obrero o empleado del gobierno": "Asalariado",
    "Trabajador por cuenta propia": "Cuenta propia",
    "Trabajador de su propia finca": "Cuenta propia",
    "Trabajador de su propia finca independientemente de la forma de tenencia": "Cuenta propia",
    "Trabajador de su propia finca (en arriendo o aparcería)": "Cuenta propia",
    "Jornalero o peón": "Jornalero",
    "Jornalero o trabajador por días": "Jornalero",
    "Empleado doméstico": "Empleado doméstico",
    "Trabajador familiar sin remuneración": "Sin remuneración",
    "Trabajador sin remuneración": "Sin remuneración",
    "Otro": "Otro",
}

# Categorias de actividad_ppal que implican estar ocupado (trabajo >=1 hora,
# incluye trabajador familiar sin remuneracion -- criterio OIT/DANE) o tener
# empleo del que esta temporalmente ausente. Construida DESPUES de
# normalizar_espacios(); cubre las variantes de redaccion de las 3 olas.
ACTIVIDAD_OCUPADO = {
    "Trabajó en forma remunerada por lo menos una hora",
    "Trabajó en forma remunerada por lo menos UNA hora en una actividad que le generó",
    "Trabajó por lo menos UNA hora en una actividad que le generó algún ingreso",
    "Trabajó como ayudante familiar sin remuneración por lo menos una hora",
    "Trabajó como ayudante familiar sin que le pagaran por lo menos UNA hora",
    "Trabajó por lo menos una hora y buscó trabajo",
    "Trabajó por lo menos UNA HORA y buscó trabajo",
    "No trabajó pero tenía un empleo o trabajo de por lo menos una hora",
    "No trabajó pero tenía un empleo o trabajo por el que recibe ingresos",
    # Variantes EXCLUSIVAS de ola 3 (2016), no relacionadas con U+FFFD -- ver
    # docstring del modulo. No afectan al benchmark (ola 3 nunca es fuente
    # de features), se agregan solo para que ocupado/tasa_ocupacion_hogar no
    # queden erroneamente sub-estimadas si alguien usa ola 3 para otra cosa
    # (ej. una especificacion dinamica futura, EDA, o validacion).
    "Trabajó en forma remunerada por lo menos UNA hora en una actividad que le gene",  # truncado en la fuente, sin '�'
    "Trabajó por lo menos UNA hora en una actividad que le generó alg???n ingreso",  # "???" literal, no U+FFFD
}
ACTIVIDAD_NO_OCUPADO = {
    "Ninguna de las anteriores",
    "Es incapacitado permanente para trabajar",
    "Es incapacitado(a) permanente para trabajar",
}


def normalizar_espacios(serie: pd.Series) -> pd.Series:
    """Colapsa espacios multiples internos a uno solo y recorta extremos."""
    return serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def cargar_personas() -> pd.DataFrame:
    personas = pd.read_parquet(PERSONAS_PATH)
    personas["llave_c"] = _llave_compuesta(personas)
    personas["edad"] = pd.to_numeric(personas["edad"], errors="coerce")

    personas["nivel_educ"] = normalizar_espacios(personas["nivel_educ"])
    personas["ocupacion"] = normalizar_espacios(personas["ocupacion"])
    personas["actividad_ppal"] = normalizar_espacios(personas["actividad_ppal"])

    for col in ["lee_escribe", "estudia"]:
        s = normalizar_espacios(personas[col])
        s = s.replace({"Si": "Sí"})
        personas[col] = s.where(s.isin(["Sí", "No"]))

    personas["nivel_educ_ordinal"] = personas["nivel_educ"].map(ESCALA_NIVEL_EDUC)
    personas["categoria_ocupacional"] = personas["ocupacion"].map(CATEGORIA_OCUPACIONAL_MAPA)

    jornalero_armonizado = personas["fue_jornalero"].where(
        personas["ola"] == 1, personas["jornalero"]
    )
    jornalero_armonizado = normalizar_espacios(jornalero_armonizado).replace({"Si": "Sí"})
    personas["jornalero_armonizado"] = jornalero_armonizado.where(
        jornalero_armonizado.isin(["Sí", "No"])
    )

    ocupado = pd.Series(np.nan, index=personas.index)
    ocupado[personas["actividad_ppal"].isin(ACTIVIDAD_OCUPADO)] = 1
    ocupado[personas["actividad_ppal"].isin(ACTIVIDAD_NO_OCUPADO)] = 0
    personas["ocupado"] = ocupado

    return personas


def construir_variables_jefe(personas: pd.DataFrame) -> pd.DataFrame:
    es_jefe = personas["parentesco"].astype(str).str.strip().str.lower().isin(JEFE_TOKENS)
    jefes = personas[es_jefe].set_index("llave_c")
    if jefes.index.duplicated().any():
        raise ValueError("Mas de un jefe de hogar por sub-hogar: revisar supuesto de unicidad.")

    resultado = jefes[["nivel_educ", "nivel_educ_ordinal", "ocupado", "categoria_ocupacional"]].rename(
        columns={
            "nivel_educ": "nivel_educ_jefe",
            "nivel_educ_ordinal": "nivel_educ_ordinal_jefe",
            "ocupado": "ocupado_jefe",
            "categoria_ocupacional": "categoria_ocupacional_jefe",
        }
    )
    resultado["horas_trabajo_jefe"] = pd.to_numeric(jefes["horas_normal"], errors="coerce")
    resultado["jornalero_jefe"] = (jefes["jornalero_armonizado"] == "Sí").astype(float)
    resultado.loc[jefes["jornalero_armonizado"].isna(), "jornalero_jefe"] = np.nan
    return resultado


def construir_variables_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """Vectorizado: precomputar indicadores 0/1 por persona, un solo groupby().agg()."""
    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]
    ind["nivel_educ_ordinal"] = personas["nivel_educ_ordinal"]

    es_adulto = personas["edad"] >= EDAD_MIN_TRABAJAR
    ind["es_alfabetizado"] = ((personas["lee_escribe"] == "Sí") & es_adulto).astype(float)
    ind["es_adulto_valido"] = (personas["lee_escribe"].notna() & es_adulto).astype(float)

    es_edad_escolar = personas["edad"].between(6, 17)
    ind["asiste"] = ((personas["estudia"] == "Sí") & es_edad_escolar).astype(float)
    ind["es_edad_escolar_valido"] = (personas["estudia"].notna() & es_edad_escolar).astype(float)

    ind["es_ocupado"] = ((personas["ocupado"] == 1) & es_adulto).astype(float)
    ind["es_adulto_ocupacion_valido"] = (personas["ocupado"].notna() & es_adulto).astype(float)

    ind["es_jornalero"] = ((personas["jornalero_armonizado"] == "Sí") & es_adulto).astype(float)
    ind["es_adulto_jornalero_valido"] = (personas["jornalero_armonizado"].notna() & es_adulto).astype(float)

    agg = ind.groupby("llave_c").agg(
        nivel_educ_max_hogar=("nivel_educ_ordinal", "max"),
        n_alfabetizados=("es_alfabetizado", "sum"),
        n_adultos_validos=("es_adulto_valido", "sum"),
        n_asisten=("asiste", "sum"),
        n_edad_escolar_validos=("es_edad_escolar_valido", "sum"),
        n_ocupados=("es_ocupado", "sum"),
        n_adultos_ocup_validos=("es_adulto_ocupacion_valido", "sum"),
        n_jornaleros=("es_jornalero", "sum"),
        n_adultos_jornalero_validos=("es_adulto_jornalero_valido", "sum"),
    )

    resultado = pd.DataFrame(index=agg.index)
    resultado["nivel_educ_max_hogar"] = agg["nivel_educ_max_hogar"]
    resultado["pct_adultos_alfabetizados"] = agg["n_alfabetizados"] / agg["n_adultos_validos"]
    resultado["tasa_asistencia_escolar"] = agg["n_asisten"] / agg["n_edad_escolar_validos"]
    resultado["tasa_ocupacion_hogar"] = agg["n_ocupados"] / agg["n_adultos_ocup_validos"]
    resultado["pct_adultos_fue_jornalero"] = agg["n_jornaleros"] / agg["n_adultos_jornalero_validos"]

    resultado.loc[agg["n_adultos_validos"] == 0, "pct_adultos_alfabetizados"] = np.nan
    resultado.loc[agg["n_edad_escolar_validos"] == 0, "tasa_asistencia_escolar"] = np.nan
    resultado.loc[agg["n_adultos_ocup_validos"] == 0, "tasa_ocupacion_hogar"] = np.nan
    resultado.loc[agg["n_adultos_jornalero_validos"] == 0, "pct_adultos_fue_jornalero"] = np.nan
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
        ["nivel_educ_ordinal_jefe", "ocupado_jefe", "nivel_educ_max_hogar",
         "pct_adultos_alfabetizados", "tasa_asistencia_escolar", "tasa_ocupacion_hogar",
         "pct_adultos_fue_jornalero"]
    ].mean())
    print()
    print("Categoria ocupacional del jefe (ola 1):")
    print(salida.loc[salida.ola == 1, "categoria_ocupacional_jefe"].value_counts())


if __name__ == "__main__":
    main()
