"""
Extension del bloque de educacion/ocupacion (build_educacion_ocupacion_hogar.py)
con variables adicionales del modulo de Personas (ELCA 2010, 2013, 2016).
Quinto bloque del inventario de 139 candidatas.

Regla de alcance aplicada en este bloque y en los siguientes (documentada
una vez aqui, no repetida bloque por bloque): las candidatas restantes de
Personas con cobertura POR OLA menor al 10% (verificado consistente entre
ola 1 y ola 2, no una caida espuria) se EXCLUYEN del benchmark -- mismo
criterio ya usado para las 8 sub-variables de motivo de ahorro en
build_ahorro_capital_social_hogar.py (5.7% de cobertura). Un hogar con
>90% de NaN en una columna no aporta señal util a un modelo de prediccion
y agrega ruido/complejidad.

Hallazgo: `poc`/`pin`/`pds` -- clasificacion OIT ya construida por la ELCA
------------------------------------------------------------------------------
Al revisar el diccionario de la encuesta se encontro que `poc`, `pin` y
`pds` NO son 3 preguntas independientes de baja cobertura (como parecia
al mirar cada una por separado, 1.4%-22.7%): son 3 categorias MUTUAMENTE
EXCLUYENTES de una clasificacion laboral que la ELCA ya construyo para
"personas de seguimiento" (miembros originales del panel, no todo
residente del hogar):
  poc = "Persona de seguimiento laboralmente OCupada"
  pin = "Persona de seguimiento laboralmente INactiva"
  pds = "Persona de seguimiento laboralmente DeSocupada"
Combinadas, la cobertura real es 20.3% (ola 1) / 40.4% (ola 2) -- mucho
mayor que cualquiera de las 3 por separado, y con la ventaja de que SI
distingue desocupado de inactivo, algo que `actividad_ppal` (usado en
build_educacion_ocupacion_hogar.py) documentaba explicitamente como
imposible de separar con confianza. Se construye
`categoria_laboral_oit_jefe` (Ocupado/Desocupado/Inactivo) como variable
COMPLEMENTARIA a `ocupado_jefe` (mayor cobertura, solo binario), no como
reemplazo -- el modelo de prediccion puede usar la version mas precisa
cuando esta disponible y la binaria cuando no.

`razon_noestudia`: identifica reason economica de desercion escolar
------------------------------------------------------------------------
Buena cobertura (38.1%/58.0%). La categoria mas frecuente es "Falta de
dinero" (12.715 casos) -- exactamente el tipo de señal directa de
vulnerabilidad que interesa a este benchmark. Se construye
`pct_ninos_no_estudia_razon_economica`: entre niños/jovenes de 6 a 17
años que NO estudian, proporcion cuya razon es economica ("Falta de
dinero" o "Necesita trabajar"). Se corrige tambien un typo menor
("Por enfermdad" vs. "Por enfermedad", categorias separadas por un error
de tipeo en la fuente) -- no afecta el calculo de razon economica pero se
documenta por si se usa esa categoria en el futuro.

`medio_consiguio`: mismo problema de doble espacio que actividad_ppal
------------------------------------------------------------------------
"No necesitó o no recurrió a ningún medio" (1.326 casos) vs. "...a ningún
 medio" con doble espacio (1.174 casos) -- mismo problema que se encontro
y corrigio en `actividad_ppal` (build_educacion_ocupacion_hogar.py). Se
aplica `normalizar_espacios()` antes de usar.

Variables construidas (todas a nivel de jefe de hogar, sin agregar --
cobertura del hogar completo es demasiado baja para agregar de forma
confiable en estas variables, ver regla de alcance arriba)
-------------------------------------------------------------------------
  categoria_laboral_oit_jefe: Ocupado / Desocupado / Inactivo (poc/pin/pds).
  grado_educ_jefe: grado educativo numerico del jefe (0-13).
  medio_consiguio_jefe: como consiguio el trabajo actual (categorica,
                                proxy de canal formal/informal de busqueda).
  registro_mercantil_jefe: si el jefe tiene registro mercantil de su
                                negocio (proxy de formalidad empresarial).
  n_empleados_jefe: tamaño de la empresa/negocio del jefe
                                (categorica, tramos).

Variable a nivel de hogar:
  pct_ninos_no_estudia_razon_economica: ver arriba.

Output: data/processed/educacion_ocupacion_hogar_ext_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "educacion_ocupacion_hogar_ext_elca_longitudinal.parquet"

JEFE_TOKENS = {"jefe de hogar", "jefe(a)"}
EDAD_ESCOLAR_MIN, EDAD_ESCOLAR_MAX = 6, 17
RAZONES_ECONOMICAS = {"Falta de dinero", "Necesita trabajar"}


def normalizar_espacios(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalizar_si_no(serie: pd.Series) -> pd.Series:
    s = normalizar_espacios(serie).replace({"Si": "Sí"})
    return s.where(s.isin(["Sí", "No"]))


# registro_mercantil NO es Sí/No simple -- tiene 4 categorias reales (ver
# docstring del modulo, hallazgo encontrado en la validacion de este bloque:
# la primera version uso normalizar_si_no() por error, que puso TODO en NaN
# porque ningun valor de esta columna es exactamente "Sí"/"No").
REGISTRO_MERCANTIL_MAPA = {
    "No lo necesita": "No lo necesita",
    "Lo necesita pero no lo tiene": "Lo necesita pero no lo tiene",
    "Si lo tiene y lo renovó este año": "Tiene, renovado",
    "Sí lo tiene y lo renovó este año": "Tiene, renovado",
    "Si lo tiene pero no lo renovó este año": "Tiene, no renovado",
    "Sí lo tiene pero no lo renovó este año": "Tiene, no renovado",
}


def normalizar_registro_mercantil(serie: pd.Series) -> pd.Series:
    s = normalizar_espacios(serie)
    return s.map(REGISTRO_MERCANTIL_MAPA)


# n_empleados: diferencia REAL de diseño de cuestionario entre olas, no
# corrupcion -- ola 1 pide el numero exacto de empleados (valores "2.0" a
# "99.0", minimo observado 2 -- no hay "1"), ola 2
# pide directamente el tramo ("De 2 a 5 personas", etc., incluyendo
# "trabaja solo" que ola 1 no captura en esta variable). Se arma la ola 1 a
# los mismos tramos de ola 2 para que sean comparables (mismo patron que
# ARMONIZACION_ARTICULOS en build_gasto_hogar.py).
TRAMOS_N_EMPLEADOS = [
    (2, 5, "De 2 a 5 personas"),
    (6, 10, "De 6 a 10 personas"),
    (11, 19, "De 11 a 19 personas"),
    (20, 49, "De 20 a 49 personas"),
    (50, float("inf"), "50 personas y más"),
]


def normalizar_n_empleados(serie: pd.Series) -> pd.Series:
    s = normalizar_espacios(serie).replace("None", np.nan)
    es_numerico = pd.to_numeric(s, errors="coerce")

    resultado = s.copy()
    resultado[s.str.contains("trabaja solo", case=False, na=False)] = "Trabaja solo"
    for lo, hi, etiqueta in TRAMOS_N_EMPLEADOS:
        mask = es_numerico.between(lo, hi)
        resultado[mask] = etiqueta
    # cualquier valor que no haya quedado en una de las categorias validas
    # (ni tramo de texto ya correcto, ni numero bucketizado) se anula
    categorias_validas = {"Trabaja solo"} | {t[2] for t in TRAMOS_N_EMPLEADOS}
    resultado = resultado.where(resultado.isin(categorias_validas))
    return resultado


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def construir_categoria_laboral_oit(personas: pd.DataFrame) -> pd.Series:
    """poc/pin/pds -> una sola categorica Ocupado/Desocupado/Inactivo."""
    resultado = pd.Series(np.nan, index=personas.index, dtype=object)
    resultado[personas["pin"] == 1] = "Inactivo"
    resultado[personas["pds"] == 1] = "Desocupado"
    resultado[personas["poc"] == 1] = "Ocupado"
    return resultado


def cargar_personas() -> pd.DataFrame:
    personas = pd.read_parquet(PERSONAS_PATH)
    personas["llave_c"] = _llave_compuesta(personas)
    personas["edad"] = pd.to_numeric(personas["edad"], errors="coerce")

    personas["categoria_laboral_oit"] = construir_categoria_laboral_oit(personas)
    personas["grado_educ"] = pd.to_numeric(personas["grado_educ"], errors="coerce")
    personas["medio_consiguio"] = normalizar_espacios(personas["medio_consiguio"]).replace("None", np.nan)
    personas["registro_mercantil"] = normalizar_registro_mercantil(personas["registro_mercantil"])
    personas["n_empleados"] = normalizar_n_empleados(personas["n_empleados"])
    personas["razon_noestudia"] = normalizar_espacios(personas["razon_noestudia"]).replace(
        {"Por enfermdad": "Por enfermedad", "None": np.nan}
    )
    return personas


def construir_variables_jefe(personas: pd.DataFrame) -> pd.DataFrame:
    es_jefe = personas["parentesco"].astype(str).str.strip().str.lower().isin(JEFE_TOKENS)
    jefes = personas[es_jefe].set_index("llave_c")
    if jefes.index.duplicated().any():
        raise ValueError("Mas de un jefe de hogar por sub-hogar: revisar supuesto de unicidad.")

    return jefes[
        ["categoria_laboral_oit", "grado_educ", "medio_consiguio", "registro_mercantil", "n_empleados"]
    ].rename(columns={
        "categoria_laboral_oit": "categoria_laboral_oit_jefe",
        "grado_educ": "grado_educ_jefe",
        "medio_consiguio": "medio_consiguio_jefe",
        "registro_mercantil": "registro_mercantil_jefe",
        "n_empleados": "n_empleados_jefe",
    })


def construir_variable_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """pct_ninos_no_estudia_razon_economica: vectorizado, un solo groupby().agg()."""
    es_edad_escolar = personas["edad"].between(EDAD_ESCOLAR_MIN, EDAD_ESCOLAR_MAX)
    no_estudia_con_razon = es_edad_escolar & personas["razon_noestudia"].notna()

    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]
    ind["es_razon_economica"] = (
        no_estudia_con_razon & personas["razon_noestudia"].isin(RAZONES_ECONOMICAS)
    ).astype(float)
    ind["tiene_razon_valida"] = no_estudia_con_razon.astype(float)

    agg = ind.groupby("llave_c").agg(
        n_razon_economica=("es_razon_economica", "sum"),
        n_razon_validos=("tiene_razon_valida", "sum"),
    )
    resultado = pd.DataFrame(index=agg.index)
    resultado["pct_ninos_no_estudia_razon_economica"] = agg["n_razon_economica"] / agg["n_razon_validos"]
    resultado.loc[agg["n_razon_validos"] == 0, "pct_ninos_no_estudia_razon_economica"] = np.nan
    return resultado


def main() -> None:
    personas = cargar_personas()

    jefe = construir_variables_jefe(personas)
    hogar = construir_variable_hogar(personas)
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
    print("categoria_laboral_oit_jefe por ola:")
    print(salida.groupby("ola")["categoria_laboral_oit_jefe"].value_counts(normalize=True))
    print()
    print(salida.groupby("ola")[
        ["grado_educ_jefe", "pct_ninos_no_estudia_razon_economica"]
    ].mean())


if __name__ == "__main__":
    main()
