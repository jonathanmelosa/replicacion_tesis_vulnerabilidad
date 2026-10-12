"""
Extension del bloque de composicion del hogar (build_personas_hogar.py)
con demografia y estructura familiar adicional, a partir del modulo de
Personas (ELCA 2010, 2013, 2016). Octavo bloque del inventario de 139
candidatas.

`estado_civil`: cobertura 100% en ola 2 es enganosa
--------------------------------------------------------
`estado_civil` salta de 39.2% (ola 1, edad minima 13) a 100.0% (ola 2,
edad minima 0) -- investigado antes de construir (misma disciplina que en
discapacidad): en ola 2, los menores de 10 años quedan codificados como
"Soltero(a)" por defecto -- una respuesta administrativa, no una
evaluacion real de estado civil. Se restringe a personas 13+ (poblacion
que ola 1 SI evalua de forma genuina) para no diluir la variable con
"Soltero(a)" automatico de niños.

`id_dpto_nac`/`id_mpio_nac`: mismo riesgo de identificador falso que
`id_dpto`/`id_mpio` del hogar (ver seccion "por que la LP se queda en 2
dominios" mas arriba, donde se documento que esos identificadores a nivel
de HOGAR son codigos anonimizados sin correspondencia real con DIVIPOLA).
No se verifico si estos identificadores de lugar de NACIMIENTO tienen el
mismo problema o son genuinos -- se EXCLUYEN de este bloque hasta poder
confirmar contra el diccionario, en vez de construir una variable de
migracion sobre una base no verificada.

`padre_vive`/`madre_vive`: dos formas de decir "no" segun la ola
------------------------------------------------------------------
Ademas de "Sí"/"No", existen las variantes "Falleció"/"Ya falleció" --
se tratan como equivalentes a "No" (el padre/madre no vive), ambas
formas indican lo mismo, solo cambia la redaccion segun la ola.

`tareas`: distingue trabajo domestico en el propio hogar vs. en OTRO hogar
--------------------------------------------------------------------------
"Sí, del hogar" (quehaceres domesticos normales) vs. "Sí, de otro hogar"
(trabajo domestico pagado en la vivienda de otra familia) son conceptos
muy distintos para un niño -- la segunda es una señal de vulnerabilidad
mucho mas fuerte (trabajo infantil domestico fuera del hogar propio). Se
construye especificamente sobre esa categoria, no sobre "cualquier tarea".

Regla de cobertura >=10% excluye `vive_conyuge`
(redundante con `tiene_conyuge_jefe` ya construido en
build_personas_hogar.py, se prefiere evitar la duplicacion) y
`mes_unionm`/`ano_unionm` (6.9% en ola 1).

Variables construidas
-------------------------
Nivel jefe (directo, sin agregar -- el jefe siempre es adulto, no hay
riesgo de mezclar poblacion infantil):
  estado_civil_jefe, etnia_jefe, edad_union_jefe (edad a la que se unio
  con su pareja por primera vez).

Nivel hogar (restringido a niños 0-17, unica poblacion con
cobertura util y comparable entre olas para estas 2 variables):
  pct_ninos_padre_vivo, pct_ninos_madre_viva.

Nivel hogar (restringido a la poblacion infantil con cobertura real,
2-14 años segun lo observado):
  pct_ninos_trabaja_otro_hogar: trabajo domestico en OTRO hogar
                                  (señal de vulnerabilidad fuerte, ver
                                  arriba), distinto de tareas domesticas
                                  normales del propio hogar.

Output: data/processed/personas_hogar_ext_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "personas_hogar_ext_elca_longitudinal.parquet"

JEFE_TOKENS = {"jefe de hogar", "jefe(a)"}
EDAD_MIN_ESTADO_CIVIL = 13
EDAD_MAX_NINOS = 17
EDAD_MAX_TAREAS = 14

VIVE_SI = {"si", "sí"}
VIVE_NO = {"no", "falleció", "ya falleció"}


def normalizar_espacios(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalizar_vive(serie: pd.Series) -> pd.Series:
    s = normalizar_espacios(serie).str.lower()
    resultado = pd.Series(np.nan, index=serie.index, dtype=object)
    resultado[s.isin(VIVE_SI)] = "Sí"
    resultado[s.isin(VIVE_NO)] = "No"
    return resultado


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def cargar_personas() -> pd.DataFrame:
    personas = pd.read_parquet(PERSONAS_PATH)
    personas["llave_c"] = _llave_compuesta(personas)
    personas["edad"] = pd.to_numeric(personas["edad"], errors="coerce")
    personas["edad_unionm"] = pd.to_numeric(personas["edad_unionm"], errors="coerce")

    personas["estado_civil"] = normalizar_espacios(personas["estado_civil"]).replace("None", np.nan)
    personas.loc[personas["edad"] < EDAD_MIN_ESTADO_CIVIL, "estado_civil"] = np.nan

    personas["etnia"] = normalizar_espacios(personas["etnia"]).replace("None", np.nan)
    personas["padre_vive"] = normalizar_vive(personas["padre_vive"])
    personas["madre_vive"] = normalizar_vive(personas["madre_vive"])

    tareas = normalizar_espacios(personas["tareas"])
    personas["trabaja_otro_hogar"] = np.where(
        tareas == "None", np.nan, (tareas == "Si, de otro hogar").astype(float)
    )
    return personas


def construir_variables_jefe(personas: pd.DataFrame) -> pd.DataFrame:
    es_jefe = personas["parentesco"].astype(str).str.strip().str.lower().isin(JEFE_TOKENS)
    jefes = personas[es_jefe].set_index("llave_c")
    if jefes.index.duplicated().any():
        raise ValueError("Mas de un jefe de hogar por sub-hogar: revisar supuesto de unicidad.")
    return jefes[["estado_civil", "etnia", "edad_unionm"]].rename(columns={
        "estado_civil": "estado_civil_jefe",
        "etnia": "etnia_jefe",
        "edad_unionm": "edad_union_jefe",
    })


def construir_variables_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """Vectorizado: precomputar indicadores 0/1 por persona, un solo groupby().agg()."""
    es_nino_17 = personas["edad"].between(0, EDAD_MAX_NINOS)
    es_nino_tareas = personas["edad"].between(0, EDAD_MAX_TAREAS)

    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]

    ind["padre_vivo_si"] = ((personas["padre_vive"] == "Sí") & es_nino_17).astype(float)
    ind["padre_vivo_valido"] = (personas["padre_vive"].notna() & es_nino_17).astype(float)
    ind["madre_viva_si"] = ((personas["madre_vive"] == "Sí") & es_nino_17).astype(float)
    ind["madre_viva_valido"] = (personas["madre_vive"].notna() & es_nino_17).astype(float)

    ind["trabaja_otro_si"] = ((personas["trabaja_otro_hogar"] == 1) & es_nino_tareas).astype(float)
    ind["trabaja_otro_valido"] = (personas["trabaja_otro_hogar"].notna() & es_nino_tareas).astype(float)

    agg = ind.groupby("llave_c").sum()

    resultado = pd.DataFrame(index=agg.index)
    resultado["pct_ninos_padre_vivo"] = agg["padre_vivo_si"] / agg["padre_vivo_valido"]
    resultado["pct_ninos_madre_viva"] = agg["madre_viva_si"] / agg["madre_viva_valido"]
    resultado["pct_ninos_trabaja_otro_hogar"] = agg["trabaja_otro_si"] / agg["trabaja_otro_valido"]

    for col, denom in [
        ("pct_ninos_padre_vivo", "padre_vivo_valido"),
        ("pct_ninos_madre_viva", "madre_viva_valido"),
        ("pct_ninos_trabaja_otro_hogar", "trabaja_otro_valido"),
    ]:
        resultado.loc[agg[denom] == 0, col] = np.nan
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
        ["edad_union_jefe", "pct_ninos_padre_vivo", "pct_ninos_madre_viva",
         "pct_ninos_trabaja_otro_hogar"]
    ].mean())
    print()
    print("estado_civil_jefe (ola 1):")
    print(salida.loc[salida.ola == 1, "estado_civil_jefe"].value_counts())


if __name__ == "__main__":
    main()
