"""
Construccion de covariable de participacion civica y politica, a partir
del modulo de Personas (ELCA 2010, 2013, 2016), para el modelo benchmark
de prediccion de transicion a la pobreza. Noveno bloque del inventario de 139
candidatas -- ultimo bloque de features de Personas antes de la
verificacion final de cobertura completa.

Candidatas de este tema: `mov_parpol` (participacion en movimiento
politico), `junta_edif` (junta de edificio/administradora), `asoc_vigil`
(asociacion de vigilancia/seguridad barrial). Las 4 restantes del mismo
tema (`jov_org_social`, `participa`, `porcentaje_participacion`,
`beca_accionsocial`) ya estan clasificadas fuera de las 139 candidatas
(no presentes en ola 1 o cobertura casi vacia) y no se reconsideran aqui.

Cobertura pareja en `asoc_vigil`/`mov_parpol` (39.2%/39.5%/39.1%, edad
minima 13 en las 3 olas), pero `junta_edif` cae de 39.2% (ola 1) a
20.3%-20.6% (olas 2-3) manteniendo el mismo rango de edad -- no se
identifico una causa de diseño (posible cambio en el filtro de la
pregunta entre rondas), se deja como observacion sin resolver ya que
20.3% aun supera el umbral minimo de 10%.

Prevalencia muy baja de "Sí" en las 3 variables (0.1%-0.4% de quienes
responden) -- se combinan con OR en un solo indicador de "participacion
civica" para tener una variable con suficiente varianza util para el
modelo, en vez de 3 indicadores casi degenerados por separado (mismo
criterio ya aplicado a apoyo_alimentario_escolar y apoyo_material_escolar
en el Bloque 6).

Variable construida (nivel hogar, restringido a personas 13+)
-------------------------------------------------------------
  tasa_participacion_civica_hogar: proporcion de miembros del hogar de
                                     13+ años que participan en al menos
                                     una de: junta de edificio, asociacion
                                     de vigilancia barrial o movimiento
                                     politico.

Output: data/processed/participacion_civica_hogar_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "participacion_civica_hogar_elca_longitudinal.parquet"

EDAD_MIN_PARTICIPACION = 13

COLUMNAS_PARTICIPACION = ["junta_edif", "asoc_vigil", "mov_parpol"]


def normalizar_espacios(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


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

    for col in COLUMNAS_PARTICIPACION:
        personas[col] = normalizar_si_no(personas[col])

    es_si = personas[COLUMNAS_PARTICIPACION].eq("Sí")
    tiene_dato = personas[COLUMNAS_PARTICIPACION].notna()
    personas["participacion_civica"] = np.where(
        tiene_dato.any(axis=1), es_si.any(axis=1).astype(float), np.nan
    )
    return personas


def construir_variables_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """Vectorizado: precomputar indicadores 0/1 por persona, un solo groupby().agg()."""
    es_13_mas = personas["edad"] >= EDAD_MIN_PARTICIPACION

    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]
    ind["civica_si"] = ((personas["participacion_civica"] == 1) & es_13_mas).astype(float)
    ind["civica_valido"] = (personas["participacion_civica"].notna() & es_13_mas).astype(float)

    agg = ind.groupby("llave_c").sum()

    resultado = pd.DataFrame(index=agg.index)
    resultado["tasa_participacion_civica_hogar"] = agg["civica_si"] / agg["civica_valido"]
    resultado.loc[agg["civica_valido"] == 0, "tasa_participacion_civica_hogar"] = np.nan
    return resultado


def main() -> None:
    personas = cargar_personas()
    hogar = construir_variables_hogar(personas)
    salida = hogar.reset_index().rename(columns={"llave_c": "llave_compuesta"})

    ids = personas.drop_duplicates("llave_c")[
        ["llave_c", "ola", "zona", "consecutivo", "llave", "llave_n16"]
    ].rename(columns={"llave_c": "llave_compuesta"})
    salida = ids.merge(salida, on="llave_compuesta", how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print()
    print(salida.groupby("ola")["tasa_participacion_civica_hogar"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
