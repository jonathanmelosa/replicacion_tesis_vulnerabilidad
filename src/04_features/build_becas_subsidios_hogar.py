"""
Construccion de covariables de becas, credito educativo y subsidios
escolares recibidos, a partir del modulo de Personas (ELCA 2010, 2013,
2016), para el modelo benchmark de prediccion de transicion a la pobreza
. Sexto
bloque del inventario de 139 candidatas.

Poblacion cubierta: nino/estudiante, no adulto en general
------------------------------------------------------------
Estas preguntas se hacen principalmente a niños en edad escolar. Hay una cola de casos con edad hasta 66-68 años (educacion
de adultos, poco frecuente) que se excluye restringiendo el calculo a
`EDAD_MIN_ESTUDIANTE`-`EDAD_MAX_ESTUDIANTE` para no diluir el indicador
con un grupo minoritario y no comparable en tamaño entre olas.

`recibio_beca`: mas categorias que Sí/No
--------------------------------------------
No es binaria: distingue "Sí, subsidio" y "Sí, beca" ademas de "No
recibió ninguno" (con la variante corrupta "No recibi???ninguno",
exclusiva de ola 3, ya conocida -- ver seccion "HALLAZGO CRITICO" de este
documento). Se construye un indicador binario "recibió algún apoyo"
(cualquiera de las dos variantes de "Sí") para uso en el benchmark, sin
distinguir tipo de apoyo (la distincion subsidio/beca no es clave para
este modelo).

Regla de alcance (cobertura minima 10%, ver seccion anterior): de las 20
candidatas originales de este tema (7 `beca_*`, `recibio_beca`,
`credito_estudiar`, `finan_educ_pago`, 9 `rec_*`), pasan el umbral:
`recibio_beca`, `credito_estudiar`, `rec_almuerzo`, `rec_balimenta`,
`rec_bfotocop`, `rec_btransp`, `rec_desayuno`, `rec_refrigerio`,
`rec_uniformes`. Las 7 `beca_*` (1.9% en ola 1), `finan_educ_pago`
(7.5% en ola 2), `rec_alimentos_pago`/`rec_vivienda_pago` (7.5% en ola 2)
y `rec_otros_pago` (7.3%/7.5%) se EXCLUYEN.

Variables construidas (nivel hogar, restringido a edad estudiantil)
-------------------------------------------------------------------------
  pct_ninos_recibio_beca_subsidio: proporcion de niños/jovenes en edad
                                     estudiantil que recibieron beca o
                                     subsidio educativo.
  pct_ninos_credito_estudiar: proporcion con credito para estudiar.
  pct_ninos_apoyo_alimentario_escolar: proporcion que recibio CUALQUIERA
                                     de los apoyos alimentarios escolares
                                     (almuerzo, "balimenta" [beca
                                     alimentaria], desayuno, refrigerio).
  pct_ninos_apoyo_material_escolar: proporcion que recibio CUALQUIERA
                                     de los apoyos materiales/logisticos
                                     (fotocopias, transporte, uniformes).

Output: data/processed/becas_subsidios_hogar_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "becas_subsidios_hogar_elca_longitudinal.parquet"

EDAD_MIN_ESTUDIANTE, EDAD_MAX_ESTUDIANTE = 4, 20

COLUMNAS_APOYO_ALIMENTARIO = ["rec_almuerzo", "rec_balimenta", "rec_desayuno", "rec_refrigerio"]
COLUMNAS_APOYO_MATERIAL = ["rec_bfotocop", "rec_btransp", "rec_uniformes"]

RECIBIO_BECA_SI = {"Sí, subsidio", "Sí, beca", "Sí, beca y subsidio"}
# "No recibi??? ninguno" es la variante exclusiva de ola 3 con el "???"
# literal ya documentado -- no
# afecta al benchmark (ola 3 nunca es fuente de features) pero se agrega
# para que la variable no quede mal si se usa ola 3 para otra cosa.
RECIBIO_BECA_NO = {"No recibió ninguno", "No recibi??? ninguno"}


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

    s = normalizar_espacios(personas["recibio_beca"])
    recibio_beca = pd.Series(np.nan, index=personas.index)
    recibio_beca[s.isin(RECIBIO_BECA_NO)] = 0
    recibio_beca[s.isin(RECIBIO_BECA_SI)] = 1
    personas["recibio_beca_subsidio"] = recibio_beca

    personas["credito_estudiar"] = normalizar_si_no(personas["credito_estudiar"])
    for col in COLUMNAS_APOYO_ALIMENTARIO + COLUMNAS_APOYO_MATERIAL:
        personas[col] = normalizar_si_no(personas[col])

    def combinar_or(cols):
        es_si = personas[cols].eq("Sí")
        tiene_dato = personas[cols].notna()
        resultado = pd.Series(np.nan, index=personas.index)
        resultado[tiene_dato.any(axis=1)] = 0
        resultado[es_si.any(axis=1)] = 1
        return resultado

    personas["apoyo_alimentario_escolar"] = combinar_or(COLUMNAS_APOYO_ALIMENTARIO)
    personas["apoyo_material_escolar"] = combinar_or(COLUMNAS_APOYO_MATERIAL)
    return personas


def construir_variables_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """Vectorizado: precomputar indicadores 0/1 por persona, un solo groupby().agg()."""
    es_estudiante = personas["edad"].between(EDAD_MIN_ESTUDIANTE, EDAD_MAX_ESTUDIANTE)

    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]
    for col, nombre in [
        ("recibio_beca_subsidio", "beca"),
        ("credito_estudiar", "credito"),
        ("apoyo_alimentario_escolar", "alimentario"),
        ("apoyo_material_escolar", "material"),
    ]:
        valores = personas[col] if col != "credito_estudiar" else (personas[col] == "Sí").astype(float)
        valores = valores.where(personas[col].notna())
        ind[f"{nombre}_si"] = ((valores == 1) & es_estudiante).astype(float)
        ind[f"{nombre}_valido"] = (valores.notna() & es_estudiante).astype(float)

    agg = ind.groupby("llave_c").sum()

    resultado = pd.DataFrame(index=agg.index)
    mapeo_salida = {
        "beca": "pct_ninos_recibio_beca_subsidio",
        "credito": "pct_ninos_credito_estudiar",
        "alimentario": "pct_ninos_apoyo_alimentario_escolar",
        "material": "pct_ninos_apoyo_material_escolar",
    }
    for nombre, col_salida in mapeo_salida.items():
        resultado[col_salida] = agg[f"{nombre}_si"] / agg[f"{nombre}_valido"]
        resultado.loc[agg[f"{nombre}_valido"] == 0, col_salida] = np.nan
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
    print(salida.groupby("ola")[
        ["pct_ninos_recibio_beca_subsidio", "pct_ninos_credito_estudiar",
         "pct_ninos_apoyo_alimentario_escolar", "pct_ninos_apoyo_material_escolar"]
    ].mean())


if __name__ == "__main__":
    main()
