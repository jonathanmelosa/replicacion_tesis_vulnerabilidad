"""
Construccion de covariables de composicion del hogar a partir del modulo de
Personas (ELCA 2010, 2013, 2016), para el modelo benchmark de prediccion de
transicion a la pobreza.

Parte de personas_elca_longitudinal_clean.parquet (NO del archivo original
sin limpiar -- por la corrupcion de codificacion
U+FFFD: usar el archivo sin limpiar produciria categorias de `parentesco`
duplicadas entre olas, ej. "Jefe(a)" vs. "Jefe de hogar" contados como
personas distintas si se buscara por texto exacto sin la limpieza previa).

Primer bloque de variables del modulo Personas a construirse.

Normalizacion de `sexo` antes de agregar
-------------------------------------------
`sexo` no tiene corrupcion de codificacion (a diferencia de `parentesco`),
pero SI tiene capitalizacion inconsistente entre olas: "Mujer"/"Hombre" en
unas filas, "MUJER"/"HOMBRE" en mayuscula en otras -- se cuentan como 4
categorias en vez de 2 si no se normaliza. Se aplica `.str.title()` antes
de cualquier agregacion, mismo patron que ya usa
`build_pobreza_desagregaciones.py` para `sexo_jefe`.

Identificacion del jefe de hogar
------------------------------------
`parentesco` usa dos etiquetas distintas para "jefe de hogar" segun la ola
("Jefe de hogar" en 2010, "Jefe(a)" en 2013/2016) -- ya verificado en
`build_pobreza_desagregaciones.py` que hay exactamente un jefe por
sub-hogar en las 3 olas (9853/9261/8818 personas = numero de hogares). Se
reutiliza el mismo criterio (JEFE_TOKENS) aqui para no depender de ese
otro script.

Variables construidas
-------------------------
Nivel de conteo/composicion:
  n_ninos_5: personas con edad < 6 (primera infancia).
  n_ninos_12: personas con edad < 12 (igual definicion que
                            t_ninos_12 en build_pobreza_desagregaciones.py,
                            recalculada aqui para que este script no
                            dependa de aquel).
  n_adultos_mayores: personas con edad >= 65.
  razon_dependencia_demografica: (menores de 15 + mayores de 65) /
                            personas de 15 a 64 años. NaN si el
                            denominador es 0 (hogar sin nadie en edad de
                            trabajar -- no se fuerza a infinito ni a 0).
  pct_mujeres_hogar: proporcion de mujeres sobre el total de
                            personas con `sexo` no nulo.

Nivel jefe de hogar (sin agregar, tomado directo -- ver "Identificacion
del jefe de hogar" arriba):
  sexo_jefe, edad_jefe

Estructura del hogar:
  tiene_conyuge_jefe: 1 si algun miembro del sub-hogar tiene
                            `parentesco` == "Cónyuge o compañera(o)", 0 si
                            no. Nombrada por lo que mide literalmente (no
                            "hogar_monoparental"): la ausencia de conyuge
                            del jefe no implica por si sola que haya hijos
                            en el hogar; combinar con n_ninos_5/n_ninos_12
                            si se quiere una definicion mas estricta de
                            monoparental en el script de modelado.

`tamano_hogar` (t_personas) NO se reconstruye aqui: ya existe en
hogar_elca_longitudinal_clean.parquet y se valido que coincide casi
exactamente con el conteo real de filas en Personas -- se usa esa fuente en el script
de modelado en vez de duplicarla.

Output: data/processed/personas_hogar_elca_longitudinal.parquet
Una fila por sub-hogar x ola (mismo grano que hogar_elca_longitudinal_clean).
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "personas_hogar_elca_longitudinal.parquet"

JEFE_TOKENS = {"jefe de hogar", "jefe(a)"}
CONYUGE_JEFE = "Cónyuge o compañera(o)"


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    """consecutivo (ola 1) / llave (ola 2) / llave_n16 (ola 3): 1 fila = 1 sub-hogar."""
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def cargar_personas() -> pd.DataFrame:
    personas = pd.read_parquet(PERSONAS_PATH)
    personas["llave_c"] = _llave_compuesta(personas)
    personas["sexo"] = personas["sexo"].astype(str).str.strip().str.title()
    personas.loc[~personas["sexo"].isin(["Hombre", "Mujer"]), "sexo"] = np.nan
    personas["edad"] = pd.to_numeric(personas["edad"], errors="coerce")
    return personas


def construir_composicion_demografica(personas: pd.DataFrame) -> pd.DataFrame:
    """
    n_ninos_5, n_ninos_12, n_adultos_mayores, razon_dependencia, pct_mujeres.

    Vectorizado: se precomputan columnas indicadoras 0/1 por persona y se
    agregan todas juntas con UN solo groupby().sum() -- groupby().apply()
    con lambdas de Python es ordenes de magnitud mas lento (probado: >100s
    para 118.824 filas con 6 apply() separados vs. <1s vectorizado).
    """
    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]
    ind["es_nino_5"] = (personas["edad"] < 6).astype(int)
    ind["es_nino_12"] = (personas["edad"] < 12).astype(int)
    ind["es_adulto_mayor"] = (personas["edad"] >= 65).astype(int)
    ind["es_menor_15"] = (personas["edad"] < 15).astype(int)
    ind["es_edad_trabajar"] = personas["edad"].between(15, 64).astype(int)
    ind["es_mujer"] = (personas["sexo"] == "Mujer").astype(int)
    ind["sexo_valido"] = personas["sexo"].notna().astype(int)

    agg = ind.groupby("llave_c").sum()

    resultado = pd.DataFrame(index=agg.index)
    resultado["n_ninos_5"] = agg["es_nino_5"]
    resultado["n_ninos_12"] = agg["es_nino_12"]
    resultado["n_adultos_mayores"] = agg["es_adulto_mayor"]
    resultado["razon_dependencia_demografica"] = (
        (agg["es_menor_15"] + agg["es_adulto_mayor"]) / agg["es_edad_trabajar"]
    )
    resultado["pct_mujeres_hogar"] = agg["es_mujer"] / agg["sexo_valido"]
    resultado.loc[agg["es_edad_trabajar"] == 0, "razon_dependencia_demografica"] = np.nan
    resultado.loc[agg["sexo_valido"] == 0, "pct_mujeres_hogar"] = np.nan
    return resultado


def construir_variables_jefe(personas: pd.DataFrame) -> pd.DataFrame:
    """sexo_jefe, edad_jefe: tomados directo del jefe, sin agregar."""
    es_jefe = personas["parentesco"].astype(str).str.strip().str.lower().isin(JEFE_TOKENS)
    jefes = personas[es_jefe].set_index("llave_c")
    if jefes.index.duplicated().any():
        raise ValueError("Mas de un jefe de hogar por sub-hogar: revisar supuesto de unicidad.")
    return jefes[["sexo", "edad"]].rename(columns={"sexo": "sexo_jefe", "edad": "edad_jefe"})


def construir_estructura_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """tiene_conyuge_jefe: 1 si algun miembro del sub-hogar es conyuge del jefe."""
    es_conyuge = (personas["parentesco"] == CONYUGE_JEFE).astype(int)
    tiene_conyuge = es_conyuge.groupby(personas["llave_c"]).max()
    return tiene_conyuge.rename("tiene_conyuge_jefe").to_frame()


def main() -> None:
    personas = cargar_personas()

    demografica = construir_composicion_demografica(personas)
    jefe = construir_variables_jefe(personas)
    estructura = construir_estructura_hogar(personas)

    salida = demografica.join(jefe, how="left").join(estructura, how="left")
    salida = salida.reset_index().rename(columns={"llave_c": "llave_compuesta"})

    # Recuperar identificadores de ola/zona desde el propio modulo de personas
    # (una fila por llave_c es suficiente, son constantes dentro del sub-hogar).
    ids = personas.drop_duplicates("llave_c")[
        ["llave_c", "ola", "zona", "consecutivo", "llave", "llave_n16"]
    ].rename(columns={"llave_c": "llave_compuesta"})
    salida = ids.merge(salida, on="llave_compuesta", how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print()
    print(salida.groupby("ola")[
        ["n_ninos_5", "n_ninos_12", "n_adultos_mayores", "razon_dependencia_demografica",
         "pct_mujeres_hogar", "tiene_conyuge_jefe"]
    ].mean())


if __name__ == "__main__":
    main()
