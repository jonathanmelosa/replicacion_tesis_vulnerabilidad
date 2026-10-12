"""
Extension del bloque de salud (build_salud_discapacidad_hogar.py) con
controles preventivos de salud y afiliacion de menores, a partir del
modulo de Personas (ELCA 2010, 2013, 2016). Septimo bloque del inventario
de 139 candidatas.

Regla de cobertura >=10%: de las 12 candidatas
originales de este tema, pasan: `prev_med/odo/opto/malter` (59.3%/65.5%,
todas las edades), `prev_pediatra` (20.1%/26.0%, SOLO niños 0-10/0-14 por
diseño -- tiene sentido, es la pregunta de control pediatrico),
`prev_planif` (39.2%/52.7%), `beneficiario_sss` (20.1%/26.1%, SOLO niños
0-10/0-14 -- ser beneficiario de la afiliacion de otra persona). Se
EXCLUYEN por cobertura <10% en al menos una ola: `hospital_veces`,
`ultima_hosp`, `ult_hosp_dias` (5.4%-5.6%/3.6%-3.9%), `dias_noasistio`
(7.8% en ola 1), `beneficiario_orden` (7.8% en ola 2).

`prev_pediatra` y `beneficiario_sss` se restringen a la poblacion infantil
(0-14 años, el maximo observado en cualquiera de las 2 olas) al agregar a
nivel de hogar -- mismo principio ya aplicado en discapacidad y
becas/subsidios: agregar sobre TODA la poblacion mezclaria personas que
nunca podrian tener dato valido (adultos) con las que si, sesgando la
proporcion hacia abajo sin motivo real.

Variables construidas (nivel hogar)
---------------------------------------
  tasa_control_preventivo_hogar: proporcion de TODOS los miembros del
                                   hogar con algun control preventivo
                                   (medico, odontologico, optico o
                                   nutricional) en el periodo de recall.
  pct_ninos_control_pediatrico: proporcion de niños 0-14 con control
                                   pediatrico.
  tasa_planificacion_familiar: proporcion de personas 13+ con algun
                                   metodo de planificacion familiar.
  pct_ninos_beneficiario_sss: proporcion de niños 0-14 que son
                                   beneficiarios (dependientes) de la
                                   afiliacion de otra persona a
                                   seguridad social en salud.

Output: data/processed/salud_discapacidad_hogar_ext_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "salud_discapacidad_hogar_ext_elca_longitudinal.parquet"

EDAD_MAX_NINOS = 14
EDAD_MIN_PLANIF = 13

COLUMNAS_PREV = ["prev_med", "prev_odo", "prev_opto", "prev_malter"]


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

    for col in COLUMNAS_PREV + ["prev_pediatra", "prev_planif", "beneficiario_sss"]:
        personas[col] = normalizar_si_no(personas[col])

    es_si = personas[COLUMNAS_PREV].eq("Sí")
    tiene_dato = personas[COLUMNAS_PREV].notna()
    personas["control_preventivo"] = np.where(tiene_dato.any(axis=1), es_si.any(axis=1).astype(float), np.nan)
    return personas


def construir_variables_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """Vectorizado: precomputar indicadores 0/1 por persona, un solo groupby().agg()."""
    es_nino = personas["edad"].between(0, EDAD_MAX_NINOS)
    es_13_mas = personas["edad"] >= EDAD_MIN_PLANIF

    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]

    ind["prev_si"] = (personas["control_preventivo"] == 1).astype(float)
    ind["prev_valido"] = personas["control_preventivo"].notna().astype(float)

    ind["pediatra_si"] = ((personas["prev_pediatra"] == "Sí") & es_nino).astype(float)
    ind["pediatra_valido"] = (personas["prev_pediatra"].notna() & es_nino).astype(float)

    ind["planif_si"] = ((personas["prev_planif"] == "Sí") & es_13_mas).astype(float)
    ind["planif_valido"] = (personas["prev_planif"].notna() & es_13_mas).astype(float)

    ind["benef_si"] = ((personas["beneficiario_sss"] == "Sí") & es_nino).astype(float)
    ind["benef_valido"] = (personas["beneficiario_sss"].notna() & es_nino).astype(float)

    agg = ind.groupby("llave_c").sum()

    resultado = pd.DataFrame(index=agg.index)
    resultado["tasa_control_preventivo_hogar"] = agg["prev_si"] / agg["prev_valido"]
    resultado["pct_ninos_control_pediatrico"] = agg["pediatra_si"] / agg["pediatra_valido"]
    resultado["tasa_planificacion_familiar"] = agg["planif_si"] / agg["planif_valido"]
    resultado["pct_ninos_beneficiario_sss"] = agg["benef_si"] / agg["benef_valido"]

    for col, denom in [
        ("tasa_control_preventivo_hogar", "prev_valido"),
        ("pct_ninos_control_pediatrico", "pediatra_valido"),
        ("tasa_planificacion_familiar", "planif_valido"),
        ("pct_ninos_beneficiario_sss", "benef_valido"),
    ]:
        resultado.loc[agg[denom] == 0, col] = np.nan
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
        ["tasa_control_preventivo_hogar", "pct_ninos_control_pediatrico",
         "tasa_planificacion_familiar", "pct_ninos_beneficiario_sss"]
    ].mean())


if __name__ == "__main__":
    main()
