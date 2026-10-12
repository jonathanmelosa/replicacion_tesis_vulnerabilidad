"""
Construccion de covariables de salud y discapacidad a partir del modulo de
Personas (ELCA 2010, 2013, 2016), para el modelo benchmark de prediccion de
transicion a la pobreza. Tercer bloque del inventario de 139 candidatas (los
anteriores: composicion del hogar en build_personas_hogar.py, educacion y
ocupacion en build_educacion_ocupacion_hogar.py).

Motivacion: los choques de salud son un mecanismo clasico de entrada a la
pobreza en la literatura de vulnerabilidad (Dercon, 2002, "Income Risk,
Coping Strategies, and Safety Nets", ya citado en paper/referencias.bib) --
un evento de salud no cubierto puede forzar la venta de activos o el
endeudamiento del hogar. Este bloque construye tanto discapacidad cronica
(estado estructural) como eventos de salud recientes (choque puntual).

Parte de personas_elca_longitudinal_clean.parquet. Mismo patron de
normalizacion "Sí"/"Si" ya usado en build_educacion_ocupacion_hogar.py
(las variables de este bloque tienen exactamente el mismo problema:
`dif_moverse`, `ceguera`, `ev_enfe`, `afiliacion`, etc. todas mezclan "Sí"
y "Si" como categorias separadas).

Discapacidad: restringida a niños de 0-10 años (hallazgo de cobertura)
--------------------------------------------------------------------------
Antes de construir el indicador de discapacidad se encontro un problema
de comparabilidad ENTRE OLAS que la auditoria de cobertura original (que
solo miraba tasas de no-nulo, no a QUIEN se le pregunta) no capturo: en
ola 1 (2010), `dif_moverse/banarse/calle/aprender` y `ceguera/sordera/
mudez` se preguntaron EXCLUSIVAMENTE a niños de 0 a 10 años (edad maxima
observada entre respuestas no-nulas = 10.0; es un sub-modulo de niños
integrado en Personas, no una pregunta general del hogar). En ola 2/3
(2013/2016) la cobertura se amplio a todas las edades (0 a 97 años). Esto
significa:
  - Un indicador "discapacidad_jefe" (el jefe de hogar nunca tiene 0-10
    años) seria 100% NaN en ola 1 -- estructuralmente inutilizable como
    feature de entrenamiento. NO se construye.
  - Un indicador "algun miembro del hogar" mezclaria poblaciones distintas
    entre olas (solo niños en 2010 vs. todos en 2013/2016) -- violaria el
    criterio de comparabilidad del constructo (Eje 1, ver seccion de
    metodologia del benchmark). Tampoco se construye tal cual.
  - Solucion: se restringe el indicador a **niños de 0 a 10 años**, el
    unico rango de edad con cobertura consistente en las 3 olas. `pct_ninos_con_discapacidad` mide la proporcion de
    niños 0-10 del hogar con alguna de las 7 preguntas en "Sí"; queda NaN
    para hogares sin ningun niño en ese rango (no se puede evaluar la
    pregunta, no es 0).

Variables construidas
-------------------------
Discapacidad cronica (7 preguntas -> 1 indicador agregado, ver arriba):
  `dif_moverse/banarse/calle/aprender` (dificultades funcionales) +
  `ceguera/sordera/mudez` (discapacidad sensorial) se combinan con un OR
  de multiples preguntas Sí/No, mismo criterio que `tiene_conyuge_jefe`
  en build_personas_hogar.py.
    pct_ninos_con_discapacidad: ver seccion anterior. NO es un indicador
                                  de discapacidad del hogar en general,
                                  es especificamente de la primera
                                  infancia -- nombrado para reflejar
                                  exactamente lo que mide.

Eventos de salud recientes (choque puntual, ventana de recall de la
encuesta):
  `ev_enfe` (enfermedad), `ev_acci` (accidente), `ev_odon` (odontologico),
  `ev_ciru` (cirugia), `hospitalizado` (hospitalizacion) se combinan en:
    tuvo_evento_salud_jefe: 1 si el jefe reporto alguno de los 5.
    n_eventos_salud_hogar: conteo de miembros del hogar con al
                                  menos un evento (no conteo de eventos,
                                  para no sobre-ponderar hogares grandes).
    tuvo_hospitalizacion_hogar: 1 si algun miembro del hogar fue
                                  hospitalizado (evento mas severo,
                                  aislado del resto por su relevancia
                                  economica -- una hospitalizacion es un
                                  choque de gasto mucho mayor que una
                                  consulta odontologica).

Afiliacion a salud (proxy de formalidad/proteccion social, NO de estado de
salud):
  tasa_afiliacion_salud_hogar: proporcion de miembros afiliados a
                                 seguridad social en salud.
  tiene_prepagada_hogar: 1 si algun miembro tiene medicina
                                 prepagada -- a diferencia de las demas
                                 variables de este bloque, esta es un
                                 indicador de MAYOR bienestar (la
                                 prepagada es un servicio privado
                                 adicional a la afiliacion basica), se
                                 incluye por su valor como marcador de
                                 riqueza, no de vulnerabilidad.

Output: data/processed/salud_discapacidad_hogar_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "salud_discapacidad_hogar_elca_longitudinal.parquet"

JEFE_TOKENS = {"jefe de hogar", "jefe(a)"}

COLUMNAS_DISCAPACIDAD = [
    "dif_moverse", "dif_banarse", "dif_calle", "dif_aprender",
    "ceguera", "sordera", "mudez",
]
COLUMNAS_EVENTO_SALUD = ["ev_enfe", "ev_acci", "ev_odon", "ev_ciru", "hospitalizado"]


def normalizar_espacios(serie: pd.Series) -> pd.Series:
    """Colapsa espacios multiples internos a uno solo y recorta extremos."""
    return serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalizar_si_no(serie: pd.Series) -> pd.Series:
    """'Si'/'Sí' -> 'Sí'; deja 'No' igual; el resto (None, 'No informa') -> NaN."""
    s = normalizar_espacios(serie).replace({"Si": "Sí"})
    return s.where(s.isin(["Sí", "No"]))


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


EDAD_MAX_DISCAPACIDAD_NINOS = 10  # ver docstring: unico rango con cobertura en las 3 olas


def cargar_personas() -> pd.DataFrame:
    personas = pd.read_parquet(PERSONAS_PATH)
    personas["llave_c"] = _llave_compuesta(personas)
    personas["edad"] = pd.to_numeric(personas["edad"], errors="coerce")

    for col in COLUMNAS_DISCAPACIDAD + COLUMNAS_EVENTO_SALUD + ["afiliacion", "prepagada"]:
        personas[col] = normalizar_si_no(personas[col])

    # Indicador combinado por persona: 1 si CUALQUIERA de las preguntas del
    # grupo es "Sí"; 0 si todas las respondidas son "No" (sin ningun "Sí");
    # NaN si ninguna pregunta del grupo tiene dato valido para esa persona.
    def combinar_or(cols):
        es_si = personas[cols].eq("Sí")
        tiene_dato = personas[cols].notna()
        resultado = pd.Series(np.nan, index=personas.index)
        resultado[tiene_dato.any(axis=1)] = 0
        resultado[es_si.any(axis=1)] = 1
        return resultado

    personas["tiene_discapacidad"] = combinar_or(COLUMNAS_DISCAPACIDAD)
    # Discapacidad SOLO es evaluable para niños 0-10 (unico rango comparable
    # entre las 3 olas, ver docstring) -- se anula fuera de ese rango para
    # que la agregacion posterior no mezcle poblaciones distintas por ola.
    personas.loc[~personas["edad"].between(0, EDAD_MAX_DISCAPACIDAD_NINOS), "tiene_discapacidad"] = np.nan

    personas["tuvo_evento_salud"] = combinar_or(COLUMNAS_EVENTO_SALUD)
    return personas


def construir_variables_jefe(personas: pd.DataFrame) -> pd.DataFrame:
    """
    NOTA: discapacidad NO se incluye a nivel de jefe -- el jefe de hogar
    nunca tiene 0-10 años, asi que quedaria 100% NaN en las 3 olas por
    construccion (ver docstring del modulo). Solo se lleva el indicador de
    eventos de salud del jefe, que si tiene cobertura de todas las edades.
    """
    es_jefe = personas["parentesco"].astype(str).str.strip().str.lower().isin(JEFE_TOKENS)
    jefes = personas[es_jefe].set_index("llave_c")
    if jefes.index.duplicated().any():
        raise ValueError("Mas de un jefe de hogar por sub-hogar: revisar supuesto de unicidad.")
    return jefes[["tuvo_evento_salud"]].rename(columns={"tuvo_evento_salud": "tuvo_evento_salud_jefe"})


def construir_variables_hogar(personas: pd.DataFrame) -> pd.DataFrame:
    """Vectorizado: precomputar indicadores 0/1 por persona, un solo groupby().agg()."""
    ind = pd.DataFrame(index=personas.index)
    ind["llave_c"] = personas["llave_c"]
    ind["es_discapacitado"] = (personas["tiene_discapacidad"] == 1).astype(float)
    ind["discapacidad_valida"] = personas["tiene_discapacidad"].notna().astype(float)
    ind["tuvo_evento"] = (personas["tuvo_evento_salud"] == 1).astype(float)
    ind["hospitalizado"] = (personas["hospitalizado"] == "Sí").astype(float)
    ind["afiliado"] = (personas["afiliacion"] == "Sí").astype(float)
    ind["afiliacion_valida"] = personas["afiliacion"].notna().astype(float)
    ind["prepagada"] = (personas["prepagada"] == "Sí").astype(float)

    agg = ind.groupby("llave_c").agg(
        n_con_discapacidad=("es_discapacitado", "sum"),
        n_discapacidad_validos=("discapacidad_valida", "sum"),
        n_eventos_salud_hogar=("tuvo_evento", "sum"),
        n_hospitalizados=("hospitalizado", "sum"),
        n_afiliados=("afiliado", "sum"),
        n_afiliacion_validos=("afiliacion_valida", "sum"),
        n_prepagada=("prepagada", "sum"),
    )

    resultado = pd.DataFrame(index=agg.index)
    resultado["pct_ninos_con_discapacidad"] = agg["n_con_discapacidad"] / agg["n_discapacidad_validos"]
    resultado.loc[agg["n_discapacidad_validos"] == 0, "pct_ninos_con_discapacidad"] = np.nan
    resultado["n_eventos_salud_hogar"] = agg["n_eventos_salud_hogar"]
    resultado["tuvo_hospitalizacion_hogar"] = (agg["n_hospitalizados"] > 0).astype(int)
    resultado["tasa_afiliacion_salud_hogar"] = agg["n_afiliados"] / agg["n_afiliacion_validos"]
    resultado.loc[agg["n_afiliacion_validos"] == 0, "tasa_afiliacion_salud_hogar"] = np.nan
    resultado["tiene_prepagada_hogar"] = (agg["n_prepagada"] > 0).astype(int)
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
        ["tuvo_evento_salud_jefe", "pct_ninos_con_discapacidad",
         "n_eventos_salud_hogar", "tuvo_hospitalizacion_hogar",
         "tasa_afiliacion_salud_hogar", "tiene_prepagada_hogar"]
    ].mean())


if __name__ == "__main__":
    main()
