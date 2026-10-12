"""
Construccion de covariables del modulo de Niños (ELCA 2010, 2013, 2016)
para el modelo benchmark de prediccion de transicion a la pobreza. Mismo
nivel de auditoria que Personas y Comunidades (4 etapas completas).

Auditoria completa
------------------------
1. **Limpieza de corrupcion U+FFFD** (`04_limpieza_base_ninos.py`): de 433
   columnas, 80 tienen "�" (79 cerradas + 1 de texto libre, `descrip_oficio`,
   sin tocar). Correccion automatica: 131 valores. Residual (16 columnas,
   25 valores): 6 rescatados via diccionario PDF de 2013
   (`{U,R}Ninos0a13.pdf`), 3 via la regla Sí/No de Personas (`S�` con
   candidato ambiguo en el diccionario pero inequivoco por diseño de
   pregunta), resto corregidos a mano por ser reconstrucciones de acento
   sin ambiguedad. Toda la corrupcion residual esta en ola 2 (2013).
2. **Clasificacion completa de las 433 columnas**
   (`docs/variable_audit/ninos_construccion.csv`, mismo criterio >1% de
   cobertura por ola): 83 `CANDIDATO_BENCHMARK`, 256 `EXCLUIDA_NO_EN_OLA1`
   (el modulo crecio mucho de 2010->2013: nuevo tramo de edad 0-13 y
   nuevas baterias de estimulacion/cuidado), 56 `EXCLUIDA_SOLO_OLA3`, 22
   `EXCLUIDA_NO_EN_OLA2`, 11 `IDENTIFICADOR`, 4 `EXCLUIDA_CASI_VACIA`
   (verificadas triviales, <1% en las 3 olas), 1 `EXCLUIDA_OTRO_PATRON`
   (`ano_nac_m`, verificada trivial). Suma verificada: 433.
3. **Busqueda de renombrados entre olas** (nombre relajado cutoff 0.55 +
   diccionarios PDF de `elca_{2010,2013}/{U,R}Ninos*.pdf`): **5
   renombrados reales encontrados**, todos del mismo bloque tematico de
   "estimulacion en el hogar" -- en ola 1 la pregunta era una unica
   variable de frecuencia; en ola 2 se dividio en "quien" (quien lo hace)
   y "freq_*" (frecuencia). Se rescata la parte de frecuencia, comparable
   entre olas:
     - `conversa` (ola 1) = `freq_conversa` (ola 2)
     - `ensena` (ola 1) = `freq_ensena` (ola 2)
     - `juega_encasa` (ola 1) = `freq_juegadentro` (ola 2) -- encontrado
       SOLO por el diccionario PDF, el nombre no tiene similitud textual
       suficiente para el metodo de nombre relajado (encasa vs dentro).
     - `juega_fueradecasa` (ola 1) = `freq_juegafuera` (ola 2)
     - `lee_libros` (ola 1) = `freq_lee` (ola 2)
   Mismas 4 categorias de frecuencia en ambas olas ("Todos los días"/
   "2-3 veces a la semana"/"Una vez a la semana"/"Nunca", con variantes
   menores de redaccion) -- verificado antes de armonizar.
4. **Verificacion de edad de la poblacion antes de construir** (mismo
   principio de comparabilidad de Personas -- discapacidad, Bloque 3):
   `edad_ames` viene codificada como años*100+meses (ej. 207 = 2 años 7
   meses) -- decodificada antes de cualquier chequeo. El bloque de
   salud/vacunacion/estimulacion se concentra en niños 0-7 años en ambas
   olas (mediana 2.6-4.8 años); el bloque de oficios/trabajo domestico se
   concentra en niños 5-11 años en ambas olas (mediana 5.1-8.2 años) --
   ambos comparables entre ola 1 y ola 2 (las dos UNICAS fuentes de
   features del benchmark; ola 3 solo aporta el resultado observado, no
   necesita comparabilidad de edad con las anteriores).

**Redundancia deliberadamente NO construida**: `padre_vive`, `madre_vive`,
`educ_padre`, `educ_madre`, `trabajo_padre`, `trabajo_madre`,
`orden_padre`, `orden_madre`, `ano_nac_p` -- estas preguntas TAMBIEN
existen en el modulo de Personas (via el propio informante) y ya se
construyeron variables equivalentes a nivel de hogar (`pct_ninos_padre_vivo`/
`pct_ninos_madre_viva` en Personas Bloque 8, `nivel_educ_max_hogar` en
Bloque 2). Duplicar esta informacion desde el modulo de Niños no aporta
señal adicional; se documenta como decision de alcance, no un descarte
accidental.

**Test cognitivo TVIP**: `puntoinicio`/`itemtope`/`menoserrores`/
`puntuaciondirecta` corresponden al Test de Vocabulario en Imágenes
Peabody (confirmado contra el diccionario PDF 2013: "variable: es igual a
la puntuación... prueba TVIP"). `puntuaciondirecta` (puntaje bruto) es la
variable resumen mas directamente interpretable -- se usa esa, sin
construir un indice compuesto con los items auxiliares de aplicacion del
test (puntoinicio/itemtope son parametros de administracion del test, no
resultado).

Variables construidas (nivel hogar, agregadas sobre niños con dato valido)
---------------------------------------------------------------------------
  tasa_vacunacion_basica_hogar: proporcion de niños 0-7 con las 3
                                    vacunas de referencia (antituberculosa,
                                    triple viral, hepatitis B recien
                                    nacido) todas = Sí.
  tasa_control_crecimiento_hogar: proporcion de niños 0-7 que asistieron
                                    a control de crecimiento y desarrollo.
  tasa_vacuna_fiebreamarilla_hogar: proporcion de niños 0-7 con vacuna
                                    contra la fiebre amarilla aplicada.
                                    CORRECCION (2026-08-09): `fiebrea` NO
                                    es un sintoma de fiebre reciente --
                                    verificado contra el diccionario PDF
                                    2013 ("303-g. ¿Recibió la vacuna
                                    contra la fiebre amarilla?"), es una
                                    vacuna mas de la bateria, mal
                                    etiquetada inicialmente como
                                    "pct_ninos_fiebre_reciente_hogar" por
                                    el nombre de columna enganoso.
  talla_promedio_nino_hogar: talla promedio (cm) de los niños
                                    medidos -- sin ajustar por edad/sexo
                                    (no es un z-score OMS), covariable
                                    cruda, documentado como limitacion.
  peso_promedio_nino_hogar: peso promedio (kg) de los niños
                                    medidos, misma limitacion.
  tasa_asistencia_escolar_nino_hogar: proporcion de niños con `asiste`=Sí
                                    (cobertura solo 0-7, pregunta distinta
                                    de `tasa_asistencia_escolar` de
                                    Personas que cubre 6-17).
  pct_ninos_oficios_hogar: proporcion de niños 5-11 que realizan
                                    CUALQUIER oficio domestico (cocinar,
                                    limpiar, lavar, planchar, mandados,
                                    traer agua, cuidar niños/enfermos).
                                    Saturado (93%-99.7%) porque `limpieza`
                                    sola ya tiene ~87% de "Sí" (barra baja:
                                    cualquier ayuda de aseo cuenta) --
                                    ver `n_oficios_promedio_nino_hogar`
                                    para una version con mas varianza.
  n_oficios_promedio_nino_hogar: conteo promedio (0-8) de tipos de
                                    oficio domestico realizados -- version
                                    de INTENSIDAD, no solo presencia.
  pct_ninos_trabajo_remunerado_hogar: proporcion de niños 5-11 con
                                    `trabajo`=Sí (trabajo fuera del hogar).
  horas_oficio_promedio_nino_hogar: horas semanales promedio dedicadas a
                                    oficios domesticos, entre los niños
                                    que reportan alguno.
  indice_estimulacion_hogar_nino: promedio (0-4) de la frecuencia de 5
                                    actividades de estimulacion (conversa,
                                    enseña, juega dentro/fuera de casa, lee
                                    libros) -- escala 0=Nunca a
                                    4=Todos los días, armonizada entre
                                    olas (ver hallazgo de renombrados).
  tvip_puntaje_directo_hogar: puntaje bruto promedio del test TVIP
                                    entre los niños evaluados -- sin
                                    estandarizar por edad (no es un
                                    puntaje normado), misma limitacion que
                                    `talla_promedio_nino_hogar`/
                                    `peso_promedio_nino_hogar`: solo
                                    comparable entre ola 1 y ola 2 en la
                                    medida en que la distribucion de edad
                                    de los niños evaluados sea similar
                                    (verificado en el punto 4 de la
                                    auditoria).
  pct_ninos_cuidado_terceros_hogar: proporcion de niños cuyo cuidador
                                    principal NO es el padre ni la madre
                                    (abuelos, otros familiares, terceros)
                                    mientras los padres trabajan --
                                    indicador de carga de cuidado infantil.

Output: data/processed/ninos_hogar_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NINOS_PATH = PROJECT_ROOT / "data" / "processed" / "ninos_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ninos_hogar_elca_longitudinal.parquet"

EDAD_MAX_SALUD = 7
EDAD_MIN_OFICIOS, EDAD_MAX_OFICIOS = 5, 11

COLUMNAS_VACUNACION_BASICA = ["antituberculosa", "triplev", "hepatitisrn"]
COLUMNAS_OFICIOS = [
    "cocinar", "limpieza", "lavar", "planchar", "mandados", "traer_agua",
    "cuidar_ninos", "cuidar_enfermos",
]

FREQ_MAPA = {
    "Nunca": 0, "Una vez a la semana": 1, "2 o 3 veces a la semana": 2,
    "2 – 3 veces a la semana": 2, "De vez en cuando": 2,
    "Casi todos los días": 3, "Todos los días": 4,
}
COLUMNAS_ESTIMULACION_ARMONIZADAS = {
    "conversa": "freq_conversa",
    "ensena": "freq_ensena",
    "juega_encasa": "freq_juegadentro",
    "juega_fueradecasa": "freq_juegafuera",
    "lee_libros": "freq_lee",
}

PADRES_TOKENS = {"la madre", "el padre", "su madre", "su padre", "madre", "padre"}


def normalizar_espacios(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalizar_si_no(serie: pd.Series) -> pd.Series:
    s = normalizar_espacios(serie).replace({"Si": "Sí"})
    return s.where(s.isin(["Sí", "No"]))


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def cargar_ninos() -> pd.DataFrame:
    ninos = pd.read_parquet(NINOS_PATH)
    ninos["llave_c"] = _llave_compuesta(ninos)

    edad_raw = pd.to_numeric(ninos["edad_ames"], errors="coerce")
    ninos["edad_a"] = (edad_raw // 100) + (edad_raw % 100) / 12

    for col in COLUMNAS_VACUNACION_BASICA + ["asistio_control", "fiebrea", "asiste", "trabajo"]:
        ninos[col] = normalizar_si_no(ninos[col])
    for col in COLUMNAS_OFICIOS:
        ninos[col] = normalizar_si_no(ninos[col])

    es_si_vacuna = ninos[COLUMNAS_VACUNACION_BASICA].eq("Sí")
    tiene_dato_vacuna = ninos[COLUMNAS_VACUNACION_BASICA].notna()
    ninos["vacunacion_basica_completa"] = np.where(
        tiene_dato_vacuna.all(axis=1), es_si_vacuna.all(axis=1).astype(float), np.nan
    )

    ninos["talla_cm"] = pd.to_numeric(ninos["talla_cm"], errors="coerce")
    ninos["pesonino"] = pd.to_numeric(ninos["pesonino"], errors="coerce")

    es_si_oficio = ninos[COLUMNAS_OFICIOS].eq("Sí")
    tiene_dato_oficio = ninos[COLUMNAS_OFICIOS].notna()
    ninos["hace_oficio"] = np.where(
        tiene_dato_oficio.any(axis=1), es_si_oficio.any(axis=1).astype(float), np.nan
    )
    # `limpieza` por si sola tiene ~87% de "Sí" (verificado con value_counts,
    # la pregunta es "de los siguientes oficios, ¿cuáles hizo la semana
    # pasada?" -- barra baja: cualquier ayuda de aseo cuenta), lo que satura
    # el indicador binario ANY (93%-99.7% entre olas, poca varianza util
    # para el modelo). Se agrega tambien el CONTEO (0-8) para preservar
    # varianza como covariable de intensidad, no solo presencia/ausencia.
    ninos["n_oficios"] = np.where(tiene_dato_oficio.any(axis=1), es_si_oficio.sum(axis=1), np.nan)
    ninos["horas_oficio"] = pd.to_numeric(ninos["horas_oficio"], errors="coerce")

    for col_ola1, col_ola2 in COLUMNAS_ESTIMULACION_ARMONIZADAS.items():
        armonizada = ninos[col_ola1].where(ninos["ola"] == 1, ninos[col_ola2])
        armonizada_norm = normalizar_espacios(armonizada)
        ninos[f"_estim_{col_ola1}"] = armonizada_norm.map(FREQ_MAPA)

    cols_estim = [f"_estim_{c}" for c in COLUMNAS_ESTIMULACION_ARMONIZADAS]
    ninos["indice_estimulacion"] = ninos[cols_estim].mean(axis=1, skipna=True)
    ninos.loc[ninos[cols_estim].isna().all(axis=1), "indice_estimulacion"] = np.nan

    ninos["tvip_puntaje_directo"] = pd.to_numeric(ninos["puntuaciondirecta"], errors="coerce")

    quien_cuida_norm = normalizar_espacios(ninos["quien_cuida"]).str.lower()
    ninos["cuidado_terceros"] = np.where(
        ninos["quien_cuida"].notna(), (~quien_cuida_norm.isin(PADRES_TOKENS)).astype(float), np.nan
    )

    return ninos


def construir_variables_hogar(ninos: pd.DataFrame) -> pd.DataFrame:
    """Vectorizado: precomputar indicadores 0/1 por niño, un solo groupby().agg()."""
    es_salud = ninos["edad_a"].between(0, EDAD_MAX_SALUD)
    es_oficio_edad = ninos["edad_a"].between(EDAD_MIN_OFICIOS, EDAD_MAX_OFICIOS)

    ind = pd.DataFrame(index=ninos.index)
    ind["llave_c"] = ninos["llave_c"]

    ind["vacuna_si"] = ((ninos["vacunacion_basica_completa"] == 1) & es_salud).astype(float)
    ind["vacuna_valido"] = (ninos["vacunacion_basica_completa"].notna() & es_salud).astype(float)

    ind["control_si"] = ((ninos["asistio_control"] == "Sí") & es_salud).astype(float)
    ind["control_valido"] = (ninos["asistio_control"].notna() & es_salud).astype(float)

    ind["fiebreamarilla_si"] = ((ninos["fiebrea"] == "Sí") & es_salud).astype(float)
    ind["fiebreamarilla_valido"] = (ninos["fiebrea"].notna() & es_salud).astype(float)

    ind["asiste_si"] = ((ninos["asiste"] == "Sí") & es_salud).astype(float)
    ind["asiste_valido"] = (ninos["asiste"].notna() & es_salud).astype(float)

    ind["talla"] = ninos["talla_cm"].where(es_salud)
    ind["peso"] = ninos["pesonino"].where(es_salud)

    ind["oficio_si"] = ((ninos["hace_oficio"] == 1) & es_oficio_edad).astype(float)
    ind["oficio_valido"] = (ninos["hace_oficio"].notna() & es_oficio_edad).astype(float)
    ind["n_oficios"] = ninos["n_oficios"].where(es_oficio_edad)
    ind["horas_oficio"] = ninos["horas_oficio"].where(es_oficio_edad & (ninos["hace_oficio"] == 1))

    ind["trab_rem_si"] = ((ninos["trabajo"] == "Sí") & es_oficio_edad).astype(float)
    ind["trab_rem_valido"] = (ninos["trabajo"].notna() & es_oficio_edad).astype(float)

    ind["estim"] = ninos["indice_estimulacion"]
    ind["tvip"] = ninos["tvip_puntaje_directo"]

    ind["cuidado_terceros_si"] = ninos["cuidado_terceros"]

    agg = ind.groupby("llave_c").agg(
        n_vacuna=("vacuna_si", "sum"), n_vacuna_validos=("vacuna_valido", "sum"),
        n_control=("control_si", "sum"), n_control_validos=("control_valido", "sum"),
        n_fiebreamarilla=("fiebreamarilla_si", "sum"), n_fiebreamarilla_validos=("fiebreamarilla_valido", "sum"),
        n_asiste=("asiste_si", "sum"), n_asiste_validos=("asiste_valido", "sum"),
        talla_promedio=("talla", "mean"), peso_promedio=("peso", "mean"),
        n_oficio=("oficio_si", "sum"), n_oficio_validos=("oficio_valido", "sum"),
        n_oficios_promedio=("n_oficios", "mean"),
        horas_oficio_promedio=("horas_oficio", "mean"),
        n_trab_rem=("trab_rem_si", "sum"), n_trab_rem_validos=("trab_rem_valido", "sum"),
        indice_estimulacion_hogar=("estim", "mean"),
        tvip_promedio=("tvip", "mean"),
        cuidado_terceros_promedio=("cuidado_terceros_si", "mean"),
    )

    resultado = pd.DataFrame(index=agg.index)
    resultado["tasa_vacunacion_basica_hogar"] = agg["n_vacuna"] / agg["n_vacuna_validos"]
    resultado["tasa_control_crecimiento_hogar"] = agg["n_control"] / agg["n_control_validos"]
    resultado["tasa_vacuna_fiebreamarilla_hogar"] = agg["n_fiebreamarilla"] / agg["n_fiebreamarilla_validos"]
    resultado["tasa_asistencia_escolar_nino_hogar"] = agg["n_asiste"] / agg["n_asiste_validos"]
    resultado["talla_promedio_nino_hogar"] = agg["talla_promedio"]
    resultado["peso_promedio_nino_hogar"] = agg["peso_promedio"]
    resultado["pct_ninos_oficios_hogar"] = agg["n_oficio"] / agg["n_oficio_validos"]
    resultado["n_oficios_promedio_nino_hogar"] = agg["n_oficios_promedio"]
    resultado["horas_oficio_promedio_nino_hogar"] = agg["horas_oficio_promedio"]
    resultado["pct_ninos_trabajo_remunerado_hogar"] = agg["n_trab_rem"] / agg["n_trab_rem_validos"]
    resultado["indice_estimulacion_hogar_nino"] = agg["indice_estimulacion_hogar"]
    resultado["tvip_puntaje_directo_hogar"] = agg["tvip_promedio"]
    resultado["pct_ninos_cuidado_terceros_hogar"] = agg["cuidado_terceros_promedio"]

    for col, denom in [
        ("tasa_vacunacion_basica_hogar", "n_vacuna_validos"),
        ("tasa_control_crecimiento_hogar", "n_control_validos"),
        ("tasa_vacuna_fiebreamarilla_hogar", "n_fiebreamarilla_validos"),
        ("tasa_asistencia_escolar_nino_hogar", "n_asiste_validos"),
        ("pct_ninos_oficios_hogar", "n_oficio_validos"),
        ("pct_ninos_trabajo_remunerado_hogar", "n_trab_rem_validos"),
    ]:
        resultado.loc[agg[denom] == 0, col] = np.nan

    return resultado


def main() -> None:
    ninos = cargar_ninos()
    hogar = construir_variables_hogar(ninos)
    salida = hogar.reset_index().rename(columns={"llave_c": "llave_compuesta"})

    ids = ninos.drop_duplicates("llave_c")[
        ["llave_c", "ola", "zona", "consecutivo", "llave", "llave_n16"]
    ].rename(columns={"llave_c": "llave_compuesta"})
    salida = ids.merge(salida, on="llave_compuesta", how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print()
    print(salida.groupby("ola")[[
        "tasa_vacunacion_basica_hogar", "tasa_control_crecimiento_hogar",
        "tasa_vacuna_fiebreamarilla_hogar", "tasa_asistencia_escolar_nino_hogar",
        "talla_promedio_nino_hogar", "peso_promedio_nino_hogar",
        "pct_ninos_oficios_hogar", "n_oficios_promedio_nino_hogar", "horas_oficio_promedio_nino_hogar",
        "pct_ninos_trabajo_remunerado_hogar", "indice_estimulacion_hogar_nino",
        "tvip_puntaje_directo_hogar", "pct_ninos_cuidado_terceros_hogar",
    ]].mean())


if __name__ == "__main__":
    main()
