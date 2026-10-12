"""
Construccion del ingreso total mensual del hogar (ELCA 2010, 2013, 2016).

Parte de hogar_elca_longitudinal_clean.parquet. Las columnas de ingreso
(`ing_*`) llegan como texto, mezclando montos numericos con dos codigos
categoricos que hay que resolver ANTES de sumar:

  - "No Recibe" / "No recibe" -> cero explicito (el hogar no tiene esa
    fuente de ingreso; NO es informacion faltante).
  - "No informa" / "No sabe"  -> no-respuesta real (<15 casos por
    variable). Se deja como NaN.

Ingreso laboral (2013 y 2016): el cuestionario pasa de una sola pregunta
(`ing_trabajo`) a una version desagregada agropecuaria/no agropecuaria
(`ing_trabagr` + `ing_trabnoagr`). Se verificó que son mutuamente
excluyentes por hogar (ningun caso con ambas vías respondidas a la vez),
por lo que se combinan sin riesgo de doble conteo.

AUDITORÍA 2026-08-06: se revisaron TODAS las columnas ing_/act_/gan_/gto_
de hogar_elca_longitudinal_clean.parquet (69 en total) contra las listas de
este script, cruzando ademas los .tab crudos y los diccionarios especificos
de cada ola (no solo los "_unido"), para verificar que ninguna se estuviera
perdiendo en silencio y que las exclusiones fueran reales y no un cambio de
nombre entre olas. El detalle completo de la investigacion (fuentes
revisadas, tablas de cruce indicador/valor, verificacion triple del hueco
de ing_ayudas) se resume a continuacion. Resumen de lo que cambio:

  - ingreso_excepcional (NUEVO): construir_ingreso_excepcional() suma
    herencias, polizas, venta de inmueble/negocio/otros activos y "otros
    ingresos" no clasificados -- 6 preguntas retrospectivas de 12 meses
    que SI existen en las 3 olas (se habia asumido erroneamente que solo
    existian en 2010). Lo que cambia entre olas es el DISEÑO de la
    pregunta, no su existencia:
      * Rural 2010: valor pedido directamente a todos los hogares
        (respuesta ~100%, con muchos ceros explicitos).
      * Urbano 2010 y TODAS las olas 2013/2016: primero un filtro Si/No,
        el valor solo se pide si "Si" (por eso el valor sin el filtro
        parece "casi no existir" en 2013/2016 -- es que casi nadie llega
        a esa pregunta, no que la pregunta no este).
    Se verifico que el indicador coincide 100% con la presencia de valor
    en todos los casos filtrados antes de usar esta logica. Convertido a
    mensual (/12) porque es un valor de 12 meses, a diferencia del resto
    del ingreso que ya es mensual.
  - ingreso_ocasional (NUEVO): act_ocasional_vr, "ingreso ocasional",
    existe SOLO en 2013 y 2016 (confirmado ausente en los .tab crudos de
    2010). Se incluye solo para esas dos olas (queda NaN en 2010, no 0,
    para no fingir comparabilidad); se deja documentada la asimetria.
  - act_vtaanimdes (venta de animales de desecho): confirmado que solo
    existe en 2010 (ausente en los .tab crudos 2013/2016); se agrego a
    EXCLUIDAS_NO_COMPARABLES (antes se perdia sin documentar).
  - Las 14 variables act_*_vr restantes (bonos, cesantias, dinero,
    fondos, inversiones, roscas, seguros de vehiculo/vida/vivienda/
    maquinaria/cosechas) siguen excluidas: se confirmo en los .tab crudos
    que la pregunta de VALOR fue eliminada del cuestionario a partir de
    2013 (solo queda el indicador Si/No, sin monto que sumar).
  - ing_ayudas: se confirmo (tres fuentes independientes: columnas del
    .tab crudo rural 2010, diccionario especifico RHogar.pdf, comparacion
    de letra de item con UHogar.pdf) que la pregunta "e. Ayudas en
    dinero" NUNCA se hizo a hogares rurales en 2010 -- no es
    no-respuesta, es una pregunta que no existe para ese universo. No hay
    forma de recuperar el dato real, asi que sigue tratandose como 0
    implicito (no distinguible de "no recibe" bajo el supuesto general),
    pero ahora queda documentada la causa real en vez de mezclarla con el
    supuesto generico de No informa/No sabe.

Componentes que se mantienen EXCLUIDOS por no ser comparables entre olas:
ing_t_actagr, ing_t_actnag, ing_t_vtapastos, ing_t_vtaanimales,
act_vtaanimdes, gan_actagr, gan_actnag, gan_vtaanimales (mas gto_t_* del
lado del gasto) -- confirmado que estas preguntas retrospectivas de 12
meses solo existen en 2010, sin ningun equivalente en 2013/2016. Y las 14
act_*_vr listadas arriba, donde la pregunta de valor fue eliminada del
cuestionario despues de 2010.

Ademas del ingreso nominal, se agregan versiones deflactadas a pesos de 2010
(ingreso_total_hogar_real* / ingreso_percapita_hogar_real*), usando los dos
deflactores documentados en build_deflactor_ipc.py: IPC total nacional (un
solo valor por ola, no varia por zona) e IPC de ingresos bajos (uno distinto
por ola Y por zona Urbano/Rural). Por eso el merge con el deflactor se hace
por ["ola", "zona"] y no solo por "ola". Estas series NO se usan para
clasificar pobreza -- esa comparacion es nominal contra la LP del ano y zona
correspondientes, ver build_pobreza_monetaria.py -- son solo para analisis
descriptivo y de tendencia entre olas.

Output: data/processed/ingreso_hogar_elca_longitudinal.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ingreso_hogar_elca_longitudinal.parquet"
DOC_PATH = PROJECT_ROOT / "docs" / "variable_audit" / "ingreso_hogar_construccion.csv"
DEFLACTOR_PATH = PROJECT_ROOT / "data" / "processed" / "deflactor_ipc_elca.parquet"

ZERO_TOKENS = {"no recibe"}
MISSING_TOKENS = {"no informa", "no sabe"}
SI_TOKENS = {"si", "sí", "1"}
NO_TOKENS = {"no", "2"}

COMPONENTES_LABORALES = ["ing_trabajo", "ing_trabagr", "ing_trabnoagr"]
COMPONENTES_DIRECTOS = [
    "ing_pensiones",
    "ing_arriendos",
    "ing_intereses_div",
    "ing_ayudas",
    "ing_otros_nrem",
]

# Componentes "excepcionales" retrospectivos de 12 meses: {nombre_final: (indicador, valor)}.
# Se verifico (ver docstring del modulo) que el indicador coincide 100% con la
# presencia de valor en todos los casos donde se aplica el diseño de filtro.
COMPONENTES_EXCEPCIONALES = {
    "herencias": ("act_herencias", "act_herencias_vr"),
    "polizas": ("act_polizas", "act_polizas_vr"),
    "vtainm": ("act_vtainm", "act_vtainm_vr"),
    "vtaneg": ("act_vtaneg", "act_vtaneg_vr"),
    "otrosing": ("act_otrosing", "act_otrosing_vr"),
    "vtaotros": ("act_vtaotros", "act_vtaotros_vr"),
}
# En ola 1 urbana, el indicador de 'vtaotros' se llama act_vgtaotros (no
# act_vtaotros, que no existe para esa ola/zona). Se verifico que coincide
# 100% con la presencia de valor en act_vtaotros_vr para ese subconjunto.
INDICADOR_VTAOTROS_OLA1_URBANO = "act_vgtaotros"

EXCLUIDAS_NO_COMPARABLES = [
    "ing_t_actagr",
    "ing_t_actnag",
    "ing_t_vtapastos",
    "ing_t_vtaanimales",
    "act_vtaanimdes",
    "gan_actagr",
    "gan_actnag",
    "gan_vtaanimales",
    "gto_t_actagr",
    "gto_t_actnag",
    "gto_t_vtaanimales",
    "act_bonos_emp_vr",
    "act_bonos_gob_vr",
    "act_cesantias_vr",
    "act_dinero_vr",
    "act_dineroprestado_vr",
    "act_fondos_vr",
    "act_inversiones_vr",
    "act_roscas_vr",
    "act_segcosechas_vr",
    "act_segmaq_vr",
    "act_segotros_vr",
    "act_segveh_vr",
    "act_segvida_vr",
    "act_segviv_vr",
]


def parse_ingreso(serie: pd.Series) -> pd.Series:
    """Texto ELCA -> pesos. 'No recibe' = 0, 'No informa'/'No sabe' = NaN."""
    texto = serie.astype(str).str.strip().str.lower()
    numerico = pd.to_numeric(serie, errors="coerce")
    numerico[texto.isin(ZERO_TOKENS)] = 0.0
    numerico[texto.isin(MISSING_TOKENS)] = np.nan
    return numerico


def _es_si(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip().str.lower().isin(SI_TOKENS)


def _es_no(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip().str.lower().isin(NO_TOKENS)


def construir_ingreso_laboral(df: pd.DataFrame) -> pd.Series:
    trabajo = parse_ingreso(df["ing_trabajo"])
    agr = parse_ingreso(df["ing_trabagr"])
    noagr = parse_ingreso(df["ing_trabnoagr"])

    ambos = trabajo.notna() & (agr.notna() | noagr.notna())
    if ambos.any():
        raise ValueError(
            f"{ambos.sum()} hogares responden ing_trabajo Y ing_trabagr/noagr a la "
            "vez: revisar el supuesto de mutua exclusividad antes de sumar."
        )

    todo_faltante = trabajo.isna() & agr.isna() & noagr.isna()
    total = trabajo.fillna(0) + agr.fillna(0) + noagr.fillna(0)
    total[todo_faltante] = np.nan
    return total


def construir_ingreso_excepcional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Herencias, polizas, venta de inmueble/negocio/otros activos, otros
    ingresos no clasificados. Ver docstring del modulo para la
    justificacion completa; en resumen:

      - Rural 2010: el valor se pedia directamente a todos los hogares
        (sin filtro previo) -> se usa vr_* tal cual (via parse_ingreso).
      - Urbano 2010 y todas las filas de 2013/2016: hay un filtro Si/No
        antes del monto -> "No" se traduce a 0, "Si" usa vr_*.

    Todas son preguntas de 12 meses ("Durante los ultimos 12 meses...");
    se dividen entre 12 para expresarlas en la misma base mensual que el
    resto del ingreso.
    """
    resultado = pd.DataFrame(index=df.index)
    es_rural_ola1 = (df["ola"] == 1) & (df["zona"] == "Rural")
    es_urbano_ola1 = (df["ola"] == 1) & (df["zona"] == "Urbano")

    for nombre, (ind_col, vr_col) in COMPONENTES_EXCEPCIONALES.items():
        valor_directo = parse_ingreso(df[vr_col])

        indicador = df[ind_col].copy()
        if nombre == "vtaotros":
            # act_vtaotros no existe para ola 1; el indicador equivalente
            # en esa ola/zona es act_vgtaotros (solo aplica a urbano, que
            # es el unico subconjunto de ola 1 que usa el filtro Si/No).
            indicador.loc[es_urbano_ola1] = df.loc[es_urbano_ola1, INDICADOR_VTAOTROS_OLA1_URBANO]

        valor_filtrado = pd.Series(np.nan, index=df.index)
        si = _es_si(indicador)
        no = _es_no(indicador)
        valor_filtrado[si] = valor_directo[si]
        valor_filtrado[no] = 0.0

        valor_anual = valor_directo.where(es_rural_ola1, valor_filtrado)
        resultado[f"ingreso_{nombre}"] = valor_anual / 12.0

    return resultado


def construir_ingreso_ocasional(df: pd.DataFrame) -> pd.Series:
    """
    act_ocasional_vr ("ingreso ocasional") solo existe en 2013 y 2016
    (confirmado ausente en los .tab crudos de 2010: la columna ni siquiera
    existe para esa ola). Queda NaN para ola 1 de forma natural (no hay
    columna que parsear); no se fuerza a 0 para no simular que la pregunta
    existio en 2010. Pregunta de 12 meses -> se divide entre 12.
    """
    return parse_ingreso(df["act_ocasional_vr"]) / 12.0


def construir_ingreso_total(df: pd.DataFrame) -> pd.DataFrame:
    resultado = pd.DataFrame(index=df.index)
    resultado["ingreso_laboral"] = construir_ingreso_laboral(df)

    for col in COMPONENTES_DIRECTOS:
        resultado[f"ingreso_{col.replace('ing_', '')}"] = parse_ingreso(df[col])

    resultado = pd.concat(
        [resultado, construir_ingreso_excepcional(df)],
        axis=1,
    )
    resultado["ingreso_ocasional"] = construir_ingreso_ocasional(df)

    componentes = [c for c in resultado.columns]
    todo_faltante = resultado[componentes].isna().all(axis=1)
    resultado["ingreso_total_hogar"] = resultado[componentes].sum(axis=1, skipna=True)
    resultado.loc[todo_faltante, "ingreso_total_hogar"] = np.nan
    resultado["n_componentes_reportados"] = resultado[componentes].notna().sum(axis=1)

    return resultado


def documentar_exclusiones() -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    motivos = {
        "ing_t_actagr": "Solo existe en ola 2010 (bloque agropecuario retrospectivo de 12 "
            "meses); confirmado ausente en los .tab crudos de 2013/2016.",
        "ing_t_actnag": "Solo existe en ola 2010; confirmado ausente en los .tab crudos de "
            "2013/2016.",
        "ing_t_vtapastos": "Solo existe en ola 2010; confirmado ausente en los .tab crudos "
            "de 2013/2016.",
        "ing_t_vtaanimales": "Solo existe en ola 2010; confirmado ausente en los .tab crudos "
            "de 2013/2016.",
        "act_vtaanimdes": "Venta de animales de desecho: solo existe en ola 2010; "
            "confirmado ausente en los .tab crudos de 2013/2016.",
        "gan_actagr": "Solo existe en ola 2010; confirmado ausente en los .tab crudos de "
            "2013/2016.",
        "gan_actnag": "Solo existe en ola 2010; confirmado ausente en los .tab crudos de "
            "2013/2016.",
        "gan_vtaanimales": "Solo existe en ola 2010; confirmado ausente en los .tab crudos "
            "de 2013/2016.",
        "gto_t_actagr": "Solo existe en ola 2010; confirmado ausente en los .tab crudos de "
            "2013/2016.",
        "gto_t_actnag": "Solo existe en ola 2010; confirmado ausente en los .tab crudos de "
            "2013/2016.",
        "gto_t_vtaanimales": "Solo existe en ola 2010; confirmado ausente en los .tab crudos "
            "de 2013/2016.",
    }
    motivo_valor_eliminado = (
        "El indicador Si/No sigue existiendo en 2013/2016, pero la pregunta de VALOR "
        "($) fue eliminada del cuestionario a partir de 2013 (confirmado: la columna "
        "_vr no existe en los .tab crudos de 2013/2016, solo el indicador)."
    )
    for var in [
        "act_bonos_emp_vr", "act_bonos_gob_vr", "act_cesantias_vr", "act_dinero_vr",
        "act_dineroprestado_vr", "act_fondos_vr", "act_inversiones_vr", "act_roscas_vr",
        "act_segcosechas_vr", "act_segmaq_vr", "act_segotros_vr", "act_segveh_vr",
        "act_segvida_vr", "act_segviv_vr",
    ]:
        motivos[var] = motivo_valor_eliminado

    pd.DataFrame(
        {"variable": EXCLUIDAS_NO_COMPARABLES, "motivo": [motivos[v] for v in EXCLUIDAS_NO_COMPARABLES]}
    ).to_csv(DOC_PATH, index=False)


def agregar_series_reales(salida: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ingreso_{total_hogar,percapita_hogar}_real_{ipctotal,ipcbajos}.

    deflactor_ipc_total es nacional (no varia por zona); deflactor_ipc_ing_bajos
    es especifico de la zona del hogar (ver build_deflactor_ipc.py), por eso el
    merge es por ola Y zona.
    """
    deflactor = pd.read_parquet(DEFLACTOR_PATH)[
        ["ola", "zona", "deflactor_ipc_total", "deflactor_ipc_ing_bajos"]
    ]
    salida = salida.merge(deflactor, on=["ola", "zona"], how="left")

    for sufijo, columna in [
        ("ipctotal", "deflactor_ipc_total"),
        ("ipcbajos", "deflactor_ipc_ing_bajos"),
    ]:
        salida[f"ingreso_total_hogar_real_{sufijo}"] = (
            salida["ingreso_total_hogar"] / salida[columna]
        )
        salida[f"ingreso_percapita_hogar_real_{sufijo}"] = (
            salida["ingreso_percapita_hogar"] / salida[columna]
        )

    return salida.drop(columns=["deflactor_ipc_total", "deflactor_ipc_ing_bajos"])


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    ingreso = construir_ingreso_total(df)
    salida = pd.concat(
        [df[["consecutivo", "ola", "hogar", "hogar_n16", "llave", "llave_n16", "zona", "t_personas"]], ingreso],
        axis=1,
    )
    salida["ingreso_percapita_hogar"] = salida["ingreso_total_hogar"] / salida["t_personas"]
    salida = agregar_series_reales(salida)

    salida.to_parquet(OUTPUT_PATH, index=False)
    documentar_exclusiones()

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print(f"Exclusiones documentadas en: {DOC_PATH}")
    print()
    print(salida.groupby("ola")["ingreso_total_hogar"].describe())


if __name__ == "__main__":
    main()
