"""
Desagregaciones de la pobreza monetaria (ELCA 2010, 2013, 2016).

Parte de pobreza_monetaria_elca_longitudinal.parquet (build_pobreza_monetaria.py)
y agrega cuatro tipos de analisis:

1. Indicadores FGT (Foster-Greer-Thorbecke): ademas de la incidencia (P0,
   ya calculada como pobre_ingreso/pobre_gasto), se agregan la BRECHA (P1)
   y la SEVERIDAD (P2) de la pobreza, exactamente como las reporta el DANE
   en sus boletines (ver docs/fuentes_dane/). Formula estandar:

       P_alpha = (1/N) * sum_i [ ((LP - y_i) / LP)^alpha * 1(y_i < LP) ]

   alpha=0 -> incidencia (proporcion de pobres)
   alpha=1 -> brecha (que tan lejos esta el ingreso promedio de los pobres
              respecto a la LP, como fraccion de la LP)
   alpha=2 -> severidad (igual que la brecha, pero pondera mas a los
              hogares mas alejados de la LP -- sensible a la desigualdad
              entre los pobres)

   Se calculan para ingreso Y gasto, contra LP y LI, por ola.

2. Desagregacion geografica: por zona (Urbano/Rural, ya en el pipeline) y
   por las 8 macro-regiones propias de la ELCA (Atlantica, Pacifica,
   Oriental, etc. -- columna `region` de hogar_elca_longitudinal_clean).
   Es el nivel geografico mas fino que los datos permiten.

3. Desagregacion por caracteristicas del jefe de hogar y del hogar,
   replicando el tipo de tablas que publica el DANE (bol_pobreza_16.pdf,
   Tablas 3-9): sexo, grupo de edad y nivel educativo del jefe de hogar
   (identificado via parentesco="Jefe de hogar"/"Jefe(a)" en
   personas_elca_longitudinal.parquet -- se verifico que hay exactamente
   un jefe por hogar en las 3 olas) y numero de niños menores de 12 años
   en el hogar.

4. Matrices de transicion de pobreza entre olas consecutivas (2010->2013,
   2013->2016), siguiendo la metodologia de Lopez-Calva y Ortiz-Juarez
   (2014), "A Vulnerability Approach to the Definition of the Middle
   Class", Tabla 3: tabla cruzada de estado de pobreza en el periodo
   inicial (filas) contra el periodo final (columnas), en porcentajes de
   fila (cada fila suma 100%). De esa tabla se deriva la clasificacion en
   4 categorias: nunca pobre, siempre pobre, sale de la pobreza (pobre ->
   no pobre) y entra en pobreza (no pobre -> pobre).

   Emparejamiento de hogares entre olas: se usa `consecutivo` (el
   identificador de hogar de 2010, que se mantiene estable en las 3 olas).
   Importante: los hogares que se DIVIDEN entre una
   ola y la siguiente se EXCLUYEN de la matriz de
   transicion -- solo se usan los matches 1 a 1. El numero de hogares
   excluidos por division se reporta aparte.

AUDITORIA 2026-08-08 -- ponderacion y robustez de las transiciones:

  - Pesos muestrales: se agregan `peso_transversal` (fexhog/fexhog_2013/
    fexhog_2010 segun ola, verificados contra los diccionarios especificos
    de cada ola -- HR254/HU253/HU324/HU250 -- son los factores de
    expansion "Unidad" a nivel de hogar) y `peso_longitudinal` (fexhog_2010,
    el factor longitudinal ancorado a la ola 1, presente en olas 2 y 3).
    LIMITACION: ola 3 (2016) no tiene un peso transversal propio en los
    datos consolidados (solo existe fhog_2016, en otra escala, no la
    variante "Factor Unidad"); para esa ola se usa el peso longitudinal
    como mejor aproximacion disponible, lo cual subrepresenta hogares que
    entraron a la muestra despues de 2010. Todas las tablas FGT se generan
    en version sin ponderar y ponderada (sufijo `_ponderado.csv`).
  - `pobre_sin_excepcional`: reclasificacion de pobreza usando
    ingreso_total_hogar menos los 6 componentes de ingreso_excepcional
    (herencias, polizas, venta de inmueble/negocio/otros, otros ingresos),
    para separar transiciones de pobreza "estructurales" de choques
    puntuales de liquidez de 12 meses que se prorratean como flujo mensual.
  - `pobre_sin_ayudas`: reclasificacion excluyendo ingreso_ayudas de las 3
    olas, para chequear cuanto del patron de transicion rural 2010->2013
    depende del hueco de cobertura de esa pregunta en rural 2010.
  - `pobre_banda_baja`/`pobre_banda_alta`: reclasificacion con LP*0.9 y
    LP*1.1, para dimensionar cuanta transicion observada es sensible a
    error de medicion cerca del umbral oficial.

Todas las tablas de salida son CSV en outputs/tables/pobreza/, no parquet:
son resultados descriptivos para el documento de tesis, no un insumo para
otro script del pipeline. Las graficas correspondientes se guardan como PNG
en outputs/figures/pobreza/ -- ver graficar_resultados() al final del
archivo. Paleta y reglas de color siguen la skill de dataviz del proyecto
(colores categoricos en orden fijo, validados para daltonismo; nunca doble
eje; etiquetas directas en vez de solo leyenda cuando el espacio alcanza).

Con `--solo-figuras-robustez`, el script no abre los datos por hogar: redibuja
las tres figuras del Anexo de robustez de las transiciones (07, 08 y 09) a
partir de las tablas agregadas ya guardadas en outputs/tables/pobreza/.

    python build_pobreza_desagregaciones.py --solo-figuras-robustez
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POBREZA_PATH = PROJECT_ROOT / "data" / "processed" / "pobreza_monetaria_elca_longitudinal.parquet"
HOGAR_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal.parquet"
INGRESO_PATH = PROJECT_ROOT / "data" / "processed" / "ingreso_hogar_elca_longitudinal.parquet"

# 6 componentes de ingreso_excepcional (ver build_ingreso_hogar.py): eventos
# retrospectivos de 12 meses (venta de inmueble/negocio, herencias, polizas,
# otros ingresos no clasificados), prorrateados como flujo mensual. Se
# excluyen para la version de robustez "sin_excepcional" de las matrices de
# transicion (ver docstring del modulo).
COMPONENTES_EXCEPCIONALES_COLS = [
    "ingreso_herencias", "ingreso_polizas", "ingreso_vtainm",
    "ingreso_vtaneg", "ingreso_otrosing", "ingreso_vtaotros",
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "pobreza"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "pobreza"

# Paleta categorica validada (dataviz skill, references/palette.md), modo claro.
# Orden fijo: nunca se reasigna por categoria, siempre en este orden de slot.
PALETA = {
    "azul": "#2a78d6",
    "naranja": "#eb6834",
    "aguamarina": "#1baf7a",
    "amarillo": "#eda100",
    "magenta": "#e87ba4",
    "verde": "#008300",
    "violeta": "#4a3aa7",
    "rojo": "#e34948",
}
INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# Estilo comun de las figuras del documento (2026-10-01): misma paleta,
# letra en puntos de impresion y sin titulo dentro de la imagen.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import estilo_figuras as ef  # noqa: E402

ef.aplicar_estilo()

JEFE_TOKENS = {"jefe de hogar", "jefe(a)"}

NIVEL_EDUC_GRUPOS = [
    ("preescolar", "Ninguno/preescolar"),
    ("ninguno", "Ninguno/preescolar"),
    ("primaria", "Primaria"),
    ("secundaria", "Secundaria"),
    ("t.cnico", "Tecnica"),
    ("tecnol.gico", "Tecnologica"),
    ("universitario", "Universitaria"),
    ("posgrado", "Posgrado"),
]


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    """consecutivo (ola 1) / llave (ola 2) / llave_n16 (ola 3): 1 fila = 1 sub-hogar."""
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def _agrupar_nivel_educ(serie: pd.Series) -> pd.Series:
    """Colapsa las variantes de codificacion (acentos rotos '?'/'�') a categorias limpias."""
    texto = serie.astype(str).str.strip().str.lower().str.replace("�", ".", regex=False)
    texto = texto.where(~texto.isin(["nan", "no informa"]), np.nan)
    resultado = pd.Series(np.nan, index=serie.index, dtype=object)
    for patron, etiqueta in NIVEL_EDUC_GRUPOS:
        resultado[texto.str.contains(patron, na=False, regex=True)] = etiqueta
    return resultado


def _grupo_edad(edad: pd.Series) -> pd.Series:
    edad = pd.to_numeric(edad, errors="coerce")
    return pd.cut(
        edad,
        bins=[0, 25, 35, 45, 55, 65, 200],
        labels=["Hasta 25", "26 a 35", "36 a 45", "46 a 55", "56 a 65", "Mayor a 65"],
        right=True,
        include_lowest=True,
    )


def cargar_pesos_muestrales(df: pd.DataFrame, llave: pd.Series) -> pd.DataFrame:
    """
    Agrega a `df` (in place, devuelve el mismo df) `peso_transversal` y
    `peso_longitudinal` (ver docstring del modulo). Generico sobre cualquier
    dataframe con columnas consecutivo/ola/llave/llave_n16 (usado por
    pobreza monetaria via `cargar_pesos_ingreso_robustez`, y por IPM en
    `eda_transicion_covariables.py` -- misma logica, sin las reclasificaciones
    de robustez de ingreso que solo aplican a pobreza monetaria).

    Caveat observado en validacion (2026-09-10, no bloquea, queda como
    observacion): `peso_longitudinal` (`fexhog_2010`) se documenta como
    "anclado a la ola 1, constante en las olas 2 y 3", pero para 43 de
    6,911 hogares (0.6%, verificado 1 a 1, sin division de hogar) el valor
    de `fexhog_2010` difiere ligeramente entre su fila de ola 2 y su fila
    de ola 3 en `HOGAR_PATH` (diferencia maxima ~3.5 unidades) -- verificado
    que la discrepancia ya esta en la fuente (HOGAR_PATH), no es un
    artefacto de esta funcion ni de `_llave_compuesta` (0 colisiones de
    llave en 27,932 filas). Probable recalibracion/redondeo distinto del
    factor de expansion entre los archivos de pesos publicados por ola de
    la ELCA. Efecto en cualquier analisis ponderado que use
    `peso_longitudinal`: inmaterial (0.6% de los hogares, magnitud minima
    frente al peso tipico), pero cualquier resultado se calcula con el
    valor de la fila de la ola que se este usando en ese momento (ola 2 o
    ola 3 segun el caso), no con un valor "canonico" unico por hogar.
    """
    hogar = pd.read_parquet(HOGAR_PATH)[
        ["consecutivo", "ola", "llave", "llave_n16", "fexhog", "fexhog_2013", "fexhog_2010"]
    ]
    llave_hogar = _llave_compuesta(hogar)
    hogar_por_llave = hogar.set_index(llave_hogar)[["fexhog", "fexhog_2013", "fexhog_2010"]]
    df["fexhog"] = llave.map(hogar_por_llave["fexhog"])
    df["fexhog_2013"] = llave.map(hogar_por_llave["fexhog_2013"])
    df["fexhog_2010"] = llave.map(hogar_por_llave["fexhog_2010"])
    # Transversal: fexhog (ola 1) / fexhog_2013 (ola 2) / fexhog_2010 (ola 3,
    # unico disponible -- ver limitacion documentada en el docstring del modulo).
    df["peso_transversal"] = df["fexhog"].where(
        df["ola"] == 1, df["fexhog_2013"].where(df["ola"] == 2, df["fexhog_2010"])
    )
    # Longitudinal: fexhog_2010, ancorado a la ola 1 (NaN en ola 1 misma, por diseno).
    df["peso_longitudinal"] = df["fexhog_2010"]
    df.drop(columns=["fexhog", "fexhog_2013", "fexhog_2010"], inplace=True)
    return df


def cargar_pesos_ingreso_robustez(pobreza: pd.DataFrame, llave_pobreza: pd.Series) -> pd.DataFrame:
    """
    Agrega a `pobreza` (in place, devuelve el mismo df):
      - peso_transversal / peso_longitudinal (ver `cargar_pesos_muestrales`)
      - ingreso_percapita_sin_excepcional / _sin_ayudas y sus reclasificaciones
        pobre_sin_excepcional / pobre_sin_ayudas
      - pobre_banda_baja / pobre_banda_alta (LP*0.9 / LP*1.1)
    """
    cargar_pesos_muestrales(pobreza, llave_pobreza)

    ingreso = pd.read_parquet(INGRESO_PATH)
    llave_ingreso = _llave_compuesta(ingreso)
    total_original_nulo = ingreso["ingreso_total_hogar"].isna()

    ingreso_sin_excepcional = (
        ingreso["ingreso_total_hogar"] - ingreso[COMPONENTES_EXCEPCIONALES_COLS].fillna(0).sum(axis=1)
    )
    ingreso_sin_ayudas = ingreso["ingreso_total_hogar"] - ingreso["ingreso_ayudas"].fillna(0)
    percapita_sin_excepcional = (ingreso_sin_excepcional / ingreso["t_personas"]).where(~total_original_nulo)
    percapita_sin_ayudas = (ingreso_sin_ayudas / ingreso["t_personas"]).where(~total_original_nulo)

    robustez = pd.DataFrame(
        {
            "ingreso_percapita_sin_excepcional": percapita_sin_excepcional.to_numpy(),
            "ingreso_percapita_sin_ayudas": percapita_sin_ayudas.to_numpy(),
        },
        index=llave_ingreso,
    )
    pobreza["ingreso_percapita_sin_excepcional"] = llave_pobreza.map(robustez["ingreso_percapita_sin_excepcional"])
    pobreza["ingreso_percapita_sin_ayudas"] = llave_pobreza.map(robustez["ingreso_percapita_sin_ayudas"])

    for col_ingreso, col_pobre in [
        ("ingreso_percapita_sin_excepcional", "pobre_sin_excepcional"),
        ("ingreso_percapita_sin_ayudas", "pobre_sin_ayudas"),
    ]:
        pobreza[col_pobre] = (pobreza[col_ingreso] < pobreza["lp"]).astype("boolean")
        pobreza.loc[pobreza[col_ingreso].isna(), col_pobre] = pd.NA

    pobreza["pobre_banda_baja"] = (pobreza["ingreso_percapita_hogar"] < pobreza["lp"] * 0.9).astype("boolean")
    pobreza["pobre_banda_alta"] = (pobreza["ingreso_percapita_hogar"] < pobreza["lp"] * 1.1).astype("boolean")
    pobreza.loc[
        pobreza["ingreso_percapita_hogar"].isna(), ["pobre_banda_baja", "pobre_banda_alta"]
    ] = pd.NA

    return pobreza


def cargar_pobreza_con_covariables() -> pd.DataFrame:
    pobreza = pd.read_parquet(POBREZA_PATH)

    hogar = pd.read_parquet(HOGAR_PATH)[
        ["consecutivo", "ola", "llave", "llave_n16", "region", "t_personas"]
    ]
    llave_pobreza = _llave_compuesta(pobreza)
    llave_hogar = _llave_compuesta(hogar)
    hogar_por_llave = hogar.set_index(llave_hogar)[["region", "t_personas"]]
    pobreza["region"] = llave_pobreza.map(hogar_por_llave["region"])
    pobreza["t_personas"] = llave_pobreza.map(hogar_por_llave["t_personas"])

    pobreza = cargar_pesos_ingreso_robustez(pobreza, llave_pobreza)

    personas = pd.read_parquet(PERSONAS_PATH)
    llave_personas = _llave_compuesta(personas)

    es_jefe = personas["parentesco"].astype(str).str.strip().str.lower().isin(JEFE_TOKENS)
    jefes = personas[es_jefe].set_index(llave_personas[es_jefe])
    if jefes.index.duplicated().any():
        raise ValueError("Mas de un jefe de hogar por sub-hogar: revisar supuesto de unicidad.")

    sexo_jefe = jefes["sexo"].astype(str).str.strip().str.title()
    sexo_jefe = sexo_jefe.where(sexo_jefe.isin(["Hombre", "Mujer"]), np.nan)
    pobreza["sexo_jefe"] = llave_pobreza.map(sexo_jefe)
    pobreza["edad_jefe"] = llave_pobreza.map(pd.to_numeric(jefes["edad"], errors="coerce"))
    pobreza["grupo_edad_jefe"] = _grupo_edad(pobreza["edad_jefe"])
    pobreza["nivel_educ_jefe"] = llave_pobreza.map(_agrupar_nivel_educ(jefes["nivel_educ"]))

    edad_personas = pd.to_numeric(personas["edad"], errors="coerce")
    ninos = personas.assign(llave=llave_personas, es_nino=edad_personas < 12)
    t_ninos = ninos.groupby("llave")["es_nino"].sum()
    pobreza["t_ninos_12"] = llave_pobreza.map(t_ninos).fillna(0).astype(int)

    return pobreza


def fgt(bienestar: pd.Series, linea: pd.Series, alpha: int, peso: pd.Series = None) -> float:
    """
    P_alpha de Foster-Greer-Thorbecke. bienestar/linea en las mismas unidades
    (pesos nominales). Si se pasa `peso` (factor de expansion), se calcula la
    version ponderada (representativa de poblacion) en vez del promedio
    muestral simple.
    """
    valido = bienestar.notna() & linea.notna()
    if peso is not None:
        valido = valido & peso.notna()
    b, l = bienestar[valido], linea[valido]
    if len(b) == 0:
        return np.nan
    brecha_relativa = ((l - b) / l).clip(lower=0)
    valores = (brecha_relativa > 0).astype(float) if alpha == 0 else brecha_relativa**alpha
    if peso is None:
        return valores.mean()
    return np.average(valores, weights=peso[valido])


def tabla_fgt(df: pd.DataFrame, groupby_cols: list, peso_col: str = None) -> pd.DataFrame:
    """
    P0/P1/P2 para ingreso y gasto, contra LP y LI, por los grupos indicados.
    `peso_col`: nombre de la columna de factor de expansion a usar (None =
    muestral sin ponderar).
    """
    combinaciones = [
        ("ingreso", "lp", "ingreso_percapita_hogar"),
        ("ingreso", "li", "ingreso_percapita_hogar"),
        ("gasto", "lp", "gasto_percapita_hogar"),
        ("gasto", "li", "gasto_percapita_hogar"),
    ]
    filas = []
    for claves, grupo in df.groupby(groupby_cols, dropna=False):
        claves = claves if isinstance(claves, tuple) else (claves,)
        peso = grupo[peso_col] if peso_col else None
        for medida, linea_nombre, columna_bienestar in combinaciones:
            fila = dict(zip(groupby_cols, claves))
            fila["medida"] = medida
            fila["linea"] = linea_nombre
            fila["n_hogares"] = grupo[columna_bienestar].notna().sum()
            for alpha in (0, 1, 2):
                fila[f"P{alpha}"] = fgt(grupo[columna_bienestar], grupo[linea_nombre], alpha, peso=peso)
            filas.append(fila)
    return pd.DataFrame(filas)


def construir_matriz_transicion(
    pobreza: pd.DataFrame,
    ola_ini: int,
    ola_fin: int,
    col_pobre: str = "pobre_ingreso",
    peso_col: str = None,
) -> dict:
    """
    Matriz de transicion pobre/no pobre entre dos olas consecutivas, siguiendo
    Lopez-Calva y Ortiz-Juarez (2014), Tabla 3. Solo hogares con match 1 a 1
    por `consecutivo` (decision confirmada: excluir hogares que se dividieron).

    `col_pobre`: variable de clasificacion a usar (permite reusar esta funcion
    para las versiones de robustez sin_excepcional / sin_ayudas / banda).
    `peso_col`: si se pasa, la matriz y la distribucion de categorias se
    ponderan por esa columna (factor de expansion de la OLA FINAL del
    periodo, practica estandar para paneles longitudinales) en vez de usar
    conteos muestrales simples.
    """
    ini = pobreza[pobreza["ola"] == ola_ini][["consecutivo", col_pobre]].dropna(subset=[col_pobre])
    cols_fin = ["consecutivo", col_pobre] + ([peso_col] if peso_col else [])
    fin = pobreza[pobreza["ola"] == ola_fin][cols_fin].dropna(subset=[col_pobre])

    ini_unicos = ini[~ini["consecutivo"].duplicated(keep=False)]
    fin_unicos = fin[~fin["consecutivo"].duplicated(keep=False)]
    n_excluidos_division = (
        ini["consecutivo"].nunique() - ini_unicos["consecutivo"].nunique()
        + fin["consecutivo"].nunique() - fin_unicos["consecutivo"].nunique()
    )

    panel = ini_unicos.merge(
        fin_unicos, on="consecutivo", suffixes=(f"_{ola_ini}", f"_{ola_fin}")
    )
    col_ini, col_fin = f"{col_pobre}_{ola_ini}", f"{col_pobre}_{ola_fin}"
    panel[col_ini] = panel[col_ini].map({True: "Pobre", False: "No pobre"})
    panel[col_fin] = panel[col_fin].map({True: "Pobre", False: "No pobre"})

    if peso_col:
        matriz_peso = panel.pivot_table(
            index=col_ini, columns=col_fin, values=peso_col, aggfunc="sum", fill_value=0
        )
        matriz_pct = (matriz_peso.div(matriz_peso.sum(axis=1), axis=0) * 100).round(1)
    else:
        matriz_pct = (
            pd.crosstab(panel[col_ini], panel[col_fin], normalize="index") * 100
        ).round(1)
    matriz_n = pd.crosstab(panel[col_ini], panel[col_fin])

    def clasificar(fila):
        if fila[col_ini] == "No pobre" and fila[col_fin] == "No pobre":
            return "Nunca pobre"
        if fila[col_ini] == "Pobre" and fila[col_fin] == "Pobre":
            return "Siempre pobre"
        if fila[col_ini] == "Pobre" and fila[col_fin] == "No pobre":
            return "Sale de la pobreza"
        return "Entra en pobreza"

    panel["categoria"] = panel.apply(clasificar, axis=1)
    if peso_col:
        pesos_por_categoria = panel.groupby("categoria")[peso_col].sum()
        distribucion_categorias = (pesos_por_categoria / pesos_por_categoria.sum() * 100).round(1)
    else:
        distribucion_categorias = (panel["categoria"].value_counts(normalize=True) * 100).round(1)
    # conteo SIEMPRE sin ponderar (n de hogares del panel emparejado) -- el
    # panel absoluto de graf_transiciones_generico usa esto incluso cuando
    # `peso_col` esta presente, porque "cuantos hogares" no admite una
    # version ponderada coherente (el factor de expansion ya se usa para
    # el panel de porcentajes).
    distribucion_categorias_n = panel["categoria"].value_counts()

    cols_panel = ["consecutivo", "categoria"] + ([peso_col] if peso_col else [])
    return {
        "ola_inicial": ola_ini,
        "ola_final": ola_fin,
        "n_hogares_panel": len(panel),
        "n_excluidos_por_division": n_excluidos_division,
        "matriz_porcentaje_fila": matriz_pct,
        "matriz_conteo": matriz_n,
        "distribucion_categorias": distribucion_categorias,
        "distribucion_categorias_n": distribucion_categorias_n,
        # 1 fila = 1 hogar del panel emparejado, con su categoria de transicion
        # (consecutivo = id estable de ola 1) -- usado por
        # eda_transicion_covariables.py para caracterizar los 4 grupos con
        # covariables externas (region, DMSP, etc.) que no viven en `pobreza`.
        "panel_categorias": panel[cols_panel].copy(),
    }


def tabla_atricion(pobreza: pd.DataFrame) -> pd.DataFrame:
    """
    Atricion TOTAL del panel entre olas consecutivas: hogares de la ola
    inicial cuyo `consecutivo` no aparece en absoluto en la ola final (ni
    como match 1 a 1 ni como division) -- complementa el conteo de
    "excluidos por division" que ya reportan las matrices de transicion.
    """
    ids = {
        ola: set(pobreza.loc[pobreza["ola"] == ola, "consecutivo"].dropna().unique())
        for ola in (1, 2, 3)
    }
    filas = []
    for ola_ini, ola_fin in [(1, 2), (2, 3)]:
        inicial, final = ids[ola_ini], ids[ola_fin]
        n_perdidos = len(inicial - final)
        filas.append(
            {
                "ola_inicial": ola_ini,
                "ola_final": ola_fin,
                "n_hogares_ola_inicial": len(inicial),
                "n_encontrados_ola_final": len(inicial & final),
                "n_atricion_total": n_perdidos,
                "pct_atricion": round(100 * n_perdidos / len(inicial), 1),
            }
        )
    return pd.DataFrame(filas)


ANO_POR_OLA = {1: 2010, 2: 2013, 3: 2016}


def _guardar(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    ruta = FIGURES_DIR / nombre
    ef.guardar(fig, ruta)
    print(f"Guardado: {ruta}")


def graf_incidencia_series(tabla_ola: pd.DataFrame) -> None:
    """Incidencia (P0) de pobreza por ingreso vs. gasto, contra la LP, en el tiempo."""
    sub = tabla_ola[tabla_ola["linea"] == "lp"]
    fig, ax = plt.subplots(figsize=(ef.ancho(0.75), 3.0))
    colores = {"ingreso": PALETA["azul"], "gasto": PALETA["naranja"]}
    for medida, color in colores.items():
        serie = sub[sub["medida"] == medida].sort_values("ola")
        anos = serie["ola"].map(ANO_POR_OLA)
        ax.plot(anos, serie["P0"] * 100, marker="o", markersize=6, linewidth=2, color=color, label=medida.capitalize())
        for x, y in zip(anos, serie["P0"] * 100):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=ef.TAM_LETRA_PEQUENA, color=INK_SECUNDARIO)
    ax.set_ylim(0, max(sub["P0"]) * 100 * 1.25)
    ax.set_xticks(list(ANO_POR_OLA.values()))
    ax.set_ylabel("Incidencia de pobreza (%)")
    ax.legend(loc="upper right")
    _guardar(fig, "01_incidencia_ingreso_vs_gasto.png")


def graf_fgt_trio(tabla_ola: pd.DataFrame) -> None:
    """P0/P1/P2 del ingreso contra la LP, por ola -- incidencia, brecha y severidad."""
    sub = tabla_ola[(tabla_ola["medida"] == "ingreso") & (tabla_ola["linea"] == "lp")].sort_values("ola")
    anos = sub["ola"].map(ANO_POR_OLA).astype(str)
    indicadores = [("P0", "Incidencia", PALETA["azul"]), ("P1", "Brecha", PALETA["aguamarina"]), ("P2", "Severidad", PALETA["violeta"])]

    x = np.arange(len(anos))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(ef.ancho(0.8), 3.0))
    for i, (col, etiqueta, color) in enumerate(indicadores):
        valores = sub[col].to_numpy() * 100
        barras = ax.bar(x + (i - 1) * ancho, valores, width=ancho * 0.9, color=color, label=etiqueta)
        ax.bar_label(barras, fmt="%.1f", padding=2, fontsize=ef.TAM_LETRA_PEQUENA, color=INK_SECUNDARIO)
    ax.set_xticks(x)
    ax.set_xticklabels(anos)
    ax.set_ylabel("% de la línea de pobreza")
    ax.legend(loc="upper right")
    _guardar(fig, "02_fgt_incidencia_brecha_severidad.png")


def graf_incidencia_zona(tabla_zona: pd.DataFrame) -> None:
    """Incidencia (P0) de pobreza por ingreso, contra la LP, por zona Urbano/Rural."""
    sub = tabla_zona[(tabla_zona["medida"] == "ingreso") & (tabla_zona["linea"] == "lp")]
    fig, ax = plt.subplots(figsize=(6, 4))
    colores = {"Urbano": PALETA["azul"], "Rural": PALETA["naranja"]}
    for zona, color in colores.items():
        serie = sub[sub["zona"] == zona].sort_values("ola")
        anos = serie["ola"].map(ANO_POR_OLA)
        ax.plot(anos, serie["P0"] * 100, marker="o", markersize=6, linewidth=2, color=color, label=zona)
        for x, y in zip(anos, serie["P0"] * 100):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=INK_SECUNDARIO)
    ax.set_ylim(0, max(sub["P0"]) * 100 * 1.25)
    ax.set_xticks(list(ANO_POR_OLA.values()))
    ax.set_ylabel("Incidencia de pobreza (%)")
    ax.set_title("Pobreza monetaria por ingreso, Urbano vs. Rural (P0, vs. LP)")
    ax.legend(frameon=False, loc="upper right")
    _guardar(fig, "03_incidencia_zona.png")


def graf_incidencia_region(tabla_region: pd.DataFrame, ola: int) -> None:
    """Incidencia (P0) de pobreza por ingreso, contra la LP, por region, en una ola dada."""
    sub = tabla_region[
        (tabla_region["medida"] == "ingreso")
        & (tabla_region["linea"] == "lp")
        & (tabla_region["ola"] == ola)
        & tabla_region["region"].notna()
    ].sort_values("P0")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    barras = ax.barh(sub["region"], sub["P0"] * 100, color=PALETA["azul"])
    ax.bar_label(barras, fmt="%.1f%%", padding=4, fontsize=9, color=INK_SECUNDARIO)
    ax.set_xlabel("Incidencia de pobreza (%)")
    ax.set_title(f"Pobreza monetaria por ingreso, por region -- {ANO_POR_OLA[ola]} (P0, vs. LP)")
    ax.set_xlim(0, sub["P0"].max() * 100 * 1.2)
    _guardar(fig, f"04_incidencia_region_{ANO_POR_OLA[ola]}.png")


def graf_incidencia_jefe(tabla_sexo: pd.DataFrame, tabla_edad: pd.DataFrame, tabla_educ: pd.DataFrame, ola: int) -> None:
    """3 paneles: incidencia por sexo, grupo de edad y nivel educativo del jefe de hogar."""
    filtro = lambda t, col: t[  # noqa: E731
        (t["medida"] == "ingreso") & (t["linea"] == "lp") & (t["ola"] == ola) & t[col].notna()
    ]

    orden_educ = ["Ninguno/preescolar", "Primaria", "Secundaria", "Tecnica", "Tecnologica", "Universitaria", "Posgrado"]
    orden_edad = ["Hasta 25", "26 a 35", "36 a 45", "46 a 55", "56 a 65", "Mayor a 65"]

    sexo = filtro(tabla_sexo, "sexo_jefe").set_index("sexo_jefe").reindex(["Mujer", "Hombre"])
    edad = filtro(tabla_edad, "grupo_edad_jefe").set_index("grupo_edad_jefe").reindex(orden_edad)
    educ = filtro(tabla_educ, "nivel_educ_jefe").set_index("nivel_educ_jefe").reindex(orden_educ)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    paneles = [
        (axes[0], sexo, "Sexo del jefe de hogar", False),
        (axes[1], edad, "Edad del jefe de hogar", True),
        (axes[2], educ, "Nivel educativo del jefe de hogar", True),
    ]
    for ax, tabla, titulo, rotar in paneles:
        valores = tabla["P0"] * 100
        barras = ax.bar(tabla.index.astype(str), valores, color=PALETA["azul"])
        ax.bar_label(barras, fmt="%.0f", padding=2, fontsize=8, color=INK_SECUNDARIO)
        ax.set_title(titulo, fontsize=10)
        if rotar:
            ax.tick_params(axis="x", rotation=40)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    axes[0].set_ylabel("Incidencia de pobreza (%)")
    fig.suptitle(f"Pobreza monetaria por ingreso segun jefe de hogar -- {ANO_POR_OLA[ola]} (P0, vs. LP)")
    _guardar(fig, f"05_incidencia_jefe_hogar_{ANO_POR_OLA[ola]}.png")


def graf_transiciones_generico(resumenes: list, titulo: str, nombre_archivo: str) -> None:
    """Distribucion de categorias de transicion (nunca/siempre pobre,
    entra/sale) por periodo -- version generica, reusada tanto para pobreza
    monetaria (`graf_transiciones`, wrapper mas abajo) como para IPM
    (`build_matriz_transicion_ipm.py`, que la importa directamente en vez
    de duplicar esta logica de graficacion).

    UN solo grafico, barras en escala ABSOLUTA (numero de hogares del panel
    emparejado, `distribucion_categorias_n` de `construir_matriz_transicion`)
    -- el alto de cada barra ya refleja el tamano real del panel en cada
    periodo (relevante al comparar monetaria, con paneles de ~3,000-3,200
    hogares en el ejercicio de modelos, contra el panel completo de
    transicion, ~6,900-8,200). Cada segmento se etiqueta con AMBOS valores,
    `n (pct%)`, para poder leer el porcentaje directamente sobre la misma
    barra sin necesitar un panel aparte
    (2026-09-02: ``una gráfica de absolutos que se pueda leer en la misma
    los porcentajes'')."""
    categorias = ["Nunca pobre", "Sale de la pobreza", "Entra en pobreza", "Siempre pobre"]
    colores = [PALETA["azul"], PALETA["aguamarina"], PALETA["naranja"], PALETA["rojo"]]

    periodos = [f"{ANO_POR_OLA[r['ola_inicial']]}-{ANO_POR_OLA[r['ola_final']]}" for r in resumenes]
    matriz_n = np.array([
        [r["distribucion_categorias_n"].get(cat, 0) for cat in categorias] for r in resumenes
    ])
    matriz_pct = np.array([
        [r["distribucion_categorias"].get(cat, 0.0) for cat in categorias] for r in resumenes
    ])

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    base = np.zeros(len(periodos))
    umbral = 0.03 * matriz_n.sum(axis=1).max()
    for i, (cat, color) in enumerate(zip(categorias, colores)):
        valores_n = matriz_n[:, i]
        valores_pct = matriz_pct[:, i]
        ax.bar(periodos, valores_n, bottom=base, color=color, label=cat, width=0.55)
        for j, (n_val, pct_val) in enumerate(zip(valores_n, valores_pct)):
            if n_val > umbral:
                ax.text(j, base[j] + n_val / 2, f"{int(n_val):,}\n({pct_val:.1f}%)",
                        ha="center", va="center", fontsize=8, color="white")
        base += valores_n

    ax.set_ylabel("Número de hogares (panel emparejado)")
    ax.set_title(titulo)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
    _guardar(fig, nombre_archivo)


def graf_transiciones(resumenes: list) -> None:
    """Wrapper de `graf_transiciones_generico` para pobreza monetaria --
    mantiene el nombre/firma originales para no romper `graficar_resultados`."""
    graf_transiciones_generico(
        resumenes,
        titulo="Matriz de transicion de pobreza monetaria entre olas\n(Lopez-Calva y Ortiz-Juarez, 2014)",
        nombre_archivo="06_transiciones_pobreza.png",
    )


def graf_robustez_transiciones(resumenes: list, sin_excepcional: list, sin_ayudas: dict) -> None:
    """
    Compara la distribucion de categorias de transicion bajo 4 especificaciones:
    linea base (muestral), ponderada por factor de expansion, ingreso sin
    componentes excepcionales, e ingreso sin ingreso_ayudas (solo 2010->2013).
    """
    categorias = ["Nunca pobre", "Sale de la pobreza", "Entra en pobreza", "Siempre pobre"]
    periodos = [f"{ANO_POR_OLA[r['ola_inicial']]}-{ANO_POR_OLA[r['ola_final']]}" for r in resumenes]

    fig, axes = plt.subplots(1, len(periodos), figsize=(ef.ancho(0.95), 3.2), sharey=True)
    if len(periodos) == 1:
        axes = [axes]

    for i, (ax, resumen, sin_exc) in enumerate(zip(axes, resumenes, sin_excepcional)):
        especificaciones = [("Base\n(muestral)", resumen["distribucion_categorias"])]
        especificaciones.append(("Sin\nexcepcional", sin_exc["distribucion_categorias"]))
        if resumen["ola_inicial"] == 1 and resumen["ola_final"] == 2 and sin_ayudas is not None:
            especificaciones.append(("Sin\ning. ayudas", sin_ayudas["distribucion_categorias"]))

        x = np.arange(len(especificaciones))
        base = np.zeros(len(especificaciones))
        colores = [PALETA["azul"], PALETA["aguamarina"], PALETA["naranja"], PALETA["rojo"]]
        for cat, color in zip(categorias, colores):
            valores = np.array([dist.get(cat, 0.0) for _, dist in especificaciones])
            ax.bar(x, valores, bottom=base, color=color, label=cat, width=0.6)
            for j, v in enumerate(valores):
                if v > 3:
                    ax.text(j, base[j] + v / 2, f"{v:.1f}", ha="center", va="center",
                            fontsize=ef.TAM_LETRA_PEQUENA - 1, color="white")
            base += valores
        ax.set_xticks(x)
        ax.set_xticklabels([nombre for nombre, _ in especificaciones])
        ax.set_title(periodos[i].replace("-", f" {ef.FLECHA} "))
        ax.set_ylim(0, 100)
        if i == 0:
            ax.set_ylabel("% de hogares (panel emparejado)")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4)
    _guardar(fig, "07_robustez_transiciones.png")


def graf_sensibilidad_banda(sensibilidad_por_periodo: dict) -> None:
    """Sensibilidad de la distribucion de categorias a una banda +-10% alrededor de la LP."""
    categorias = ["Nunca pobre", "Sale de la pobreza", "Entra en pobreza", "Siempre pobre"]
    periodos = list(sensibilidad_por_periodo.keys())

    fig, axes = plt.subplots(1, len(periodos), figsize=(ef.ancho(0.95), 3.2), sharey=True)
    if len(periodos) == 1:
        axes = [axes]

    columnas = [("banda_lp90", "LP −10%"), ("baseline_lp", "LP"), ("banda_lp110", "LP +10%")]
    for i, (ax, periodo) in enumerate(zip(axes, periodos)):
        tabla = sensibilidad_por_periodo[periodo].set_index("categoria")
        x = np.arange(len(columnas))
        base = np.zeros(len(columnas))
        colores = [PALETA["azul"], PALETA["aguamarina"], PALETA["naranja"], PALETA["rojo"]]
        for cat, color in zip(categorias, colores):
            valores = np.array([tabla.loc[cat, col] if cat in tabla.index else 0.0 for col, _ in columnas])
            ax.bar(x, valores, bottom=base, color=color, label=cat, width=0.6)
            for j, v in enumerate(valores):
                if v > 3:
                    ax.text(j, base[j] + v / 2, f"{v:.1f}", ha="center", va="center",
                            fontsize=ef.TAM_LETRA_PEQUENA - 1, color="white")
            base += valores
        ax.set_xticks(x)
        ax.set_xticklabels([etiqueta for _, etiqueta in columnas])
        ax.set_title(str(periodo).replace("-", f" {ef.FLECHA} "))
        ax.set_ylim(0, 100)
        if i == 0:
            ax.set_ylabel("% de hogares (panel emparejado)")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4)
    _guardar(fig, "08_sensibilidad_banda.png")


def graf_atricion(atricion: pd.DataFrame) -> None:
    """Atricion total del panel (hogares que no aparecen en absoluto en la ola siguiente)."""
    periodos = [
        f"{ANO_POR_OLA[fila.ola_inicial]} {ef.FLECHA} {ANO_POR_OLA[fila.ola_final]}"
        for fila in atricion.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(ef.ancho(0.55), 2.6))
    barras = ax.bar(periodos, atricion["pct_atricion"], color=ef.COLOR_SERIE, width=0.5)
    ax.bar_label(barras, fmt="%.1f%%", padding=3, fontsize=ef.TAM_LETRA_PEQUENA, color=INK_SECUNDARIO)
    ax.set_ylabel("Atrición del panel (%)")
    ax.set_ylim(0, max(atricion["pct_atricion"]) * 1.4)
    _guardar(fig, "09_atricion_panel.png")


def graficar_resultados(
    tabla_ola: pd.DataFrame,
    tabla_zona: pd.DataFrame,
    tabla_region: pd.DataFrame,
    tabla_sexo: pd.DataFrame,
    tabla_edad: pd.DataFrame,
    tabla_educ: pd.DataFrame,
    resumen_transiciones: list,
    resumen_sin_excepcional: list,
    resumen_sin_ayudas: dict,
    sensibilidad_por_periodo: dict,
    atricion: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    graf_incidencia_series(tabla_ola)
    graf_fgt_trio(tabla_ola)
    graf_incidencia_zona(tabla_zona)
    graf_incidencia_region(tabla_region, ola=3)
    graf_incidencia_jefe(tabla_sexo, tabla_edad, tabla_educ, ola=3)
    graf_transiciones(resumen_transiciones)
    graf_robustez_transiciones(resumen_transiciones, resumen_sin_excepcional, resumen_sin_ayudas)
    graf_sensibilidad_banda(sensibilidad_por_periodo)
    graf_atricion(atricion)


def graficar_robustez_desde_csv() -> None:
    """Redibuja las tres figuras del Anexo de robustez de las transiciones
    (07_robustez_transiciones, 08_sensibilidad_banda, 09_atricion_panel) solo
    a partir de las tablas agregadas que `main` guarda en OUTPUT_DIR, sin
    abrir los datos por hogar. Para quien regenere las figuras del paper con
    los resultados agregados publicados."""
    def distribucion(nombre: str) -> pd.Series:
        return pd.read_csv(OUTPUT_DIR / nombre, index_col="categoria")["porcentaje"]

    resumenes, sin_excepcional, sensibilidad_por_periodo = [], [], {}
    for ola_ini, ola_fin in [(1, 2), (2, 3)]:
        sufijo = f"{ola_ini}_a_{ola_fin}"
        periodo = f"{ANO_POR_OLA[ola_ini]}-{ANO_POR_OLA[ola_fin]}"
        resumenes.append({"ola_inicial": ola_ini, "ola_final": ola_fin,
                          "distribucion_categorias": distribucion(f"transicion_categorias_ola{sufijo}.csv")})
        sin_excepcional.append({"distribucion_categorias": distribucion(f"transicion_categorias_sin_excepcional_ola{sufijo}.csv")})
        sensibilidad_por_periodo[periodo] = pd.read_csv(OUTPUT_DIR / f"transicion_sensibilidad_banda_ola{sufijo}.csv")
    sin_ayudas = {"distribucion_categorias": distribucion("transicion_categorias_sin_ayudas_ola1_a_2.csv")}
    atricion = pd.read_csv(OUTPUT_DIR / "atricion_panel.csv")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    graf_robustez_transiciones(resumenes, sin_excepcional, sin_ayudas)
    graf_sensibilidad_banda(sensibilidad_por_periodo)
    graf_atricion(atricion)


def main() -> None:
    if "--solo-figuras-robustez" in sys.argv:
        graficar_robustez_desde_csv()
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pobreza = cargar_pobreza_con_covariables()

    # 1) FGT por ola (linea base) + 2) por zona/region + 3) por jefe de hogar/ninos
    especificaciones = {
        "fgt_por_ola.csv": ["ola"],
        "fgt_por_ola_zona.csv": ["ola", "zona"],
        "fgt_por_ola_region.csv": ["ola", "region"],
        "fgt_por_ola_sexo_jefe.csv": ["ola", "sexo_jefe"],
        "fgt_por_ola_grupo_edad_jefe.csv": ["ola", "grupo_edad_jefe"],
        "fgt_por_ola_nivel_educ_jefe.csv": ["ola", "nivel_educ_jefe"],
        "fgt_por_ola_t_ninos_12.csv": ["ola", "t_ninos_12"],
    }
    tablas = {}
    for nombre_archivo, cols in especificaciones.items():
        tabla = tabla_fgt(pobreza, cols)
        tabla.to_csv(OUTPUT_DIR / nombre_archivo, index=False)
        tablas[nombre_archivo] = tabla
        print(f"Guardado: {OUTPUT_DIR / nombre_archivo} ({len(tabla)} filas)")

        # Version ponderada (factor de expansion transversal por ola). Ver
        # docstring del modulo para la limitacion de ola 3 (2016).
        nombre_pond = nombre_archivo.replace(".csv", "_ponderado.csv")
        tabla_pond = tabla_fgt(pobreza, cols, peso_col="peso_transversal")
        tabla_pond.to_csv(OUTPUT_DIR / nombre_pond, index=False)
        print(f"Guardado: {OUTPUT_DIR / nombre_pond} ({len(tabla_pond)} filas)")

    # 4) Matrices de transicion
    resumen_transiciones = []
    resumen_sin_excepcional = []
    sensibilidad_por_periodo = {}
    for ola_ini, ola_fin in [(1, 2), (2, 3)]:
        sufijo = f"{ola_ini}_a_{ola_fin}"
        periodo = f"{ANO_POR_OLA[ola_ini]}-{ANO_POR_OLA[ola_fin]}"

        resultado = construir_matriz_transicion(pobreza, ola_ini, ola_fin)
        resultado["matriz_porcentaje_fila"].to_csv(OUTPUT_DIR / f"transicion_pct_ola{sufijo}.csv")
        resultado["matriz_conteo"].to_csv(OUTPUT_DIR / f"transicion_conteo_ola{sufijo}.csv")
        resultado["distribucion_categorias"].to_csv(
            OUTPUT_DIR / f"transicion_categorias_ola{sufijo}.csv", header=["porcentaje"]
        )
        print(
            f"\nTransicion ola {ola_ini} -> ola {ola_fin} "
            f"(n={resultado['n_hogares_panel']}, excluidos por division={resultado['n_excluidos_por_division']}):"
        )
        print(resultado["matriz_porcentaje_fila"])
        print(resultado["distribucion_categorias"])
        resumen_transiciones.append(resultado)

        # -- Robustez: ponderada por factor de expansion longitudinal --
        resultado_pond = construir_matriz_transicion(
            pobreza, ola_ini, ola_fin, peso_col="peso_longitudinal"
        )
        resultado_pond["matriz_porcentaje_fila"].to_csv(
            OUTPUT_DIR / f"transicion_pct_ponderado_ola{sufijo}.csv"
        )
        resultado_pond["distribucion_categorias"].to_csv(
            OUTPUT_DIR / f"transicion_categorias_ponderado_ola{sufijo}.csv", header=["porcentaje"]
        )

        # -- Robustez: ingreso sin componentes excepcionales --
        resultado_exc = construir_matriz_transicion(
            pobreza, ola_ini, ola_fin, col_pobre="pobre_sin_excepcional"
        )
        resultado_exc["distribucion_categorias"].to_csv(
            OUTPUT_DIR / f"transicion_categorias_sin_excepcional_ola{sufijo}.csv", header=["porcentaje"]
        )
        resumen_sin_excepcional.append(resultado_exc)

        # -- Sensibilidad a banda +-10% alrededor de la LP --
        resultado_b90 = construir_matriz_transicion(pobreza, ola_ini, ola_fin, col_pobre="pobre_banda_baja")
        resultado_b110 = construir_matriz_transicion(pobreza, ola_ini, ola_fin, col_pobre="pobre_banda_alta")
        sensibilidad = pd.DataFrame(
            {
                "baseline_lp": resultado["distribucion_categorias"],
                "banda_lp90": resultado_b90["distribucion_categorias"],
                "banda_lp110": resultado_b110["distribucion_categorias"],
            }
        )
        sensibilidad.to_csv(OUTPUT_DIR / f"transicion_sensibilidad_banda_ola{sufijo}.csv")
        sensibilidad_por_periodo[periodo] = sensibilidad.reset_index().rename(columns={"index": "categoria"})
        print(f"\nSensibilidad a banda +-10% de LP, ola {ola_ini} -> ola {ola_fin}:")
        print(sensibilidad)

    # -- Robustez: ingreso sin ingreso_ayudas (hueco de cobertura rural 2010), solo 2010->2013 --
    resultado_ayudas = construir_matriz_transicion(pobreza, 1, 2, col_pobre="pobre_sin_ayudas")
    resultado_ayudas["distribucion_categorias"].to_csv(
        OUTPUT_DIR / "transicion_categorias_sin_ayudas_ola1_a_2.csv", header=["porcentaje"]
    )
    print("\nRobustez sin ingreso_ayudas, ola 1 -> ola 2:")
    print(resultado_ayudas["distribucion_categorias"])

    # 5) Atricion total del panel
    atricion = tabla_atricion(pobreza)
    atricion.to_csv(OUTPUT_DIR / "atricion_panel.csv", index=False)
    print("\nAtricion total del panel (hogares que no aparecen en absoluto en la ola siguiente):")
    print(atricion)

    print(f"\nTodas las tablas guardadas en: {OUTPUT_DIR}")

    graficar_resultados(
        tabla_ola=tablas["fgt_por_ola.csv"],
        tabla_zona=tablas["fgt_por_ola_zona.csv"],
        tabla_region=tablas["fgt_por_ola_region.csv"],
        tabla_sexo=tablas["fgt_por_ola_sexo_jefe.csv"],
        tabla_edad=tablas["fgt_por_ola_grupo_edad_jefe.csv"],
        tabla_educ=tablas["fgt_por_ola_nivel_educ_jefe.csv"],
        resumen_transiciones=resumen_transiciones,
        resumen_sin_excepcional=resumen_sin_excepcional,
        resumen_sin_ayudas=resultado_ayudas,
        sensibilidad_por_periodo=sensibilidad_por_periodo,
        atricion=atricion,
    )
    print(f"Todas las graficas guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
