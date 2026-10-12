"""
panel_transicion.py
=====================================
Carga del panel de la matriz de transicion de pobreza y de las covariables
de la ola base, compartida por los scripts vigentes del paper (figura del
nucleo SHAP por grupo, Cuarto hallazgo, trayectorias de 3 olas).

Creado 2026-09-30: al retirar el perfil univariado del
paper, los scripts vigentes no deben depender de los scripts del perfil.
Las rutas, `COLS_ID` y las funciones de carga vivian en
`eda_transicion_covariables.py` (script del ranking univariado), y
`construir_panel_split` en `eda_perfil_split_trayectoria.py`; se movieron
aqui sin cambios y esos scripts ahora las importan de este modulo.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import (  # noqa: E402
    HOGAR_PATH,
    _llave_compuesta,
    cargar_pesos_muestrales,
    construir_matriz_transicion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POBREZA_PATH = PROJECT_ROOT / "data" / "processed" / "pobreza_monetaria_elca_longitudinal.parquet"
IPM_PATH = PROJECT_ROOT / "data" / "processed" / "ipm_multidimensional_elca_longitudinal.parquet"
GEO_PATH = PROJECT_ROOT / "data" / "processed" / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"
CONSOLIDADO_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_consolidado_elca_longitudinal.parquet"
INVENTARIO_PATH = PROJECT_ROOT / "outputs" / "tables" / "eda_variables_modelo" / "01_inventario_variables.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"

CATEGORIAS_ORDEN = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]

COLS_ID = {"consecutivo", "consecutivo_c", "ola", "llave", "llave_n16", "llave_compuesta"}


def cargar_peso_longitudinal_por_consecutivo(df: pd.DataFrame, ola_fin: int) -> pd.DataFrame:
    """
    Equivalente a `cargar_pesos_muestrales` pero para dataframes SIN
    `llave`/`llave_n16` (caso de `ipm_multidimensional_elca_longitudinal.parquet`
    -- a diferencia de pobreza monetaria, IPM no trae el desglose de
    sub-hogar por division). Se une por (consecutivo, ola) directamente en
    vez de por llave compuesta: es seguro porque `construir_matriz_transicion`
    excluye los hogares con `consecutivo` duplicado (division) ANTES de leer
    `peso_col`, asi que una fila duplicada temporal con el mismo peso no
    afecta el resultado.

    `ola_fin`: ola final de la transicion en curso (2 o 3) -- el factor de
    expansion se toma de esa ola, practica estandar para paneles
    longitudinales (ya usada por `construir_matriz_transicion`).
    """
    hogar = pd.read_parquet(HOGAR_PATH, columns=["consecutivo", "ola", "fexhog_2010"])
    hogar_ola_fin = hogar[hogar["ola"] == ola_fin].drop_duplicates(subset=["consecutivo", "ola"])
    df["peso_longitudinal"] = df.merge(
        hogar_ola_fin, on=["consecutivo", "ola"], how="left", validate="many_to_one"
    )["fexhog_2010"].to_numpy()
    return df


def _excluir_hogares_divididos(df: pd.DataFrame, ola_base: int) -> pd.DataFrame:
    """Excluye TODAS las filas de un `consecutivo` que aparece mas de una
    vez en la ola base (hogar dividido entre olas) -- misma regla exacta
    que usan los modelos predictivos ya publicados
    (`build_benchmark_train_test.py`, `construir_transicion`:
    `ini[~ini["consecutivo"].duplicated(keep=False)]`), aplicada aqui por
    igual a region, DMSP y covariables para que la Seccion 5.2 hable de la
    MISMA poblacion base que los modelos, no de una definida por una regla
    distinta. Ola 1 nunca tiene duplicados (no aplica)."""
    return df[~df["consecutivo"].duplicated(keep=False)]


def cargar_region_por_consecutivo(ola_base: int) -> pd.Series:
    """`region` de la ola BASE de la transicion (1=2010, 2=2013), indexada
    por `consecutivo` -- unica variable geografica valida en ELCA (ver
    docstring del modulo). Hogares divididos excluidos via
    `_excluir_hogares_divididos` (ver esa funcion)."""
    hogar = pd.read_parquet(HOGAR_PATH, columns=["consecutivo", "ola", "region"])
    hogar_ola = _excluir_hogares_divididos(hogar[hogar["ola"] == ola_base], ola_base)
    assert not hogar_ola["consecutivo"].duplicated().any(), f"consecutivo debe ser unico en ola {ola_base}"
    return hogar_ola.set_index("consecutivo")["region"]


def cargar_dmsp_por_consecutivo(anio_base: int) -> pd.Series:
    """dmsp_stable_lights del anio BASE de la transicion (2010 o 2013 --
    100% de cobertura en ambas, ver eda_variables_modelo.py; 2016 no
    aplica porque ninguna transicion lo usa como ola base, ver docstring
    del modulo), indexada por `consecutivo`. Hogares divididos excluidos
    via `_excluir_hogares_divididos` (438 consecutivos en 2013 -- ver esa
    funcion; verificado 2026-09-10 que ninguno de esos 438 sobrevive de
    todas formas el filtro de division de `construir_matriz_transicion`
    sobre la tabla de pobreza/IPM, asi que este filtro no mueve el
    resultado del panel de transicion, solo mantiene esta funcion
    consistente con las demas)."""
    geo = pd.read_parquet(GEO_PATH, columns=["consecutivo", "ola", "dmsp_stable_lights"])
    geo_base = _excluir_hogares_divididos(geo[geo["ola"] == anio_base], anio_base)
    assert not geo_base["consecutivo"].duplicated().any(), f"consecutivo debe ser unico en ola {anio_base}"
    return geo_base.set_index("consecutivo")["dmsp_stable_lights"]


def cargar_tipos_variables() -> dict:
    """variable -> tipo (Numerica/Categorica/Booleana), del inventario ya
    construido por eda_variables_modelo.py sobre este mismo consolidado."""
    inv = pd.read_csv(INVENTARIO_PATH)
    tipos = dict(zip(inv["variable"], inv["tipo"]))
    tipos["dmsp_stable_lights"] = "Numerica"
    tipos.setdefault("zona", "Categorica")
    return tipos


def cargar_covariables_ola_base(dmsp: pd.Series, ola_base: int) -> pd.DataFrame:
    """Universo de covariables candidatas: consolidado ML (ola BASE de la
    transicion) + DMSP (no vive en el consolidado, se agrega aparte).
    Indexado por consecutivo, 1 fila = 1 hogar -- mismo insumo (antes del
    filtro no-pobre-en-base) que usan los modelos de ML, para que el
    ranking hable de las MISMAS covariables que ya se evaluan en el
    modelado. Hogares divididos excluidos via `_excluir_hogares_divididos`
    (532 consecutivos en ola 2, 732 en ola 3 -- misma regla exacta que
    `build_benchmark_train_test.py` aplica al mismo consolidado)."""
    consolidado = pd.read_parquet(CONSOLIDADO_PATH)
    base = _excluir_hogares_divididos(consolidado[consolidado["ola"] == ola_base], ola_base)
    base = base.set_index("consecutivo")
    base = base.drop(columns=[c for c in COLS_ID if c in base.columns], errors="ignore")
    base["dmsp_stable_lights"] = dmsp
    return base


MAPA_TRAYECTORIA = {
    "Entra y sale (transitorio)": "Entra (transitorio)",
    "Entra y se queda (persistente)": "Entra (persistente)",
}
CATEGORIAS_ORDEN_SPLIT = [
    "Siempre pobre", "Sale de la pobreza",
    "Entra (transitorio)", "Entra (persistente)", "Nunca pobre",
]


def construir_panel_split() -> pd.DataFrame:
    pobreza = pd.read_parquet(POBREZA_PATH)
    llave = _llave_compuesta(pobreza)
    cargar_pesos_muestrales(pobreza, llave)
    resultado = construir_matriz_transicion(pobreza, 1, 2, col_pobre="pobre_ingreso", peso_col="peso_longitudinal")
    panel = resultado["panel_categorias"].copy()

    tray = pd.read_csv(TABLES_DIR / "trayectorias_3olas_monetaria_hogares.csv")
    tray_map = tray.set_index("consecutivo")["trayectoria"]

    mask_entra = panel["categoria"] == "Entra en pobreza"
    sub_tray = panel.loc[mask_entra, "consecutivo"].map(tray_map).map(MAPA_TRAYECTORIA)

    nueva_cat = panel["categoria"].astype(str).copy()
    nueva_cat[mask_entra] = sub_tray.fillna("Entra (sin dato 2016)")
    panel["categoria"] = nueva_cat
    return panel
