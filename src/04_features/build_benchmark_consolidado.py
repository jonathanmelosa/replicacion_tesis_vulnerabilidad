"""
Consolidacion final: une todos los parquets de features (Personas x9
bloques, Comunidades, Niños, Choques, Hogar) mas las variables monetarias
(ingreso, gasto, pobreza) en un unico dataset hogar-ola para el modelo
benchmark de prediccion de transicion a la pobreza.

Este script SOLO consolida (join ancho por llave de hogar). NO construye
todavia la matriz de entrenamiento train/test (eso requiere desplazar el
outcome de pobreza a la ola siguiente por hogar y filtrar a la poblacion
no-pobre en la ola base -- ver metodologia, punto 1-2 -- que queda como
paso siguiente, deliberadamente separado de la consolidacion).

Llave de union
------------------
Todos los parquets de features comparten el mismo esquema de identidad
entre olas ya usado en todo el proyecto: `consecutivo` (ola 1), `llave`
(ola 2), `llave_n16` (ola 3). Se construye `llave_compuesta` de forma
identica en cada archivo antes de unir (mismo criterio que cada script de
`build_*_hogar.py` individual). El panel ancla es
`hogar_elca_longitudinal_clean.parquet` (827 columnas RAW del modulo
Hogar, de las cuales aqui solo se usan las columnas de identidad -- las
827 columnas de contenido de Hogar no se traen a este consolidado; no
fueron auditadas en esta sesion y traerlas sin auditar violaria el mismo
estandar de rigor aplicado a Personas/Comunidades/Niños/Choques).

Variables monetarias: nominal (para el label de pobreza) vs. real (para
covariable de nivel)
--------------------------------------------------------------------------
`pobreza_monetaria_elca_longitudinal.parquet` tiene ingreso/gasto per
capita NOMINAL (comparado contra LP/LI nominal del mismo año -- la
comparacion correcta para determinar pobreza, ver auditoria de
`build_ingreso_hogar.py`/`build_gasto_hogar.py` mas arriba). Para una
covariable de NIVEL comparable entre olas (2010 vs 2013 vs 2016) hace
falta deflactar -- se usa la version REAL con el deflactor "ingresos
bajos" (`_real_ipcbajos`, de `ingreso_hogar_elca_longitudinal.parquet`/
`gasto_hogar_elca_longitudinal.parquet`) en vez de IPC total, porque es
metodologicamente mas cercano al gasto de los hogares en riesgo de
pobreza (ver "Deflactor IPC: metodologia de construccion completa" mas
arriba) -- consistente con como el DANE actualiza la LP oficial.

Se agrega tambien la BRECHA A LA LP (`ingreso_percapita_hogar / lp`,
`gasto_percapita_hogar / lp`, ambas en terminos NOMINALES del mismo año)
-- un ratio escala-invariante que no necesita deflactar porque compara al
hogar contra su propio umbral contemporaneo, siguiendo el enfoque de
vulnerabilidad a la pobreza de Chaudhuri, Jalan y Suryahadi (2002) ya
citado en la metodologia del benchmark. Es, junto con el ingreso real,
uno de los 2 candidatos a covariable principal del Modelo A (con
ingreso/gasto) vs. Modelo B (sin ingreso/gasto) que se comparara en el
benchmark.

Cobertura esperada por modulo (no todos los hogares tienen dato en todos
los modulos, por diseño -- NO se imputa nada en este script)
--------------------------------------------------------------------------
  - Hogar/Ingreso/Gasto/Pobreza, Personas (9 bloques), Comunidades,
    Choques: 27.932 filas (100% del panel).
  - Niños: 15.473 filas -- solo hogares con al menos un niño en el rango
    de edad relevante. El resto queda en NaN tras el join, correctamente
    (no significa "sin informacion", significa "no aplica: sin niños").
  - Comunidades: ver caveat de cobertura del join hogar->comunidad ya
    documentado en `build_comunidades_hogar.py` (8.1%/2.9%/3.6% de
    hogares sin comunidad emparejada, aun con ID valido).

Output: data/processed/benchmark_consolidado_elca_longitudinal.parquet
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED / "benchmark_consolidado_elca_longitudinal.parquet"

HOGAR_PATH = PROCESSED / "hogar_elca_longitudinal_clean.parquet"
POBREZA_PATH = PROCESSED / "pobreza_monetaria_elca_longitudinal.parquet"
INGRESO_PATH = PROCESSED / "ingreso_hogar_elca_longitudinal.parquet"
GASTO_PATH = PROCESSED / "gasto_hogar_elca_longitudinal.parquet"

FEATURE_PARQUETS = [
    "personas_hogar_elca_longitudinal.parquet",
    "educacion_ocupacion_hogar_elca_longitudinal.parquet",
    "salud_discapacidad_hogar_elca_longitudinal.parquet",
    "ahorro_capital_social_hogar_elca_longitudinal.parquet",
    "educacion_ocupacion_hogar_ext_elca_longitudinal.parquet",
    "becas_subsidios_hogar_elca_longitudinal.parquet",
    "salud_discapacidad_hogar_ext_elca_longitudinal.parquet",
    "personas_hogar_ext_elca_longitudinal.parquet",
    "participacion_civica_hogar_elca_longitudinal.parquet",
    "comunidades_hogar_elca_longitudinal.parquet",
    "ninos_hogar_elca_longitudinal.parquet",
    "choques_hogar_elca_longitudinal.parquet",
    "hogar_features_elca_longitudinal.parquet",
]

ID_COLS = {"consecutivo", "llave", "llave_n16", "hogar", "hogar_n16", "ola", "zona", "llave_compuesta"}


def llave_compuesta(df: pd.DataFrame) -> pd.Series:
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def cargar_base() -> pd.DataFrame:
    base = pd.read_parquet(HOGAR_PATH, columns=["consecutivo", "llave", "llave_n16", "ola", "zona"])
    base["llave_compuesta"] = llave_compuesta(base)
    if base["llave_compuesta"].isna().any():
        raise ValueError("llave_compuesta con nulos en la base ancla: revisar identidad de hogar.")
    if base.duplicated(subset=["llave_compuesta", "ola"]).any():
        raise ValueError("llave_compuesta+ola no es unico en la base ancla.")
    return base


def cargar_monetarias() -> pd.DataFrame:
    pobreza = pd.read_parquet(POBREZA_PATH)
    pobreza["llave_compuesta"] = llave_compuesta(pobreza)
    pobreza["brecha_lp_ingreso"] = pobreza["ingreso_percapita_hogar"] / pobreza["lp"]
    pobreza["brecha_lp_gasto"] = pobreza["gasto_percapita_hogar"] / pobreza["lp"]
    cols_pobreza = [
        "llave_compuesta", "ola",
        "ingreso_percapita_hogar", "gasto_percapita_hogar", "lp", "li",
        "pobre_ingreso", "pobre_extremo_ingreso", "pobre_gasto", "pobre_extremo_gasto",
        "concuerdan_ingreso_gasto", "brecha_lp_ingreso", "brecha_lp_gasto",
    ]
    pobreza = pobreza[cols_pobreza]

    ingreso = pd.read_parquet(INGRESO_PATH, columns=[
        "consecutivo", "llave", "llave_n16", "ola",
        "ingreso_percapita_hogar_real_ipcbajos",
    ])
    ingreso["llave_compuesta"] = llave_compuesta(ingreso)
    ingreso = ingreso[["llave_compuesta", "ola", "ingreso_percapita_hogar_real_ipcbajos"]]
    ingreso = ingreso.rename(columns={"ingreso_percapita_hogar_real_ipcbajos": "ingreso_percapita_hogar_real"})

    gasto = pd.read_parquet(GASTO_PATH, columns=[
        "consecutivo", "llave", "llave_n16", "ola",
        "gasto_percapita_hogar_real_ipcbajos",
    ])
    gasto["llave_compuesta"] = llave_compuesta(gasto)
    gasto = gasto[["llave_compuesta", "ola", "gasto_percapita_hogar_real_ipcbajos"]]
    gasto = gasto.rename(columns={"gasto_percapita_hogar_real_ipcbajos": "gasto_percapita_hogar_real"})

    monetarias = pobreza.merge(ingreso, on=["llave_compuesta", "ola"], how="left")
    monetarias = monetarias.merge(gasto, on=["llave_compuesta", "ola"], how="left")
    return monetarias


def cargar_feature(nombre_archivo: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / nombre_archivo)
    if "llave_compuesta" not in df.columns:
        df["llave_compuesta"] = llave_compuesta(df)
    cols_id_presentes = [c for c in df.columns if c in ID_COLS and c not in ("llave_compuesta", "ola")]
    df = df.drop(columns=cols_id_presentes)

    cols_contenido = [c for c in df.columns if c not in ("llave_compuesta", "ola")]
    solapadas = set(cols_contenido)
    return df, solapadas


def main() -> None:
    base = cargar_base()
    monetarias = cargar_monetarias()

    salida = base.merge(monetarias, on=["llave_compuesta", "ola"], how="left", validate="one_to_one")

    columnas_vistas: dict[str, str] = {}
    for nombre_archivo in FEATURE_PARQUETS:
        df, cols_contenido = cargar_feature(nombre_archivo)

        choques_nombre = cols_contenido & columnas_vistas.keys()
        if choques_nombre:
            raise ValueError(
                f"Colision de nombres de columna al agregar {nombre_archivo}: "
                f"{choques_nombre} ya vienen de {[columnas_vistas[c] for c in choques_nombre]}"
            )
        for c in cols_contenido:
            columnas_vistas[c] = nombre_archivo

        antes = len(salida)
        salida = salida.merge(df, on=["llave_compuesta", "ola"], how="left", validate="one_to_one")
        assert len(salida) == antes, f"El join de {nombre_archivo} cambio el numero de filas."

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH}")
    print(f"Dimensiones: {salida.shape[0]:,} filas x {salida.shape[1]:,} columnas")
    print()
    print("Filas por ola:")
    print(salida.groupby("ola").size())
    print()
    print("Cobertura (no-nulo) por archivo de origen, promedio de sus columnas:")
    for archivo in FEATURE_PARQUETS:
        cols = [c for c, origen in columnas_vistas.items() if origen == archivo]
        if cols:
            cov = salida[cols].notna().mean().mean()
            print(f"  {archivo:55s} {len(cols):3d} columnas, cobertura promedio {cov:.1%}")


if __name__ == "__main__":
    main()
