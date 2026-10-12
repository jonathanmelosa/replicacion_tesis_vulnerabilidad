"""
Clasificacion de pobreza monetaria del hogar (ELCA 2010, 2013, 2016),
metodologia oficial DANE.

Variable principal (oficial): pobreza por INGRESO
----------------------------------------------------
Siguiendo al DANE, un hogar es pobre (pobre extremo) si su ingreso NOMINAL
per capita del año de la ola es menor a la Linea de Pobreza -- LP -- (Linea
de pobreza extrema/indigencia -- LI) NOMINAL de ese mismo año y de su zona
(Urbano/Rural). La comparacion es siempre nominal contra nominal, dentro del
mismo año: el DANE NO deflacta para esta clasificacion (los deflactores de
build_deflactor_ipc.py se usan solo para series descriptivas, no para esta
comparacion). Ver docs/fuentes_dane/README.md para las fuentes exactas de
cada LP/LI y src/04_features/config_dane.py para los valores.

Variable secundaria (robustez): pobreza por GASTO
----------------------------------------------------
Además de la pobreza oficial por
ingreso, se construye una clasificacion paralela comparando el gasto NOMINAL
per capita del hogar (build_gasto_hogar.py) contra la misma LP/LI nominal de
su año y zona. No es una medida oficial del DANE -- el DANE mide pobreza
monetaria solo con ingreso -- se incluye como chequeo de robustez, util
cuando se sospecha subreporte de ingreso (frecuente en encuestas con
componente rural), y para contrastar cuantos hogares cambian de
clasificacion segun el bienestar que se use.

Salida: una fila por sub-hogar x ola con:
  ingreso_percapita_hogar, gasto_percapita_hogar   (insumos, en pesos nominales)
  lp, li                                            (linea aplicable, pesos nominales del año/zona)
  pobre_ingreso, pobre_extremo_ingreso              (1/0, medida oficial DANE)
  pobre_gasto, pobre_extremo_gasto                  (1/0, medida de robustez)
  concuerdan_ingreso_gasto                          (1 si pobre_ingreso == pobre_gasto)

Output: data/processed/pobreza_monetaria_elca_longitudinal.parquet
"""

from pathlib import Path

import pandas as pd

import config_dane as cfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGRESO_PATH = PROJECT_ROOT / "data" / "processed" / "ingreso_hogar_elca_longitudinal.parquet"
GASTO_PATH = PROJECT_ROOT / "data" / "processed" / "gasto_hogar_elca_longitudinal.parquet"
HOGAR_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pobreza_monetaria_elca_longitudinal.parquet"


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    """consecutivo (ola 1) / llave (ola 2) / llave_n16 (ola 3): 1 fila = 1 sub-hogar."""
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def cargar_insumos() -> pd.DataFrame:
    ingreso = pd.read_parquet(INGRESO_PATH)[
        ["consecutivo", "ola", "hogar", "hogar_n16", "llave", "llave_n16", "ingreso_percapita_hogar"]
    ]
    gasto = pd.read_parquet(GASTO_PATH)[
        ["consecutivo", "ola", "hogar", "hogar_n16", "llave", "llave_n16",
         "zona", "gasto_percapita_hogar"]
    ]

    llave_ingreso = _llave_compuesta(ingreso)
    llave_gasto = _llave_compuesta(gasto)

    if llave_ingreso.duplicated().any() or llave_ingreso.isna().any():
        raise ValueError("La llave compuesta de ingreso_hogar no es unica / tiene nulos.")

    ingreso_por_llave = ingreso.set_index(llave_ingreso)["ingreso_percapita_hogar"]
    gasto = gasto.copy()
    gasto["ingreso_percapita_hogar"] = llave_gasto.map(ingreso_por_llave)

    return gasto


def clasificar_pobreza(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ano"] = df["ola"].map(cfg.OLA_A_ANO)

    lineas = cfg.LINEAS_POBREZA
    df = df.merge(lineas[["ola", "zona", "lp", "li"]], on=["ola", "zona"], how="left")

    if df["lp"].isna().any():
        faltantes = df.loc[df["lp"].isna(), ["ola", "zona"]].drop_duplicates()
        raise ValueError(
            "Hogares sin linea de pobreza asignada (ola/zona sin match en "
            f"config_dane.LINEAS_POBREZA):\n{faltantes}"
        )

    df["pobre_ingreso"] = (df["ingreso_percapita_hogar"] < df["lp"]).astype("boolean")
    df["pobre_extremo_ingreso"] = (df["ingreso_percapita_hogar"] < df["li"]).astype("boolean")
    df.loc[df["ingreso_percapita_hogar"].isna(), ["pobre_ingreso", "pobre_extremo_ingreso"]] = pd.NA

    df["pobre_gasto"] = (df["gasto_percapita_hogar"] < df["lp"]).astype("boolean")
    df["pobre_extremo_gasto"] = (df["gasto_percapita_hogar"] < df["li"]).astype("boolean")
    df.loc[df["gasto_percapita_hogar"].isna(), ["pobre_gasto", "pobre_extremo_gasto"]] = pd.NA

    df["concuerdan_ingreso_gasto"] = (df["pobre_ingreso"] == df["pobre_gasto"]).astype("boolean")
    df.loc[df["pobre_ingreso"].isna() | df["pobre_gasto"].isna(), "concuerdan_ingreso_gasto"] = pd.NA

    return df


def main() -> None:
    insumos = cargar_insumos()
    salida = clasificar_pobreza(insumos)

    columnas = [
        "consecutivo", "ola", "ano", "hogar", "hogar_n16", "llave", "llave_n16", "zona",
        "ingreso_percapita_hogar", "gasto_percapita_hogar", "lp", "li",
        "pobre_ingreso", "pobre_extremo_ingreso",
        "pobre_gasto", "pobre_extremo_gasto",
        "concuerdan_ingreso_gasto",
    ]
    salida = salida[columnas]
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print()
    print("Incidencia de pobreza monetaria por ingreso (metodologia oficial DANE):")
    print(
        salida.groupby("ola")["pobre_ingreso"].apply(lambda s: s.dropna().astype(float).mean())
    )
    print()
    print("Incidencia de pobreza monetaria por gasto (medida de robustez, no oficial):")
    print(
        salida.groupby("ola")["pobre_gasto"].apply(lambda s: s.dropna().astype(float).mean())
    )
    print()
    print("Concordancia ingreso vs. gasto:")
    print(
        salida.groupby("ola")["concuerdan_ingreso_gasto"].apply(lambda s: s.dropna().astype(float).mean())
    )


if __name__ == "__main__":
    main()
