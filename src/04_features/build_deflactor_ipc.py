"""
Deflactores de precios para hacer comparables en terminos reales el ingreso y
el gasto del hogar entre las olas ELCA 2010, 2013 y 2016.

Se construyen DOS deflactores independientes, documentados por separado
porque miden cosas distintas:

  deflactor_ipc_total       : IPC total nacional, sin distincion por nivel de
                               ingreso del hogar NI por zona. Es el deflactor
                               general estandar en la literatura para hacer
                               comparable el poder adquisitivo entre anos,
                               independientemente de si el hogar es pobre o
                               no, o vive en zona urbana o rural. No se
                               encontro una serie de IPC total oficial
                               desagregada por zona urbano/rural descargable
                               (el IPC del DANE se desagrega por ciudad, no
                               por dominio urbano/rural); por eso este
                               deflactor queda nacional unico, a diferencia
                               del de ingresos bajos.
  deflactor_ipc_ing_bajos   : el mismo indice que el DANE usa para actualizar
                               la linea de pobreza (LP) cada año, calculado
                               POR ZONA (Urbano/Rural) ademas del nacional.
                               Tiene sentido si se quiere consistencia exacta
                               con la metodologia de pobreza, dado que la
                               mayoria de hogares ELCA estan cerca o por
                               debajo de la LP de su zona.

Precision temporal: se usa el PROMEDIO ANUAL de cada ola (2010, 2013, 2016),
no el mes exacto de entrevista de cada hogar.

Como se calcula cada deflactor
--------------------------------
1) IPC total nacional
   El archivo fuente (docs/fuentes_dane/IPC_Variacion.xls) trae variaciones
   PORCENTUALES mes a mes, no el indice en niveles. Se reconstruye el indice
   encadenando esas variaciones desde una base arbitraria (100 en diciembre
   de 2008, mes de referencia de la metodologia IPC-08 vigente sin quiebre
   entre 2010 y 2016). El promedio de los 12 meses de cada año es el nivel
   anual de ese año. El deflactor es la razon de ese nivel frente al año base.

2) IPC ingresos bajos, nacional y por zona (derivado)
   No existe un archivo historico descargable unico con el IPC especifico
   para el grupo de "ingresos bajos", ni desagregado por zona. Ese vinculo
   si esta documentado de forma explicita en los boletines de pobreza del
   DANE ("linea base ENIG 2006-2007, actualizadas por IPC total de ingresos
   bajos"): por construccion, la razon entre los valores oficiales de LP de
   dos años (nacional, o de una misma zona) ES la inflacion acumulada de
   ingresos bajos entre esos dos años para ese dominio. Por eso este
   deflactor se deriva directamente de config_dane.LP_NACIONAL (version
   nacional) y de config_dane.LINEAS_POBREZA (version por zona) en lugar de
   construirse a partir de una serie de IPC independiente. Ver
   docs/fuentes_dane/README.md seccion 2 para el detalle de esta decision.

   La version por zona no es identica a la nacional: la LP urbana y la LP
   rural no siempre crecen al mismo ritmo de un año a otro (aunque en el
   tramo 2010-2016 la diferencia es pequeña, ver tabla impresa al final de
   este script), asi que usar el deflactor de la zona propia del hogar es
   mas preciso que usar siempre el nacional.

Año base: 2010 (primera ola). "Real" en las salidas de este pipeline
significa "en pesos de 2010" salvo que se indique lo contrario.

Uso: estos deflactores NO se usan para clasificar pobreza (esa comparacion
es nominal, ingreso/gasto del hogar del año X contra la LP/LI nominal de ese
mismo año y zona, igual que lo hace el DANE oficialmente). Se usan solo para
construir series de ingreso y gasto en terminos reales, para analisis
descriptivo y de tendencia entre olas.

Output: data/processed/deflactor_ipc_elca.parquet
Una fila por (ola, zona) -- 3 olas x 2 zonas = 6 filas. ipc_total_promedio_anual
y deflactor_ipc_total se repiten para ambas zonas de una misma ola porque son
nacionales; deflactor_ipc_ing_bajos_zona varia por zona.
"""

from pathlib import Path
import pandas as pd
import config_dane as cfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "deflactor_ipc_elca.parquet"

ANO_BASE = 2010

MESES_ORDEN = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


def _indice_ipc_total_mensual() -> pd.DataFrame:
    """
    Reconstruye el indice del IPC total nacional en niveles, encadenando las
    variaciones porcentuales mensuales oficiales desde 100 en diciembre de
    2008. Retorna una fila por (ano, mes) con el nivel del indice a cierre de
    ese mes.
    """
    crudo = pd.read_excel(cfg.IPC_VARIACION_XLS, sheet_name="VariaNal", header=None)

    fila_anos = crudo.iloc[13, 1:].dropna()
    anos = fila_anos.astype(int).tolist()
    variaciones = crudo.iloc[14:26, 1:1 + len(anos)].to_numpy(dtype=float)

    registros = []
    indice = 100.0
    for col, ano in enumerate(anos):
        for fila, mes in enumerate(MESES_ORDEN):
            variacion_pct = variaciones[fila, col]
            indice *= 1 + variacion_pct / 100.0
            registros.append({"ano": ano, "mes": mes, "indice_ipc_total": indice})

    return pd.DataFrame(registros)


def construir_deflactores() -> pd.DataFrame:
    ipc_mensual = _indice_ipc_total_mensual()

    ipc_anual = (
        ipc_mensual[ipc_mensual["ano"].isin(cfg.OLA_A_ANO.values())]
        .groupby("ano", as_index=False)["indice_ipc_total"]
        .mean()
        .rename(columns={"indice_ipc_total": "ipc_total_promedio_anual"})
    )
    base_ipc_total = ipc_anual.loc[ipc_anual["ano"] == ANO_BASE, "ipc_total_promedio_anual"].iloc[0]
    ipc_anual["deflactor_ipc_total"] = ipc_anual["ipc_total_promedio_anual"] / base_ipc_total

    lp_nacional = pd.Series(cfg.LP_NACIONAL, name="lp_nacional").rename_axis("ano").reset_index()
    base_lp_nacional = lp_nacional.loc[lp_nacional["ano"] == ANO_BASE, "lp_nacional"].iloc[0]
    lp_nacional["deflactor_ipc_ing_bajos_nacional"] = lp_nacional["lp_nacional"] / base_lp_nacional

    # LP por zona (config_dane.LINEAS_POBREZA) -> deflactor ingresos bajos por zona,
    # cada zona contra su propia LP de 2010 (no contra la nacional).
    lp_zona = cfg.LINEAS_POBREZA[["ola", "ano", "zona", "lp"]].rename(columns={"lp": "lp_zona"})
    base_lp_zona = lp_zona.loc[lp_zona["ano"] == ANO_BASE].set_index("zona")["lp_zona"]
    lp_zona["deflactor_ipc_ing_bajos"] = lp_zona.apply(
        lambda fila: fila["lp_zona"] / base_lp_zona[fila["zona"]], axis=1
    )

    tabla = lp_zona.merge(ipc_anual, on="ano", how="left").merge(lp_nacional, on="ano", how="left")

    return tabla[
        [
            "ola", "ano", "zona",
            "ipc_total_promedio_anual", "deflactor_ipc_total",
            "lp_zona", "deflactor_ipc_ing_bajos",
            "lp_nacional", "deflactor_ipc_ing_bajos_nacional",
        ]
    ].sort_values(["ola", "zona"]).reset_index(drop=True)


def main() -> None:
    tabla = construir_deflactores()
    tabla.to_parquet(OUTPUT_PATH, index=False)

    print(f"Año base: {ANO_BASE}")
    print(f"Guardado: {OUTPUT_PATH}")
    print()
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
