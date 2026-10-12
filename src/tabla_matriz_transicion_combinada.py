"""
tabla_matriz_transicion_combinada.py
=====================================
Version COMPACTA de las matrices de transición monetaria (`tabla_matriz_
transicion.py`) e IPM (`tabla_matriz_transicion_ipm.py`): en vez de dos
tablas separadas, pone ambas definiciones de pobreza lado a lado en una
sola tabla -- para la Sección "Dinámica de transiciones a la pobreza",
que tiene un límite de espacio de 2 páginas y no puede permitirse dos
tablas completas más sus notas al pie por separado.

Actualizado: cada celda muestra el
numero de hogares y el % de fila, mas una columna Total por definicion;
los periodos van como subtitulos de fila (sin columna "Periodo") para que
quepa en el ancho de pagina; nota sin rutas de archivo y con la fuente
comun de `estilo_tablas.py`. Reemplaza al diagrama Sankey en el cuerpo.

No recalcula nada: reusa los mismos CSV ya producidos por
`build_pobreza_desagregaciones.py` (monetaria) y
`build_matriz_transicion_ipm.py` (IPM).

INPUTS

    outputs/tables/pobreza/transicion_{pct,conteo}_ola{1_a_2,2_a_3}.csv
    outputs/tables/pobreza/transicion_{pct,conteo}_ipm_ola{1_a_2,2_a_3}.csv

OUTPUTS

    paper/tables/tab_matriz_transicion_combinada.tex

CÓMO CORRER

    python src/tabla_matriz_transicion_combinada.py
"""

import sys
from pathlib import Path

import pandas as pd

import estilo_tablas as et

REPO_ROOT = Path(__file__).resolve().parents[1]
POBREZA_DIR = REPO_ROOT / "outputs" / "tables" / "pobreza"

CONFIG = {
    "monetaria": {
        "pct_1_2": POBREZA_DIR / "transicion_pct_ola1_a_2.csv",
        "pct_2_3": POBREZA_DIR / "transicion_pct_ola2_a_3.csv",
        "conteo_1_2": POBREZA_DIR / "transicion_conteo_ola1_a_2.csv",
        "conteo_2_3": POBREZA_DIR / "transicion_conteo_ola2_a_3.csv",
    },
    "ipm": {
        "pct_1_2": POBREZA_DIR / "transicion_pct_ipm_ola1_a_2.csv",
        "pct_2_3": POBREZA_DIR / "transicion_pct_ipm_ola2_a_3.csv",
        "conteo_1_2": POBREZA_DIR / "transicion_conteo_ipm_ola1_a_2.csv",
        "conteo_2_3": POBREZA_DIR / "transicion_conteo_ipm_ola2_a_3.csv",
    },
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar_matriz(ruta_pct: Path, ruta_conteo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ruta_pct.exists() or not ruta_conteo.exists():
        print(f"ERROR: no se encontró {ruta_pct} o {ruta_conteo}", file=sys.stderr)
        sys.exit(1)
    pct = pd.read_csv(ruta_pct, index_col=0)
    conteo = pd.read_csv(ruta_conteo, index_col=0)
    # El % de fila del CSV debe salir exactamente de los conteos (sin ponderar).
    recalculado = conteo.div(conteo.sum(axis=1), axis=0) * 100
    assert (recalculado.round(1) - pct).abs().to_numpy().max() < 0.051, f"% de {ruta_pct.name} no cuadra con sus conteos"
    return pct, conteo


def miles(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def celdas_fila(pct: pd.DataFrame, conteo: pd.DataFrame, fila: str) -> str:
    """'n (%)' para No pobre y Pobre en la siguiente encuesta, mas el total de la fila."""
    partes = [f"{miles(int(conteo.loc[fila, col]))} ({pct.loc[fila, col]:.1f}\\%)" for col in ("No pobre", "Pobre")]
    partes.append(miles(int(conteo.loc[fila].sum())))
    return " & ".join(partes)


def generar_tex(datos: dict) -> str:
    m12, cm12 = cargar_matriz(CONFIG["monetaria"]["pct_1_2"], CONFIG["monetaria"]["conteo_1_2"])
    m23, cm23 = cargar_matriz(CONFIG["monetaria"]["pct_2_3"], CONFIG["monetaria"]["conteo_2_3"])
    i12, ci12 = cargar_matriz(CONFIG["ipm"]["pct_1_2"], CONFIG["ipm"]["conteo_1_2"])
    i23, ci23 = cargar_matriz(CONFIG["ipm"]["pct_2_3"], CONFIG["ipm"]["conteo_2_3"])
    n_m12, n_i12 = int(cm12.to_numpy().sum()), int(ci12.to_numpy().sum())
    n_m23, n_i23 = int(cm23.to_numpy().sum()), int(ci23.to_numpy().sum())
    assert n_m12 == n_i12, f"universo monetaria ({n_m12}) != IPM ({n_i12}) en 2010->2013 -- revisar supuesto de mismo panel"
    assert n_m23 == n_i23, f"universo monetaria ({n_m23}) != IPM ({n_i23}) en 2013->2016 -- revisar supuesto de mismo panel"

    def bloque(periodo: str, mon_pct, mon_cnt, ipm_pct, ipm_cnt) -> list:
        return [
            rf"    \multicolumn{{7}}{{l}}{{\textit{{{periodo}}}}} \\",
            rf"    \quad No pobre & {celdas_fila(mon_pct, mon_cnt, 'No pobre')} & {celdas_fila(ipm_pct, ipm_cnt, 'No pobre')} \\",
            rf"    \quad Pobre & {celdas_fila(mon_pct, mon_cnt, 'Pobre')} & {celdas_fila(ipm_pct, ipm_cnt, 'Pobre')} \\",
        ]

    lineas = [
        # [tbp] y no [H] (2026-10-05): con [H] la tabla saltaba de pagina y
        # dejaba un hueco de ~1/4 de pagina en la Seccion 3.3.
        r"\begin{table}[tbp]",
        r"  \centering",
        r"  \caption{Matriz de transición de pobreza, pobreza monetaria y",
        r"  multidimensional (IPM), 2010$\to$2013 y 2013$\to$2016.}",
        r"  \label{tab:matriz_transicion}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l rrr rrr}",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{\textbf{Monetaria}} & \multicolumn{3}{c}{\textbf{IPM}} \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"    \textbf{Primera encuesta} & \textbf{No pobre} & \textbf{Pobre} & \textbf{Total} & \textbf{No pobre} & \textbf{Pobre} & \textbf{Total} \\",
        r"    \midrule",
        *bloque(r"2010$\to$2013", m12, cm12, i12, ci12),
        r"    \addlinespace",
        *bloque(r"2013$\to$2016", m23, cm23, i23, ci23),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Número de hogares y, entre paréntesis,",
        r"    porcentaje de la fila: de los hogares en cada estado en la primera",
        r"    encuesta del periodo, qué porcentaje está en cada estado en la",
        r"    siguiente. Panel emparejado 1 a 1, con los mismos hogares para las dos",
        rf"    definiciones de pobreza ($n={miles(n_m12)}$ en 2010$\to$2013 y $n={miles(n_m23)}$",
        r"    en 2013$\to$2016); excluye a los hogares que se dividieron entre olas",
        r"    (511 en 2010$\to$2013 y 1{,}190 en 2013$\to$2016). Sin ponderar.",
        rf"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lineas)


def main() -> None:
    out_dir = Path(CONFIG["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tex = generar_tex(CONFIG)
    ruta_tex = out_dir / "tab_matriz_transicion_combinada.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
