"""
tabla_variacion_dmsp_comunidad.py
=====================================
Tabla del anexo sobre de donde viene la variacion de la iluminacion
nocturna. Por definicion de pobreza, ventana y zona: hogares,
parte de la variacion de la luz y del riesgo que concentra la comunidad
(correlacion intraclase), comunidades con luz identica, hogares saturados y
AUC-ROC de la luz sola. Respalda las cifras de la Seccion 5.3.

No recalcula nada: lee `diagnostico_variacion_dmsp_comunidad.csv`
(`src/05_model/diagnostico_variacion_dmsp_comunidad.py`).

INPUT

    data/processed/benchmark_resultados/diagnostico_variacion_dmsp_comunidad.csv

OUTPUT

    paper/tables/tab_variacion_dmsp_comunidad.tex

COMO CORRER

    python src/tabla_variacion_dmsp_comunidad.py
"""

from pathlib import Path

import pandas as pd

import estilo_tablas as et

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_variacion_dmsp_comunidad.csv"
OUTPUT = REPO_ROOT / "paper" / "tables" / "tab_variacion_dmsp_comunidad.tex"

DEFINICIONES = [("monetaria", "Pobreza monetaria"), ("ipm", "Pobreza multidimensional (IPM)")]
VENTANAS = [("2010_2013", r"2010$\to$2013"), ("2013_2016", r"2013$\to$2016")]
ZONAS = ["Total", "Urbano", "Rural"]


def miles(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def generar_tex(datos: pd.DataFrame) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Variación de las luces nocturnas de baja resolución (DMSP-OLS) y del riesgo de pobreza entre",
        r"  comunidades, pobreza monetaria e IPM, 2010$\to$2013 y 2013$\to$2016.}",
        r"  \label{tab:variacion_dmsp_comunidad}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{ll c cc cc c}",
        r"    \toprule",
        r"    & & & \multicolumn{2}{c}{\textbf{\% en la comunidad}} & \textbf{DMSP-OLS idéntico} & \textbf{Saturados} & \textbf{AUC-ROC} \\",
        r"    \cmidrule(lr){4-5}",
        r"    \textbf{Ventana} & \textbf{Zona} & \textbf{Hogares} & \textbf{DMSP-OLS} & \textbf{Riesgo}"
        r" & \textbf{(\%)} & \textbf{(\%)} & \textbf{DMSP-OLS} \\",
        r"    \midrule",
    ]
    for i, (definicion, titulo) in enumerate(DEFINICIONES):
        if i > 0:
            lineas.append(r"    \addlinespace")
        lineas.append(rf"    \multicolumn{{8}}{{l}}{{\textit{{{titulo}}}}} \\")
        for ventana, etiqueta_ventana in VENTANAS:
            for j, zona in enumerate(ZONAS):
                r = datos[(datos["definicion"] == definicion) & (datos["ventana"] == ventana) & (datos["zona"] == zona)]
                assert len(r) == 1, f"falta {definicion} {ventana} {zona}"
                r = r.iloc[0]
                celda_ventana = etiqueta_ventana if j == 0 else ""
                lineas.append(
                    f"    \\quad {celda_ventana} & {zona} & {miles(int(r['n_hogares']))} & {pct(r['icc_luz'])} & "
                    f"{pct(r['icc_y'])} & {pct(r['pct_comunidades_luz_identica'])} & {pct(r['pct_saturado'])} & "
                    f"{r['auc_luz']:.2f} \\\\"
                )
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Hogares no pobres al comienzo de cada ventana y",
        r"    con dato de DMSP-OLS. \% en la comunidad: correlación intraclase de un",
        r"    análisis de varianza de un factor, es decir, la parte de la variación",
        r"    entre hogares que corresponde a diferencias entre comunidades; para el",
        r"    riesgo se calcula sobre la variable que indica si el hogar es pobre en",
        r"    la encuesta siguiente. Excluye a los hogares sin comunidad identificada.",
        r"    DMSP-OLS idéntico: comunidades con al menos dos hogares en las que todos",
        r"    tienen el mismo valor de DMSP-OLS. Saturados: hogares con el valor máximo de",
        r"    la escala (63). AUC-ROC DMSP-OLS: capacidad de DMSP-OLS por sí solo para ordenar el",
        r"    riesgo (menor valor, más riesgo). Sin ponderar.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lineas) + "\n"


def main() -> None:
    tex = generar_tex(pd.read_csv(INPUT))
    OUTPUT.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {OUTPUT}\n\n{tex}")


if __name__ == "__main__":
    main()
