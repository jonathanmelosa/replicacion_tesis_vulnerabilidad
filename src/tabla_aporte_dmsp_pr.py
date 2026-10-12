"""
tabla_aporte_dmsp_pr.py
=====================================
Aporte de la iluminacion nocturna DMSP-OLS en AUC-PR: AUC-PR sin luz y con luz, cambio e IC95% del
cambio (bootstrap pareado por comunidad), para los cinco algoritmos con y
sin ingreso, bajo pobreza monetaria e IPM, en el conjunto de prueba
2013->2016. Seccion 5.3 del paper.

No recalcula nada: lee `diagnostico_curva_pr.csv`
(`src/05_model/diagnostico_curva_pr.py`).

INPUT

    data/processed/benchmark_resultados/diagnostico_curva_pr.csv

OUTPUT

    paper/tables/tab_aporte_dmsp_pr.tex

COMO CORRER

    python src/tabla_aporte_dmsp_pr.py
"""

from pathlib import Path

import pandas as pd

import estilo_tablas as et

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_curva_pr.csv"
OUTPUT = REPO_ROOT / "paper" / "tables" / "tab_aporte_dmsp_pr.tex"

# Mismo orden y nombres que `tabla_desempeno_pr.py`.
ALGORITMOS = [
    ("Logistica", r"\emph{Elastic net}"),
    ("Random Forest", "Random Forest"),
    ("XGBoost", "XGBoost"),
    ("LightGBM", "LightGBM"),
    ("HistGradientBoosting", "HistGradientBoosting"),
]
PANELES = [("B", "Sin ingreso ni gasto (Modelo B)"), ("A", "Con ingreso y gasto (Modelo A)")]


def celdas(r: pd.Series) -> str:
    marca = "" if r["delta_ap_cruza_cero"] else r"\textsuperscript{*}"
    return (f"{r['ap_base']:.3f} & {r['delta_ap']:+.3f}{marca} & "
            f"{r['delta_ap_ci95_low']:.3f}, {r['delta_ap_ci95_high']:.3f}")


def generar_tex(datos: pd.DataFrame) -> str:
    lineas = [
        # [tbp] y no [H] (2026-10-05): con [H] saltaba de pagina y dejaba un
        # hueco en la Seccion 5.3.
        r"\begin{table}[tbp]",
        r"  \centering",
        r"  \caption{AUC-PR sin y con luces nocturnas de baja resolución (DMSP-OLS) en el conjunto de prueba,",
        r"  pobreza monetaria e IPM, 2013$\to$2016.}",
        r"  \label{tab:aporte_dmsp_pr}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l ccc ccc}",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{\textbf{Monetaria}} & \multicolumn{3}{c}{\textbf{IPM}} \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"    \textbf{Modelo} & \textbf{AUC-PR} & \textbf{Cambio} & \textbf{IC95\%}"
        r" & \textbf{AUC-PR} & \textbf{Cambio} & \textbf{IC95\%} \\",
        r"    \midrule",
    ]
    for i, (espec, titulo) in enumerate(PANELES):
        if i > 0:
            lineas.append(r"    \addlinespace")
        lineas.append(rf"    \multicolumn{{7}}{{l}}{{\textit{{{titulo}}}}} \\")
        for clave, nombre in ALGORITMOS:
            fila = []
            for definicion in ("monetaria", "ipm"):
                r = datos[(datos["definicion"] == definicion) & (datos["especificacion"] == espec) & (datos["algoritmo"] == clave)]
                assert len(r) == 1, f"falta {definicion} {espec} {clave}"
                fila.append(celdas(r.iloc[0]))
            etiqueta = nombre + (" (principal)" if espec == "B" and clave == "Logistica" else "")
            lineas.append(f"    \\quad {etiqueta} & {fila[0]} & {fila[1]} \\\\")
    prev_mon = datos.loc[datos["definicion"] == "monetaria", "prevalencia"].iloc[0]
    prev_ipm = datos.loc[datos["definicion"] == "ipm", "prevalencia"].iloc[0]
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} AUC-PR: modelo con la encuesta, sin DMSP-OLS.",
        r"    Cambio: diferencia al agregar las 23 variables de DMSP-OLS. IC95\%:",
        r"    intervalo del cambio por \emph{bootstrap}",
        r"    pareado de 2{,}000 remuestras por comunidad. \textsuperscript{*}: el",
        rf"    intervalo excluye el cero. Referencia al azar: {prev_mon:.3f} (monetaria) y",
        rf"    {prev_ipm:.3f} (IPM). El modelo principal se marca solo bajo pobreza",
        r"    monetaria.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def main() -> None:
    tex = generar_tex(pd.read_csv(INPUT))
    OUTPUT.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {OUTPUT}\n\n{tex}")


if __name__ == "__main__":
    main()
