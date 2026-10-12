"""
tabla_sensibilidad_shap.py
=====================================
Tabla de apendice con la sensibilidad del nucleo SHAP y de su direccion a
los umbrales elegidos (Seccion 5.2 de `main.tex`, Primer y Segundo
hallazgo). Solo formatea: los calculos estan en
`src/05_model/sensibilidad_shap_nucleo.py`.

Panel A: tamano del nucleo (fuente Monetaria 2010->2013) segun la masa
SHAP acumulada y el minimo de combinaciones (de 10), cuantas de las 22
variables del nucleo base conserva, y cuantas de las 7 universales siguen
dentro del nucleo en las cuatro fuentes.
Panel B: cuantas de las 20 variables numericas del nucleo quedan con
direccion consistente en las cuatro fuentes segun el |rho| minimo y la
proporcion minima de signo.

INPUTS

    data/processed/benchmark_resultados/sensibilidad_nucleo_shap.csv
    data/processed/benchmark_resultados/sensibilidad_signo_nucleo.csv

OUTPUTS

    paper/tables/tab_sensibilidad_shap.tex

COMO CORRER

    python src/tabla_sensibilidad_shap.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
RUTA_TEX = REPO_ROOT / "paper" / "tables" / "tab_sensibilidad_shap.tex"


def panel_nucleo(df: pd.DataFrame) -> list:
    lineas = [
        r"    \multicolumn{5}{l}{\textit{A. Núcleo: masa SHAP acumulada y mínimo de combinaciones (de 10)}} \\",
        r"    \midrule",
        r"    Masa & Mín. comb. & Núcleo & Conserva de las 22 & \shortstack{Universales en las\\4 fuentes (de 7)} \\",
        r"    \midrule",
    ]
    for _, f in df.iterrows():
        base = " (base)" if (f["masa"] == 0.80 and f["min_combinaciones"] == 5) else ""
        lineas.append(
            f"    {int(round(100 * f['masa']))}\\% & {int(f['min_combinaciones'])} & {int(f['n_nucleo_tabla'])}{base} & "
            f"{int(f['de_las_22_conserva'])} & {int(f['de_las_7_universales_en_las_4'])} \\\\"
        )
    return lineas


def panel_signo(df: pd.DataFrame) -> list:
    lineas = [
        r"    \midrule",
        r"    \multicolumn{5}{l}{\textit{B. Dirección: $|\rho|$ mínimo y proporción mínima de signo}} \\",
        r"    \midrule",
        r"    $|\rho|$ mín. & Prop. mín. & Consistentes (de 20) & & \\",
        r"    \midrule",
    ]
    for _, f in df.iterrows():
        base = " (base)" if (abs(f["rho_min"] - 0.10) < 1e-9 and abs(f["prop_min"] - 0.80) < 1e-9) else ""
        lineas.append(f"    {f['rho_min']:.2f} & {f['prop_min']:.2f} & {int(f['n_consistentes'])}{base} & & \\\\")
    return lineas


def main() -> None:
    nucleo = pd.read_csv(RESULTADOS / "sensibilidad_nucleo_shap.csv")
    signo = pd.read_csv(RESULTADOS / "sensibilidad_signo_nucleo.csv")
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Sensibilidad del núcleo SHAP y de su dirección a los umbrales",
        r"  elegidos, pobreza monetaria e IPM, 2010$\to$2013 y 2013$\to$2016.}",
        r"  \label{tab:sensibilidad_shap}",
        r"  \footnotesize",
        r"  \begin{tabular}{ccccc}",
        r"    \toprule",
    ]
    lineas += panel_nucleo(nucleo) + panel_signo(signo) + [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} ``(base)'': criterio usado en el texto",
        r"    (Sección~\ref{subsec:importancia_resultados}). El núcleo de 22 variables",
        r"    se conserva por completo con criterios más laxos; con los más estrictos",
        r"    queda un subconjunto. Las cuatro fuentes son las dos definiciones de",
        r"    pobreza en las dos ventanas. " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    tex = aplicar_signo_menos("\n".join(lineas))
    RUTA_TEX.parent.mkdir(parents=True, exist_ok=True)
    RUTA_TEX.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {RUTA_TEX}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
