"""
tabla_shap_variables_necesarias.py
=====================================
Tabla para la Sección de importancia de variables (Sección~5.3 de
`main.tex`): de las ~90 variables originales del benchmark, cuántas bastan
para capturar la mayor parte de la señal predictiva medida por SHAP, por
algoritmo y especificación (A/B). Lee el CSV ya agregado por
`src/05_model/diagnostico_shap_variables_necesarias.py` (que resume las
columnas one-hot/missingindicator de vuelta a su variable original antes
de calcular la masa acumulada) -- este script solo formatea, no calcula.

INPUTS

    data/processed/benchmark_resultados/diagnostico_shap_variables_necesarias.csv

OUTPUTS

    paper/tables/tab_shap_variables_necesarias.tex

COMO CORRER

    python src/tabla_shap_variables_necesarias.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "entrada": REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_shap_variables_necesarias.csv",
    "nombres_algoritmo": {
        "Random Forest": "Random Forest",
        "XGBoost": "XGBoost",
        "HistGradientBoosting": "HistGradientBoosting",
        "LightGBM": "LightGBM",
        "Logistica": "Logística regularizada",
    },
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar(cfg: dict) -> pd.DataFrame:
    ruta = Path(cfg["entrada"])
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta} -- correr primero "
              f"src/05_model/diagnostico_shap_variables_necesarias.py", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ruta)
    df["algoritmo"] = df["algoritmo"].map(cfg["nombres_algoritmo"]).fillna(df["algoritmo"])
    return df


def formatear_fila(fila: pd.Series) -> str:
    return (
        f"    {fila['algoritmo']:<24s} & {fila['especificacion']} & "
        f"{fila['n_variables_originales']:d} & "
        f"{fila['pct_shap_top10']*100:.1f}\\% & "
        f"{fila['n_variables_para_80pct']:d} \\\\"
    )


def generar_tex(df: pd.DataFrame) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Concentración de la importancia SHAP por algoritmo y",
        r"  especificación, pobreza monetaria, 2013$\to$2016.}",
        r"  \label{tab:shap_variables_necesarias}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Espec.} & \textbf{\shortstack{Variables\\(total)}} & "
        r"\textbf{\shortstack{\% SHAP en las\\10 primeras}} & \textbf{\shortstack{Variables para el\\80\% del SHAP}} \\",
        r"    \midrule",
    ]
    especificaciones = sorted(df["especificacion"].unique())
    for i, espec in enumerate(especificaciones):
        bloque = df[df["especificacion"] == espec].sort_values("n_variables_para_80pct")
        for _, fila in bloque.iterrows():
            lineas.append(formatear_fila(fila))
        if i < len(especificaciones) - 1:
            lineas.append(r"    \addlinespace")
    # Nota y cierre del entorno dentro del .tex (2026-10-01): antes el
    # \end{table} vivia en main.tex y las definiciones iban en el titulo.
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Importancia SHAP calculada sobre el conjunto",
        r"    de prueba. Variables originales: las \emph{dummies one-hot} y los",
        r"    indicadores de faltante se suman a su variable de origen antes de",
        r"    calcular la masa acumulada. Espec.: especificación (A con ingreso y",
        r"    gasto del hogar, B sin ellos). " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return aplicar_signo_menos("\n".join(lineas))


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = cargar(cfg)
    tex = generar_tex(df)
    ruta_tex = out_dir / "tab_shap_variables_necesarias.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
