"""
tabla_geo3_cv.py
=====================================
Formatea la comparacion A/B vs. Ageo3/Bgeo3 (las 127 variables de las
tres fuentes geoespaciales juntas: DMSP-OLS, ALOS PALSAR y Landsat 5 TM)
bajo pobreza monetaria, para el Anexo de la tesis
(Tabla~\\ref{tab:geo3_cv}). No reentrena nada: lee dos registros ya
calculados por `src/05_model/modelo_geo3_baseline_cv.py` y
`src/05_model/modelo_geo3_robusto_cv.py`.

Ambos registros usan el MISMO esquema (validacion cruzada de 10
particiones dentro de la transicion 2010->2013, metricas sobre
probabilidades out-of-fold, 5 semillas) porque ALOS PALSAR y Landsat 5 TM
no tienen dato en 2013 y no admiten el holdout temporal. Por eso estas
cifras son comparables entre si, pero NO contra la tabla de DMSP-OLS con
holdout (tab:marginal_dmsp_fbeta2_cv10). Solo se corrieron HistGradientBoosting,
XGBoost y logistica regularizada.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_geo3_baseline.csv
    data/processed/benchmark_resultados/registro_modelos_geo3_robusto.csv

OUTPUTS

    paper/tables/tab_geo3_cv.tex

COMO CORRER

    python src/tabla_geo3_cv.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

NOMBRES_ALGORITMO = {
    "XGBoost": "XGBoost",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
}
ALGORITMOS_ORDEN = ["XGBoost", "HistGradientBoosting", "Logística regularizada"]
PARES = [("A", "Ageo3", "A + 3 fuentes"), ("B", "Bgeo3", "B + 3 fuentes")]


def cargar(nombre: str) -> pd.DataFrame:
    ruta = RESULTADOS_DIR / nombre
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ruta)
    df["algoritmo"] = df["algoritmo"].map(NOMBRES_ALGORITMO).fillna(df["algoritmo"])
    return df


def fila_tex(algoritmo: str, etiqueta: str, fila: pd.Series, delta: str) -> str:
    return (
        f"    {algoritmo:<24s} & {etiqueta:<14s} & {int(fila['n_covariables_modelo'])} & "
        f"{fila['auc_roc_media']:.3f} & [{fila['auc_roc_ci95_low']:.3f}, {fila['auc_roc_ci95_high']:.3f}] & "
        f"{delta} & {fila['precision_top10_media']:.3f} \\\\"
    )


def generar_tex(base: pd.DataFrame, geo: pd.DataFrame) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{AUC-ROC y precisión en el decil de mayor riesgo con y sin las",
        r"  tres fuentes geoespaciales en validación cruzada, pobreza monetaria,",
        r"  2010$\to$2013.}",
        r"  \label{tab:geo3_cv}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{llccccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Especificación} & \textbf{N.º var.} & \textbf{AUC-ROC} & \textbf{IC95\%} & \textbf{$\Delta$AUC} & \textbf{Prec.-top10} \\",
        r"    \midrule",
    ]
    bloques = []
    for espec_base, espec_geo, etiqueta_geo in PARES:
        for algoritmo in ALGORITMOS_ORDEN:
            f_base = base[(base["algoritmo"] == algoritmo) & (base["especificacion"] == espec_base)]
            f_geo = geo[(geo["algoritmo"] == algoritmo) & (geo["especificacion"] == espec_geo)]
            if len(f_base) != 1 or len(f_geo) != 1:
                print(f"ERROR: se esperaba una fila para {algoritmo} {espec_base}/{espec_geo}", file=sys.stderr)
                sys.exit(1)
            f_base, f_geo = f_base.iloc[0], f_geo.iloc[0]
            delta = f"{f_geo['auc_roc_media'] - f_base['auc_roc_media']:+.3f}"
            bloques.append([
                fila_tex(algoritmo, espec_base, f_base, "--"),
                fila_tex(algoritmo, etiqueta_geo, f_geo, delta),
            ])
    for i, bloque in enumerate(bloques):
        lineas += bloque
        if i < len(bloques) - 1:
            lineas.append(r"    \addlinespace")
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Las tres fuentes suman 127 variables (DMSP-OLS,",
        r"    ALOS PALSAR y Landsat 5 TM). Validación cruzada de 10 particiones dentro",
        r"    de 2010$\to$2013; media e IC95\% sobre 5 semillas. ALOS PALSAR y Landsat 5 TM",
        r"    no tienen dato en 2013, por lo que esta variante no admite la evaluación en",
        r"    el conjunto de prueba 2013$\to$2016"
        r"    de la Tabla~\ref{tab:marginal_dmsp_fbeta2_cv10}: las métricas se calculan sobre",
        r"    probabilidades \emph{out-of-fold} y son comparables entre sí, no contra",
        r"    esa tabla. Solo se estimaron modelos con los tres algoritmos mostrados.",
        r"    N.º var.: covariables que recibe el modelo. Prec.-top10: precisión en el",
        r"    10\% de hogares con mayor riesgo predicho. " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return aplicar_signo_menos("\n".join(lineas))


def main() -> None:
    base = cargar("registro_modelos_geo3_baseline.csv")
    geo = cargar("registro_modelos_geo3_robusto.csv")
    tex = generar_tex(base, geo)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta = OUTPUT_DIR / "tab_geo3_cv.tex"
    ruta.write_text(tex + "\n", encoding="utf-8")
    print(f"Tabla exportada: {ruta}\n\n{tex}")


if __name__ == "__main__":
    main()
