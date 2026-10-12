"""
tabla_desempeno_pr.py
=====================================
Tablas de desempeno en el conjunto de prueba 2013->2016 centradas en la
curva de precision-recall: AUC-PR
con IC95% y AUC-ROC; recall, precision y F1 al umbral elegido por
validacion cruzada maximizando F2 (la regla de operacion del paper); y
precision y recall en el decil de mayor riesgo (10% de hogares). Cinco algoritmos x
Modelo B (sin ingreso ni gasto; el elastic net B es el modelo principal) y
Modelo A (con ingreso y gasto).

Genera dos tablas con el mismo formato: pobreza monetaria (cuerpo,
Seccion 5.1) y pobreza multidimensional (anexo).

No recalcula nada: lee `diagnostico_curva_pr.csv`
(`src/05_model/diagnostico_curva_pr.py`), columnas `*_base` (modelos sin
DMSP-OLS).

INPUT

    data/processed/benchmark_resultados/diagnostico_curva_pr.csv

OUTPUTS

    paper/tables/tab_desempeno_pr_monetaria.tex
    paper/tables/tab_desempeno_pr_ipm.tex

COMO CORRER

    python src/tabla_desempeno_pr.py
"""

from pathlib import Path

import pandas as pd

import estilo_tablas as et

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_curva_pr.csv"
OUT_DIR = REPO_ROOT / "paper" / "tables"

# Orden fijo: el modelo principal primero, luego el orden de la Seccion 4.2.
ALGORITMOS = [
    ("Logistica", r"\emph{Elastic net}"),
    ("Random Forest", "Random Forest"),
    ("XGBoost", "XGBoost"),
    ("LightGBM", "LightGBM"),
    ("HistGradientBoosting", "HistGradientBoosting"),
]
PANELES = [("B", "Sin ingreso ni gasto (Modelo B)"), ("A", "Con ingreso y gasto (Modelo A)")]

CONFIG = {
    "monetaria": {
        "archivo": "tab_desempeno_pr_monetaria.tex",
        "label": "tab:desempeno_pr",
        "titulo": r"Desempeño en el conjunto de prueba, pobreza monetaria, 2013$\to$2016.",
        "nota_extra": "",
    },
    "ipm": {
        "archivo": "tab_desempeno_pr_ipm.tex",
        "label": "tab:desempeno_pr_ipm",
        "titulo": r"Desempeño en el conjunto de prueba, pobreza multidimensional (IPM), 2013$\to$2016.",
        "nota_extra": "",
    },
}


def miles(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def fila(r: pd.Series, etiqueta: str) -> str:
    ic = f"{r.ap_base_ci95_low:.2f}--{r.ap_base_ci95_high:.2f}"
    umbral = f"{r.recall_f2_base:.2f} & {r.precision_f2_base:.2f} & {r.f1_f2_base:.2f}"
    decil = f"{r.precision_k10_base:.2f} & {r.recall_k10_base:.2f}"
    return f"    \\quad {etiqueta} & {r.ap_base:.3f} & {ic} & {r.auc_roc_base:.3f} & {umbral} & {decil} \\\\"


def generar_tex(datos: pd.DataFrame, definicion: str) -> str:
    cfg = CONFIG[definicion]
    sub = datos[datos["definicion"] == definicion]
    prevalencias = sub["prevalencia"].unique()
    tamanos = sub["n_hogares"].unique()
    assert len(prevalencias) == 1 and len(tamanos) == 1, "prevalencia o n distintos entre modelos"
    prevalencia, n_hogares = float(prevalencias[0]), int(tamanos[0])

    lineas = [
        # Monetaria (cuerpo) en [tbp]: con [H] dejaba media pagina en blanco en
        # la Seccion 5.1 (2026-10-05). La del IPM, en el anexo, sigue en [H].
        r"\begin{table}[tbp]" if definicion == "monetaria" else r"\begin{table}[H]",
        r"  \centering",
        f"  \\caption{{{cfg['titulo']}}}",
        f"  \\label{{{cfg['label']}}}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l ccc ccc cc}",
        r"    \toprule",
        r"    & & & & \multicolumn{3}{c}{\textbf{Umbral F$_2$}} & \multicolumn{2}{c}{\textbf{Decil superior}} \\",
        r"    \cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"    \textbf{Modelo} & \textbf{AUC-PR} & \textbf{IC95\%} & \textbf{AUC-ROC} & \textbf{Rec.} & \textbf{Prec.} & \textbf{F$_1$} & \textbf{Prec.} & \textbf{Rec.} \\",
        r"    \midrule",
    ]
    for i, (espec, titulo_panel) in enumerate(PANELES):
        if i > 0:
            lineas.append(r"    \addlinespace")
        lineas.append(rf"    \multicolumn{{9}}{{l}}{{\textit{{{titulo_panel}}}}} \\")
        for clave, etiqueta in ALGORITMOS:
            r = sub[(sub["especificacion"] == espec) & (sub["algoritmo"] == clave)]
            assert len(r) == 1, f"falta {definicion} {espec} {clave}"
            # El modelo principal es solo el elastic net, Modelo B, monetaria.
            if definicion == "monetaria" and espec == "B" and clave == "Logistica":
                etiqueta += " (principal)"
            lineas.append(fila(r.iloc[0], etiqueta))
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} AUC-PR: área bajo la curva de",
        r"    \emph{precision}-\emph{recall}, estimada como \emph{average precision};",
        rf"    un modelo que asigna el riesgo al azar obtiene {prevalencia:.3f}, la",
        r"    proporción de hogares vulnerables, y 0.5 de AUC-ROC. IC95\%:",
        r"    \emph{bootstrap} de 2{,}000 remuestras por comunidad. Umbral F$_2$:",
        r"    métricas al umbral elegido por validación cruzada maximizando F$_2$.",
        r"    Decil superior: si un programa atiende al 10\% de los hogares con",
        r"    mayor riesgo predicho. Rec.: \emph{recall}; Prec.: \emph{precision}.",
        rf"    {miles(n_hogares)} hogares no pobres en 2013; una estimación por modelo.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lineas)


def main() -> None:
    datos = pd.read_csv(INPUT)
    for definicion, cfg in CONFIG.items():
        tex = generar_tex(datos, definicion)
        ruta = OUT_DIR / cfg["archivo"]
        ruta.write_text(tex, encoding="utf-8")
        print(f"Tabla exportada: {ruta}\n\n{tex}\n")


if __name__ == "__main__":
    main()
