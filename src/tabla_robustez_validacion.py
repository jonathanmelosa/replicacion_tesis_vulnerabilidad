"""
tabla_robustez_validacion.py
=====================================
AUC-PR con validacion cruzada por fila, agrupada por hogar y agrupada por
comunidad, con las dos ventanas juntas. Anexo
"Robustez de la validacion y margenes de informacion", Seccion 4.3 del
paper.

No recalcula nada: lee las salidas de
`src/05_model/robustez_hogares_no_repetidos.py`. El "Cambio" se reporta
como agrupada MENOS por fila (negativo = agrupar baja el desempeño), con
el intervalo del bootstrap pareado por grupo hogar-comunidad.

INPUTS

    data/processed/benchmark_resultados/robustez_asesor_2026_10/no_repetidos_b_desempeno.csv
    data/processed/benchmark_resultados/robustez_asesor_2026_10/no_repetidos_b_diferencias.csv
    data/processed/benchmark_resultados/robustez_asesor_2026_10/prevalencia_validacion_b.csv
        (referencia al azar; la genera src/05_model/prevalencia_validacion_b.py)

OUTPUT

    paper/tables/tab_robustez_validacion.tex

COMO CORRER

    python src/tabla_robustez_validacion.py
"""

from pathlib import Path

import pandas as pd

import estilo_tablas as et

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "robustez_asesor_2026_10"
OUTPUT = REPO_ROOT / "paper" / "tables" / "tab_robustez_validacion.tex"

# Mismo orden y nombres que `tabla_aporte_dmsp_pr.py`.
ALGORITMOS = [
    ("Logistica", r"\emph{Elastic net}"),
    ("Random Forest", "Random Forest"),
    ("XGBoost", "XGBoost"),
    ("LightGBM", "LightGBM"),
    ("HistGradientBoosting", "HistGradientBoosting"),
]
PANELES = [
    ("B", "Sin ingreso ni gasto (Modelo B)"),
    ("BgeoDMSP", "Modelo B con DMSP-OLS"),
    ("A", "Con ingreso y gasto (Modelo A)"),
    ("AgeoDMSP", "Modelo A con DMSP-OLS"),
]


def num(x: float) -> str:
    """Tres decimales; un negativo que redondea a cero conserva el signo
    ("-0.000"), para que un intervalo marcado con asterisco no parezca
    incluir el cero."""
    if x < 0 and round(x, 3) == 0:
        return "-0.000"
    return f"{round(x, 3) + 0.0:.3f}"


def cambio(dif: pd.DataFrame, espec: str, algoritmo: str, forma: str) -> str:
    """Agrupada menos por fila: el CSV guarda fila menos agrupada."""
    r = dif[(dif.especificacion == espec) & (dif.algoritmo == algoritmo) & (dif.comparacion == f"fila - {forma}")]
    assert len(r) == 1, f"falta {espec} {algoritmo} {forma}"
    r = r.iloc[0]
    marca = "" if r["delta_ap_cruza_cero"] else r"\textsuperscript{*}"
    return f"{num(-r['delta_ap'])}{marca} ({num(-r['delta_ap_ci95_high'])}, {num(-r['delta_ap_ci95_low'])})"


def generar_tex(des: pd.DataFrame, dif: pd.DataFrame, prevalencia: float) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{AUC-PR con validación cruzada por fila, agrupada por hogar y agrupada por",
        r"  comunidad, pobreza monetaria, 2010$\to$2013 y 2013$\to$2016 juntos.}",
        r"  \label{tab:robustez_validacion}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l c cc cc}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{\textbf{Agrupada por hogar}} & \multicolumn{2}{c}{\textbf{Agrupada por comunidad}} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"    \textbf{Modelo} & \textbf{Por fila} & \textbf{AUC-PR} & \textbf{Cambio (IC95\%)}"
        r" & \textbf{AUC-PR} & \textbf{Cambio (IC95\%)} \\",
        r"    \midrule",
    ]
    for i, (espec, titulo) in enumerate(PANELES):
        if i > 0:
            lineas.append(r"    \addlinespace")
        lineas.append(rf"    \multicolumn{{6}}{{l}}{{\textit{{{titulo}}}}} \\")
        for clave, nombre in ALGORITMOS:
            ap = des[(des.especificacion == espec) & (des.algoritmo == clave)].set_index("forma")["ap"]
            assert set(ap.index) == {"fila", "hogar", "comunidad"}, f"faltan formas en {espec} {clave}"
            etiqueta = nombre + (" (principal)" if espec == "B" and clave == "Logistica" else "")
            lineas.append(
                f"    \\quad {etiqueta} & {num(ap['fila'])} & {num(ap['hogar'])} & {cambio(dif, espec, clave, 'hogar')}"
                f" & {num(ap['comunidad'])} & {cambio(dif, espec, clave, 'comunidad')} \\\\"
            )
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Cambio: AUC-PR de la validación agrupada menos el",
        r"    de la validación por fila. IC95\%: \emph{bootstrap} pareado de 2{,}000",
        r"    remuestras por grupo de hogar y comunidad. \textsuperscript{*}: el intervalo",
        rf"    excluye el cero. Referencia al azar: {prevalencia:.3f}.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def main() -> None:
    des = pd.read_csv(RES / "no_repetidos_b_desempeno.csv")
    dif = pd.read_csv(RES / "no_repetidos_b_diferencias.csv")
    prevalencia = pd.read_csv(RES / "prevalencia_validacion_b.csv")["prevalencia"].iloc[0]
    tex = generar_tex(des, dif, prevalencia)
    OUTPUT.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {OUTPUT}\n\n{tex}")


if __name__ == "__main__":
    main()
