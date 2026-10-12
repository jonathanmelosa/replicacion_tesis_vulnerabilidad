"""
tabla_robustez_techo.py
=====================================
Techo de la informacion de comunidad.
Anexo "Robustez de la validacion y margenes de informacion", Seccion 5.3
del paper. Genera dos tablas:

  1. AUC-PR sin informacion de comunidad y cambio al agregar DMSP-OLS, la
     tasa de entrada de la comunidad en 2010->2013 o la de los vecinos en
     el mismo periodo (validacion cruzada dentro de 2013->2016).
  2. Correlacion intraclase por comunidad del nivel de pobreza y de la
     entrada en pobreza, por definicion y zona.

No recalcula nada: lee las salidas de
`src/05_model/robustez_techo_comunidad.py`.

INPUTS

    data/processed/benchmark_resultados/robustez_asesor_2026_10/techo_resultados.csv
    data/processed/benchmark_resultados/robustez_asesor_2026_10/techo_icc.csv

OUTPUTS

    paper/tables/tab_robustez_techo.tex
    paper/tables/tab_robustez_icc.tex

COMO CORRER

    python src/tabla_robustez_techo.py
"""

from pathlib import Path

import pandas as pd

import estilo_tablas as et
from tabla_robustez_historia import celda
from tabla_robustez_validacion import ALGORITMOS, num

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "robustez_asesor_2026_10"
OUT_TECHO = REPO_ROOT / "paper" / "tables" / "tab_robustez_techo.tex"
OUT_ICC = REPO_ROOT / "paper" / "tables" / "tab_robustez_icc.tex"

PANELES = [
    ("monetaria", "B", "Pobreza monetaria, sin ingreso ni gasto (Modelo B)"),
    ("monetaria", "A", "Pobreza monetaria, con ingreso y gasto (Modelo A)"),
    ("ipm", "B", "IPM, sin ingreso ni gasto (Modelo B)"),
    ("ipm", "A", "IPM, con ingreso y gasto (Modelo A)"),
]
VARIANTES = ["dmsp", "tasa_previa", "oraculo"]
ZONAS = ["Total", "Urbano", "Rural"]


def tabla_techo(datos: pd.DataFrame) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{AUC-PR sin información de comunidad y cambio al agregarla, pobreza monetaria",
        r"  e IPM, validación cruzada dentro de 2013$\to$2016.}",
        r"  \label{tab:robustez_techo}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l c ccc}",
        r"    \toprule",
        r"    \textbf{Modelo} & \textbf{Sin información} & \textbf{DMSP-OLS} & \textbf{Tasa 2010$\to$2013}"
        r" & \textbf{Tasa de los vecinos} \\",
        r"    \midrule",
    ]
    for i, (definicion, espec, titulo) in enumerate(PANELES):
        if i > 0:
            lineas.append(r"    \addlinespace")
        lineas.append(rf"    \multicolumn{{5}}{{l}}{{\textit{{{titulo}}}}} \\")
        for clave, nombre in ALGORITMOS:
            sub = datos[(datos.definicion == definicion) & (datos.especificacion == espec)
                        & (datos.algoritmo == clave)].set_index("variante")
            assert set(sub.index) == set(VARIANTES), f"faltan variantes en {definicion} {espec} {clave}"
            base = sub["ap_ref"].unique()
            assert len(base) == 1, "la referencia debe ser la misma en las tres variantes"
            principal = definicion == "monetaria" and espec == "B" and clave == "Logistica"
            etiqueta = nombre + (" (principal)" if principal else "")
            celdas = " & ".join(celda(sub.loc[v]) for v in VARIANTES)
            lineas.append(f"    \\quad {etiqueta} & {num(base[0])} & {celdas} \\\\")
    prev = datos.groupby("definicion")["prevalencia"].unique()
    n = datos.groupby("definicion")["n_hogares"].unique()
    assert all(len(v) == 1 for v in prev) and all(len(v) == 1 for v in n)
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Sin información: AUC-PR del modelo con las covariables de",
        r"    2013. DMSP-OLS: se agregan sus 23 variables. Tasa 2010$\to$2013: tasa de entrada en",
        r"    pobreza de la comunidad en la ventana anterior. Tasa de los vecinos: tasa de entrada",
        r"    de los demás hogares de la comunidad en 2013$\to$2016, calculada en cada partición",
        r"    solo con los hogares de entrenamiento. Las dos tasas se ajustan hacia el promedio de",
        r"    la zona con un estimador bayesiano empírico. Las celdas reportan el cambio en el",
        r"    AUC-PR y su IC95\% (\emph{bootstrap} pareado de 2{,}000 remuestras por comunidad).",
        r"    \textsuperscript{*}: el intervalo excluye el cero. Hogares con comunidad",
        rf"    identificada: {int(n['monetaria'][0]):,} (monetaria) y {int(n['ipm'][0]):,} (IPM). Referencia al azar:".replace(",", "{,}"),
        rf"    {prev['monetaria'][0]:.3f} (monetaria) y {prev['ipm'][0]:.3f} (IPM).",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def tabla_icc(icc: pd.DataFrame) -> str:
    filas = [("nivel", "2010", "Nivel de pobreza, 2010"), ("nivel", "2013", "Nivel de pobreza, 2013"),
             ("nivel", "2016", "Nivel de pobreza, 2016"),
             ("entrada", "2010-2013", r"Entrada en pobreza, 2010$\to$2013"),
             ("entrada", "2013-2016", r"Entrada en pobreza, 2013$\to$2016")]
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Parte de la variación del nivel de pobreza y de la entrada en pobreza que",
        r"  concentra la comunidad, pobreza monetaria e IPM, 2010 a 2016.}",
        r"  \label{tab:robustez_icc}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{l ccc ccc}",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{\textbf{Monetaria}} & \multicolumn{3}{c}{\textbf{IPM}} \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"    & \textbf{Total} & \textbf{Urbano} & \textbf{Rural} & \textbf{Total} & \textbf{Urbano} & \textbf{Rural} \\",
        r"    \midrule",
    ]
    for medida, periodo, etiqueta in filas:
        valores = []
        for definicion in ("monetaria", "ipm"):
            for zona in ZONAS:
                r = icc[(icc.medida == medida) & (icc.periodo == periodo) & (icc.definicion == definicion) & (icc.zona == zona)]
                assert len(r) == 1, f"falta {medida} {periodo} {definicion} {zona}"
                valores.append(f"{100 * r['icc'].iloc[0]:.1f}")
        lineas.append(f"    {etiqueta} & " + " & ".join(valores) + r" \\")
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Porcentaje de la variación que corresponde a la",
        r"    comunidad: correlación intraclase de un análisis de varianza de un factor, corregida",
        r"    por el tamaño desigual de las comunidades (Anexo~\ref{apx:variacion_dmsp}). Nivel de",
        r"    pobreza: todos los hogares de cada ola. Entrada en pobreza: hogares no pobres al",
        r"    comienzo de cada ventana, con o sin dato de DMSP-OLS. Se excluyen los hogares sin",
        r"    comunidad identificada.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def main() -> None:
    techo = tabla_techo(pd.read_csv(RES / "techo_resultados.csv"))
    icc = tabla_icc(pd.read_csv(RES / "techo_icc.csv", dtype={"periodo": str}))
    OUT_TECHO.write_text(techo, encoding="utf-8")
    OUT_ICC.write_text(icc, encoding="utf-8")
    print(f"Tablas exportadas: {OUT_TECHO}, {OUT_ICC}\n\n{techo}\n{icc}")


if __name__ == "__main__":
    main()
