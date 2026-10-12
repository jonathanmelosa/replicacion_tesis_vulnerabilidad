"""
tabla_robustez_historia.py
=====================================
AUC-PR sin historia del hogar y cambio al agregarla. Anexo "Robustez de
la validacion y margenes de informacion", Seccion 5.2 del paper.

No recalcula nada: lee las salidas de
`src/05_model/robustez_historia_hogar.py` (validacion cruzada de 10
particiones x 5 semillas dentro de 2013->2016). Ademas imprime el conteo
de diferencias dentro de grupo que cita el texto del anexo (ninguna
positiva y significativa).

INPUT

    data/processed/benchmark_resultados/robustez_asesor_2026_10/historia_resultados.csv

OUTPUT

    paper/tables/tab_robustez_historia.tex

COMO CORRER

    python src/tabla_robustez_historia.py
"""

from pathlib import Path

import pandas as pd

import estilo_tablas as et
from tabla_robustez_validacion import ALGORITMOS, num

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "robustez_asesor_2026_10" / "historia_resultados.csv"
OUTPUT = REPO_ROOT / "paper" / "tables" / "tab_robustez_historia.tex"

PANELES = [("B", "Sin ingreso ni gasto (Modelo B)"), ("A", "Con ingreso y gasto (Modelo A)")]
VARIANTES = ["estado_previo", "cambios", "ambas"]


def celda(r: pd.Series) -> str:
    marca = "" if r["delta_ap_cruza_cero"] else r"\textsuperscript{*}"
    return f"{num(r['delta_ap'])}{marca} ({num(r['delta_ap_ci95_low'])}, {num(r['delta_ap_ci95_high'])})"


def generar_tex(datos: pd.DataFrame) -> str:
    todos = datos[datos["grupo"] == "todos"]
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{AUC-PR sin historia del hogar y cambio al agregarla, pobreza monetaria,",
        r"  validación cruzada dentro de 2013$\to$2016.}",
        r"  \label{tab:robustez_historia}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l c ccc}",
        r"    \toprule",
        r"    \textbf{Modelo} & \textbf{Sin historia} & \textbf{Estado previo} & \textbf{Cambios} & \textbf{Ambas} \\",
        r"    \midrule",
    ]
    for i, (espec, titulo) in enumerate(PANELES):
        if i > 0:
            lineas.append(r"    \addlinespace")
        lineas.append(rf"    \multicolumn{{5}}{{l}}{{\textit{{{titulo}}}}} \\")
        for clave, nombre in ALGORITMOS:
            sub = todos[(todos.especificacion == espec) & (todos.algoritmo == clave)].set_index("variante")
            assert set(sub.index) == set(VARIANTES), f"faltan variantes en {espec} {clave}"
            base = sub["ap_ref"].unique()
            assert len(base) == 1, "la referencia debe ser la misma en las tres variantes"
            etiqueta = nombre + (" (principal)" if espec == "B" and clave == "Logistica" else "")
            celdas = " & ".join(celda(sub.loc[v]) for v in VARIANTES)
            lineas.append(f"    \\quad {etiqueta} & {num(base[0])} & {celdas} \\\\")
    prevalencia = todos["prevalencia"].unique()
    assert len(prevalencia) == 1
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Sin historia: AUC-PR del modelo con las covariables de",
        r"    2013. Estado previo: si el hogar era pobre en 2010 y su brecha frente a la línea de",
        r"    pobreza. Cambios: diferencia entre 2013 y 2010 en las variables numéricas no",
        r"    monetarias que más pesan en los modelos de la especificación. Las celdas reportan",
        r"    el cambio en el AUC-PR y su IC95\% (\emph{bootstrap} pareado de 2{,}000 remuestras",
        r"    por comunidad). \textsuperscript{*}: el intervalo excluye el cero. Referencia al",
        rf"    azar: {prevalencia[0]:.3f}.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def main() -> None:
    datos = pd.read_csv(INPUT)
    tex = generar_tex(datos)
    OUTPUT.write_text(tex, encoding="utf-8")
    dentro = datos[datos["grupo"] != "todos"]
    sig = ~dentro["delta_ap_cruza_cero"]
    print(f"Diferencias dentro de grupo: {len(dentro)}; positivas y significativas: "
          f"{int((sig & (dentro['delta_ap'] > 0)).sum())}; negativas y significativas: "
          f"{int((sig & (dentro['delta_ap'] < 0)).sum())}")
    print(f"Tabla exportada: {OUTPUT}\n\n{tex}")


if __name__ == "__main__":
    main()
