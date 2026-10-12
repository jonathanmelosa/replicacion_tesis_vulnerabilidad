"""
tabla_split_trayectoria_nucleo.py
=====================================
Tabla LaTeX del Cuarto hallazgo de la Seccion 5.2: hogares que entran en
pobreza en 2010->2013, transitorios (vuelven a salir antes de 2016) frente a
persistentes (siguen pobres en 2016), en las variables del nucleo SHAP.
Reemplaza a `tabla_perfil_split_trayectoria.py` (perfil univariado de 53
variables, retirado del paper el 2026-09-30).

Se muestran los indicadores con |diferencia estandarizada| >= 0.20 (columna
`sustantiva` de `eda_split_trayectoria_nucleo.py`), ordenados por su valor
absoluto. Se marcan con $^{\ddagger}$ las variables sin direccion estable o
moderada en SHAP (menos de 8 de 12 celdas consistentes en
`sensibilidad_signo_nucleo_por_variable.csv`, o categoricas, a las que no se
les mide direccion). Todos los numeros salen de los CSV fuente.

INPUTS
    outputs/tables/eda_transicion_covariables/split_trayectoria_nucleo_monetaria.csv
    data/processed/benchmark_resultados/sensibilidad_signo_nucleo_por_variable.csv

OUTPUTS
    paper/tables/tab_split_trayectoria_nucleo.tex

COMO CORRER
    python src/tabla_split_trayectoria_nucleo.py
"""

from pathlib import Path

import pandas as pd

from etiquetas_variables import etiqueta
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
RUTA_SPLIT = REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables" / "split_trayectoria_nucleo_monetaria.csv"
RUTA_SIGNO = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "sensibilidad_signo_nucleo_por_variable.csv"
OUTPUT_PATH = REPO_ROOT / "paper" / "tables" / "tab_split_trayectoria_nucleo.tex"

MIN_CELDAS_DIRECCION = 8  # de 12, igual que graf_perfil_nucleo_grupos.py
GRUPOS = ["Siempre pobre", "Sale de la pobreza", "Entra (transitorio)", "Entra (persistente)", "Nunca pobre"]


def rotulo(fila: pd.Series) -> str:
    """Etiqueta legible; para categoricas agrega el nivel mostrado."""
    base = etiqueta(fila["variable"])
    if " = " in fila["indicador"]:
        nivel = fila["indicador"].split(" = ", 1)[1].removesuffix(" (%)")
        return f"{base}: {nivel.lower()} (\\%)"
    if fila["indicador"].endswith("(%)"):
        return f"{base} (\\%)"
    return base


def main() -> None:
    df = pd.read_csv(RUTA_SPLIT)
    signo = pd.read_csv(RUTA_SIGNO).set_index("variable")["celdas_consistente"]
    estables = set(signo[signo >= MIN_CELDAS_DIRECCION].index)
    sub = df[df["sustantiva"]].copy()
    n_total = len(df)

    filas = []
    for _, f in sub.iterrows():
        marca = "" if f["variable"] in estables else "$^{\\ddagger}$"
        decimales = 1 if f["indicador"].endswith("(%)") or abs(f["Nunca pobre"]) >= 10 else 2
        valores = " & ".join(f"{f[g]:.{decimales}f}" for g in GRUPOS)
        filas.append(f"    {rotulo(f)}{marca} & {valores} & {f['dif_estandarizada']:+.2f} \\\\")

    n_t, n_p = int(sub["n_transitorio"].max()), int(sub["n_persistente"].max())
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Hogares que entran en pobreza según su trayectoria posterior: "
        r"transitorios frente a persistentes, pobreza monetaria, 2010$\to$2013$\to$2016.}",
        r"  \label{tab:split_trayectoria_nucleo}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        # Sin \resizebox (2026-10-01): primera columna en parrafo para que la
        # tabla quepa con el mismo tamano de letra que las demas.
        r"  \begin{tabular}{>{\raggedright\arraybackslash}p{5.2cm}rrrrrr}",
        r"    \toprule",
        r"    \textbf{Variable (ola 2010)} & \textbf{Siempre} & \textbf{Sale} & \textbf{Transitorio} "
        r"& \textbf{Persistente} & \textbf{Nunca} & \textbf{\shortstack{Dif.\\estand.}} \\",
        r"    \midrule",
        *filas,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        rf"    \footnotesize \textit{{Nota:}} Transitorios ($n={n_t}$): entran en pobreza en "
        rf"2010$\to$2013 y vuelven a salir en 2016; persistentes ($n={n_p}$): siguen pobres. "
        r"Promedios ponderados por el factor de expansión longitudinal. Se muestran los "
        rf"{len(sub)} de {n_total} indicadores del núcleo SHAP (las variables categóricas, un "
        r"indicador por nivel) cuya diferencia estandarizada entre persistentes y transitorios es de "
        r"al menos 0.20 en valor absoluto; el signo negativo indica un valor menor en los persistentes. "
        r"$^{\ddagger}$: variable sin dirección estable o moderada en SHAP (o categórica, sin "
        r"dirección medida), que se lee solo como asociación. De los 723 hogares que entran en "
        r"pobreza entre 2010 y 2013, 610 tienen dato de 2016. " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    OUTPUT_PATH.write_text(aplicar_signo_menos("\n".join(lineas)) + "\n", encoding="utf-8")
    print("\n".join(filas))
    print(f"Tabla exportada: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
