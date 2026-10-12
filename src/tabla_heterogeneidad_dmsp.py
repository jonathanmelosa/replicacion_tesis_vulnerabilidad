"""
tabla_heterogeneidad_dmsp.py
=====================================
Formatea `diagnostico_heterogeneidad_dmsp.csv` (pobreza monetaria) y
`diagnostico_heterogeneidad_ipm.csv` (IPM) -- ya calculados por
`src/05_model/diagnostico_heterogeneidad_{dmsp,ipm}.py` -- en dos tablas
LaTeX para la Sección "Heterogeneidad territorial" de la tesis: el
$\\Delta$AUC-ROC de agregar DMSP-OLS, por los 4 ejes de heterogeneidad
(zona, cercanía a la LP, estrato, proxy de riqueza) x las 2
especificaciones (A, B) x los 5 algoritmos.

Excluye los grupos marcados `confiable=False` en el CSV fuente (n
insuficiente para que la métrica sea informativa, ej. Estrato 0/4/5 con
menos de 30 hogares) -- documentado en la nota de la tabla, no omitido en
silencio.

INPUTS

    data/processed/benchmark_resultados/diagnostico_heterogeneidad_dmsp.csv
    data/processed/benchmark_resultados/diagnostico_heterogeneidad_ipm.csv

OUTPUTS

    paper/tables/tab_heterogeneidad_dmsp.tex
    paper/tables/tab_heterogeneidad_ipm.tex

CÓMO CORRER

    python src/tabla_heterogeneidad_dmsp.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

ALGORITMOS_ORDEN = ["XGBoost", "Random Forest", "LightGBM", "HistGradientBoosting", "Logistica"]
NOMBRES_CORTOS = {
    "XGBoost": "XGB", "Random Forest": "RF", "LightGBM": "LGBM",
    "HistGradientBoosting": "HGB", "Logistica": "Log.",
}
EJES_ORDEN = [
    ("eje_zona", "Zona"),
    ("eje_brecha_lp", "Cercanía a LP"),
    ("eje_estrato", "Estrato"),
    ("eje_riqueza_proxy", "Servicios y bienes"),
]
# Las etiquetas de los terciles vienen de los scripts de diagnostico y
# decian "(proxy menos discriminante)" en los dos extremos, lo que no
# describe nada: son terciles de un indice de servicios publicos y bienes
# durables.
GRUPOS_RENOMBRADOS = {
    "Bajo (proxy menos discriminante)": "Tercil bajo",
    "Medio": "Tercil medio",
    "Alto (proxy menos discriminante)": "Tercil alto",
}


def cargar(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(ruta)


def generar_tex(df: pd.DataFrame, especificaciones: list, etiquetas_espec: dict, label: str, caption: str) -> str:
    df = df[(df["eje"] != "Global") & (df["confiable"])].copy()

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \footnotesize",
        # Sin \resizebox (2026-10-01): la tabla cabe al ancho del texto y el
        # escalado la agrandaba, con letra mayor que el resto de tablas.
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{llrr" + "r" * len(ALGORITMOS_ORDEN) + "}",
        r"    \toprule",
    ]
    encabezado_algos = " & ".join([f"\\textbf{{{NOMBRES_CORTOS[a]}}}" for a in ALGORITMOS_ORDEN])
    lineas.append(
        r"    \textbf{Eje} & \textbf{Grupo} & \textbf{Espec.} & $\boldsymbol{n}$ & " + encabezado_algos + r" \\"
    )
    lineas.append(r"    \midrule")

    for eje_raw, eje_bonito in EJES_ORDEN:
        grupos = df.loc[df["eje"] == eje_raw, "grupo"].drop_duplicates().tolist()
        for gi, grupo in enumerate(grupos):
            for ei, espec in enumerate(especificaciones):
                sub = df[(df.eje == eje_raw) & (df.grupo == grupo) & (df.especificacion_base == espec)]
                if sub.empty:
                    continue
                n = int(sub["n"].iloc[0])
                celdas = []
                for algo in ALGORITMOS_ORDEN:
                    fila_algo = sub[sub.algoritmo == algo]
                    if fila_algo.empty:
                        celdas.append("--")
                        continue
                    delta = fila_algo.iloc[0]["delta"]
                    celdas.append(f"\\textbf{{{delta:+.3f}}}" if abs(delta) >= 0.02 else f"{delta:+.3f}")
                prefijo_eje = eje_bonito if (gi == 0 and ei == 0) else ""
                lineas.append(
                    f"    {prefijo_eje} & {GRUPOS_RENOMBRADOS.get(grupo, grupo)} & {etiquetas_espec[espec]} & {n:,} & ".replace(",", "{,}")
                    + " & ".join(celdas) + r" \\"
                )
            if not (gi == len(grupos) - 1):
                lineas.append(r"    \addlinespace")
        lineas.append(r"    \midrule")
    if lineas[-1] == r"    \midrule":
        lineas.pop()

    lineas += [
        r"    \bottomrule", r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Cambio en el AUC-ROC del conjunto de prueba al",
        r"    agregar DMSP-OLS, por subgrupo. En \textbf{negrita}: $|\Delta|\geq 0.02$.",
        r"    Se omiten los grupos con $n<30$. Espec.: especificación (A con ingreso y",
        r"    gasto del hogar, B sin ellos). XGB: XGBoost; RF: Random Forest; LGBM:",
        r"    LightGBM; HGB: HistGradientBoosting; Log.: logística regularizada. " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return aplicar_signo_menos("\n".join(lineas))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dmsp = cargar(RESULTADOS_DIR / "diagnostico_heterogeneidad_dmsp.csv")
    tex_dmsp = generar_tex(
        dmsp, ["A", "B"], {"A": "A", "B": "B"},
        label="tab:heterogeneidad_dmsp",
        caption=(
            r"Cambio en el AUC-ROC al agregar DMSP-OLS por subgrupo, pobreza "
            r"monetaria, 2013$\to$2016."
        ),
    )
    (OUTPUT_DIR / "tab_heterogeneidad_dmsp.tex").write_text(tex_dmsp, encoding="utf-8")
    print(f"Tabla exportada: {OUTPUT_DIR / 'tab_heterogeneidad_dmsp.tex'}")

    ipm = cargar(RESULTADOS_DIR / "diagnostico_heterogeneidad_ipm.csv")
    tex_ipm = generar_tex(
        ipm, ["Aipm", "Bipm"], {"Aipm": "A", "Bipm": "B"},
        label="tab:heterogeneidad_ipm",
        caption=(
            r"Cambio en el AUC-ROC al agregar DMSP-OLS por subgrupo, pobreza "
            r"multidimensional (IPM), 2013$\to$2016."
        ),
    )
    (OUTPUT_DIR / "tab_heterogeneidad_ipm.tex").write_text(tex_ipm, encoding="utf-8")
    print(f"Tabla exportada: {OUTPUT_DIR / 'tab_heterogeneidad_ipm.tex'}")


if __name__ == "__main__":
    main()
