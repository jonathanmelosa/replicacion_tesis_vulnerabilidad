"""
tabla_multiclase_anexo.py
=====================================
Tablas del anexo del modelo multiclase:
la Seccion 5.2 (Tercer hallazgo) cita sus AUC por par de grupos, pero
ningun anexo los reportaba. Solo la especificacion que usa el texto:
Modelo B (sin ingreso ni gasto), holdout temporal (entrenamiento
2010->2013, prueba 2013->2016), sin variables geoespaciales. El Modelo A
se omite porque el ingreso de la ola base define "entra" frente a "sale"
y su AUC de ese par (~1.0) es una tautologia. Los hiperparametros de
ambos modelos ya estan en `tab_hiperparametros.tex`; no se repiten aqui.

INPUTS

    data/processed/benchmark_resultados/multiclase/registro_modelos_multiclase.csv
        (metricas: promedio de 5 semillas)
    data/processed/benchmark_resultados/multiclase/diagnostico_bootstrap_cluster_multiclase.csv
        (AUC por par, semilla 42, con IC95 por bootstrap por comunidad, y
        cambio al agregar DMSP-OLS con su IC95; lo genera
        `src/05_model/diagnostico_bootstrap_cluster_multiclase.py`)
    data/processed/benchmark_resultados/multiclase/auc_pares_multiclase_semillas_detalle.csv
        (AUC por par con 5 semillas, para reportar en la nota cuanto difiere
        la semilla 42 del promedio; `src/05_model/auc_pares_multiclase_semillas.py`)

Cambio 2026-09-30: la tabla de AUC por par pasa de una sola corrida sin
intervalos a AUC con IC95 por bootstrap por comunidad (misma convencion
que el modelo binario). El punto es el de la semilla 42 para que coincida
con su intervalo; la nota informa la diferencia maxima con el promedio de
5 semillas.

OUTPUTS

    paper/tables/tab_multiclase_metricas.tex
    paper/tables/tab_multiclase_auc_pares.tex

COMO CORRER

    python src/tabla_multiclase_anexo.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
MULTICLASE = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "multiclase"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"
ESPECIFICACION = "B4"

ALGORITMOS = {
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "Random Forest": "Random Forest",
    "Red neuronal (MLP)": "Red neuronal (MLP)",
    "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
}

METRICAS = [
    ("accuracy_media", r"\shortstack{Exactitud}"),
    ("balanced_accuracy_media", r"\shortstack{Exactitud\\balanceada}"),
    ("f1_macro_media", r"\shortstack{F1\\macro}"),
    ("auc_ovr_macro_media", r"\shortstack{AUC-ROC\\macro}"),
    ("auc_entra_vs_resto_media", r"\shortstack{AUC entra\\frente al resto}"),
    ("recall_entra_media", r"\shortstack{Recall\\entra}"),
    ("precision_top10_entra_media", r"\shortstack{Prec.-top10\\entra}"),
]

PARES = [
    ("auc_nunca_pobre_vs_siempre_pobre", r"\shortstack{Nunca frente\\a siempre}"),
    ("auc_nunca_pobre_vs_entra", r"\shortstack{Nunca frente\\a entra}"),
    ("auc_nunca_pobre_vs_sale", r"\shortstack{Nunca frente\\a sale}"),
    ("auc_entra_vs_siempre_pobre", r"\shortstack{Entra frente\\a siempre}"),
    ("auc_sale_vs_siempre_pobre", r"\shortstack{Sale frente\\a siempre}"),
    ("auc_entra_vs_sale", r"\shortstack{\textbf{Entra frente}\\\textbf{a sale}}"),
]


def _cargar(nombre: str, especificacion: str = ESPECIFICACION) -> pd.DataFrame:
    df = pd.read_csv(MULTICLASE / nombre)
    df = df[df["especificacion"] == especificacion].copy()
    df["algoritmo"] = df["algoritmo"].map(ALGORITMOS)
    orden = list(ALGORITMOS.values())
    return df.sort_values("algoritmo", key=lambda s: s.map(orden.index))


def _tabla(caption: str, label: str, columnas: list, filas: list, nota: str) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        "  \\begin{tabular}{l" + "c" * len(columnas) + "}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & " + " & ".join(columnas) + r" \\",
        r"    \midrule",
        *filas,
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        f"    \\footnotesize \\textit{{Nota:}} {nota} {FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return aplicar_signo_menos("\n".join(lineas)) + "\n"


def tabla_metricas(registro: pd.DataFrame) -> str:
    columnas = [r"\textbf{Balanceo}"] + [f"\\textbf{{{t}}}" for _, t in METRICAS]
    filas = [
        f"    {f['algoritmo']} & {f['balanceo_elegido']} & "
        + " & ".join(f"{f[c]:.3f}" for c, _ in METRICAS) + r" \\"
        for _, f in registro.iterrows()
    ]
    n_test = f"{int(registro['n_test'].iloc[0]):,}".replace(",", "{,}")
    nota = (
        "Modelo B (sin ingreso ni gasto), entrenado en 2010$\\to$2013 y "
        f"evaluado en 2013$\\to$2016 ($n={n_test}$ hogares de los cuatro "
        "grupos), promedio de 5 semillas. El modelo "
        "asigna a cada hogar el grupo más probable, sin umbral ajustable, de "
        "modo que casi nunca asigna ``entra'', el grupo más pequeño "
        "(\\emph{recall} cercano a cero); por eso se usa como diagnóstico y "
        "no como modelo principal. Prec.-top10 entra: fracción de hogares "
        "que efectivamente entran en pobreza entre el 10\\% con mayor "
        "probabilidad de ``entra''. Hiperparámetros en la "
        "Tabla~\\ref{tab:hiperparametros}."
    )
    return _tabla(
        "Desempeño del modelo multiclase (cuatro grupos) en el conjunto de prueba, "
        "pobreza monetaria, 2013$\\to$2016.",
        "tab:multiclase_metricas", columnas, filas, nota,
    )


def tabla_auc_pares(boot: pd.DataFrame, semillas: pd.DataFrame) -> str:
    """Panel superior: AUC por par del Modelo B con IC95 (bootstrap por
    comunidad). Panel inferior: cambio al agregar DMSP-OLS (B4geoDMSP - B4)
    con su IC95; $^{\\dagger}$ marca los intervalos que excluyen el cero."""
    columnas = [f"\\textbf{{{t}}}" if "textbf" not in t else t for _, t in PARES]
    n_col = len(PARES) + 1
    orden = list(ALGORITMOS.values())
    boot = boot.assign(algoritmo=boot["algoritmo"].map(ALGORITMOS), par="auc_" + boot["par"])
    punto = boot.pivot(index="algoritmo", columns="par", values="auc_B4").loc[orden]
    lo = boot.pivot(index="algoritmo", columns="par", values="auc_B4_ci95_low").loc[orden]
    hi = boot.pivot(index="algoritmo", columns="par", values="auc_B4_ci95_high").loc[orden]
    d = boot.pivot(index="algoritmo", columns="par", values="delta_auc").loc[orden]
    d_lo = boot.pivot(index="algoritmo", columns="par", values="delta_ci95_low").loc[orden]
    d_hi = boot.pivot(index="algoritmo", columns="par", values="delta_ci95_high").loc[orden]
    cruza = boot.pivot(index="algoritmo", columns="par", values="cruza_cero").loc[orden]
    ic = lambda a, b, fmt: f"{{\\scriptsize [{a:{fmt}}, {b:{fmt}}]}}"

    filas = [f"    \\multicolumn{{{n_col}}}{{l}}{{\\textit{{AUC-ROC, Modelo B}}}} \\\\"]
    for alg in orden:
        filas.append(f"    {alg} & " + " & ".join(f"{punto.loc[alg, c]:.3f}" for c, _ in PARES) + r" \\")
        filas.append("     & " + " & ".join(ic(lo.loc[alg, c], hi.loc[alg, c], ".3f") for c, _ in PARES) + r" \\")
    filas += [r"    \addlinespace", f"    \\multicolumn{{{n_col}}}{{l}}{{\\textit{{Cambio al agregar DMSP-OLS}}}} \\\\"]
    for alg in orden:
        celdas = [f"{d.loc[alg, c]:+.3f}" + ("" if cruza.loc[alg, c] else "$^{\\dagger}$") for c, _ in PARES]
        filas.append(f"    {alg} & " + " & ".join(celdas) + r" \\")
        filas.append("     & " + " & ".join(ic(d_lo.loc[alg, c], d_hi.loc[alg, c], "+.3f") for c, _ in PARES) + r" \\")

    # Diferencia maxima entre la semilla 42 y el promedio de 5 semillas (B4).
    det = semillas[semillas["especificacion"] == "B4"]
    cols = [c for c, _ in PARES]
    media = det.groupby("algoritmo")[cols].mean()
    s42 = det[det["semilla"] == 42].set_index("algoritmo")[cols]
    dif_max = float((s42 - media).abs().to_numpy().max())
    n_boot = int(boot["n_boot_validas"].min())
    n_clusters = int(boot["n_clusters"].iloc[0])
    miles = lambda n: f"{n:,}".replace(",", "{,}")
    n_hogares = miles(int(boot["n_hogares"].iloc[0]))

    nota = (
        "AUC-ROC de cada par de grupos: se restringe a los hogares de los dos "
        "grupos y se usa como puntaje la probabilidad que el modelo asigna al "
        "primero. Modelo B, conjunto de prueba 2013$\\to$2016 "
        f"($n={n_hogares}$ hogares), semilla 42; el promedio de 5 semillas difiere "
        f"como máximo en {dif_max:.3f}. Entre corchetes, intervalo de confianza del "
        f"95\\% por \\emph{{bootstrap}} por comunidad ({miles(n_boot)} remuestras de "
        f"{miles(n_clusters)} comunidades; los hogares sin comunidad identificada cuentan "
        "cada uno como una), pareado entre especificaciones en el panel inferior. "
        "El panel inferior es la diferencia entre el mismo modelo con las 23 "
        "variables de DMSP-OLS y sin ellas; $^{\\dagger}$: intervalo que excluye el cero."
    )
    return _tabla(
        "AUC-ROC del modelo multiclase por par de grupos de la matriz de transición, "
        "pobreza monetaria, 2013$\\to$2016.",
        "tab:multiclase_auc_pares", columnas, filas, nota,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    salidas = {
        "tab_multiclase_metricas.tex": tabla_metricas(_cargar("registro_modelos_multiclase.csv")),
        "tab_multiclase_auc_pares.tex": tabla_auc_pares(
            pd.read_csv(MULTICLASE / "diagnostico_bootstrap_cluster_multiclase.csv"),
            pd.read_csv(MULTICLASE / "auc_pares_multiclase_semillas_detalle.csv")),
    }
    for nombre, tex in salidas.items():
        (OUTPUT_DIR / nombre).write_text(tex, encoding="utf-8")
        print(f"Tabla exportada: {OUTPUT_DIR / nombre}")


if __name__ == "__main__":
    main()
