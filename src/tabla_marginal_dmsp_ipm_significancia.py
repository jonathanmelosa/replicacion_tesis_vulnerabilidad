"""
tabla_marginal_dmsp_ipm_significancia.py
=====================================
Tabla de contribucion marginal de DMSP-OLS bajo pobreza MULTIDIMENSIONAL
(IPM) -- analoga a `tabla_marginal_dmsp_fbeta2_cv10.py` (monetaria), pero
con el foco puesto en la SIGNIFICANCIA del cambio de AUC-ROC al agregar
DMSP-OLS (columna Delta del bootstrap pareado), no solo en el valor de AUC
de cada especificacion por separado -- por eso el formato es una fila por
PAR (base vs. +DMSP-OLS), no una fila por especificacion como en la
version monetaria. El IC95% y los valores p (sin y con corregir por
clustering comunitario) NO van en columnas propias de la tabla -- se resumen en
la nota al pie, generada dinamicamente en `_nota_significancia()` a partir
de los pares con Delta AUC-ROC significativo, junto con la explicacion de
por que se corrige por clustering comunitario (resolucion espacial gruesa
de DMSP-OLS, verificada: 30 arcosegundos/~1km vs. 15 arcosegundos/~500m de
VIIRS, su sucesor, con mejor calibracion y menos saturacion -- la
limitacion de granularidad de DMSP-OLS es precisamente lo que motivo el
diseno de VIIRS).

Por que esta tabla existe: a diferencia de monetaria (efecto agregado nulo
en las 10 combinaciones, bootstrap sin cluster y con cluster), bajo IPM el
bootstrap de `diagnostico_bootstrap_ipm.py`/`diagnostico_bootstrap_cluster_
ipm.py` encuentra un efecto ESTADISTICAMENTE SIGNIFICATIVO (IC95% no cruza
cero) en 3 de 10 combinaciones (XGBoost-A, XGBoost-B, LightGBM-B) -- un
resultado que hasta ahora no tenia tabla propia en el documento.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_ipm.csv
    data/processed/benchmark_resultados/diagnostico_bootstrap_ipm.csv
    data/processed/benchmark_resultados/diagnostico_bootstrap_cluster_ipm.csv
    data/processed/benchmark_resultados/diagnostico_bootstrap_dmsp.csv
    data/processed/benchmark_resultados/diagnostico_bootstrap_cluster_dmsp.csv
        (panel monetario de la tabla del anexo)
        (cobertura parcial: solo XGBoost, HistGradientBoosting, Logistica
        -- Random Forest y LightGBM no tienen bootstrap cluster-robusto
        corrido; se resume como "sin cobertura" en la nota si aplica a un
        par significativo)

OUTPUTS

    paper/tables/tab_marginal_dmsp_ipm_significancia.tex
    paper/tables/tab_significancia_dmsp_ipm.tex (detalle para el anexo)

COMO CORRER

    python src/tabla_marginal_dmsp_ipm_significancia.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = REPO_ROOT / "data" / "processed" / "benchmark_resultados"

CONFIG = {
    "registro": RESULTADOS_DIR / "registro_modelos_ipm.csv",
    "bootstrap": RESULTADOS_DIR / "diagnostico_bootstrap_ipm.csv",
    "bootstrap_cluster": RESULTADOS_DIR / "diagnostico_bootstrap_cluster_ipm.csv",
    "bootstrap_monetaria": RESULTADOS_DIR / "diagnostico_bootstrap_dmsp.csv",
    "bootstrap_cluster_monetaria": RESULTADOS_DIR / "diagnostico_bootstrap_cluster_dmsp.csv",
    "algoritmos_orden": [
        "XGBoost",
        "Random Forest",
        "LightGBM",
        "HistGradientBoosting (sklearn)",
        "Logistica regularizada (elastic net, benchmark)",
    ],
    "nombres_algoritmo": {
        "XGBoost": "XGBoost",
        "Random Forest": "Random Forest",
        "LightGBM": "LightGBM",
        "HistGradientBoosting (sklearn)": "HistGradientBoosting",
        "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
    },
    # nombres cortos usados en diagnostico_bootstrap*_ipm.csv (columna "algoritmo")
    "nombres_bootstrap": {
        "XGBoost": "XGBoost",
        "Random Forest": "Random Forest",
        "LightGBM": "LightGBM",
        "HistGradientBoosting (sklearn)": "HistGradientBoosting",
        "Logistica regularizada (elastic net, benchmark)": "Logistica",
    },
    "pares_especificacion": [("Aipm", "AipmgeoDMSP"), ("Bipm", "BipmgeoDMSP")],
    "etiqueta_especificacion": {"Aipm": "A", "Bipm": "B"},
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar(cfg: dict) -> tuple:
    faltantes = [p for p in (cfg["registro"], cfg["bootstrap"], cfg["bootstrap_cluster"]) if not p.exists()]
    if faltantes:
        print(f"ERROR: no se encontraron {faltantes}", file=sys.stderr)
        sys.exit(1)
    return (
        pd.read_csv(cfg["registro"]),
        pd.read_csv(cfg["bootstrap"]),
        pd.read_csv(cfg["bootstrap_cluster"]),
    )


def fila_tex(registro: pd.DataFrame, boot: pd.DataFrame,
             algoritmo_raw: str, espec_base: str, espec_geo: str, cfg: dict) -> str:
    nombre = cfg["nombres_algoritmo"][algoritmo_raw]
    nombre_boot = cfg["nombres_bootstrap"][algoritmo_raw]
    etiqueta = cfg["etiqueta_especificacion"][espec_base]

    r_base = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec_base)].iloc[0]
    r_geo = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec_geo)].iloc[0]

    b = boot[(boot.algoritmo == nombre_boot) & (boot.especificacion_base == espec_base)]
    if b.empty:
        print(f"ERROR: sin fila de bootstrap para {nombre_boot}/{espec_base}", file=sys.stderr)
        sys.exit(1)
    b = b.iloc[0]

    delta_top10 = r_geo["precision_top10_media"] - r_base["precision_top10_media"]

    negrita = not bool(b["cruza_cero"])
    fmt_delta = f"\\textbf{{{b['delta_auc']:+.3f}}}" if negrita else f"{b['delta_auc']:+.3f}"

    return (
        f"    {nombre:<24s} & {etiqueta} & {r_base['auc_roc_media']:.3f} & {r_geo['auc_roc_media']:.3f} & "
        f"{fmt_delta} & {delta_top10:+.3f} \\\\"
    )


def generar_tex(registro: pd.DataFrame, boot: pd.DataFrame, cfg: dict) -> str:
    # Formato alineado con el resto de tablas: sin \\resizebox (con 6 columnas estiraba la tabla y agrandaba
    # la letra), sin espacio entre cada fila, 3 decimales, y una nota corta
    # que remite al anexo para los IC, valores p y la correccion por
    # clustering (antes iban en la nota, ~12 lineas).
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Contribución marginal de DMSP-OLS en el conjunto de prueba,",
        r"  pobreza multidimensional (IPM), 2013$\to$2016.}",
        r"  \label{tab:marginal_dmsp_ipm_significancia}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Espec.} & \textbf{AUC base} & \textbf{AUC +DMSP} & "
        r"$\boldsymbol{\Delta}$\textbf{AUC} & $\boldsymbol{\Delta}$\textbf{Prec.-top10} \\",
        r"    \midrule",
    ]
    for i, (espec_base, espec_geo) in enumerate(cfg["pares_especificacion"]):
        for algoritmo_raw in cfg["algoritmos_orden"]:
            lineas.append(fila_tex(registro, boot, algoritmo_raw, espec_base, espec_geo, cfg))
        if i < len(cfg["pares_especificacion"]) - 1:
            lineas.append(r"    \addlinespace")
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Entrenamiento en 2010$\to$2013. AUC: promedio de 5 semillas.",
        r"    $\Delta$AUC: \emph{bootstrap} pareado sobre el conjunto de prueba",
        r"    (una semilla); en \textbf{negrita}, diferencias cuyo IC95\% no",
        r"    cruza cero. Intervalos, valores $p$ y corrección por",
        r"    conglomerados de comunidad en la",
        r"    Tabla~\ref{tab:significancia_dmsp_ipm} del",
        r"    Anexo~\ref{apx:marginal_dmsp}. " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return aplicar_signo_menos("\n".join(lineas))


def _filas_significancia(boot: pd.DataFrame, boot_cl: pd.DataFrame, especs: list, cfg: dict) -> list:
    """Una fila por algoritmo y especificacion: Delta AUC (negrita si su
    IC95% no cruza cero), IC95%, p sin corregir y p con remuestreo agrupado
    por comunidad ("--" donde no se estimo)."""
    filas = []
    for i, espec_base in enumerate(especs):
        for algoritmo_raw in cfg["algoritmos_orden"]:
            nombre_boot = cfg["nombres_bootstrap"][algoritmo_raw]
            b = boot[(boot.algoritmo == nombre_boot) & (boot.especificacion_base == espec_base)].iloc[0]
            c = boot_cl[(boot_cl.algoritmo == nombre_boot) & (boot_cl.especificacion_base == espec_base)]
            p_cl = f"{c.iloc[0]['p_valor_cluster']:.3f}" if not c.empty else "--"
            delta = f"{b['delta_auc']:+.3f}"
            if not bool(b["cruza_cero"]):
                delta = f"\\textbf{{{delta}}}"
            filas.append(
                f"    {cfg['nombres_algoritmo'][algoritmo_raw]:<24s} & {espec_base[0]} & "
                f"{delta} & [{b['ci95_low']:+.3f}, {b['ci95_high']:+.3f}] & {b['p_valor']:.3f} & {p_cl} \\\\"
            )
        if i < len(especs) - 1:
            filas.append(r"    \addlinespace")
    return filas


def generar_tex_anexo(boot: pd.DataFrame, boot_cl: pd.DataFrame,
                      boot_mon: pd.DataFrame, boot_cl_mon: pd.DataFrame, cfg: dict) -> str:
    """Detalle de significancia del cambio de AUC-ROC al agregar DMSP-OLS.
    Panel IPM: los 10 pares de la tabla principal. Panel monetario
: las Conclusiones afirman que
    bajo pobreza monetaria el IC95% cruza cero en las 10 combinaciones, y
    ninguna tabla del documento lo mostraba. La correccion por comunidad se
    corrio para XGBoost, HistGradientBoosting y Logistica, no para Random
    Forest ni LightGBM."""
    n_col = 6
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Significancia del cambio en el AUC-ROC al agregar DMSP-OLS,",
        r"  pobreza monetaria e IPM, 2013$\to$2016.}",
        r"  \label{tab:significancia_dmsp_ipm}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Espec.} & $\boldsymbol{\Delta}$\textbf{AUC} & "
        r"\textbf{IC95\%} & $\boldsymbol{p}$ & $\boldsymbol{p}$ \textbf{por comunidad} \\",
        r"    \midrule",
        f"    \\multicolumn{{{n_col}}}{{l}}{{\\textit{{Pobreza monetaria}}}} \\\\",
        *_filas_significancia(boot_mon, boot_cl_mon, ["A", "B"], cfg),
        r"    \addlinespace",
        f"    \\multicolumn{{{n_col}}}{{l}}{{\\textit{{Pobreza multidimensional (IPM)}}}} \\\\",
        *_filas_significancia(boot, boot_cl, [b for b, _ in cfg["pares_especificacion"]], cfg),
    ]
    n_cl = int(boot_cl["n_clusters"].iloc[0])
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} $\Delta$AUC, IC95\% y $p$: \emph{bootstrap} pareado sobre el",
        r"    conjunto de prueba (una semilla). $p$ por comunidad:",
        f"    remuestreo agrupado por comunidad ({n_cl:,} comunidades)".replace(",", "{,}") + ",",
        r"    porque hogares de una misma comunidad comparten en buena medida el",
        r"    mismo píxel de DMSP-OLS (resolución de $\sim$1\,km); ``--'':",
        r"    corrección no estimada. " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return aplicar_signo_menos("\n".join(lineas))


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    registro, boot, boot_cl = cargar(cfg)
    for nombre, tex in {
        "tab_marginal_dmsp_ipm_significancia.tex": generar_tex(registro, boot, cfg),
        "tab_significancia_dmsp_ipm.tex": generar_tex_anexo(
            boot, boot_cl,
            pd.read_csv(cfg["bootstrap_monetaria"]), pd.read_csv(cfg["bootstrap_cluster_monetaria"]), cfg),
    }.items():
        (out_dir / nombre).write_text(tex, encoding="utf-8")
        print(f"Tabla exportada: {out_dir / nombre}")


if __name__ == "__main__":
    main()
