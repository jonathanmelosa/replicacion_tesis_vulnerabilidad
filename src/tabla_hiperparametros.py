"""
tabla_hiperparametros.py
=====================================
Tabla del Anexo con los hiperparametros finales (y la estrategia de
balanceo y el umbral de clasificacion elegidos; el umbral agregado el
2026-09-25 para que las notas de las matrices de confusion remitan solo
a este anexo) de TODOS los modelos cuyos resultados se reportan en la
tesis
incluyendo los de multiclase, deben tener los hiperparametros reportados
en ese anexo". No reentrena nada: lee la columna `hiperparametros` (JSON)
de cada registro de resultados.

Bloques (y donde se usan en `main.tex`)
    1. Pobreza monetaria, holdout temporal -- A, B, A+DMSP, B+DMSP
       (Tabla de desempeno de la Seccion 5.1 y tabla marginal de DMSP-OLS).
       Registro: registro_modelos_fbeta2_cv10.csv
    2. IPM, holdout temporal -- A, B, A+DMSP, B+DMSP (Anexo de desempeno
       IPM y significancia de DMSP-OLS bajo IPM).
       Registro: registro_modelos_ipm.csv
    3. Tres fuentes geoespaciales, validacion cruzada dentro de 2010->2013
       -- A, B (sin geo) y A+3 fuentes, B+3 fuentes (Anexo geo3).
       Registros: registro_modelos_geo3_baseline.csv, registro_modelos_geo3_robusto.csv
    4. Multiclase (4 grupos) -- A4 y B4 (citados en las Secciones 4.1 y
       5.2; las variantes geo del multiclase no se reportan).
       Registro: multiclase/registro_modelos_multiclase.csv

Los analisis SHAP de 2013->2016, la heterogeneidad y el bootstrap reusan
los hiperparametros de estos mismos modelos, asi que no agregan filas.

INPUTS
    data/processed/benchmark_resultados/{registros de arriba}

OUTPUTS
    paper/tables/tab_hiperparametros.tex (longtable)

COMO CORRER
    python src/tabla_hiperparametros.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
OUTPUT_PATH = REPO_ROOT / "paper" / "tables" / "tab_hiperparametros.tex"

BLOQUES = [
    ("Pobreza monetaria, conjunto de prueba 2013$\\to$2016",
     [("registro_modelos_fbeta2_cv10.csv", None)],
     {"A": "A", "B": "B", "AgeoDMSP": "A + DMSP", "BgeoDMSP": "B + DMSP"}),
    ("Pobreza multidimensional (IPM), conjunto de prueba 2013$\\to$2016",
     [("registro_modelos_ipm.csv", None)],
     {"Aipm": "A", "Bipm": "B", "AipmgeoDMSP": "A + DMSP", "BipmgeoDMSP": "B + DMSP"}),
    ("Tres fuentes geoespaciales, pobreza monetaria, validación cruzada en 2010$\\to$2013",
     [("registro_modelos_geo3_baseline.csv", None), ("registro_modelos_geo3_robusto.csv", None)],
     {"A": "A", "B": "B", "Ageo3": "A + 3 fuentes", "Bgeo3": "B + 3 fuentes"}),
    ("Multiclase (cuatro grupos), pobreza monetaria, conjunto de prueba 2013$\\to$2016",
     [("multiclase/registro_modelos_multiclase.csv", ["A4", "B4"])],
     {"A4": "A", "B4": "B"}),
]

ALGORITMOS = {
    "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "Red neuronal (MLP)": "Red neuronal (MLP)",
}
ORDEN_ALG = list(ALGORITMOS.values())

# class_weight / scale_pos_weight ya estan en la columna de balanceo.
CLAVES_OMITIDAS = {"class_weight", "scale_pos_weight"}


def _formatear_valor(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    if isinstance(v, list):
        return "(" + ", ".join(_formatear_valor(x) for x in v) + ")"
    return str(v)


def formatear_hiperparametros(texto_json: str) -> str:
    h = json.loads(texto_json)
    partes = []
    for k in sorted(h):
        nombre = k.replace("modelo__", "")
        if nombre in CLAVES_OMITIDAS:
            continue
        partes.append(f"{nombre.replace('_', chr(92) + '_')} = {_formatear_valor(h[k])}")
    return "; ".join(partes)


def _umbral(fila) -> str:
    """Umbral de clasificacion promedio sobre las semillas (el mismo de las
    tablas de desempeno). El multiclase no tiene umbral: asigna la clase
    mas probable."""
    u = fila.get("umbral_clasificacion_media")
    return "--" if pd.isna(u) else f"{u:.2f}"


def cargar_bloque(fuentes, especs_validas) -> pd.DataFrame:
    frames = []
    for archivo, filtro in fuentes:
        ruta = RESULTADOS / archivo
        if not ruta.exists():
            print(f"ERROR: no se encontró {ruta}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(ruta)
        if filtro:
            df = df[df["especificacion"].isin(filtro)]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["especificacion"].isin(especs_validas)].copy()
    dup = df.duplicated(["algoritmo", "especificacion"]).sum()
    if dup:
        print(f"ERROR: {dup} filas duplicadas algoritmo x especificacion", file=sys.stderr)
        sys.exit(1)
    return df


def main() -> None:
    # Mismo tamano de letra y formato de nota que el resto de tablas (2026-10-01).
    lineas = [
        r"{\footnotesize",
        r"\captionsetup{font=normalsize}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{lllc>{\raggedright\arraybackslash}p{0.36\textwidth}}",
        r"  \caption{Hiperparámetros finales, estrategia de balanceo y umbral de clasificación de los modelos reportados.}",
        r"  \label{tab:hiperparametros} \\",
        r"  \toprule",
        r"  \textbf{Algoritmo} & \textbf{Espec.} & \textbf{Balanceo} & \textbf{Umbral} & \textbf{Hiperparámetros} \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \toprule",
        r"  \textbf{Algoritmo} & \textbf{Espec.} & \textbf{Balanceo} & \textbf{Umbral} & \textbf{Hiperparámetros} \\",
        r"  \midrule",
        r"  \endhead",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]
    n_total = 0
    for i, (titulo, fuentes, especs) in enumerate(BLOQUES):
        df = cargar_bloque(fuentes, set(especs))
        df["alg"] = df["algoritmo"].map(ALGORITMOS)
        if df["alg"].isna().any():
            print(f"ERROR: algoritmo sin nombre: {df.loc[df['alg'].isna(), 'algoritmo'].unique()}", file=sys.stderr)
            sys.exit(1)
        df["orden_alg"] = df["alg"].map(ORDEN_ALG.index)
        df["orden_esp"] = df["especificacion"].map(list(especs).index)
        df = df.sort_values(["orden_alg", "orden_esp"])
        if i > 0:
            lineas.append(r"  \midrule")
        lineas.append(f"  \\multicolumn{{5}}{{l}}{{\\textit{{{titulo}}}}} \\\\*")
        for _, f in df.iterrows():
            lineas.append(
                f"  {f['alg']} & {especs[f['especificacion']]} & {f['balanceo_elegido']} & {_umbral(f)} & "
                f"{formatear_hiperparametros(f['hiperparametros'])} \\\\"
            )
        n_total += len(df)
        print(f"{titulo}: {len(df)} modelos")
    lineas += [
        r"\end{longtable}",
        r"\vspace{-\LTpost}",
        r"\textit{Nota:} Espec.: especificación (A con ingreso y gasto del hogar, B sin",
        r"ellos). Umbral: promedio sobre las semillas del umbral elegido por validación",
        r"cruzada maximizando F$_2$; ``--'': sin umbral (el modelo multiclase asigna el",
        r"grupo más probable) o no registrado. " + FUENTE + r"\par",
        r"}",
    ]
    OUTPUT_PATH.write_text(aplicar_signo_menos("\n".join(lineas)) + "\n", encoding="utf-8")
    print(f"\nGuardado: {OUTPUT_PATH} ({n_total} modelos)")


if __name__ == "__main__":
    main()
