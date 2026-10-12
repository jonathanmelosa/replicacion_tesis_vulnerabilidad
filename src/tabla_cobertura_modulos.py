"""
tabla_cobertura_modulos.py
=====================================
Genera la tabla de cobertura de covariables por módulo de la ELCA
(Anexo "Cobertura de covariables por módulo",
Tabla~\\ref{tab:cobertura_modulos}).

Las tres columnas se calculan desde los datos, con una sola definición por
columna para todos los módulos:

  - "Candidatas auditadas": filas con `clasificacion == "CANDIDATO_BENCHMARK"`
    en el archivo de auditoría del módulo (`docs/variable_audit/*_construccion.csv`).
    Choques no pasó por esa auditoría columna por columna, de ahí el "--".
  - "Usadas en el modelo": columnas de contenido (sin llaves de identidad)
    que el módulo aporta al consolidado final.
  - "Cobertura (% del panel)": proporción de filas hogar x ola del
    consolidado con al menos una variable del módulo no vacía. Si varía
    entre olas, se reporta el rango de las tres olas; para Niños se agrega
    el conteo.

Cambio 2026-10-11: Hogar reportaba 103 usadas, el
número de candidatas originales absorbidas en variables compuestas, anotado
a mano porque esa correspondencia no quedó tabulada; un reconteo desde el
código de `build_hogar_features.py` da 102, no 103. Se unifica la
definición: Hogar reporta sus columnas finales (56), como los demás módulos
(los cinco suman las 164 covariables de módulos del consolidado). La
cobertura, antes escrita a mano, se calcula ahora desde el consolidado:
Comunidades pasa de 91.9--96.4% (hogares con comunidad identificada) a
67.9--92.4% del panel, y Niños de 55.4% (hogares con niños de 6 a 9 años) a
50.1% (hogares con algún dato del módulo).

INPUTS

    docs/variable_audit/{hogar,personas_hogar,comunidades,ninos}_construccion.csv
    data/processed/*_hogar_elca_longitudinal.parquet (bloques de features por módulo)
    data/processed/benchmark_consolidado_elca_longitudinal.parquet

OUTPUTS

    paper/tables/tab_cobertura_modulos.tex

CÓMO CORRER

    python src/tabla_cobertura_modulos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estilo_tablas as et  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "docs" / "variable_audit"
PROCESSED = REPO_ROOT / "data" / "processed"
CONSOLIDADO = PROCESSED / "benchmark_consolidado_elca_longitudinal.parquet"
OUTPUT = REPO_ROOT / "paper" / "tables" / "tab_cobertura_modulos.tex"

# Columnas de identidad/llave que no cuentan como covariable de contenido
# (mismo criterio que build_benchmark_consolidado.py y eda_variables_modelo.py).
ID_COLS = {"consecutivo", "llave", "llave_n16", "ola", "zona", "llave_compuesta", "consecutivo_c"}

# (rótulo en la tabla, bloques de features del módulo, archivo de auditoría,
#  formato de la cobertura cuando no es 100%; ver `cobertura`)
MODULOS = [
    ("Hogar", ["hogar_features_elca_longitudinal.parquet"], "hogar_construccion.csv", "total"),
    ("Personas (9 bloques)", [
        "personas_hogar_elca_longitudinal.parquet",
        "educacion_ocupacion_hogar_elca_longitudinal.parquet",
        "salud_discapacidad_hogar_elca_longitudinal.parquet",
        "ahorro_capital_social_hogar_elca_longitudinal.parquet",
        "educacion_ocupacion_hogar_ext_elca_longitudinal.parquet",
        "becas_subsidios_hogar_elca_longitudinal.parquet",
        "salud_discapacidad_hogar_ext_elca_longitudinal.parquet",
        "personas_hogar_ext_elca_longitudinal.parquet",
        "participacion_civica_hogar_elca_longitudinal.parquet",
    ], "personas_hogar_construccion.csv", "total"),
    ("Comunidades", ["comunidades_hogar_elca_longitudinal.parquet"], "comunidades_construccion.csv", "rango"),
    ("Niños (6--9 años)", ["ninos_hogar_elca_longitudinal.parquet"], "ninos_construccion.csv", "total"),
    ("Choques", ["choques_hogar_elca_longitudinal.parquet"], None, "total"),
]


def contar_candidatas(archivo: str | None) -> str:
    if archivo is None:
        return "--"
    df = pd.read_csv(AUDIT_DIR / archivo)
    return str(int((df["clasificacion"] == "CANDIDATO_BENCHMARK").sum()))


def columnas_modulo(bloques: list[str], consolidado: pd.DataFrame) -> list[str]:
    cols = []
    for bloque in bloques:
        cols += [c for c in pd.read_parquet(PROCESSED / bloque).columns if c not in ID_COLS]
    faltan = [c for c in cols if c not in consolidado.columns]
    if faltan:
        sys.exit(f"ERROR: columnas del módulo ausentes en el consolidado: {faltan[:5]}")
    return cols


def cobertura(cols: list[str], consolidado: pd.DataFrame, formato: str) -> str:
    """formato "rango": mínimo--máximo entre olas (Comunidades, cuya cobertura
    cae de 2010 a 2016); "total": porcentaje y conteo del panel completo
    (Niños). Si las tres olas tienen 100%, se reporta "100%"."""
    con_dato = consolidado[cols].notna().any(axis=1)
    por_ola = (con_dato.groupby(consolidado["ola"]).mean() * 100).round(1)
    if (por_ola == 100).all():
        return "100\\%"
    if formato == "rango":
        return f"{por_ola.min():.1f}--{por_ola.max():.1f}\\%"
    n, total = int(con_dato.sum()), len(con_dato)
    return f"{100 * n / total:.1f}\\% ({n:,} / {total:,})".replace(",", "{,}")


def generar_tex(filas: list[dict]) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Cobertura de las covariables por módulo de la ELCA, 2010, 2013 y 2016.}",
        r"  \label{tab:cobertura_modulos}",
        r"  \footnotesize",
        r"  \begin{tabular}{lp{2.6cm}p{2.4cm}p{3.2cm}}",
        r"    \toprule",
        r"    \textbf{Módulo} & \textbf{Candidatas auditadas} & \textbf{Usadas en el modelo} & \textbf{Cobertura (\% del panel)} \\",
        r"    \midrule",
    ]
    for f in filas:
        lineas.append(f"    {f['modulo']} & {f['candidatas']} & {f['usadas']} & {f['cobertura']} \\\\")
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.9\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} ``Candidatas auditadas'': columnas del módulo que",
        r"    superaron los cuatro filtros de calidad de la auditoría descrita arriba.",
        r"    ``Usadas en el modelo'': variables finales que el módulo aporta al",
        r"    panel, sin llaves de identidad. Cobertura: proporción de observaciones",
        r"    hogar$\times$ola con al menos una variable del módulo; un rango indica",
        r"    el mínimo y el máximo entre las tres olas.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lineas) + "\n"


def main() -> None:
    consolidado = pd.read_parquet(CONSOLIDADO)
    filas = []
    for rotulo, bloques, auditoria, formato in MODULOS:
        cols = columnas_modulo(bloques, consolidado)
        filas.append({
            "modulo": rotulo,
            "candidatas": contar_candidatas(auditoria),
            "usadas": len(cols),
            "cobertura": cobertura(cols, consolidado, formato),
        })
    total_usadas = sum(f["usadas"] for f in filas)
    print("Módulo | candidatas | usadas | cobertura")
    for f in filas:
        print(f"  {f['modulo']:<22s} {f['candidatas']:>4s}  {f['usadas']:>4}  {f['cobertura']}")
    print(f"Total usadas en los cinco módulos: {total_usadas}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(generar_tex(filas), encoding="utf-8")
    print(f"Tabla exportada: {OUTPUT}")


if __name__ == "__main__":
    main()
