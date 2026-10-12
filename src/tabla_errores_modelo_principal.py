"""
tabla_errores_modelo_principal.py
=====================================
Tablas del anexo de errores del modelo principal (elastic net, Modelo B,
pobreza monetaria, prueba 2013->2016, umbral F2):

  1. tab_errores_nucleo: las 20 variables numericas del nucleo SHAP en
     2013, con la media de los verdaderos negativos (VN), los falsos
     positivos (FP) y los verdaderos positivos (VP), y las diferencias
     estandarizadas FP - VN (los FP son mas precarios) y VP - FP (cautela:
     los que caen eran aun mas precarios que los FP).
  2. tab_errores_choques: las 17 medidas de choques de 2016 (posteriores
     a la prediccion), VP frente a FP.

Marca con asterisco las diferencias cuyo IC95% (bootstrap por comunidad)
excluye el cero.

No recalcula nada: lee `diagnostico_errores_modelo_principal.csv`
(`src/05_model/diagnostico_errores_modelo_principal.py`), regla
`umbral_f2`.

INPUT

    data/processed/benchmark_resultados/diagnostico_errores_modelo_principal.csv

OUTPUTS

    paper/tables/tab_errores_nucleo.tex
    paper/tables/tab_errores_choques.tex

COMO CORRER

    python src/tabla_errores_modelo_principal.py
"""

from pathlib import Path

import pandas as pd

import estilo_tablas as et
from etiquetas_variables import etiqueta

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_errores_modelo_principal.csv"
OUT_DIR = REPO_ROOT / "paper" / "tables"
REGLA = "umbral_f2"

# Las variables de choques de 2016 no estan en el nucleo y no tienen
# etiqueta en `etiquetas_variables.py`.
ETIQUETAS_CHOQUES = {
    "total_choques_hogar": "N.\\textsuperscript{o} de choques",
    "tuvo_algun_choque_hogar": "Tuvo al menos un choque",
    "n_tipos_choque_hogar": "N.\\textsuperscript{o} de tipos de choque",
    "tuvo_choque_salud_hogar": "Choque de salud",
    "tuvo_choque_economico_hogar": "Choque económico",
    "tuvo_choque_patrimonial_hogar": "Choque patrimonial",
    "tuvo_choque_agropecuario_hogar": "Choque agropecuario",
    "tuvo_choque_familiar_hogar": "Choque familiar",
    "tuvo_choque_severo_hogar": "Choque severo",
    "afrontamiento_erosivo_hogar": "Respuesta que erosiona activos",
    "afrontamiento_protector_hogar": "Respuesta que protege activos",
    "intensifico_trabajo_hogar": "Intensificó el trabajo",
    "retiro_hijos_colegio_choque_hogar": "Retiró hijos del colegio",
    "redujo_alimentos_choque_hogar": "Redujo alimentos",
    "se_endeudo_formal_choque_hogar": "Se endeudó con entidad formal",
    "se_endeudo_informal_choque_hogar": "Se endeudó informalmente",
    "no_ajusto_choque_hogar": "No hizo ningún ajuste",
}


def numero(valor: float) -> str:
    if abs(valor) >= 1000:
        return f"{valor:,.0f}".replace(",", "{,}")
    return f"{valor:.2f}"


def dif(fila: pd.Series) -> str:
    marca = r"\textsuperscript{*}" if not fila["ic_cruza_cero"] else ""
    return f"{fila['diferencia_estandarizada']:.2f}{marca}"


def nota(texto: str) -> list:
    return [
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        f"    \\footnotesize \\textit{{Nota:}} {texto}",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]


def tabla_nucleo(datos: pd.DataFrame) -> str:
    fp_vn = datos[datos["comparacion"] == "FP_vs_VN_nucleo_2013"].set_index("variable")
    vp_fp = datos[datos["comparacion"] == "VP_vs_FP_nucleo_2013"].set_index("variable")
    orden = fp_vn["diferencia_estandarizada"].abs().sort_values(ascending=False).index
    n_vn, n_fp, n_vp = int(fp_vn["n_b"].iloc[0]), int(fp_vn["n_a"].iloc[0]), int(vp_fp["n_a"].iloc[0])
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Variables del núcleo en 2013 según el resultado de la clasificación, modelo",
        r"  principal, pobreza monetaria, 2013$\to$2016.}",
        r"  \label{tab:errores_nucleo}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l ccc cc}",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{\textbf{Media en 2013}} & \multicolumn{2}{c}{\textbf{Dif. estandarizada}} \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"    \textbf{Variable} & \textbf{VN} & \textbf{FP} & \textbf{VP} & \textbf{FP $-$ VN} & \textbf{VP $-$ FP} \\",
        r"    \midrule",
    ]
    for var in orden:
        a, b = fp_vn.loc[var], vp_fp.loc[var]
        lineas.append(f"    {etiqueta(var)} & {numero(a['media_b'])} & {numero(a['media_a'])} & {numero(b['media_a'])} & {dif(a)} & {dif(b)} \\\\")
    lineas += [r"    \bottomrule", r"  \end{tabular}"]
    lineas += nota(
        "Hogares no pobres en 2013 clasificados por el modelo principal con el "
        "umbral F$_2$. VN: no marcados que no son pobres en 2016 "
        f"($n={n_vn:,}$); FP: marcados que no son pobres en 2016 ($n={n_fp:,}$); "
        f"VP: marcados que son pobres en 2016 ($n={n_vp:,}$). Diferencia "
        "estandarizada: diferencia de medias dividida por la desviación estándar "
        "combinada de los dos grupos. \\textsuperscript{*}: el IC95\\% de la "
        "diferencia de medias (\\emph{bootstrap} de 2{,}000 remuestras por "
        "comunidad) excluye el cero. Sin ponderar.".replace(",", "{,}").replace("{{,}}", "{,}")
    )
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def tabla_choques(datos: pd.DataFrame) -> str:
    vp_fp = datos[datos["comparacion"] == "VP_vs_FP_choques_2016"].copy()
    vp_fp["base"] = vp_fp["variable"].str.replace("_2016", "", regex=False)
    vp_fp = vp_fp.set_index("base").loc[list(ETIQUETAS_CHOQUES)]
    n_vp, n_fp = int(vp_fp["n_a"].iloc[0]), int(vp_fp["n_b"].iloc[0])
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Choques reportados en 2016 por los hogares marcados que caen y que no caen",
        r"  en pobreza, modelo principal, pobreza monetaria, 2013$\to$2016.}",
        r"  \label{tab:errores_choques}",
        r"  \footnotesize",
        r"  \begin{tabular}{l cc c}",
        r"    \toprule",
        r"    \textbf{Medida en 2016} & \textbf{VP} & \textbf{FP} & \textbf{Dif. estandarizada} \\",
        r"    \midrule",
    ]
    for base, fila in vp_fp.iterrows():
        lineas.append(f"    {ETIQUETAS_CHOQUES[base]} & {numero(fila['media_a'])} & {numero(fila['media_b'])} & {dif(fila)} \\\\")
    lineas += [r"    \bottomrule", r"  \end{tabular}"]
    lineas += nota(
        "Hogares marcados por el modelo principal con el umbral F$_2$. VP: son "
        f"pobres en 2016 ($n={n_vp}$); FP: no son pobres en 2016 "
        f"($n={n_fp:,}$). Choques reportados en la encuesta de 2016, posteriores "
        "a la predicción y no usados por el modelo; las variables binarias se "
        "reportan como proporciones. Diferencia estandarizada y "
        "\\textsuperscript{*} como en la Tabla~\\ref{tab:errores_nucleo}. Sin "
        "ponderar.".replace(f"{n_fp:,}", f"{n_fp:,}".replace(",", "{,}"))
    )
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def main() -> None:
    datos = pd.read_csv(INPUT)
    datos = datos[datos["regla"] == REGLA]
    for nombre, tex in [("tab_errores_nucleo.tex", tabla_nucleo(datos)), ("tab_errores_choques.tex", tabla_choques(datos))]:
        ruta = OUT_DIR / nombre
        ruta.write_text(tex, encoding="utf-8")
        print(f"Tabla exportada: {ruta}")


if __name__ == "__main__":
    main()
