"""
tabla_robustez_desactualizada.py
=====================================
Encuesta desactualizada:
¿compensa DMSP-OLS de 2013 una encuesta con tres años de rezago? Anexo
"Robustez de la validacion y margenes de informacion", Seccion 5.3 del
paper. Genera dos tablas:

  1. AUC-PR con la encuesta de 2013 (E1) y con la de 2010 (E2), diferencia
     E2 - E1 y cambio al agregar DMSP-OLS de 2013 a la encuesta de 2010
     (E3 - E2), con IC95%.
  2. Descomposicion por grupos: razon del riesgo predicho de los hogares
     que salieron de la pobreza frente al resto con cada encuesta, y
     diferencia de AUC-PR E1 - E2 dentro de cada grupo.

No recalcula nada: lee las salidas de
`src/05_model/robustez_encuesta_desactualizada.py` y
`src/05_model/robustez_encuesta_desactualizada_diagnostico.py`. Imprime
ademas la cifra del modelo principal bajo IPM sin las variables que
cambiaron de cobertura, que cita el texto del anexo.

INPUTS

    data/processed/benchmark_resultados/robustez_asesor_2026_10/desactualizada_escenarios.csv
    data/processed/benchmark_resultados/robustez_asesor_2026_10/desactualizada_resultados.csv
    data/processed/benchmark_resultados/robustez_asesor_2026_10/diagnostico_e1_e2_por_grupo.csv
    data/processed/benchmark_resultados/robustez_asesor_2026_10/diagnostico_e1_e2_sin_cambiadas.csv

OUTPUTS

    paper/tables/tab_robustez_desactualizada.tex
    paper/tables/tab_robustez_desactualizada_grupos.tex

COMO CORRER

    python src/tabla_robustez_desactualizada.py
"""

from pathlib import Path

import pandas as pd

import estilo_tablas as et
from tabla_robustez_validacion import ALGORITMOS, num

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "robustez_asesor_2026_10"
OUT_1 = REPO_ROOT / "paper" / "tables" / "tab_robustez_desactualizada.tex"
OUT_2 = REPO_ROOT / "paper" / "tables" / "tab_robustez_desactualizada_grupos.tex"

PANELES = [
    ("monetaria", "B", "Pobreza monetaria, sin ingreso ni gasto (Modelo B)"),
    ("monetaria", "A", "Pobreza monetaria, con ingreso y gasto (Modelo A)"),
    ("ipm", "B", "IPM, sin ingreso ni gasto (Modelo B)"),
    ("ipm", "A", "IPM, con ingreso y gasto (Modelo A)"),
]


def significativo(valor) -> bool:
    """`delta_ap_cruza_cero` se lee del CSV como texto cuando la columna
    mezcla filas sin dato (las de 'parte_recuperada')."""
    return str(valor) == "False"


def celda(delta: float, low: float, high: float, sig: bool) -> str:
    marca = r"\textsuperscript{*}" if sig else ""
    return f"{num(delta)}{marca} ({num(low)}, {num(high)})"


def etiqueta(definicion: str, espec: str, nombre: str, clave: str) -> str:
    principal = definicion == "monetaria" and espec == "B" and clave == "Logistica"
    return nombre + (" (principal)" if principal else "")


def tabla_1(esc: pd.DataFrame, res: pd.DataFrame) -> str:
    esc, res = esc[esc.zona == "Total"], res[res.zona == "Total"]
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{AUC-PR con la encuesta de 2013 y con la de 2010, y cambio al agregar DMSP-OLS",
        r"  de 2013, pobreza monetaria e IPM, 2013$\to$2016.}",
        r"  \label{tab:robustez_desactualizada}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l cc cc}",
        r"    \toprule",
        r"    \textbf{Modelo} & \textbf{Encuesta 2013} & \textbf{Encuesta 2010}"
        r" & \textbf{2010 menos 2013} & \textbf{Con DMSP-OLS de 2013} \\",
        r"    \midrule",
    ]
    for i, (definicion, espec, titulo) in enumerate(PANELES):
        if i > 0:
            lineas.append(r"    \addlinespace")
        lineas.append(rf"    \multicolumn{{5}}{{l}}{{\textit{{{titulo}}}}} \\")
        for clave, nombre in ALGORITMOS:
            ap = esc[(esc.definicion == definicion) & (esc.especificacion == espec) & (esc.algoritmo == clave)].set_index("escenario")["ap"]
            r = res[(res.definicion == definicion) & (res.especificacion == espec) & (res.algoritmo == clave)].set_index("medida")
            # el CSV guarda E1 - E2 ("perdida"); la tabla reporta E2 - E1
            p, l = r.loc["perdida_encuesta_vieja"], r.loc["aporte_luz_actualizada"]
            dif = celda(-p["delta_ap"], -p["delta_ap_ci95_high"], -p["delta_ap_ci95_low"], significativo(p["delta_ap_cruza_cero"]))
            luz = celda(l["delta_ap"], l["delta_ap_ci95_low"], l["delta_ap_ci95_high"], significativo(l["delta_ap_cruza_cero"]))
            lineas.append(f"    \\quad {etiqueta(definicion, espec, nombre, clave)} & {num(ap['E1'])} & {num(ap['E2'])} & {dif} & {luz} \\\\")
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Modelos estimados con 2010$\to$2013 y aplicados a los",
        r"    hogares no pobres en 2013 con la encuesta de 2013 o con la de 2010. 2010 menos 2013:",
        r"    diferencia de AUC-PR entre las dos encuestas (negativa si la encuesta vieja predice",
        r"    peor). Con DMSP-OLS de 2013: cambio en el AUC-PR al agregar DMSP-OLS de 2013 a la",
        r"    encuesta de 2010. Entre paréntesis, IC95\% por \emph{bootstrap} pareado de 2{,}000",
        r"    remuestras por comunidad. \textsuperscript{*}: el intervalo excluye el cero.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def tabla_2(grupos: pd.DataFrame) -> str:
    obs = grupos.groupby("definicion")["razon_tasas_observada"].unique()
    assert all(len(v) == 1 for v in obs)
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Riesgo predicho de los hogares que salieron de la pobreza frente al resto, y",
        r"  diferencia de AUC-PR dentro de cada grupo, encuesta de 2013 y de 2010, pobreza",
        r"  monetaria e IPM, 2013$\to$2016.}",
        r"  \label{tab:robustez_desactualizada_grupos}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l cc cc}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{\textbf{Razón de riesgo predicho}} & \multicolumn{2}{c}{\textbf{2013 menos 2010, dentro de grupo}} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"    \textbf{Modelo} & \textbf{Encuesta 2013} & \textbf{Encuesta 2010} & \textbf{Salieron} & \textbf{No pobres al inicio} \\",
        r"    \midrule",
    ]
    for i, (definicion, espec, titulo) in enumerate(PANELES):
        if i > 0:
            lineas.append(r"    \addlinespace")
        lineas.append(rf"    \multicolumn{{5}}{{l}}{{\textit{{{titulo}}}}} \\")
        for clave, nombre in ALGORITMOS:
            g = grupos[(grupos.definicion == definicion) & (grupos.especificacion == espec) & (grupos.algoritmo == clave)]
            assert len(g) == 1, f"falta {definicion} {espec} {clave}"
            g = g.iloc[0]
            celdas = []
            for grupo in ("salio", "no_pobre_previo"):
                low, high = g[f"dif_{grupo}_ci95_low"], g[f"dif_{grupo}_ci95_high"]
                marca = r"\textsuperscript{*}" if (low > 0 or high < 0) else ""
                celdas.append(f"{num(g[f'dif_e1_menos_e2_{grupo}'])}{marca}")
            lineas.append(f"    \\quad {etiqueta(definicion, espec, nombre, clave)} & {g['razon_riesgo_predicho_E1']:.2f}"
                          f" & {g['razon_riesgo_predicho_E2']:.2f} & {celdas[0]} & {celdas[1]} \\\\")
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Razón de riesgo predicho: riesgo predicho promedio de",
        r"    los hogares que salieron de la pobreza en la ventana anterior dividido por el de los",
        r"    que no eran pobres al inicio de ella. Razón observada de las tasas de entrada en",
        rf"    pobreza: {obs['monetaria'][0]:.2f} (monetaria) y {obs['ipm'][0]:.2f} (IPM). 2013 menos 2010, dentro de",
        r"    grupo: diferencia de AUC-PR entre las dos encuestas, calculada por separado en cada",
        r"    grupo. \textsuperscript{*}: el IC95\% por \emph{bootstrap} pareado de 2{,}000 remuestras",
        r"    por comunidad excluye el cero.",
        f"    {et.FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return et.aplicar_signo_menos("\n".join(lineas)) + "\n"


def main() -> None:
    t1 = tabla_1(pd.read_csv(RES / "desactualizada_escenarios.csv"), pd.read_csv(RES / "desactualizada_resultados.csv"))
    t2 = tabla_2(pd.read_csv(RES / "diagnostico_e1_e2_por_grupo.csv"))
    OUT_1.write_text(t1, encoding="utf-8")
    OUT_2.write_text(t2, encoding="utf-8")
    sin = pd.read_csv(RES / "diagnostico_e1_e2_sin_cambiadas.csv")
    p = sin[(sin.definicion == "ipm") & (sin.especificacion == "B") & (sin.algoritmo == "Logistica")].iloc[0]
    print(f"IPM, modelo principal: 2013 menos 2010 = {p['dif_e1_menos_e2_original']:.3f} con todas las variables, "
          f"{p['dif_e1_menos_e2_sin_cambiadas']:.3f} sin las {int(p['n_variables_quitadas'])} cambiadas")
    print(f"Tablas exportadas: {OUT_1}, {OUT_2}\n\n{t1}\n{t2}")


if __name__ == "__main__":
    main()
