"""
graf_perfil_nucleo_grupos.py
===================================
Figura de la Seccion 5.2 (Tercer hallazgo): donde queda el grupo
"Entra en pobreza" (los vulnerables) respecto a "Sale de la pobreza" y a
los dos extremos, para las variables del nucleo SHAP cuya DIRECCION es
estable o moderada (consistentes en >= 8 de las 12 celdas de la malla de
umbrales de `sensibilidad_shap_nucleo.py`).

Cambio 2026-09-30: las medias por grupo ya no se leen de las tablas del perfil de 53
variables (`perfil_completo_monetaria_*.csv`), que dejaban fuera a dos
variables estables (grado educativo del jefe, controles preventivos) por
no estar en esa lista. Se calculan aqui, directamente, como media
ponderada por `peso_longitudinal` de la covariable de la ola base en cada
grupo de la matriz de transicion -- misma formula que la rama numerica de
`construir_tabla_comparativa` (`eda_transicion_covariables.py`), mismo
panel (`construir_matriz_transicion`) y mismas covariables
(`cargar_covariables_ola_base`, en `02_build/panel_transicion.py`).

POSICION RELATIVA (para poder comparar variables de escalas distintas):

    posicion = (media del grupo - media de Siempre pobre)
               / (media de Nunca pobre - media de Siempre pobre)

    0 = media de Siempre pobre, 1 = media de Nunca pobre. No depende del
    sentido de la variable (mas hacinamiento es peor, mas riqueza es
    mejor): ambos extremos quedan fijos. Un grupo esta "mas cerca de sale
    que de nunca pobre" si su posicion queda mas cerca de la de "Sale" que
    de 1.

INPUTS

    data/processed/pobreza_elca_longitudinal*.parquet y covariables de la ola
        base (via `eda_transicion_covariables`, ver POBREZA_PATH alli)
    data/processed/benchmark_resultados/sensibilidad_signo_nucleo_por_variable.csv
    outputs/tables/pobreza/transicion_conteo_{ola1_a_2,ola2_a_3}.csv (n de cada panel)

OUTPUTS

    outputs/figures/eda_transicion_covariables/perfil_nucleo_grupos_monetaria.png
    outputs/tables/eda_transicion_covariables/perfil_nucleo_grupos_posicion_monetaria.csv
    (posicion relativa de cada grupo por variable y ventana, detras de la figura)

COMO CORRER

    python src/graf_perfil_nucleo_grupos.py

Con `--solo-figura`, no abre los datos por hogar: redibuja la figura a partir
de perfil_nucleo_grupos_posicion_monetaria.csv ya guardado.

    python src/graf_perfil_nucleo_grupos.py --solo-figura
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "02_build"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "04_features"))
from build_pobreza_desagregaciones import _llave_compuesta, cargar_pesos_muestrales, construir_matriz_transicion  # noqa: E402
from panel_transicion import POBREZA_PATH, cargar_covariables_ola_base, cargar_dmsp_por_consecutivo  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda_transicion_covariables"
RUTA_SIGNO_VARIABLE = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "sensibilidad_signo_nucleo_por_variable.csv"
RUTA_CONTEOS = PROJECT_ROOT / "outputs" / "tables" / "pobreza"

MIN_CELDAS_DIRECCION = 8  # de 12: direccion estable o moderada
# (sufijo, rotulo, archivo de conteos, ola base, ola final, anio DMSP de la ola base)
VENTANAS = [("2010_2013", "2010 → 2013", "transicion_conteo_ola1_a_2.csv", 1, 2, 2010),
            ("2013_2016", "2013 → 2016", "transicion_conteo_ola2_a_3.csv", 2, 3, 2013)]

ETIQUETAS = {
    "riqueza_pca_hogar": "Índice de riqueza (PCA)",
    "n_activos_financieros_hogar": "N.º de activos financieros",
    "estrato_verificado_hogar": "Estrato (verificado)",
    "nivel_educ_max_hogar": "Máx. nivel educativo del hogar",
    "nivel_educ_ordinal_jefe": "Nivel educativo del jefe",
    "grado_educ_jefe": "Último grado aprobado por el jefe",
    "tasa_control_preventivo_hogar": "Tasa de controles preventivos",
    "personas_por_cuarto_hogar": "Personas por habitación",
    "valor_arriendo_pagado_hogar": "Valor del arriendo pagado",
    "razon_dependencia_demografica": "Razón de dependencia",
}

# Estilo comun de las figuras del documento (2026-10-01).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import estilo_figuras as ef  # noqa: E402

COLOR_SALE = ef.COLOR_CATEGORIA["Sale de la pobreza"]
COLOR_ENTRA = ef.COLOR_CATEGORIA["Entra en pobreza"]
INK_SECUNDARIO = ef.INK_SECUNDARIO
INK_MUTED = ef.INK_MUTED

ef.aplicar_estilo()
plt.rcParams.update({"axes.spines.left": False})


def variables_a_graficar() -> list:
    por_variable = pd.read_csv(RUTA_SIGNO_VARIABLE)
    return sorted(por_variable.loc[por_variable["celdas_consistente"] >= MIN_CELDAS_DIRECCION, "variable"])


def medias_por_grupo(variables: list, ola_ini: int, ola_fin: int, anio_dmsp: int) -> pd.DataFrame:
    """Media ponderada por `peso_longitudinal` de cada variable (ola base)
    en cada grupo de la matriz de transicion monetaria."""
    pobreza = pd.read_parquet(POBREZA_PATH)
    cargar_pesos_muestrales(pobreza, _llave_compuesta(pobreza))
    panel = construir_matriz_transicion(
        pobreza, ola_ini, ola_fin, col_pobre="pobre_ingreso", peso_col="peso_longitudinal"
    )["panel_categorias"]
    covariables = cargar_covariables_ola_base(cargar_dmsp_por_consecutivo(anio_dmsp), ola_ini)
    faltantes = set(variables) - set(covariables.columns)
    if faltantes:
        raise ValueError(f"Variables sin covariable en la ola base: {faltantes}")
    df = panel.merge(covariables[variables], left_on="consecutivo", right_index=True, how="left")
    filas = {}
    for cat, g in df.groupby("categoria", observed=True):
        filas[cat] = {}
        for v in variables:
            mask = g[v].notna() & g["peso_longitudinal"].notna()
            filas[cat][v] = np.average(g.loc[mask, v].astype(float), weights=g.loc[mask, "peso_longitudinal"])
    return pd.DataFrame(filas)  # filas = variables, columnas = grupos


def posiciones(variables: list) -> pd.DataFrame:
    filas = []
    for ventana, _, _, ola_ini, ola_fin, anio_dmsp in VENTANAS:
        medias = medias_por_grupo(variables, ola_ini, ola_fin, anio_dmsp)
        for v in variables:
            f = medias.loc[v]
            siempre, nunca = f["Siempre pobre"], f["Nunca pobre"]
            filas.append({
                "ventana": ventana, "variable": v,
                **{f"media_{c}": f[c] for c in ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]},
                "pos_sale": (f["Sale de la pobreza"] - siempre) / (nunca - siempre),
                "pos_entra": (f["Entra en pobreza"] - siempre) / (nunca - siempre),
            })
    pos = pd.DataFrame(filas)
    # "Entra" mas cerca de "Sale" que de "Nunca pobre" (posicion 1).
    pos["entra_mas_cerca_de_sale"] = (pos["pos_entra"] - pos["pos_sale"]).abs() < (1 - pos["pos_entra"]).abs()
    return pos


def n_panel(archivo: str) -> int:
    return int(pd.read_csv(RUTA_CONTEOS / archivo, index_col=0).to_numpy().sum())


def main() -> None:
    ruta_pos = TABLES_DIR / "perfil_nucleo_grupos_posicion_monetaria.csv"
    if "--solo-figura" in sys.argv:
        pos = pd.read_csv(ruta_pos)
    else:
        variables = variables_a_graficar()
        pos = posiciones(variables)
        pos.to_csv(ruta_pos, index=False)

    orden = pos[pos["ventana"] == VENTANAS[0][0]].sort_values("pos_entra")["variable"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(ef.ancho(0.95), 3.4), sharey=True)
    for ax, (ventana, rotulo, archivo, *_) in zip(axes, VENTANAS):
        g = pos[pos["ventana"] == ventana].set_index("variable").loc[orden]
        y = range(len(orden))
        ax.hlines(y, g["pos_sale"], g["pos_entra"], color=INK_MUTED, linewidth=1.4, zorder=2)
        ax.scatter(g["pos_sale"], y, s=28, color=COLOR_SALE, marker="o", zorder=3, label="Sale de la pobreza")
        ax.scatter(g["pos_entra"], y, s=24, color=COLOR_ENTRA, marker="D", zorder=4, label="Entra en pobreza")
        ax.axvline(0, color=INK_MUTED, linewidth=1.0)
        ax.axvline(1, color=INK_MUTED, linewidth=1.0)
        ax.set_yticks(list(y))
        ax.set_yticklabels([ETIQUETAS.get(v, v) for v in orden])
        ax.set_xlim(-0.12, 1.12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["Siempre\npobre", "0.5", "Nunca\npobre"])
        ax.set_title(f"{rotulo} (n = {n_panel(archivo):,})")
    fig.supxlabel("Posición entre las medias de Siempre pobre (0) y Nunca pobre (1)",
                  fontsize=ef.TAM_LETRA, color=INK_SECUNDARIO, y=0.02)
    handles, rotulos = axes[0].get_legend_handles_labels()
    fig.legend(handles, rotulos, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.subplots_adjust(wspace=0.06)

    out = FIGURES_DIR / "perfil_nucleo_grupos_monetaria.png"
    ef.guardar(fig, out)
    print(pos.round(3).to_string(index=False))
    n_cerca = int(pos["entra_mas_cerca_de_sale"].sum())
    print(f"'Entra' mas cerca de 'Sale' que de 'Nunca pobre': {n_cerca} de {len(pos)} combinaciones variable x ventana")
    print(pos.loc[~pos["entra_mas_cerca_de_sale"], ["ventana", "variable"]].to_string(index=False))
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
