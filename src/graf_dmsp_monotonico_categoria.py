"""
graf_dmsp_monotonico_categoria.py
===================================
Grafica el HALLAZGO PRINCIPAL de la covariable DMSP (iluminacion nocturna)
sobre los 4 grupos de transicion de pobreza: relacion MONOTONICA -- Siempre
pobre tiene la iluminacion mas baja, Nunca pobre la mas alta, con Sale de
la pobreza y Entra en pobreza en el orden intermedio esperado, tanto para
pobreza monetaria como IPM (2010->2013).

POR QUE UN GRAFICO DE BARRAS Y NO UN MAPA
    Se probaron varias variantes de mapa territorial (tramas, hexagonos,
    glifos por region, small multiples) para mostrar los 4 grupos sobre el
    fondo de brillo DMSP. Ninguna deja ver con
    claridad el hallazgo que se quiere comunicar: la relacion monotonica
    categoria->iluminacion. Un mapa dispersa la atencion en la geografia;
    el hallazgo es estadistico (una relacion ORDENADA entre 4 categorias y
    una variable continua), y un grafico de barras ordenado por valor lo
    muestra sin ambiguedad, sin competir con ningun fondo.

DATOS (reales, ya calculados por eda_transicion_covariables.py)
    Siempre pobre < Sale de la pobreza < Entra en pobreza < Nunca pobre,
    tanto en media ponderada como en mediana, en monetaria E IPM (4 de 4
    combinaciones monotonicas) -- ver
    outputs/tables/eda_transicion_covariables/dmsp_por_categoria_{monetaria,ipm}_2010_2013.csv

INPUTS
    outputs/tables/eda_transicion_covariables/dmsp_por_categoria_monetaria_2010_2013.csv
    outputs/tables/eda_transicion_covariables/dmsp_por_categoria_ipm_2010_2013.csv

OUTPUTS
    outputs/figures/eda_transicion_covariables/dmsp_monotonico_categoria.png

COMO CORRER
    python src/graf_dmsp_monotonico_categoria.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda_transicion_covariables"

# Estilo comun de las figuras del documento (2026-10-01): misma paleta por
# grupo, letra en puntos de impresion y sin titulo dentro de la imagen (lo
# que decia el titulo -- barra = media ponderada, linea = mediana -- pasa al
# caption de main.tex).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import estilo_figuras as ef  # noqa: E402

COLOR_CATEGORIA = ef.COLOR_CATEGORIA
INK_PRIMARIO = ef.INK_PRIMARIO
ef.aplicar_estilo()
plt.rcParams.update({"axes.spines.left": False, "axes.grid.axis": "x"})


def graficar_panel(ax, tabla: pd.DataFrame, titulo: str) -> None:
    tabla = tabla.sort_values("dmsp_media_ponderada")
    colores = [COLOR_CATEGORIA[c] for c in tabla["categoria"]]
    barras = ax.barh(tabla["categoria"], tabla["dmsp_media_ponderada"], color=colores, height=0.62)
    # Valor de la media despues de lo que termine mas a la derecha (la barra
    # o la marca de la mediana), para que ninguno de los dos lo tape.
    for barra, valor, mediana in zip(barras, tabla["dmsp_media_ponderada"], tabla["dmsp_mediana"]):
        ax.text(max(valor, mediana) + 1.5, barra.get_y() + barra.get_height() / 2, f"{valor:.1f}",
                ha="left", va="center", fontsize=ef.TAM_LETRA_PEQUENA, color=INK_PRIMARIO)

    # mediana como marcador secundario, para dejar ver que el patron no
    # depende de valores atipicos (misma relacion monotonica en ambas metricas)
    ax.scatter(tabla["dmsp_mediana"], tabla["categoria"], marker="|", s=160,
               color=INK_PRIMARIO, linewidths=1.6, zorder=5, label="Mediana")

    ax.set_xlim(0, 72)
    ax.set_title(titulo)


def main() -> None:
    monetaria = pd.read_csv(TABLES_DIR / "dmsp_por_categoria_monetaria_2010_2013.csv")
    ipm = pd.read_csv(TABLES_DIR / "dmsp_por_categoria_ipm_2010_2013.csv")

    fig, axes = plt.subplots(1, 2, figsize=(ef.ancho(0.95), 2.6))
    graficar_panel(axes[0], monetaria, "Pobreza monetaria")
    graficar_panel(axes[1], ipm, "Pobreza multidimensional (IPM)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=ef.INK_MUTED, label="Media ponderada"),
               plt.Line2D([0], [0], marker="|", linestyle="", color=INK_PRIMARIO,
                          markersize=9, markeredgewidth=1.6, label="Mediana")]
    fig.supxlabel("Luces nocturnas, DMSP-OLS (0 a 63)", fontsize=ef.TAM_LETRA,
                  color=ef.INK_SECUNDARIO, y=0.1)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out = FIGURES_DIR / "dmsp_monotonico_categoria.png"
    ef.guardar(fig, out)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
