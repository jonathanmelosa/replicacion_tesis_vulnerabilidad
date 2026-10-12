"""
graf_variacion_dmsp_comunidad.py
=====================================
Figura del argumento de escala de la Seccion 5.3: parte de la
variacion entre hogares que corresponde a la comunidad (correlacion
intraclase) para la iluminacion nocturna y para el riesgo de pobreza, por
zona, en las dos ventanas, bajo pobreza monetaria. La luz es casi del todo
una variable de comunidad; el riesgo varia sobre todo entre vecinos.

No recalcula nada: lee `diagnostico_variacion_dmsp_comunidad.csv`
(`src/05_model/diagnostico_variacion_dmsp_comunidad.py`).

INPUT

    data/processed/benchmark_resultados/diagnostico_variacion_dmsp_comunidad.csv

OUTPUT

    outputs/figures/modelos/variacion_dmsp_comunidad.png

COMO CORRER

    python src/graf_variacion_dmsp_comunidad.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import estilo_figuras as ef

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_variacion_dmsp_comunidad.csv"
OUTPUT = REPO_ROOT / "outputs" / "figures" / "modelos" / "variacion_dmsp_comunidad.png"

DEFINICION = "monetaria"
VENTANAS = [("2010_2013", f"2010 {ef.FLECHA} 2013"), ("2013_2016", f"2013 {ef.FLECHA} 2016")]
ZONAS = ["Total", "Urbano", "Rural"]
ANCHO_BARRA = 0.36
FRACCION_ANCHO = 0.9


def main() -> None:
    datos = pd.read_csv(INPUT)
    datos = datos[datos["definicion"] == DEFINICION]

    ef.aplicar_estilo()
    fig, ejes = plt.subplots(1, 2, figsize=(ef.ancho(FRACCION_ANCHO), 3.0), sharey=True)
    x = np.arange(len(ZONAS))
    for ax, (ventana, titulo) in zip(ejes, VENTANAS):
        sub = datos[datos["ventana"] == ventana].set_index("zona").loc[ZONAS]
        barras_luz = ax.bar(x - ANCHO_BARRA / 2 - 0.01, 100 * sub["icc_luz"], ANCHO_BARRA,
                            color=ef.COLOR_SERIE, label="DMSP-OLS")
        barras_riesgo = ax.bar(x + ANCHO_BARRA / 2 + 0.01, 100 * sub["icc_y"], ANCHO_BARRA,
                               color=ef.PALETA["naranja"], label="Riesgo de pobreza")
        for barra in list(barras_luz) + list(barras_riesgo):
            ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 1.5, f"{barra.get_height():.0f}",
                    ha="center", va="bottom", fontsize=ef.TAM_LETRA_PEQUENA, color=ef.INK_SECUNDARIO)
        ax.set_xticks(x)
        ax.set_xticklabels(ZONAS)
        ax.set_title(titulo)
        ax.set_ylim(0, 110)
    ejes[0].set_ylabel("% de la variación que\ncorresponde a la comunidad")
    manijas, etiquetas = ejes[0].get_legend_handles_labels()
    fig.legend(manijas, etiquetas, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.06))
    ef.guardar(fig, OUTPUT)
    print(f"Figura guardada: {OUTPUT}")


if __name__ == "__main__":
    main()
