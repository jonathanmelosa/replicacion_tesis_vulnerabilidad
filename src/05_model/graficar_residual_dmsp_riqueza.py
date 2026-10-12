"""
graficar_residual_dmsp_riqueza.py
==================================
Grafica el diagnostico de `diagnostico_residual_dmsp_riqueza.py`: la
correlacion de dmsp_stable_lights con caer en pobreza es, en crudo,
comparable en magnitud a la de las 4 variables que dominan el nucleo de
importancia multivariada (servicios publicos, estrato verificado, bienes
durables, personas por habitación), pero colapsa de -0.214 a +0.010 al
residualizar contra las 4 -- la evidencia visual de que la redundancia,
no la ausencia de señal, explica por que DMSP-OLS no aporta poder
predictivo incremental bajo pobreza monetaria (paper/main.tex, seccion
"Por que DMSP-OLS no aporta").

El color codifica el contraste que importa para el argumento: correlacion
bruta (gris, el punto de comparacion) vs. residualizada (rojo, el
resultado clave), no de quien es cada variable.

NO recalcula nada: solo lee el CSV que ya produjo
`diagnostico_residual_dmsp_riqueza.py`.

INPUT

    data/processed/benchmark_resultados/diagnostico_residual_dmsp_riqueza.csv

OUTPUT

    outputs/figures/modelos/05_residual_dmsp_riqueza.png

COMO CORRER

    cd src/05_model && python graficar_residual_dmsp_riqueza.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Estilo comun de las figuras del documento (2026-10-01).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import estilo_figuras as ef  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_residual_dmsp_riqueza.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "modelos" / "05_residual_dmsp_riqueza.png"

# Gris oscuro / gris claro / rojo: el caption de main.tex los nombra.
COLOR_DMSP_BRUTA = ef.INK_SECUNDARIO
COLOR_REFERENCIA_BRUTA = "#b5b3ad"
COLOR_RESIDUALIZADA = ef.PALETA["rojo"]

ETIQUETAS = [
    ("DMSP-OLS", "cruda", "dmsp_stable_lights", COLOR_DMSP_BRUTA),
    ("N.º de servicios públicos\n(referencia)", "cruda (control vs outcome)", "n_servicios_publicos_hogar", COLOR_REFERENCIA_BRUTA),
    ("Estrato verificado\n(referencia)", "cruda (control vs outcome)", "estrato_verificado_hogar", COLOR_REFERENCIA_BRUTA),
    ("N.º de bienes durables\n(referencia)", "cruda (control vs outcome)", "n_bienes_durables_hogar", COLOR_REFERENCIA_BRUTA),
    ("Personas por habitación\n(referencia)", "cruda (control vs outcome)", "personas_por_cuarto_hogar", COLOR_REFERENCIA_BRUTA),
    ("DMSP-OLS,\nresidualizada", "parcial (sklearn.LinearRegression)", "dmsp_stable_lights", COLOR_RESIDUALIZADA),
]


def main() -> None:
    tabla = pd.read_csv(INPUT_PATH)

    filas = []
    for etiqueta, comparacion, variable, color in ETIQUETAS:
        fila = tabla[(tabla["comparacion"] == comparacion) & (tabla["variable"] == variable)].iloc[0]
        filas.append({"etiqueta": etiqueta, "r": fila["r"], "n": int(fila["n"]), "color": color})
    datos = pd.DataFrame(filas).iloc[::-1]  # para que la primera quede arriba en barh

    ef.aplicar_estilo()
    plt.rcParams.update({"axes.grid.axis": "x"})
    fig, ax = plt.subplots(figsize=(ef.ancho(0.62), 3.6))

    barras = ax.barh(datos["etiqueta"], datos["r"], color=datos["color"], height=0.6)
    ax.axvline(0, color=ef.INK_SECUNDARIO, linewidth=0.8)
    ax.set_xlim(-0.3, 0.3)
    ax.set_xlabel("Correlación de Pearson ($r$) con caer en pobreza monetaria")

    UMBRAL_BARRA_CORTA = 0.05  # bajo este |r|, la barra es muy angosta para el texto adentro
    for barra, r in zip(barras, datos["r"]):
        y_centro = barra.get_y() + barra.get_height() / 2
        if abs(r) >= UMBRAL_BARRA_CORTA:
            ax.text(r / 2, y_centro, f"{r:+.3f}".replace("-", "\u2212"), va="center", ha="center",
                    fontsize=ef.TAM_LETRA_PEQUENA, color="white")
        else:
            desplazamiento = 0.012 if r >= 0 else -0.012
            alineacion = "left" if r >= 0 else "right"
            ax.text(r + desplazamiento, y_centro, f"{r:+.3f}".replace("-", "\u2212"), va="center", ha=alineacion,
                    fontsize=ef.TAM_LETRA_PEQUENA, color=ef.INK_PRIMARIO)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_DMSP_BRUTA, label="DMSP-OLS, correlación bruta"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_REFERENCIA_BRUTA, label="Referencias, correlación bruta"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RESIDUALIZADA, label="DMSP-OLS, residualizada"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=1)
    # La nota al pie (asterisco, correlacion parcial via residuos / FWL) no se
    # leia en el tamano del documento; va en el caption de main.tex
    #.
    fig.tight_layout()

    ef.guardar(fig, OUTPUT_PATH)
    print(f"Guardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
