"""
graf_curva_pr_dmsp.py
=====================================
Curva de precision-recall del modelo principal (elastic net, Modelo B,
pobreza monetaria, prueba 2013->2016) sin y con las 23 variables de
iluminacion nocturna DMSP-OLS. Las dos curvas
se superponen: la luz no cambia el desempeno en ningun punto de operacion.

No recalcula nada: lee los puntos de la curva y el AUC-PR de
`src/05_model/diagnostico_curva_pr.py`.

INPUTS

    data/processed/benchmark_resultados/diagnostico_curva_pr_puntos.csv
    data/processed/benchmark_resultados/diagnostico_curva_pr.csv

OUTPUT

    outputs/figures/modelos/curva_pr_dmsp_monetaria.png

COMO CORRER

    python src/graf_curva_pr_dmsp.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import estilo_figuras as ef

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
OUTPUT = REPO_ROOT / "outputs" / "figures" / "modelos" / "curva_pr_dmsp_monetaria.png"

DEFINICION, ESPECIFICACION, PRINCIPAL = "monetaria", "B", "Logistica"
FRACCION_ANCHO = 0.72


def main() -> None:
    puntos = pd.read_csv(RESULTADOS / "diagnostico_curva_pr_puntos.csv")
    puntos = puntos[(puntos.definicion == DEFINICION) & (puntos.especificacion == ESPECIFICACION)
                    & (puntos.algoritmo == PRINCIPAL)]
    metricas = pd.read_csv(RESULTADOS / "diagnostico_curva_pr.csv")
    principal = metricas[(metricas.definicion == DEFINICION) & (metricas.especificacion == ESPECIFICACION)
                         & (metricas.algoritmo == PRINCIPAL)].iloc[0]

    ef.aplicar_estilo()
    fig, ax = plt.subplots(figsize=(ef.ancho(FRACCION_ANCHO), 3.1))
    series = [
        ("base", ef.COLOR_SERIE, "-", f"Solo encuesta (AUC-PR {principal.ap_base:.3f})"),
        ("geo", ef.PALETA["naranja"], "--", f"Encuesta y DMSP-OLS (AUC-PR {principal.ap_geo:.3f})"),
    ]
    for modelo, color, estilo, etiqueta in series:
        curva = puntos[puntos.modelo == modelo].sort_values("recall")
        ax.plot(curva.recall, curva.precision, color=color, linewidth=2, linestyle=estilo,
                drawstyle="steps-post", label=etiqueta)

    prevalencia = principal.prevalencia
    ax.axhline(prevalencia, color=ef.INK_MUTED, linewidth=1, linestyle=":")
    ax.text(0.01, prevalencia + 0.015, f"Azar ({prevalencia:.3f})", ha="left", va="bottom",
            color=ef.INK_SECUNDARIO, fontsize=ef.TAM_LETRA_PEQUENA)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall: fracción de los hogares vulnerables marcados")
    ax.set_ylabel("Precision: fracción de los marcados\nque es vulnerable")
    ax.legend(loc="upper right")
    ef.guardar(fig, OUTPUT)
    print(f"Figura guardada: {OUTPUT}")


if __name__ == "__main__":
    main()
