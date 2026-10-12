"""
graf_curva_pr.py
=====================================
Curva de precision-recall del modelo principal (elastic net, Modelo B,
pobreza monetaria) en el conjunto de prueba 2013->2016, con los otros
cuatro algoritmos del Modelo B en gris como referencia, la linea del azar
(la proporcion de hogares vulnerables) y dos puntos de operacion del
modelo principal: el umbral elegido por validacion cruzada maximizando F2
(la regla del paper) y el decil de mayor riesgo.

No recalcula nada: lee los puntos de la curva y las metricas en
el umbral F2 y el decil de mayor riesgo de `src/05_model/diagnostico_curva_pr.py`.

INPUTS

    data/processed/benchmark_resultados/diagnostico_curva_pr_puntos.csv
    data/processed/benchmark_resultados/diagnostico_curva_pr.csv

OUTPUT

    outputs/figures/modelos/curva_pr_monetaria.png

COMO CORRER

    python src/graf_curva_pr.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import estilo_figuras as ef

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
OUTPUT = REPO_ROOT / "outputs" / "figures" / "modelos" / "curva_pr_monetaria.png"

DEFINICION, ESPECIFICACION, PRINCIPAL = "monetaria", "B", "Logistica"
COLOR_REFERENCIA = "#c4c2bb"
FRACCION_ANCHO = 0.72


def main() -> None:
    puntos = pd.read_csv(RESULTADOS / "diagnostico_curva_pr_puntos.csv")
    metricas = pd.read_csv(RESULTADOS / "diagnostico_curva_pr.csv")
    puntos = puntos[(puntos.definicion == DEFINICION) & (puntos.especificacion == ESPECIFICACION) & (puntos.modelo == "base")]
    metricas = metricas[(metricas.definicion == DEFINICION) & (metricas.especificacion == ESPECIFICACION)]

    ef.aplicar_estilo()
    fig, ax = plt.subplots(figsize=(ef.ancho(FRACCION_ANCHO), 3.3))

    otros = [a for a in puntos.algoritmo.unique() if a != PRINCIPAL]
    for i, algoritmo in enumerate(otros):
        curva = puntos[puntos.algoritmo == algoritmo].sort_values("recall")
        ax.plot(curva.recall, curva.precision, color=COLOR_REFERENCIA, linewidth=1, drawstyle="steps-post",
                label="Otros cuatro algoritmos" if i == 0 else None)
    curva = puntos[puntos.algoritmo == PRINCIPAL].sort_values("recall")
    ax.plot(curva.recall, curva.precision, color=ef.COLOR_SERIE, linewidth=2, drawstyle="steps-post",
            label="Elastic net (modelo principal)")

    prevalencia = metricas.prevalencia.iloc[0]
    ax.axhline(prevalencia, color=ef.INK_MUTED, linewidth=1, linestyle="--")
    ax.text(0.01, prevalencia + 0.015, f"Azar ({prevalencia:.3f})", ha="left", va="bottom",
            color=ef.INK_SECUNDARIO, fontsize=ef.TAM_LETRA_PEQUENA)

    principal = metricas[metricas.algoritmo == PRINCIPAL].iloc[0]
    puntos_operacion = [
        ("Umbral F$_2$", principal["recall_f2_base"], principal["precision_f2_base"], (0, 10), "center"),
        ("Decil superior", principal["recall_k10_base"], principal["precision_k10_base"], (0, -14), "center"),
    ]
    for etiqueta, x, y, desplazamiento, alineacion in puntos_operacion:
        ax.plot(x, y, "o", markersize=7, color=ef.COLOR_SERIE, markeredgecolor=ef.SURFACE, markeredgewidth=1.5, zorder=5)
        ax.annotate(etiqueta, (x, y), xytext=desplazamiento, textcoords="offset points", ha=alineacion,
                    fontsize=ef.TAM_LETRA_PEQUENA, color=ef.INK_PRIMARIO)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall: fracción de los hogares vulnerables marcados")
    ax.set_ylabel("Precision: fracción de los marcados\nque es vulnerable")
    ax.legend(loc="upper right")
    ef.guardar(fig, OUTPUT)
    print(f"Figura guardada: {OUTPUT}")


if __name__ == "__main__":
    main()
