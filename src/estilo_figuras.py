"""
estilo_figuras.py
=====================================
Estilo comun de las figuras del documento. Toma la paleta y los colores
de tinta que ya usaba la mayoria de figuras (build_pobreza_desagregaciones.py,
graf_dmsp_monotonico_categoria.py, graf_perfil_nucleo_grupos.py) y fija el
tamano de letra en puntos de IMPRESION:

    Cada figura se genera con el ancho fisico con que se inserta en
    main.tex (`ancho(fraccion)` = fraccion x ancho de texto), de modo que
    9 pt en matplotlib son 9 pt en el PDF, cerca del \\footnotesize (10 pt)
    de las tablas. Asi ninguna figura queda con letra diminuta ni enorme.

Convenciones:
    - Sin titulo dentro de la imagen: el titulo es el \\caption de LaTeX
      (excepcion: los Sankey de transicion, ver graf_sankey_transicion_pobreza.py).
      Los subtitulos de panel (p. ej. "2010 -> 2013") si se mantienen.
    - Texto en espanol con tildes y flechas reales (->).
    - Etiquetas de variables legibles (etiquetas_variables.py), nunca nombres
      de columna.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# Ancho de texto de main.tex: carta (21.59 cm) menos margenes de 3 y 2.5 cm.
ANCHO_TEXTO_IN = (21.59 - 3.0 - 2.5) / 2.54

# Paleta categorica (orden fijo de slots) y colores de tinta.
PALETA = {
    "azul": "#2a78d6",
    "naranja": "#eb6834",
    "aguamarina": "#1baf7a",
    "amarillo": "#eda100",
    "magenta": "#e87ba4",
    "verde": "#008300",
    "violeta": "#4a3aa7",
    "rojo": "#e34948",
}
COLOR_CATEGORIA = {
    "Nunca pobre": PALETA["azul"],
    "Sale de la pobreza": PALETA["aguamarina"],
    "Entra en pobreza": PALETA["naranja"],
    "Siempre pobre": PALETA["rojo"],
}
COLOR_SERIE = PALETA["azul"]
INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

TAM_LETRA = 9
TAM_LETRA_PEQUENA = 8

FLECHA = "→"


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": INK_SECUNDARIO,
        "text.color": INK_PRIMARIO,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECUNDARIO,
        "ytick.labelcolor": INK_SECUNDARIO,
        "font.family": "sans-serif",
        "font.size": TAM_LETRA,
        "axes.titlesize": TAM_LETRA,
        "axes.labelsize": TAM_LETRA,
        "xtick.labelsize": TAM_LETRA_PEQUENA,
        "ytick.labelsize": TAM_LETRA_PEQUENA,
        "legend.fontsize": TAM_LETRA_PEQUENA,
        "legend.frameon": False,
        "axes.grid": True,
        # Solo lineas horizontales y detras de los datos; los graficos de
        # barras horizontales piden ax.grid(axis="x") explicitamente.
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })


def ancho(fraccion: float) -> float:
    """Ancho en pulgadas de una figura insertada con width=fraccion\\textwidth."""
    return fraccion * ANCHO_TEXTO_IN


def guardar(fig: plt.Figure, ruta: Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
