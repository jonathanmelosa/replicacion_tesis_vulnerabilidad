"""
Timeline de cobertura temporal real de las fuentes geoespaciales evaluadas
para complementar/reemplazar Google Street View (Sección 3.3 de la tesis).

Solo lee un dato del proyecto, el conteo agregado de fotos de Street View
por año (`outputs/tables/gsv/gsv_fotos_por_anio.csv`, generado por
`src/conteo_gsv_por_anio.py`); las fechas de inicio/fin de cada
fuente son hechos verificados contra documentación oficial (ver
paper/main.tex, Sección 3.3.2 y sus fuentes: Copernicus/ESA, NOAA, Google
Earth Engine Data Catalog).
Se deja como script independiente (no como parte de ningún pipeline) para
poder regenerar la figura si esas fechas se corrigen o se agregan fuentes.

Output: paper/figures/fig_cobertura_temporal_fuentes.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "paper" / "figures"

# ── Paleta categórica por dominio físico (fija, no ciclada) ──────────────
COLOR_FOTO = "#4a3aa7"    # violeta — GSV
COLOR_RADAR = "#2a78d6"   # azul — Sentinel-1 / ALOS PALSAR
COLOR_OPTICO = "#eb6834"  # naranja — Sentinel-2 / Landsat 5
COLOR_LUCES = "#1baf7a"   # aqua — VIIRS / DMSP-OLS
COLOR_OLA = "#d03b3b"     # rojo — franjas de las olas ELCA

X_MIN, X_MAX = 2005, 2026

# Distribución real del año de captura de las fotos GSV efectivamente
# obtenidas — 20,935 fotos; la genera `src/conteo_gsv_por_anio.py` a
# partir de descargas_final.csv (ver Sección 3.2.1).
# Se usa para pintar la barra de GSV con intensidad proporcional a la
# densidad real de fotos por año, en vez de un bloque uniforme: la
# disponibilidad TÉCNICA de Street View arranca en 2007, pero la enorme
# mayoría de las fotos realmente usadas en este proyecto son de 2023 en
# adelante (60.9% del total en 2023-2026; mediana 2023) — pintar la barra
# de forma uniforme sería engañoso.
GSV_POR_ANIO_PATH = REPO_ROOT / "outputs" / "tables" / "gsv" / "gsv_fotos_por_anio.csv"
GSV_DENSIDAD_POR_ANIO = (
    pd.read_csv(GSV_POR_ANIO_PATH).set_index("anio")["n_fotos"].to_dict()
)

# (fila, [(inicio, fin, color, alpha, hatch, etiqueta_extremo)], nombre)
# alpha=1 -> cobertura completa/verificada; alpha=0.45 -> parcial o no
# verificada con hogares reales; hatch="//" -> corrida real confirmada
# sobre hogares ELCA (ver Cuadro "Estado de verificación").
FUENTES = [
    {
        "nombre": "Google Street View",
        "color": COLOR_FOTO,
        "segmentos": [],  # se dibuja aparte, con densidad real por año — ver más abajo
        "continua": True,
        "nota": "disponible desde 2007, pero 60.9% de las fotos\nusadas son de 2023 en adelante (mediana 2023)",
        "densidad": True,
    },
    {
        "nombre": "Sentinel-1 (radar)",
        "color": COLOR_RADAR,
        "segmentos": [(2014.75, X_MAX, 1.0, "//", True)],
        "continua": True,
        "nota": None,
    },
    {
        "nombre": "ALOS PALSAR (radar)",
        "color": COLOR_RADAR,
        "segmentos": [
            (2006, 2008, 0.4, None, False),
            (2008, 2011, 1.0, "//", True),
            (2015, X_MAX, 0.4, None, False),
        ],
        "continua": True,
        "nota": "corrida real completa: 100% ola 2010;\nhueco confirmado 2011-2014 (0% ola 2013)",
    },
    {
        "nombre": "Sentinel-2 (óptico)",
        "color": COLOR_OPTICO,
        "segmentos": [(2015.5, 2017.25, 0.35, None, False), (2017.25, X_MAX, 0.75, None, False)],
        "continua": True,
        "nota": "no se llevó a producción: no resuelve\nla cobertura real de la ola 2010",
    },
    {
        "nombre": "Landsat 5 TM (óptico)",
        "color": COLOR_OPTICO,
        "segmentos": [
            (X_MIN, 2008, 0.4, None, False),
            (2008, 2011.4, 1.0, "//", True),
        ],
        "continua": False,
        "nota": "corrida real: 65.5% ola 2010 (nubosidad);\nsin escenas reales desde jun.2011 (0% ola 2013)",
    },
    {
        "nombre": "VIIRS DNB (luces)",
        "color": COLOR_LUCES,
        "segmentos": [(2012.25, X_MAX, 0.75, None, False)],
        "continua": True,
        "nota": "no se llevó a producción: no resuelve\nla cobertura real de la ola 2010",
    },
    {
        "nombre": "DMSP-OLS (luces)",
        "color": COLOR_LUCES,
        "segmentos": [
            (X_MIN, 2008, 0.4, None, False),
            (2008, 2014, 1.0, "//", True),
        ],
        "continua": False,
        "nota": "corrida real completa: 100% ola 2010\ny 100% ola 2013",
    },
]

OLAS_ELCA = [2010, 2013, 2016]

fig, ax = plt.subplots(figsize=(9.5, 5.2))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

n = len(FUENTES)
alto_barra = 0.6

# ── Franjas verticales de las olas ELCA (van primero, detrás de las barras) ─
for ola in OLAS_ELCA:
    ax.axvspan(ola - 0.35, ola + 0.35, color=COLOR_OLA, alpha=0.12, zorder=0)
    ax.text(ola, n - 0.15, f"Ola\n{ola}", ha="center", va="bottom",
            fontsize=9, color=COLOR_OLA, fontweight="bold")

# ── Barras de cobertura por fuente ────────────────────────────────────────
for i, fuente in enumerate(FUENTES):
    y = n - 1 - i

    if fuente.get("densidad"):
        # GSV: una franja delgada POR AÑO, con opacidad proporcional a la
        # densidad real de fotos de ese año (GSV_DENSIDAD_POR_ANIO). Años
        # sin ninguna foto usada (2007-2011) quedan con una opacidad
        # residual fija, solo para marcar que la fuente "existía" —
        # ninguna barra sólida ahí sería engañosa (ver nota en CONFIG).
        max_densidad = max(GSV_DENSIDAD_POR_ANIO.values())
        for anio in range(X_MIN, X_MAX + 1):
            cuenta = GSV_DENSIDAD_POR_ANIO.get(anio, 0)
            if anio < 2007:
                alpha = 0.0
            elif cuenta == 0:
                alpha = 0.07
            else:
                alpha = 0.18 + 0.82 * (cuenta / max_densidad)
            if alpha <= 0:
                continue
            ax.add_patch(Rectangle(
                (anio, y - alto_barra / 2), 1.0, alto_barra,
                facecolor=fuente["color"], alpha=alpha, edgecolor="#2b2b28",
                linewidth=0.3, hatch="//", zorder=2,
            ))
        # Contorno completo de la barra (2007 en adelante) para que el
        # patrón de hatch verificado siga siendo legible como un borde
        # continuo, independientemente de la opacidad de cada franja.
        ax.add_patch(Rectangle(
            (2007, y - alto_barra / 2), X_MAX - 2007, alto_barra,
            facecolor="none", edgecolor="#2b2b28", linewidth=1.1, zorder=3,
        ))
    else:
        for (ini, fin, alpha, hatch, verificado) in fuente["segmentos"]:
            # El patrón de hatch se dibuja en el color del borde (edgecolor), no
            # del relleno — si ambos son iguales el hatch queda invisible. Para
            # las barras verificadas (hatch != None) se fuerza un borde oscuro
            # y opaco (sin alpha) para que el patrón sea legible; las demás usan
            # el mismo color del relleno como borde, casi invisible a propósito.
            borde = "#2b2b28" if hatch else fuente["color"]
            ax.add_patch(Rectangle(
                (ini, y - alto_barra / 2), fin - ini, alto_barra,
                facecolor=fuente["color"], alpha=alpha, edgecolor=borde,
                linewidth=1.1 if hatch else 0.8, hatch=hatch, zorder=2,
            ))

    ultimo_fin = X_MAX if fuente.get("densidad") else fuente["segmentos"][-1][1]
    if fuente["continua"] and ultimo_fin >= X_MAX:
        ax.annotate("", xy=(X_MAX + 0.35, y), xytext=(X_MAX, y),
                    arrowprops=dict(arrowstyle="-|>", color=fuente["color"], lw=1.5), zorder=3)
    if fuente["nota"]:
        ax.text(X_MAX + 0.55, y, fuente["nota"], fontsize=7.5, color="#6b6a63",
                va="center", ha="left", style="italic")

ax.set_yticks([n - 1 - i for i in range(n)])
ax.set_yticklabels([f["nombre"] for f in FUENTES], fontsize=10)
ax.set_xlim(X_MIN, X_MAX + 3.6)
ax.set_ylim(-0.7, n - 0.05)
ax.set_xlabel("Año")
ax.set_xticks(range(X_MIN, X_MAX + 1, 2))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color="#e5e4dd", linewidth=0.6, zorder=0)
# Sin titulo dentro de la imagen (estilo comun, 2026-10-01): el titulo es
# el \caption de main.tex.

leyenda = [
    mpatches.Patch(facecolor="#8a8a86", alpha=1.0, hatch="//", edgecolor="#2b2b28", label="Corrida real confirmada sobre hogares ELCA"),
    mpatches.Patch(facecolor="#8a8a86", alpha=0.75, edgecolor="#8a8a86", label="Cobertura de catálogo, no llevada a producción"),
    mpatches.Patch(facecolor="#8a8a86", alpha=0.4, edgecolor="#8a8a86", label="Fuera de la ventana usada por el proyecto"),
    mpatches.Patch(facecolor=COLOR_FOTO, alpha=0.9, hatch="//", edgecolor="#2b2b28",
                    label="GSV: intensidad ∝ densidad real de fotos por año"),
]
ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.42, -0.12),
          ncol=1, frameon=False, fontsize=8.5, alignment="left")

fig.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
out_path = FIG_DIR / "fig_cobertura_temporal_fuentes.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#fcfcfb")  # SURFACE de estilo_figuras.py
print(f"Figura guardada en: {out_path}")
