"""
graf_mapa_territorial_tramas.py
=================================
VALIDACION DE ESTILO para el mapa territorial de transicion de pobreza
(mapa_territorial_transicion_divipola.py, para correr en la sala con datos
reales de hogar). Este script usa las proporciones REALES de la ELCA por
region (2010->2013, pobreza monetaria) pero la posicion de cada punto
dentro de su region sigue siendo SIMULADA -- no hay coordenadas reales de
hogar disponibles fuera de la base restringida de la universidad.

QUE ES REAL Y QUE ES SIMULADO (leer antes de interpretar la figura)
    REAL: n_hogares y % por categoria de transicion, por region (de
    eda_transicion_covariables.py, outputs/tables/eda_transicion_covariables/).
    SIMULADO: la posicion exacta de cada punto -- se dispersa al azar
    alrededor de la ubicacion aproximada de su region (COORDENADAS_REGION,
    identica a mapa_transicion_regiones.py), con mayor dispersion en las
    microrregiones rurales (mas dispersas geograficamente que una ciudad).

DECISIONES DE ESTILO
    1. Agregacion espacial (densidad suavizada por grupo dominante), NO
       puntos individuales -- para no poder recuperar la ubicacion exacta
       de un hogar a partir de la imagen (riesgo de privacidad: un mapa de
       puntos individuales es invertible con precision de ~1km/pixel a
       partir de la extension y el tamano de figura, ambos publicos en el
       codigo).
    2. Trama (patron de lineas: cruces/diagonales/puntos/horizontales) en
       vez de relleno de color solido -- dos motivos: (a) mezclar los 4
       colores por proporcion producia tonos indistinguibles (cafe/gris)
       en zonas con 2-3 grupos parejos; (b) un relleno de color grande
       competia visualmente con la paleta "inferno" del brillo DMSP.
    3. Trama CON COLOR por grupo (version final, 2026-09-12) -- se probo
       tambien monocroma (blanco); se prefirio la version con color tras
       comparar ambas.
    4. Contorno de pais/departamento en blanco solido y grueso -- mucho
       mas contraste que el gris tenue de versiones anteriores.
    5. Recorte estricto al poligono real de Colombia (shapely.vectorized.contains)
       -- el difuminado gaussiano se expande mas alla de la frontera
       (Venezuela, Panama, oceano) si no se recorta explicitamente.
    6. "Dominante" = argmax simple de densidad suavizada por grupo, SIN
       margen minimo sobre el segundo lugar (version final, 2026-09-12).
       Se probo tambien con margen minimo (12 puntos porcentuales), que
       excluye a "Entra en pobreza"/"Sale de la pobreza" de TODO el mapa
       (nunca son mayoria en ninguna de las 9 regiones, ni siquiera sin
       margen) -- se prefiere el argmax simple, que si
       deja ver esos 2 grupos en las zonas donde son el mas grande de los
       4 (aunque no lleguen a mayoria absoluta).

NOTA PENDIENTE
    Diferenciar visualmente "sin datos" (zona sin hogares cerca) de
    "hay datos pero sin dominante claro" (solo aplicaria si se reactiva
    el margen minimo, ver punto 6) -- no aplica a esta version final, que
    no usa margen.

INPUTS
    outputs/tables/eda_transicion_covariables/region_x_categoria_monetaria_2010_2013.csv
    outputs/tables/eda_transicion_covariables/dmsp_por_region_monetaria_2010_2013.csv
    data/interim/geo_referencia/gadm41_COL_0.shp
    data/interim/geo_referencia/gadm41_COL_1.shp
    data/interim/geo_referencia/fondo_dmsp_colombia_2010.npz

OUTPUTS
    outputs/figures/mapa_territorial_divipola/prueba_estilo_tramas_regional.png

COMO CORRER
    python src/graf_mapa_territorial_tramas.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from shapely.vectorized import contains

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEO_DIR = PROJECT_ROOT / "data" / "interim" / "geo_referencia"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "mapa_territorial_divipola"

COORDENADAS_REGION = {
    "Atlántica": (-74.8, 10.6), "Atlántica Media": (-75.0, 9.0),
    "Oriental": (-73.1, 7.1), "Central": (-75.6, 6.3),
    "Pacífica": (-76.5, 3.5), "Bogotá": (-74.08, 4.65),
    "Cundi-Boyacense": (-73.5, 5.6), "Eje Cafetero": (-75.75, 4.6),
    "Centro-Oriente": (-73.0, 6.6),
}
REGIONES_RURALES = {"Atlántica Media", "Cundi-Boyacense", "Eje Cafetero", "Centro-Oriente"}
CATEGORIAS = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]
HATCH_GRUPO = {"Sale de la pobreza": "...", "Nunca pobre": "///", "Siempre pobre": "xxx", "Entra en pobreza": "---"}
COLOR_GRUPO = {
    "Siempre pobre": "#e6242f", "Sale de la pobreza": "#22c55e",
    "Entra en pobreza": "#eb6834", "Nunca pobre": "#38bdf8",
}

RESOLUCION = 300
SIGMA = 11  # suavizado de la densidad -- mayor valor, fronteras mas limpias/menos fragmentadas
MARGEN_MINIMO = 0.0  # SIN margen minimo (version final, ver punto 6 del docstring)
UMBRAL_PERCENTIL_DENSIDAD = 25  # celdas bajo este percentil de densidad total quedan sin trama


def simular_puntos_con_proporciones_reales(rng: np.random.Generator) -> pd.DataFrame:
    pct = pd.read_csv(TABLES_DIR / "region_x_categoria_monetaria_2010_2013.csv", index_col=0)
    dmsp = pd.read_csv(TABLES_DIR / "dmsp_por_region_monetaria_2010_2013.csv", index_col=0)

    filas = []
    for region, (lon0, lat0) in COORDENADAS_REGION.items():
        n = int(dmsp.loc[region, "n_hogares"])
        dispersion = 1.3 if region in REGIONES_RURALES else 0.35
        lons = lon0 + rng.normal(0, dispersion, n)
        lats = lat0 + rng.normal(0, dispersion, n)
        probs = pct.loc[region, CATEGORIAS].to_numpy() / 100.0
        probs = probs / probs.sum()
        grupos_region = rng.choice(CATEGORIAS, size=n, p=probs)
        filas.append(pd.DataFrame({"longitud": lons, "latitud": lats, "grupo": grupos_region}))
    return pd.concat(filas, ignore_index=True)


def calcular_dominante(muestra: pd.DataFrame, extent: tuple, pais: gpd.GeoDataFrame) -> tuple:
    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, RESOLUCION)
    ys = np.linspace(ymin, ymax, RESOLUCION)

    densidades = {}
    for cat in CATEGORIAS:
        sub = muestra[muestra["grupo"] == cat]
        hist, _, _ = np.histogram2d(sub["latitud"], sub["longitud"], bins=[ys, xs])
        densidades[cat] = gaussian_filter(hist, sigma=SIGMA)

    pila = np.stack([densidades[c] for c in CATEGORIAS], axis=0)
    total = pila.sum(axis=0)
    proporciones = pila / np.clip(total, 1e-9, None)

    orden = np.argsort(-proporciones, axis=0)
    primero = np.take_along_axis(proporciones, orden[:1], axis=0)[0]
    segundo = np.take_along_axis(proporciones, orden[1:2], axis=0)[0]
    margen = primero - segundo

    dominante_idx = orden[0].astype(float)
    umbral_densidad = np.percentile(total[total > 0], UMBRAL_PERCENTIL_DENSIDAD)
    dominante_idx[total < umbral_densidad] = np.nan
    dominante_idx[margen < MARGEN_MINIMO] = np.nan

    h, w = dominante_idx.shape
    X = np.linspace(xmin, xmax, w)
    Y = np.linspace(ymin, ymax, h)
    Xg, Yg = np.meshgrid(X, Y)
    dentro_colombia = contains(pais.union_all(), Xg, Yg)
    dominante_idx[~dentro_colombia] = np.nan

    n_sin_dominante = np.isnan(dominante_idx[dentro_colombia]).sum()
    n_total_dentro = dentro_colombia.sum()
    print(
        f"Celdas dentro de Colombia sin dominante claro "
        f"(sin datos suficientes o margen < {MARGEN_MINIMO}): "
        f"{n_sin_dominante}/{n_total_dentro} ({100 * n_sin_dominante / n_total_dentro:.1f}%)"
    )

    return dominante_idx, X, Y


def graficar(dominante_idx: np.ndarray, X: np.ndarray, Y: np.ndarray, extent: tuple,
             compuesto_log: np.ndarray, pais: gpd.GeoDataFrame, deptos: gpd.GeoDataFrame) -> None:
    dominante_masked = np.ma.masked_invalid(dominante_idx)
    xmin, xmax, ymin, ymax = extent

    fig, ax = plt.subplots(figsize=(9, 11))
    fig.patch.set_facecolor("#05070c")
    ax.set_facecolor("#05070c")
    ax.imshow(compuesto_log, cmap="inferno", extent=extent, origin="upper", zorder=1,
              vmin=0, vmax=np.percentile(compuesto_log, 99.7))

    plt.rcParams["hatch.linewidth"] = 0.7
    for i, cat in enumerate(CATEGORIAS):
        capa = np.ma.masked_where(dominante_masked != i, dominante_masked)
        ax.contourf(X, Y, capa, levels=[i - 0.5, i + 0.5], colors="none", hatches=[HATCH_GRUPO[cat]], zorder=2)
        for coleccion in ax.collections[-1:]:
            coleccion.set_edgecolor(COLOR_GRUPO[cat])
            coleccion.set_linewidth(0.0)

    pais.boundary.plot(ax=ax, color="white", linewidth=1.8, zorder=3, alpha=1.0)
    deptos.boundary.plot(ax=ax, color="white", linewidth=0.7, zorder=3, alpha=0.75)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    ax.set_title(
        "Grupos de transición de pobreza monetaria por región (dominante), 2010 → 2013\n"
        "proporciones reales por región, posición dentro de la región simulada",
        color="white", fontsize=10.5, loc="left",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=COLOR_GRUPO[g],
                              hatch=HATCH_GRUPO[g], linewidth=0.7, label=g) for g in CATEGORIAS]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("white")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "prueba_estilo_tramas_regional.png"
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Guardado: {out}")


def main() -> None:
    pais = gpd.read_file(GEO_DIR / "gadm41_COL_0.shp")
    deptos = gpd.read_file(GEO_DIR / "gadm41_COL_1.shp")
    fondo = np.load(GEO_DIR / "fondo_dmsp_colombia_2010.npz")
    compuesto_log, extent = fondo["compuesto_log"], tuple(fondo["extent"])

    rng = np.random.default_rng(11)
    muestra = simular_puntos_con_proporciones_reales(rng)
    print(f"Total de puntos (n real de hogares por region, posicion simulada): {len(muestra):,}")

    dominante_idx, X, Y = calcular_dominante(muestra, extent, pais)
    graficar(dominante_idx, X, Y, extent, compuesto_log, pais, deptos)


if __name__ == "__main__":
    main()
