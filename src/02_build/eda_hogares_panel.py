"""
Analisis exploratorio del panel de hogares ELCA (2010, 2013, 2016).

Responde estas preguntas:
  1. Hogares unicos presentes en las tres olas (urbano/rural, estrato, region,
     departamento, municipio).
  2. Hogares que se dividieron entre 2010-2013 y entre 2013-2016.
  3. Hogares perdidos en 2013 y en 2016, y composicion de esa perdida.
  4. Personas totales por hogar y por ola.
  5. Personas que estaban en hogares que se perdieron.

Tablas -> outputs/tables/eda_hogares/
Figuras -> outputs/figures/eda_hogares/

Con `--solo-figuras`, no abre los datos por hogar: redibuja las dos figuras
del Anexo de composición del panel (01_hogares_por_ola, 03_attrition_tasas)
a partir de las tablas agregadas ya guardadas.

    python eda_hogares_panel.py --solo-figuras
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Estilo comun de las figuras del documento (2026-10-01).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import estilo_figuras as ef  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_hogares"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda_hogares"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WAVE_LABELS = {1: "2010", 2: "2013", 3: "2016"}

# Paleta fija y reducida, segura para daltonismo (evita usar color como unico
# canal para mas de dos categorias; siempre acompanada de etiquetas/leyenda).
COLOR_SINGLE = ef.COLOR_SERIE
COLOR_URBANO = ef.PALETA["azul"]
COLOR_RURAL = ef.PALETA["naranja"]
COLOR_LOST = ef.PALETA["rojo"]

ef.aplicar_estilo()


def savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    ef.guardar(fig, FIGURES_DIR / name)


def bar_with_labels(ax, x, heights, color=COLOR_SINGLE, fmt="{:,.0f}"):
    bars = ax.bar(x, heights, color=color, width=0.6)
    for rect, h in zip(bars, heights):
        ax.annotate(
            fmt.format(h),
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ef.TAM_LETRA_PEQUENA,
            color=ef.INK_SECUNDARIO,
        )
    return bars


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["ola_label"] = df["ola"].map(WAVE_LABELS)
    return df


def composition_table(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    """Conteo y % de hogares por ola, desagregado por index_col."""
    tabla = pd.pivot_table(
        df,
        values="consecutivo",
        columns="ola_label",
        index=index_col,
        aggfunc="count",
        margins=True,
        margins_name="Total",
    ).fillna(0).astype(int)
    porcentaje = (tabla.div(tabla.loc["Total"], axis=1) * 100).round(2)
    return tabla, porcentaje


def save_composition(df: pd.DataFrame, index_col: str, prefix: str) -> pd.DataFrame:
    tabla, porcentaje = composition_table(df, index_col)
    combinado = tabla.astype(str) + " (" + porcentaje.astype(str) + "%)"
    combinado.to_csv(TABLES_DIR / f"{prefix}_{index_col}.csv")
    return tabla


# ---------------------------------------------------------------------------
# 1. Hogares unicos presentes en las tres olas
# ---------------------------------------------------------------------------
def analizar_hogares_por_ola(df: pd.DataFrame) -> pd.DataFrame:
    conteo_ola = df.groupby("ola_label")["consecutivo"].nunique().reindex(
        [WAVE_LABELS[o] for o in (1, 2, 3)]
    )
    conteo_ola.to_frame("n_hogares").to_csv(TABLES_DIR / "01_hogares_por_ola.csv")
    graficar_hogares_por_ola(conteo_ola)
    return conteo_ola


def graficar_hogares_por_ola(conteo_ola: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(ef.ancho(0.48), 2.6))
    bar_with_labels(ax, conteo_ola.index, conteo_ola.values)
    ax.set_ylim(0, conteo_ola.max() * 1.15)
    ax.set_ylabel("Número de hogares")
    savefig(fig, "01_hogares_por_ola.png")


def analizar_patron_presencia(df: pd.DataFrame) -> pd.DataFrame:
    presencia = pd.crosstab(df["consecutivo"], df["ola"])
    for ola in (1, 2, 3):
        if ola not in presencia.columns:
            presencia[ola] = 0
    presencia = presencia[[1, 2, 3]]
    presencia["patron"] = (
        presencia.astype(bool).astype(int).astype(str).agg("".join, axis=1)
    )

    resumen = presencia["patron"].value_counts().to_frame(name="n_hogares")
    resumen["porcentaje"] = (resumen["n_hogares"] / resumen["n_hogares"].sum() * 100).round(2)
    resumen = resumen.sort_values("n_hogares", ascending=False)
    resumen.loc["Total"] = [resumen["n_hogares"].sum(), 100.0]
    resumen.to_csv(TABLES_DIR / "01_patron_presencia.csv")

    top = resumen.drop(index="Total").head(8)
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_with_labels(ax, top.index.astype(str), top["n_hogares"].values)
    ax.set_title("Patrones de presencia por ola (1=2010, 2=2013, 3=2016)")
    ax.set_xlabel("Patron (presente=1 / ausente=0)")
    ax.set_ylabel("Numero de hogares")
    savefig(fig, "01_patron_presencia.png")
    return presencia


def analizar_composicion_geografica(df: pd.DataFrame, presencia: pd.DataFrame) -> None:
    """Composicion (zona/estrato/region/dpto/mpio) de hogares presentes en
    mas de una ola, replicando el filtro usado en el notebook exploratorio."""
    df_presencia = df.merge(presencia[["patron"]], on="consecutivo", how="inner")
    df_filtrada = df_presencia.query("patron != '100'")

    for col in ["zona", "estrato", "region", "id_dpto", "id_mpio"]:
        save_composition(df_filtrada, col, "01_composicion")

    tabla_zona, _ = composition_table(df_filtrada, "zona")
    zonas = [z for z in tabla_zona.index if z != "Total"]
    olas = [WAVE_LABELS[o] for o in (1, 2, 3) if WAVE_LABELS[o] in tabla_zona.columns]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(olas))
    width = 0.35
    colors = {"Urbano": COLOR_URBANO, "Rural": COLOR_RURAL}
    for i, zona in enumerate(zonas):
        valores = [tabla_zona.loc[zona, o] for o in olas]
        offset = (i - (len(zonas) - 1) / 2) * width
        ax.bar(
            [xi + offset for xi in x],
            valores,
            width=width,
            label=zona,
            color=colors.get(zona, COLOR_SINGLE),
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(olas)
    ax.set_ylabel("Numero de hogares")
    ax.set_title("Composicion urbano/rural por ola\n(hogares presentes en mas de una ola)")
    ax.legend(frameon=False)
    savefig(fig, "01_zona_por_ola.png")


# ---------------------------------------------------------------------------
# 2. Hogares que se dividieron entre olas
# ---------------------------------------------------------------------------
def analizar_divisiones(df: pd.DataFrame) -> pd.DataFrame:
    tamanos = df.groupby(["consecutivo", "ola_label"]).size().rename("n_sub_hogares")
    divididos = tamanos[tamanos > 1].reset_index()

    resumen = (
        divididos.groupby("ola_label")
        .agg(hogares_divididos=("consecutivo", "nunique"), sub_hogares_generados=("n_sub_hogares", "sum"))
        .reindex([WAVE_LABELS[o] for o in (2, 3)])
        .fillna(0)
        .astype(int)
    )
    resumen.to_csv(TABLES_DIR / "02_hogares_divididos_por_ola.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    bar_with_labels(ax, resumen.index, resumen["hogares_divididos"].values)
    ax.set_title("Hogares originales que se dividieron, por ola")
    ax.set_ylabel("Numero de hogares divididos")
    savefig(fig, "02_hogares_divididos_por_ola.png")
    return resumen


# ---------------------------------------------------------------------------
# 3. Hogares perdidos (attrition) entre olas
# ---------------------------------------------------------------------------
def analizar_attrition(df: pd.DataFrame) -> dict:
    ola1 = set(df.query("ola == 1")["consecutivo"].unique())
    ola2 = set(df.query("ola == 2")["consecutivo"].unique())
    ola3 = set(df.query("ola == 3")["consecutivo"].unique())

    perdidos_2013 = ola1 - ola2
    perdidos_2016 = ola2 - ola3
    perdidos_total = ola1 - ola3

    resumen = pd.DataFrame(
        {
            "hogares_base": [len(ola1), len(ola2), len(ola1)],
            "hogares_perdidos": [len(perdidos_2013), len(perdidos_2016), len(perdidos_total)],
        },
        index=["2010->2013", "2013->2016", "2010->2016"],
    )
    resumen["tasa_perdida_%"] = (resumen["hogares_perdidos"] / resumen["hogares_base"] * 100).round(2)
    # La tasa de perdida (hogares de la ola base que no aparecen en la
    # siguiente) no coincide con la caida neta del numero de hogares:
    # algunos hogares no encontrados en 2013 vuelven a ser encuestados en
    # 2016 (agregado 2026-09-25, para explicar en el texto por que la
    # perdida 2013->2016 es 11.4% aunque el conteo solo cae 7.4%).
    resumen["hogares_regresan"] = [0, len(ola3 - ola2), 0]
    resumen["hogares_fin"] = [len(ola2), len(ola3), len(ola3)]
    resumen["caida_neta_%"] = ((1 - resumen["hogares_fin"] / resumen["hogares_base"]) * 100).round(2)
    resumen.to_csv(TABLES_DIR / "03_attrition_resumen.csv")
    graficar_attrition_tasas(resumen)

    # Composicion de la perdida 2010->2013 (caracteristicas en ola 1)
    base_ola1 = df.query("ola == 1")[
        ["consecutivo", "zona", "estrato", "region", "id_dpto", "id_mpio"]
    ].drop_duplicates("consecutivo")
    base_ola1["perdido"] = base_ola1["consecutivo"].isin(perdidos_2013)

    base_ola2 = df.query("ola == 2")[
        ["consecutivo", "zona", "estrato", "region", "id_dpto", "id_mpio"]
    ].drop_duplicates("consecutivo")
    base_ola2["perdido"] = base_ola2["consecutivo"].isin(perdidos_2016)

    for col in ["zona", "estrato", "region", "id_dpto", "id_mpio"]:
        tabla_2013 = (
            base_ola1.groupby(col)["perdido"].agg(n_perdidos="sum", tasa="mean")
            .assign(tasa_pct=lambda x: (x["tasa"] * 100).round(1))
            .drop(columns="tasa")
            .sort_values("tasa_pct", ascending=False)
        )
        tabla_2013.to_csv(TABLES_DIR / f"03_attrition_2010_2013_{col}.csv")

        tabla_2016 = (
            base_ola2.groupby(col)["perdido"].agg(n_perdidos="sum", tasa="mean")
            .assign(tasa_pct=lambda x: (x["tasa"] * 100).round(1))
            .drop(columns="tasa")
            .sort_values("tasa_pct", ascending=False)
        )
        tabla_2016.to_csv(TABLES_DIR / f"03_attrition_2013_2016_{col}.csv")

    tasa_zona_2013 = base_ola1.groupby("zona")["perdido"].mean() * 100
    tasa_zona_2016 = base_ola2.groupby("zona")["perdido"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    zonas = sorted(set(tasa_zona_2013.index) | set(tasa_zona_2016.index))
    x = range(len(zonas))
    width = 0.35
    ax.bar(
        [xi - width / 2 for xi in x],
        [tasa_zona_2013.get(z, 0) for z in zonas],
        width=width,
        label="2010->2013",
        color=COLOR_SINGLE,
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        [tasa_zona_2016.get(z, 0) for z in zonas],
        width=width,
        label="2013->2016",
        color=COLOR_LOST,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(zonas)
    ax.set_ylabel("% de hogares perdidos")
    ax.set_title("Composicion de la perdida de hogares por zona")
    ax.legend(frameon=False)
    savefig(fig, "03_attrition_zona.png")

    return {
        "ola1": ola1,
        "ola2": ola2,
        "ola3": ola3,
        "perdidos_2013": perdidos_2013,
        "perdidos_2016": perdidos_2016,
        "perdidos_total": perdidos_total,
    }


# ---------------------------------------------------------------------------
# 4. Personas totales por hogar y por ola
# ---------------------------------------------------------------------------
def analizar_personas(df: pd.DataFrame) -> None:
    resumen = df.groupby("ola_label")["t_personas"].describe().reindex(
        [WAVE_LABELS[o] for o in (1, 2, 3)]
    )
    resumen.to_csv(TABLES_DIR / "04_personas_resumen_por_ola.csv")

    total_por_ola = df.groupby("ola_label")["t_personas"].sum().reindex(
        [WAVE_LABELS[o] for o in (1, 2, 3)]
    )
    total_por_ola.to_frame("total_personas").to_csv(TABLES_DIR / "04_personas_total_por_ola.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    bar_with_labels(ax, total_por_ola.index, total_por_ola.values)
    ax.set_title("Total de personas cubiertas por ola")
    ax.set_ylabel("Numero de personas")
    savefig(fig, "04_personas_total_por_ola.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    datos = [df.query("ola_label == @o")["t_personas"].dropna() for o in (WAVE_LABELS[k] for k in (1, 2, 3))]
    bp = ax.boxplot(datos, tick_labels=[WAVE_LABELS[o] for o in (1, 2, 3)], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_SINGLE)
        patch.set_alpha(0.5)
    ax.set_title("Distribucion del tamano del hogar (personas) por ola")
    ax.set_ylabel("Personas por hogar")
    savefig(fig, "04_distribucion_personas_por_ola.png")


# ---------------------------------------------------------------------------
# 5. Personas en hogares que se perdieron
# ---------------------------------------------------------------------------
def analizar_personas_perdidas(df: pd.DataFrame, attrition: dict) -> None:
    personas_ola1 = df.query("ola == 1").drop_duplicates("consecutivo").set_index("consecutivo")["t_personas"]
    personas_ola2 = df.query("ola == 2").drop_duplicates("consecutivo").set_index("consecutivo")["t_personas"]

    personas_perdidas_2013 = personas_ola1.loc[list(attrition["perdidos_2013"])].sum()
    personas_perdidas_2016 = personas_ola2.loc[list(attrition["perdidos_2016"])].sum()
    personas_perdidas_total = personas_ola1.loc[list(attrition["perdidos_total"])].sum()

    resumen = pd.DataFrame(
        {"personas_en_hogares_perdidos": [personas_perdidas_2013, personas_perdidas_2016, personas_perdidas_total]},
        index=["2010->2013", "2013->2016", "2010->2016"],
    )
    resumen.to_csv(TABLES_DIR / "05_personas_hogares_perdidos.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    bar_with_labels(ax, resumen.index, resumen["personas_en_hogares_perdidos"].values, color=COLOR_LOST)
    ax.set_title("Personas que estaban en hogares perdidos")
    ax.set_ylabel("Numero de personas")
    savefig(fig, "05_personas_hogares_perdidos.png")


def graficar_attrition_tasas(resumen: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(ef.ancho(0.48), 2.6))
    periodos = [i.replace("->", f" {ef.FLECHA}\n") for i in resumen.index]
    bar_with_labels(ax, periodos, resumen["tasa_perdida_%"].values, fmt="{:.1f}%")
    ax.set_ylim(0, resumen["tasa_perdida_%"].max() * 1.15)
    ax.set_ylabel("% de hogares perdidos")
    savefig(fig, "03_attrition_tasas.png")


def graficar_desde_csv() -> None:
    """Redibuja las dos figuras del Anexo de composición del panel
    (01_hogares_por_ola, 03_attrition_tasas) solo a partir de las tablas
    agregadas ya guardadas en TABLES_DIR, sin abrir los datos por hogar."""
    conteo = pd.read_csv(TABLES_DIR / "01_hogares_por_ola.csv", index_col="ola_label", dtype={"ola_label": str})
    graficar_hogares_por_ola(conteo["n_hogares"])
    graficar_attrition_tasas(pd.read_csv(TABLES_DIR / "03_attrition_resumen.csv", index_col=0))


def main() -> None:
    if "--solo-figuras" in sys.argv:
        graficar_desde_csv()
        return
    df = load_data()

    analizar_hogares_por_ola(df)
    presencia = analizar_patron_presencia(df)
    analizar_composicion_geografica(df, presencia)
    analizar_divisiones(df)
    attrition = analizar_attrition(df)
    analizar_personas(df)
    analizar_personas_perdidas(df, attrition)

    print(f"Tablas guardadas en: {TABLES_DIR}")
    print(f"Figuras guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
