"""
Caracterizacion de las variables finales del panel hogar-ola que alimenta
el modelo benchmark de prediccion de transicion a la pobreza (Seccion
"Estadisticas descriptivas" de la tesis, subsec:desc).

A diferencia de eda_hogares_panel.py (que caracteriza el PANEL de hogares:
composicion, hogares divididos, atricion, personas por hogar, sobre
hogar_elca_longitudinal_clean.parquet), este script caracteriza las
VARIABLES FINALES que sobrevivieron la auditoria y quedaron en el dataset
consolidado que consume el modelo
(benchmark_consolidado_elca_longitudinal.parquet, 178 columnas de
contenido tras excluir identidad -- ver build_benchmark_consolidado.py):
cuantas son, que temas cubren, como se comportan por ola, que cambia de
forma importante en el periodo 2010-2013-2016, que tan correlacionadas
estan entre si (relevante para el modelado posterior), y como son los
hogares del panel (perfil por zona y por condicion de pobreza).

Se le unen ademas las 4 variables insignia de las tres fuentes
geoespaciales que sobrevivieron la evaluacion de cobertura (DMSP-OLS,
ALOS PALSAR, Landsat 5 TM; ver subsec:geo) -- no solo se dejan en la tabla
estatica de la ola 2010 ya existente (tabla_descriptivos_geoespaciales.py),
sino que entran a este mismo analisis tematico, de correlacion y de
perfil de hogares.

Union de la fuente geoespacial al consolidado
----------------------------------------------------------------------
variables_geoespaciales_unificadas.parquet identifica hogar x ola con
`id_ola` (dtype object): id_ola en ola=2010 == consecutivo (ola 1 del
panel), en ola=2013 == llave (ola 2), en ola=2016 == llave_n16 (ola 3) --
exactamente el mismo esquema `llave_compuesta` que ya usa
build_benchmark_consolidado.py. Convertir id_ola a float da
llave_compuesta directamente, unica por (llave_compuesta, ola real);
se remapea ola (2010/2013/2016) a las olas 1/2/3 del panel y se hace un
left join validate="one_to_one".

Cobertura real por variable insignia geoespacial (ya documentada en
subsec:geo): dmsp_stable_lights y su tendencia acumulada cubren olas 1 y
2 al 100%, nada en ola 3; alos_hh_db solo ola 1 (100%); l5_ndvi solo ola
1 (65.5%). Por eso las DMSP entran a las figuras/tablas por-ola (tienen 2
olas reales) y ALOS/Landsat 5 (una sola ola) solo entran a correlacion,
perfil de hogares y a la figura de distribuciones geoespaciales 2010 --
nunca se imputan olas sin dato real.

Output:
    outputs/tables/eda_variables_modelo/
    outputs/figures/eda_variables_modelo/

Con `--solo-figura-inventario`, no abre los datos por hogar ni las variables
geoespaciales: redibuja la figura de inventario por modulo (Anexo de
cobertura de covariables) a partir de 01_inventario_variables.csv ya guardado.

    python eda_variables_modelo.py --solo-figura-inventario
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Estilo y etiquetas comunes de las figuras del documento (2026-10-01).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import estilo_figuras as ef  # noqa: E402
from etiquetas_variables import etiqueta as _etiqueta_nucleo  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
CONSOLIDADO_PATH = PROCESSED / "benchmark_consolidado_elca_longitudinal.parquet"
GEO_PATH = PROCESSED / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_variables_modelo"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda_variables_modelo"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WAVE_LABELS = {1: "2010", 2: "2013", 3: "2016"}
YEAR_TO_WAVE = {2010: 1, 2013: 2, 2016: 3}

ID_COLS = {"consecutivo", "llave", "llave_n16", "ola", "zona", "llave_compuesta", "consecutivo_c"}

# Columnas de contenido de cada archivo fuente, en el mismo orden que
# FEATURE_PARQUETS de build_benchmark_consolidado.py, para poder etiquetar
# cada columna del consolidado con su modulo tematico de origen.
MODULE_RANGES = [
    ("Monetario/pobreza", [
        "ingreso_percapita_hogar", "gasto_percapita_hogar", "lp", "li",
        "pobre_ingreso", "pobre_extremo_ingreso", "pobre_gasto", "pobre_extremo_gasto",
        "concuerdan_ingreso_gasto", "brecha_lp_ingreso", "brecha_lp_gasto",
        "ingreso_percapita_hogar_real", "gasto_percapita_hogar_real",
    ]),
    ("Personas (9 bloques)", [
        "n_ninos_5", "n_ninos_12", "n_adultos_mayores", "razon_dependencia_demografica",
        "pct_mujeres_hogar", "sexo_jefe", "edad_jefe", "tiene_conyuge_jefe",
        "nivel_educ_jefe", "nivel_educ_ordinal_jefe", "ocupado_jefe",
        "categoria_ocupacional_jefe", "horas_trabajo_jefe", "jornalero_jefe",
        "nivel_educ_max_hogar", "pct_adultos_alfabetizados", "tasa_asistencia_escolar",
        "tasa_ocupacion_hogar", "pct_adultos_fue_jornalero", "tuvo_evento_salud_jefe",
        "pct_ninos_con_discapacidad", "n_eventos_salud_hogar", "tuvo_hospitalizacion_hogar",
        "tasa_afiliacion_salud_hogar", "tiene_prepagada_hogar", "ahorra_jefe",
        "participa_organizacion_jefe", "n_tipos_organizacion_jefe", "cotiza_pension_jefe",
        "afiliado_pension_jefe", "afiliado_salud_laboral_jefe", "tasa_ahorro_hogar",
        "pct_hogar_participa_organizacion", "tasa_cotizacion_pension_hogar",
        "tasa_afiliacion_pension_hogar", "tasa_afiliacion_salud_laboral_hogar",
        "categoria_laboral_oit_jefe", "grado_educ_jefe", "medio_consiguio_jefe",
        "registro_mercantil_jefe", "n_empleados_jefe", "pct_ninos_no_estudia_razon_economica",
        "pct_ninos_recibio_beca_subsidio", "pct_ninos_credito_estudiar",
        "pct_ninos_apoyo_alimentario_escolar", "pct_ninos_apoyo_material_escolar",
        "tasa_control_preventivo_hogar", "pct_ninos_control_pediatrico",
        "tasa_planificacion_familiar", "pct_ninos_beneficiario_sss", "estado_civil_jefe",
        "etnia_jefe", "edad_union_jefe", "pct_ninos_padre_vivo", "pct_ninos_madre_viva",
        "pct_ninos_trabaja_otro_hogar", "tasa_participacion_civica_hogar",
    ]),
    ("Comunidades", [
        "consecutivo_c", "percepcion_inseguridad_comunidad", "n_problemas_convivencia_comunidad",
        "problema_contaminacion_comunidad", "n_organizaciones_comunidad", "barrio_legal_comunidad",
        "problema_homicidios_comunidad", "riesgo_inundacion_comunidad", "acceso_agua_comunidad",
        "hay_desplazados_comunidad", "n_desplazados_comunidad", "tiene_puesto_salud_comunidad",
        "tiene_escuela_primaria_comunidad", "tiene_colegio_secundaria_comunidad",
        "tiene_transporte_publico_comunidad", "solidaridad_comunidad",
        "acude_justicia_formal_comunidad", "cortes_agua_comunidad",
        "n_obras_infraestructura_reciente_comunidad", "n_servicios_primera_infancia_comunidad",
        "n_espacios_publicos_comunidad", "n_acciones_conflicto_armado_comunidad",
    ]),
    ("Ninos (6-9 anios)", [
        "tasa_vacunacion_basica_hogar", "tasa_control_crecimiento_hogar",
        "tasa_vacuna_fiebreamarilla_hogar", "tasa_asistencia_escolar_nino_hogar",
        "talla_promedio_nino_hogar", "peso_promedio_nino_hogar", "pct_ninos_oficios_hogar",
        "n_oficios_promedio_nino_hogar", "horas_oficio_promedio_nino_hogar",
        "pct_ninos_trabajo_remunerado_hogar", "indice_estimulacion_hogar_nino",
        "tvip_puntaje_directo_hogar", "pct_ninos_cuidado_terceros_hogar",
    ]),
    ("Choques", [
        "total_choques_hogar", "tuvo_algun_choque_hogar", "n_tipos_choque_hogar",
        "tuvo_choque_salud_hogar", "tuvo_choque_economico_hogar", "tuvo_choque_patrimonial_hogar",
        "tuvo_choque_agropecuario_hogar", "tuvo_choque_familiar_hogar", "tuvo_choque_severo_hogar",
        "afrontamiento_erosivo_hogar", "afrontamiento_protector_hogar", "intensifico_trabajo_hogar",
        "retiro_hijos_colegio_choque_hogar", "redujo_alimentos_choque_hogar",
        "se_endeudo_formal_choque_hogar", "se_endeudo_informal_choque_hogar", "no_ajusto_choque_hogar",
    ]),
    ("Hogar/vivienda/activos", [
        "tenencia_vivienda_hogar", "tipo_vivienda_hogar", "material_paredes_hogar",
        "material_pisos_hogar", "servicio_sanitario_hogar", "obtencion_agua_hogar",
        "energia_cocinan_hogar", "eliminan_basura_hogar", "estrato_hogar",
        "estrato_verificado_hogar", "n_hogares_comparte_vivienda_hogar", "riqueza_pca_hogar",
        "n_servicios_publicos_hogar", "personas_por_cuarto_hogar", "personas_por_dormitorio_hogar",
        "n_bienes_durables_hogar", "tiene_vehiculo_hogar", "tiene_internet_hogar",
        "n_activos_financieros_hogar", "tiene_propiedad_rural_hogar", "tiene_transporte_carga_hogar",
        "tiene_ingreso_agropecuario_hogar", "beneficiario_familias_accion_hogar",
        "beneficiario_red_juntos_hogar", "n_programas_sociales_hogar",
        "beneficiario_algun_programa_hogar", "tuvo_desastre_natural_hogar",
        "financio_credito_formal_vivienda_hogar", "financio_recursos_propios_vivienda_hogar",
        "financio_subsidio_vivienda_hogar", "financio_otra_fuente_vivienda_hogar",
        "tiene_escritura_vivienda_hogar", "tiene_titulo_baldio_hogar",
        "valor_arriendo_pagado_hogar", "tiene_deuda_hogar", "deuda_formal_hogar",
        "deuda_informal_hogar", "recibio_ayuda_alimentos_hogar", "recibio_ayuda_fam_colombia_hogar",
        "recibio_ayuda_fam_exterior_hogar", "recibio_ayuda_ong_hogar",
        "recibio_ayuda_org_internacional_hogar", "recibio_ayuda_religiosa_hogar",
        "recibio_ayuda_desplazados_hogar", "envio_ayuda_alimentos_hogar",
        "envio_ayuda_fam_colombia_hogar", "envio_ayuda_fam_exterior_hogar",
        "envio_ayuda_otras_hogar", "uso_ayuda_alimentos_hogar", "uso_ayuda_salud_hogar",
        "uso_ayuda_educacion_hogar", "uso_ayuda_vivienda_hogar", "n_tipos_ayuda_recibida_hogar",
        "uso_ayuda_agropecuario_hogar", "uso_ayuda_ahorrar_hogar", "practica_religion_hogar",
    ]),
    ("Geoespacial", ["dmsp_stable_lights", "dmsp_stable_lights_acum_tendencia", "alos_hh_db", "l5_ndvi"]),
]
COL_TO_MODULE = {col: modulo for modulo, cols in MODULE_RANGES for col in cols}

# Variables clave, una por tema, usadas en las figuras/tablas de
# distribucion/tendencia/correlacion/perfil. dmsp_* solo tienen ola 1 y 2
# (no se completan con NaN en ola 3 para no sugerir dato inexistente en
# las figuras por-ola); alos_hh_db y l5_ndvi solo tienen ola 1, por lo que
# se excluyen de las figuras por-ola y solo entran a correlacion/perfil.
KEY_VARS_BY_WAVE = [
    "ingreso_percapita_hogar_real", "gasto_percapita_hogar_real", "brecha_lp_ingreso",
    "riqueza_pca_hogar", "n_bienes_durables_hogar", "personas_por_cuarto_hogar",
    "nivel_educ_ordinal_jefe", "tasa_ocupacion_hogar", "tasa_afiliacion_salud_hogar",
    "tasa_asistencia_escolar", "razon_dependencia_demografica", "tuvo_algun_choque_hogar",
    "percepcion_inseguridad_comunidad", "dmsp_stable_lights", "dmsp_stable_lights_acum_tendencia",
]
KEY_VARS_2010_ONLY = ["alos_hh_db", "l5_ndvi"]
KEY_VARS_ALL = KEY_VARS_BY_WAVE + KEY_VARS_2010_ONLY

GEO_INSIGNIA_2010 = ["dmsp_stable_lights", "dmsp_stable_lights_acum_tendencia", "alos_hh_db", "l5_ndvi"]

COLOR_SINGLE = "#4C72B0"
COLOR_URBANO = "#4C72B0"
COLOR_RURAL = "#DD8452"
COLOR_POBRE = "#C44E52"
COLOR_NOPOBRE = "#55A868"
PALETTE_MODULOS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#CCB974"]

ef.aplicar_estilo()

# Etiquetas legibles de las variables de la figura de correlaciones que no
# estan en etiquetas_variables.py (definiciones tomadas de los scripts de
# 04_features que construyen cada variable).
ETIQUETAS_CORRELACION = {
    "afrontamiento_erosivo_hogar": "Afrontamiento erosivo ante choques",
    "ahorra_jefe": "Jefe ahorra",
    "gasto_percapita_hogar": "Gasto per cápita (nominal)",
    "ingreso_percapita_hogar": "Ingreso per cápita (nominal)",
    "jornalero_jefe": "Jefe jornalero",
    "n_ninos_5": "N.º de niños menores de 6 años",
    "n_tipos_ayuda_recibida_hogar": "N.º de tipos de ayuda recibida",
    "n_tipos_choque_hogar": "N.º de tipos de choque",
    "n_tipos_organizacion_jefe": "N.º de organizaciones del jefe",
    "participa_organizacion_jefe": "Jefe participa en organizaciones",
    "pct_adultos_fue_jornalero": "% de adultos que fueron jornaleros",
    "pct_hogar_participa_organizacion": "% de adultos en organizaciones",
    "peso_promedio_nino_hogar": "Peso promedio de los niños",
    "recibio_ayuda_fam_colombia_hogar": "Recibió ayuda de familiares en Colombia",
    "redujo_alimentos_choque_hogar": "Redujo alimentos ante choques",
    "talla_promedio_nino_hogar": "Talla promedio de los niños",
    "tasa_ahorro_hogar": "% de adultos que ahorran",
    "tuvo_algun_choque_hogar": "Tuvo algún choque",
}


# Nombres de modulo con tildes para la figura de inventario.
NOMBRES_MODULO = {
    "Monetario/pobreza": "Monetario y pobreza",
    "Personas (9 bloques)": "Personas (9 bloques)",
    "Comunidades": "Comunidades",
    "Ninos (6-9 anios)": "Niños (6 a 9 años)",
    "Choques": "Choques",
    "Hogar/vivienda/activos": "Hogar, vivienda y activos",
}


def etiqueta_legible(variable: str) -> str:
    if variable in ETIQUETAS_CORRELACION:
        return ETIQUETAS_CORRELACION[variable]
    texto = _etiqueta_nucleo(variable)
    return texto.replace("N.\\textsuperscript{o}", "N.º").replace("\\%", "%")


def savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    ef.guardar(fig, FIGURES_DIR / name)


def llave_compuesta(df: pd.DataFrame) -> pd.Series:
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def cargar_consolidado_con_geo() -> pd.DataFrame:
    df = pd.read_parquet(CONSOLIDADO_PATH)

    geo = pd.read_parquet(GEO_PATH, columns=["id_ola", "ola"] + GEO_INSIGNIA_2010)
    geo["llave_compuesta"] = geo["id_ola"].astype(float)
    geo["ola"] = geo["ola"].map(YEAR_TO_WAVE)
    geo = geo.dropna(subset=["ola"])
    geo["ola"] = geo["ola"].astype(int)
    geo = geo[["llave_compuesta", "ola"] + GEO_INSIGNIA_2010]

    antes = len(df)
    df = df.merge(geo, on=["llave_compuesta", "ola"], how="left", validate="many_to_one")
    assert len(df) == antes, "El join geoespacial cambio el numero de filas del consolidado."
    return df


def clasificar_tipo(serie: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(serie) or str(serie.dtype) == "boolean":
        return "Booleana"
    if pd.api.types.is_numeric_dtype(serie):
        return "Numerica"
    return "Categorica"


# ── 1. Inventario de variables ──────────────────────────────────────────


def tabla_inventario(df: pd.DataFrame, content_cols: list) -> pd.DataFrame:
    """Devuelve el inventario completo (incluye las 4 variables insignia
    geoespaciales, que la matriz de correlacion si usa), pero el CSV y la
    figura de inventario solo cubren las variables de la ELCA: las 4
    insignia no son el set geoespacial de los modelos (23 DMSP-OLS en
    AgeoDMSP/BgeoDMSP, ver construir_pipeline_geo_dmsp.py), asi que
    contarlas como "modulo Geoespacial" subestimaba ese set."""
    n_total = len(df)
    filas = []
    for col in content_cols:
        filas.append({
            "variable": col,
            "modulo": COL_TO_MODULE.get(col, "Sin clasificar"),
            "tipo": clasificar_tipo(df[col]),
            "n_no_nulo": int(df[col].notna().sum()),
            "cobertura_pct": 100 * df[col].notna().mean(),
        })
    inv = pd.DataFrame(filas)
    inv_elca = inv[inv["modulo"] != "Geoespacial"]
    inv_elca.to_csv(TABLES_DIR / "01_inventario_variables.csv", index=False)
    print(f"Guardado 01_inventario_variables.csv ({len(inv_elca)} variables ELCA, n_total={n_total:,})")
    return inv


def figura_inventario_por_modulo(inv: pd.DataFrame) -> None:
    inv = inv[inv["modulo"] != "Geoespacial"]
    tabla = inv.groupby(["modulo", "tipo"]).size().unstack(fill_value=0)
    orden = [m for m, _ in MODULE_RANGES if m in tabla.index]
    tabla = tabla.loc[orden]
    tipos = [t for t in ["Numerica", "Categorica", "Booleana"] if t in tabla.columns]

    fig, ax = plt.subplots(figsize=(ef.ancho(0.8), 3.2))
    bottom = np.zeros(len(tabla))
    colores = {"Numerica": ef.PALETA["azul"], "Categorica": ef.PALETA["naranja"], "Booleana": ef.PALETA["aguamarina"]}
    nombres_tipo = {"Numerica": "Numérica", "Categorica": "Categórica", "Booleana": "Booleana"}
    for tipo in tipos:
        vals = tabla[tipo].values
        ax.bar(tabla.index, vals, bottom=bottom, label=nombres_tipo[tipo], color=colores[tipo])
        bottom += vals
    for i, total in enumerate(tabla.sum(axis=1).values):
        ax.annotate(f"{int(total)}", xy=(i, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=ef.TAM_LETRA_PEQUENA)
    ax.set_ylim(0, tabla.sum(axis=1).max() * 1.12)
    ax.set_ylabel("Número de variables")
    ax.legend(title="Tipo")
    ax.set_xticks(range(len(tabla)))
    ax.set_xticklabels([NOMBRES_MODULO.get(m, m) for m in tabla.index], rotation=30, ha="right")
    savefig(fig, "01_inventario_por_modulo.png")


# ── 2 y 3. Comportamiento por ola y cambios en el periodo ──────────────


def tabla_resumen_por_ola(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for var in KEY_VARS_ALL:
        for ola in (1, 2, 3):
            if var in KEY_VARS_2010_ONLY and ola != 1:
                continue
            if var.startswith("dmsp_") and ola == 3:
                continue
            serie = df.loc[df["ola"] == ola, var].dropna()
            if serie.empty:
                continue
            filas.append({
                "variable": var, "ola": WAVE_LABELS[ola], "n": len(serie),
                "media": serie.mean(), "sd": serie.std(),
                "min": serie.min(), "p25": serie.quantile(0.25),
                "mediana": serie.median(), "p75": serie.quantile(0.75),
                "max": serie.max(),
            })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLES_DIR / "02_resumen_estadistico_por_ola.csv", index=False)
    print(f"Guardado 02_resumen_estadistico_por_ola.csv ({len(tabla)} filas)")
    return tabla


def tabla_cambios_entre_olas(resumen: pd.DataFrame) -> pd.DataFrame:
    piv = resumen.pivot(index="variable", columns="ola", values="media")
    piv = piv.reindex(columns=[w for w in ("2010", "2013", "2016") if w in piv.columns])
    if "2010" in piv.columns and "2013" in piv.columns:
        piv["var_pct_2010_2013"] = 100 * (piv["2013"] - piv["2010"]) / piv["2010"].abs()
    if "2013" in piv.columns and "2016" in piv.columns:
        piv["var_pct_2013_2016"] = 100 * (piv["2016"] - piv["2013"]) / piv["2013"].abs()
    piv = piv.reset_index()
    piv.to_csv(TABLES_DIR / "03_cambios_entre_olas.csv", index=False)
    print(f"Guardado 03_cambios_entre_olas.csv ({len(piv)} variables)")
    return piv


def clasificar_forma(serie_global: pd.Series) -> str:
    """Decide como graficar una variable clave segun la forma de su
    distribucion global (pooled todas las olas): un boxplot es ilegible
    (caja plana o binaria sin informacion visual) cuando la variable es
    categorica de pocas categorias o una tasa concentrada en el techo/piso
    (rango intercuartilico global = 0)."""
    n_unicos = serie_global.nunique()
    if n_unicos <= 5:
        return "categorica"
    q25, q75 = serie_global.quantile(0.25), serie_global.quantile(0.75)
    if (q75 - q25) == 0 and serie_global.min() >= 0 and serie_global.max() <= 1:
        return "tasa_concentrada"
    return "continua"


def _plot_categorica(ax, df: pd.DataFrame, var: str, olas_validas: list) -> None:
    categorias = sorted(df[var].dropna().unique())
    etiquetas = [WAVE_LABELS[o] for o in olas_validas]
    bottom = np.zeros(len(olas_validas))
    cmap = plt.get_cmap("Blues")
    colores = [cmap(0.35 + 0.5 * i / max(len(categorias) - 1, 1)) for i in range(len(categorias))]
    for cat, color in zip(categorias, colores):
        props = [
            (df.loc[df["ola"] == o, var] == cat).sum() / df.loc[df["ola"] == o, var].notna().sum()
            for o in olas_validas
        ]
        ax.bar(etiquetas, props, bottom=bottom, color=color, label=str(cat))
        bottom += np.array(props)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, title="Categoria", title_fontsize=6, loc="lower right", frameon=False)


def _plot_tasa_concentrada(ax, df: pd.DataFrame, var: str, olas_validas: list) -> None:
    etiquetas = [WAVE_LABELS[o] for o in olas_validas]
    medias = [df.loc[df["ola"] == o, var].mean() for o in olas_validas]
    bars = ax.bar(etiquetas, medias, color="#4C72B0", alpha=0.8)
    for rect, m in zip(bars, medias):
        ax.annotate(f"{m:.3f}", xy=(rect.get_x() + rect.get_width() / 2, m),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)


def figura_distribuciones_por_ola(df: pd.DataFrame) -> None:
    variables = KEY_VARS_BY_WAVE
    n_cols = 3
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    axes = axes.flatten()
    formas = {}
    for ax, var in zip(axes, variables):
        olas_validas = [o for o in (1, 2, 3) if not (var.startswith("dmsp_") and o == 3)]
        forma = clasificar_forma(df[var].dropna())
        formas[var] = forma
        if forma == "categorica":
            _plot_categorica(ax, df, var, olas_validas)
        elif forma == "tasa_concentrada":
            _plot_tasa_concentrada(ax, df, var, olas_validas)
        else:
            datos = [df.loc[df["ola"] == o, var].dropna().values for o in olas_validas]
            etiquetas = [WAVE_LABELS[o] for o in olas_validas]
            ax.boxplot(datos, tick_labels=etiquetas, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "#4C72B0", "alpha": 0.6})
        ax.set_title(var, fontsize=9)
    for ax in axes[len(variables):]:
        ax.axis("off")
    fig.suptitle(
        "Distribucion de indicadores clave por ola\n"
        "(boxplot sin atipicos para continuas; barras apiladas por categoria para "
        "discretas de pocas categorias; media anotada para tasas concentradas en el techo)",
        y=1.04, fontsize=10,
    )
    savefig(fig, "02_distribuciones_clave_por_ola.png")
    print("Forma usada por variable en 02_distribuciones_clave_por_ola.png:", formas)


def figura_tendencias_indicadores(resumen: pd.DataFrame) -> None:
    piv = resumen.pivot(index="variable", columns="ola", values="media")
    piv = piv.reindex(columns=[w for w in ("2010", "2013", "2016") if w in piv.columns])
    piv = piv.loc[[v for v in KEY_VARS_BY_WAVE if v in piv.index]]
    indexado = piv.div(piv.iloc[:, 0], axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    for var in indexado.index:
        fila = indexado.loc[var].dropna()
        ax.plot(fila.index, fila.values, marker="o", label=var)
    ax.axhline(100, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Media indexada (ola 2010 = 100)")
    ax.set_title("Evolucion de indicadores clave a traves de las olas")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    savefig(fig, "03_tendencias_indicadores_clave.png")


# ── 4. Correlacion ──────────────────────────────────────────────────────


MIN_N_CORR = 500  # evita pares "perfectos" espurios por pocas observaciones conjuntas no-nulas


def tabla_y_figura_correlacion(df: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    numericas = inv.loc[inv["tipo"] == "Numerica", "variable"].tolist()
    numericas = [c for c in numericas if c not in ID_COLS]
    corr = df[numericas].corr(numeric_only=True)
    n_conjunto = df[numericas].notna().astype(int).T.dot(df[numericas].notna().astype(int))

    pares = []
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = corr.loc[c1, c2]
            n_ij = n_conjunto.loc[c1, c2]
            if pd.notna(r) and abs(r) > 0.7 and n_ij >= MIN_N_CORR:
                pares.append({
                    "variable_1": c1, "modulo_1": COL_TO_MODULE.get(c1, "Sin clasificar"),
                    "variable_2": c2, "modulo_2": COL_TO_MODULE.get(c2, "Sin clasificar"),
                    "r": r, "n_conjunto": int(n_ij),
                })
    tabla_pares = pd.DataFrame(pares).sort_values("r", key=lambda s: s.abs(), ascending=False)
    tabla_pares.to_csv(TABLES_DIR / "04_correlaciones_altas.csv", index=False)
    print(f"Guardado 04_correlaciones_altas.csv ({len(tabla_pares)} pares con |r|>0.7)")

    orden = sorted(cols, key=lambda c: (list(dict(MODULE_RANGES)).index(COL_TO_MODULE.get(c, "Sin clasificar")) if COL_TO_MODULE.get(c, "Sin clasificar") in dict(MODULE_RANGES) else 99, c))
    corr_ordenada = corr.loc[orden, orden]
    n_ordenada = n_conjunto.loc[orden, orden]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr_ordenada.values, cmap="RdBu_r", vmin=-1, vmax=1)
    mask_alta = (corr_ordenada.abs() > 0.7) & (n_ordenada >= MIN_N_CORR)
    np.fill_diagonal(mask_alta.values, False)
    ys, xs = np.where(mask_alta.values)
    ax.scatter(xs, ys, marker="o", facecolors="none", edgecolors="black", s=12, linewidths=0.6)
    ax.set_xticks(range(len(orden)))
    ax.set_yticks(range(len(orden)))
    ax.set_xticklabels(orden, fontsize=5, rotation=90)
    ax.set_yticklabels(orden, fontsize=5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Correlacion de Pearson")
    ax.set_title(f"Matriz de correlacion de variables numericas ({len(orden)} vars)\ncirculos: |r| > 0.7")
    savefig(fig, "04_matriz_correlacion.png")

    figura_top_correlaciones(tabla_pares)
    return tabla_pares


def figura_top_correlaciones(tabla_pares: pd.DataFrame) -> None:
    """Barras horizontales con los pares de variables mas correlacionados
    (|r|>0.7, mismo umbral y tabla que la matriz completa). A diferencia
    de la matriz 159x159 -- ilegible por el numero de etiquetas -- aqui
    cada barra es un par identificable, ordenado por |r|, coloreado por
    signo y anotado con el modulo tematico de origen."""
    pares = tabla_pares.sort_values("r", key=lambda s: s.abs(), ascending=True).reset_index(drop=True)
    etiquetas = [f"{etiqueta_legible(r.variable_1)} – {etiqueta_legible(r.variable_2)}" for r in pares.itertuples()]
    colores = [ef.PALETA["rojo"] if r < 0 else ef.COLOR_SERIE for r in pares["r"]]

    fig, ax = plt.subplots(figsize=(ef.ancho(0.95), 0.16 * len(pares) + 0.6))
    ax.barh(range(len(pares)), pares["r"], color=colores, height=0.7)
    ax.set_yticks(range(len(pares)))
    ax.set_yticklabels(etiquetas, fontsize=ef.TAM_LETRA_PEQUENA - 1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=ef.INK_SECUNDARIO, linewidth=0.8)
    ax.set_xlim(-1, 1.15)
    ax.set_xlabel("Correlación de Pearson ($r$)")
    for i, r in enumerate(pares["r"]):
        # Las negativas se rotulan a la derecha del cero para no chocar con
        # la etiqueta del par.
        x_texto = r if r >= 0 else 0
        ax.annotate(f"{r:.2f}".replace("-", "\u2212"), xy=(x_texto, i), xytext=(2, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=ef.TAM_LETRA_PEQUENA - 1, color=ef.INK_SECUNDARIO)
    savefig(fig, "04b_top_correlaciones.png")
    print(f"Guardado 04b_top_correlaciones.png ({len(pares)} pares)")


# ── 5. Perfil de hogares ────────────────────────────────────────────────


def tabla_y_figura_perfil(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grupo_pobreza"] = df["pobre_ingreso"].map({True: "Pobre", False: "No pobre"})

    filas = []
    for var in KEY_VARS_ALL:
        for grupo_col, grupo_val in [("zona", "Urbano"), ("zona", "Rural"),
                                      ("grupo_pobreza", "Pobre"), ("grupo_pobreza", "No pobre")]:
            serie = df.loc[df[grupo_col] == grupo_val, var].dropna()
            if serie.empty:
                continue
            filas.append({
                "variable": var, "criterio": grupo_col, "grupo": grupo_val,
                "n": len(serie), "media": serie.mean(),
            })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLES_DIR / "05_perfil_hogares.csv", index=False)
    print(f"Guardado 05_perfil_hogares.csv ({len(tabla)} filas)")

    variables_fig = [v for v in KEY_VARS_ALL if v not in ("dmsp_stable_lights_acum_tendencia",)]
    n_cols = 3
    n_rows = int(np.ceil(len(variables_fig) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.0 * n_rows))
    axes = axes.flatten()
    for ax, var in zip(axes, variables_fig):
        zona_u = tabla.query("variable == @var and criterio == 'zona' and grupo == 'Urbano'")["media"]
        zona_r = tabla.query("variable == @var and criterio == 'zona' and grupo == 'Rural'")["media"]
        pobre = tabla.query("variable == @var and criterio == 'grupo_pobreza' and grupo == 'Pobre'")["media"]
        nopobre = tabla.query("variable == @var and criterio == 'grupo_pobreza' and grupo == 'No pobre'")["media"]
        vals = [zona_u.iloc[0] if len(zona_u) else np.nan, zona_r.iloc[0] if len(zona_r) else np.nan,
                pobre.iloc[0] if len(pobre) else np.nan, nopobre.iloc[0] if len(nopobre) else np.nan]
        labels = ["Urbano", "Rural", "Pobre", "No pobre"]
        colores = [COLOR_URBANO, COLOR_RURAL, COLOR_POBRE, COLOR_NOPOBRE]
        ax.bar(labels, vals, color=colores)
        ax.set_title(var, fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    for ax in axes[len(variables_fig):]:
        ax.axis("off")
    fig.suptitle("Perfil de hogares: indicadores clave por zona y condicion de pobreza", y=1.01)
    savefig(fig, "05_perfil_hogares_zona_pobreza.png")
    return tabla


# ── 6. Distribuciones geoespaciales 2010 ────────────────────────────────


def figura_distribuciones_geo_2010(df: pd.DataFrame) -> None:
    df_2010 = df[df["ola"] == 1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, var in zip(axes, GEO_INSIGNIA_2010):
        serie = df_2010[var].dropna()
        ax.hist(serie, bins=30, color=COLOR_SINGLE, alpha=0.8)
        cobertura = 100 * df_2010[var].notna().mean()
        ax.set_title(f"{var}\n(n={len(serie):,}, cobertura {cobertura:.1f}%)", fontsize=9)
    fig.suptitle("Distribucion de las variables insignia geoespaciales, ola 2010", y=1.05)
    savefig(fig, "06_distribuciones_geoespaciales_2010.png")


def main() -> None:
    if "--solo-figura-inventario" in sys.argv:
        figura_inventario_por_modulo(pd.read_csv(TABLES_DIR / "01_inventario_variables.csv"))
        return
    df = cargar_consolidado_con_geo()
    content_cols = [c for c in df.columns if c not in ID_COLS]

    inv = tabla_inventario(df, content_cols)
    figura_inventario_por_modulo(inv)

    resumen = tabla_resumen_por_ola(df)
    tabla_cambios_entre_olas(resumen)
    figura_distribuciones_por_ola(df)
    figura_tendencias_indicadores(resumen)

    tabla_y_figura_correlacion(df, inv)
    tabla_y_figura_perfil(df)
    figura_distribuciones_geo_2010(df)

    print("\nListo. Tablas en", TABLES_DIR, "\nFiguras en", FIGURES_DIR)


if __name__ == "__main__":
    main()
