"""
Caracterizacion de los 4 grupos de transicion de pobreza, pobreza
monetaria e IPM, para la seccion de la tesis sobre diferencias en
covariables entre poblaciones de estudio.

Corre sobre AMBAS transiciones disponibles en el panel ELCA (ver
TRANSICIONES mas abajo): 2010->2013 (ola 1 -> ola 2) y 2013->2016 (ola
2 -> ola 3), para poder comparar si el perfil de "quien entra a pobreza"
se repite en un periodo distinto.

Nomenclatura de archivos: TODOS los outputs (incluidos los de la
transicion 2010->2013 que antes no llevaban sufijo) ahora incluyen el
sufijo de transicion (`_2010_2013` / `_2013_2016`), para mantener
el patron simetrico -- esto obliga a actualizar
las referencias en `mapa_transicion_regiones.py` y
`tabla_covariables_transicion.py`, ya hecho en el mismo commit que este
cambio.

Grupos (Lopez-Calva y Ortiz-Juarez 2014, Tabla 3, ya implementados en
`construir_matriz_transicion` de build_pobreza_desagregaciones.py):
Siempre pobre / Sale de la pobreza / Entra en pobreza / Nunca pobre.

DMSP-OLS por transicion: la variable geoespacial insignia se toma SIEMPRE
de la ola BASE de cada transicion (2010 para 2010->2013, 2013 para
2013->2016) -- DMSP-OLS fue descontinuado despues de 2013 (reemplazado
por VIIRS), asi que 2016 no tiene cobertura, pero eso no afecta a ninguna
de las dos transiciones porque ninguna usa 2016 como ola BASE (ver
`cargar_dmsp_por_consecutivo`, coberturas verificadas: 100% en 2010 y en
2013, 0% en 2016).

Ponderacion: `peso_longitudinal` (fexhog_2010) para AMBAS definiciones de
pobreza, calculado aqui para IPM reusando `cargar_pesos_muestrales`
(extraida de build_pobreza_desagregaciones.py, misma logica que ya usa
pobreza monetaria)
IPM sean comparables en la seccion de comparacion. Los MODELOS de ML no
usan estos pesos (confirmado: no hay peso/fexhog/sample_weight en
src/05_model/); son exclusivos de este analisis descriptivo.

Comparacion entre periodos (funcion `comparar_periodos`, nueva): para
cada definicion de pobreza, junta el tamano de efecto de cada variable
seleccionada en al menos una de las dos transiciones, marcando si fue
seleccionada en ambas -- output autonomo, no requiere recalculo manual
fuera de este script.

INPUTS
    data/processed/pobreza_monetaria_elca_longitudinal.parquet
    data/processed/ipm_multidimensional_elca_longitudinal.parquet
    data/processed/hogar_elca_longitudinal_clean.parquet (region, pesos)
    data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet (DMSP)
    data/processed/benchmark_consolidado_elca_longitudinal.parquet (covariables)

OUTPUTS (por transicion x definicion, `outputs/tables/eda_transicion_covariables/`)
    region_x_categoria_{monetaria,ipm}_{sufijo}.csv
    dmsp_por_categoria_{monetaria,ipm}_{sufijo}.csv
    dmsp_por_region_{monetaria,ipm}_{sufijo}.csv
    ranking_covariables_{monetaria,ipm}_{sufijo}.csv
    seleccion_final_{monetaria,ipm}_{sufijo}.csv
    tabla_comparativa_{monetaria,ipm}_{sufijo}.csv
    comparacion_periodos_{monetaria,ipm}.csv (comparacion entre las 2 transiciones)

COMO CORRER
    cd src/02_build && python eda_transicion_covariables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import (  # noqa: E402
    HOGAR_PATH,
    _llave_compuesta,
    cargar_pesos_muestrales,
    construir_matriz_transicion,
)

# Rutas y funciones de carga movidas a panel_transicion.py (2026-09-30).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_transicion import (  # noqa: E402
    PROJECT_ROOT,
    POBREZA_PATH,
    IPM_PATH,
    GEO_PATH,
    CONSOLIDADO_PATH,
    INVENTARIO_PATH,
    TABLES_DIR,
    CATEGORIAS_ORDEN,
    COLS_ID,
    cargar_peso_longitudinal_por_consecutivo,
    _excluir_hogares_divididos,
    cargar_region_por_consecutivo,
    cargar_dmsp_por_consecutivo,
    cargar_tipos_variables,
    cargar_covariables_ola_base,
)

# Las 2 transiciones disponibles en el panel ELCA de 3 olas. `ola_ini`/
# `ola_fin` van a `construir_matriz_transicion` (ya parametrizada por
# ola); `anio_dmsp` es el anio de la fuente geoespacial que corresponde a
# la ola BASE de cada transicion (ver docstring del modulo).
TRANSICIONES = [
    {"ola_ini": 1, "ola_fin": 2, "anio_dmsp": 2010, "sufijo": "2010_2013"},
    {"ola_ini": 2, "ola_fin": 3, "anio_dmsp": 2013, "sufijo": "2013_2016"},
]

# Variables que entran SIEMPRE a la seleccion final, sin importar su
# posicion en el ranking estadistico:
#   - dmsp_stable_lights: variable insignia de la tesis.
#   - total_choques_hogar: NO pasa el umbral de robustez en 3 de las 4
#     combinaciones transicion x definicion (ver ranking_covariables_*.csv),
#     pero muestra una brecha "Entra en pobreza" > "Nunca pobre" CONSISTENTE
#     en las 4 (choques_medias_grupo_*.csv, +0.14/+0.05/+0.26/+0.30) -- la
#     hipotesis de trabajo (2026-09-10) es que el eta^2 sobre rangos no esta
#     bien calibrado para esta variable de conteo sesgada/con muchos ceros,
#     no que la señal no exista.
#   - afrontamiento_erosivo_hogar / afrontamiento_protector_hogar: como
#     "formas de enfrentar el choque" (vender activos/endeudarse vs.
#     ahorros/redes de apoyo), que se incluyen
#     pese a que su direccion es INCONSISTENTE entre transiciones
#     (choques_dirigido_*.csv) -- a diferencia de total_choques_hogar, esto
#     se documenta como hallazgo exploratorio/ruidoso, no como patron
#     confirmado (celdas de "Entra en pobreza" chicas, n=254-343, porque
#     solo aplican a hogares que tuvieron algun choque).
VARIABLES_OBLIGATORIAS = [
    "dmsp_stable_lights", "total_choques_hogar",
    "afrontamiento_erosivo_hogar", "afrontamiento_protector_hogar",
]

# Columnas que son (casi) el propio label de pobreza monetaria en la ola
# base (ver build_benchmark_train_test.py:COLS_EXCLUIR_SIEMPRE): excluidas
# SOLO del ranking de MONETARIA por circularidad, no del de IPM (ahi son
# covariables legitimas, no el label -- el label de IPM, `pobre_ipm`/
# `ipm_score`/`priv_*`, ni siquiera vive en el consolidado que usamos como
# universo de covariables).
COLS_LABEL_MONETARIA = {
    "pobre_ingreso", "pobre_extremo_ingreso", "pobre_gasto", "pobre_extremo_gasto",
    "concuerdan_ingreso_gasto", "lp", "li",
}

UMBRAL_ETA2 = 0.01  # "efecto pequeno" (Cohen) para variables Numericas (eta2 sobre rangos)
UMBRAL_V = 0.10  # analogo para Categorica/Booleana (V de Cramer sesgo-corregida)
UMBRAL_CI_ETA2 = 0.005  # piso de "no ruido" para el limite inferior del bootstrap
UMBRAL_CI_V = 0.05
N_BOOT = 300
UMBRAL_REDUNDANCIA = 0.70  # |rho| de Spearman para colapsar variables casi-colineales
N_VARIABLES_OBJETIVO = 20  # ademas de DMSP -> 21 en tabla final (subido de 11,
# para que quepan variables robustas del modulo "Ninos" que el cupo
# anterior dejaba fuera solo por prioridad de otros modulos con efecto aun mayor, no por
# falta de senal -- ver ranking_covariables_*.csv, columna 'robusto'. Subido de 14 a 20
# tras verificar que ampliar el cupo puede desplazar variables ya
# seleccionadas -- no es una operacion puramente aditiva, ver interaccion con MAX_POR_MODULO
# en `seleccionar_variables_finales`)
MAX_POR_MODULO = 3  # diversidad tematica


def perfilar_transicion(panel_categorias: pd.DataFrame, region: pd.Series, dmsp: pd.Series) -> dict:
    """A partir de `panel_categorias` (consecutivo, categoria, peso_longitudinal),
    construye la agregacion por region y el perfil de DMSP por grupo."""
    df = panel_categorias.copy()
    df["region"] = df["consecutivo"].map(region)
    df["dmsp_stable_lights"] = df["consecutivo"].map(dmsp)
    df["categoria"] = pd.Categorical(df["categoria"], categories=CATEGORIAS_ORDEN, ordered=True)

    n_sin_peso = df["peso_longitudinal"].isna().sum()
    n_sin_region = df["region"].isna().sum()
    n_sin_dmsp = df["dmsp_stable_lights"].isna().sum()

    # Distribucion de categoria dentro de cada region (% de fila, ponderado
    # por peso_longitudinal) -- para identificar patrones espaciales.
    pivote_peso = df.pivot_table(
        index="region", columns="categoria", values="peso_longitudinal", aggfunc="sum",
        fill_value=0, observed=False,
    )
    region_x_categoria_pct = (pivote_peso.div(pivote_peso.sum(axis=1), axis=0) * 100).round(1)
    region_x_categoria_n = pd.crosstab(df["region"], df["categoria"])

    # DMSP promedio (ponderado) y mediana (sin ponderar, robusta) por grupo,
    # y por region x grupo para ver si el patron espacial se sostiene.
    def _media_ponderada(g):
        peso = g["peso_longitudinal"]
        valor = g["dmsp_stable_lights"]
        mask = valor.notna() & peso.notna()
        if not mask.any():
            return float("nan")
        return (valor[mask] * peso[mask]).sum() / peso[mask].sum()

    dmsp_por_categoria = df.groupby("categoria", observed=True).apply(
        lambda g: pd.Series(
            {
                "dmsp_media_ponderada": _media_ponderada(g),
                "dmsp_mediana": g["dmsp_stable_lights"].median(),
                "n_hogares": len(g),
            }
        ),
        include_groups=False,
    )
    dmsp_region_x_categoria = df.pivot_table(
        index="region", columns="categoria", values="dmsp_stable_lights", aggfunc="median",
        observed=False,
    ).round(1)

    # Agregado por region (SIN desagregar por categoria) -- insumo del mapa
    # esquematico: 1 valor de DMSP por region, para posicionar cada "burbuja"
    # de region independientemente del color (grupo dominante).
    dmsp_por_region = df.groupby("region", observed=False).apply(
        lambda g: pd.Series({
            "dmsp_media_ponderada": _media_ponderada(g),
            "dmsp_mediana": g["dmsp_stable_lights"].median(),
            "n_hogares": len(g),
        }),
        include_groups=False,
    )

    return {
        "region_x_categoria_pct": region_x_categoria_pct,
        "region_x_categoria_n": region_x_categoria_n,
        "dmsp_por_categoria": dmsp_por_categoria,
        "dmsp_region_x_categoria": dmsp_region_x_categoria,
        "dmsp_por_region": dmsp_por_region,
        "n_sin_peso": n_sin_peso,
        "n_sin_region": n_sin_region,
        "n_sin_dmsp": n_sin_dmsp,
        "n_total": len(df),
    }


def _kish_ess(pesos: np.ndarray) -> float:
    """Tamano de muestra efectivo de Kish: penaliza la varianza de los
    pesos (un peso muy desigual reduce la precision real por debajo del N
    crudo) -- se usa en vez de N para el sesgo de V de Cramer y como base
    del remuestreo bootstrap."""
    return float(pesos.sum() ** 2 / np.sum(pesos ** 2))


def weighted_rank_eta2(valores: pd.Series, grupo: pd.Series, peso: pd.Series) -> tuple:
    """eta^2 sobre rangos (ANOVA de rangos ponderado por `peso`): analogo
    ponderado y robusto a colas pesadas/outliers del eta^2 de Kruskal-Wallis.
    Acotado en [0,1], no crece con N (solo con la separacion real entre
    grupos)."""
    df = pd.DataFrame({"v": valores, "g": grupo, "w": peso}).dropna()
    if df["g"].nunique() < 2 or len(df) < 10:
        return float("nan"), 0.0
    df["rango"] = df["v"].rank(method="average")
    n_eff = _kish_ess(df["w"].to_numpy())
    media_global = np.average(df["rango"], weights=df["w"])
    ss_total = np.sum(df["w"] * (df["rango"] - media_global) ** 2)
    if ss_total == 0:
        return 0.0, n_eff
    ss_between = 0.0
    for _, sub in df.groupby("g", observed=True):
        media_g = np.average(sub["rango"], weights=sub["w"])
        ss_between += sub["w"].sum() * (media_g - media_global) ** 2
    return float(ss_between / ss_total), n_eff


def weighted_cramers_v(valores: pd.Series, grupo: pd.Series, peso: pd.Series) -> tuple:
    """V de Cramer con correccion de sesgo (Bergsma 2013), usando el N
    efectivo de Kish en vez del N crudo -- evita que el ponderado infle
    artificialmente la "certeza" del estadistico."""
    df = pd.DataFrame({"v": valores, "g": grupo, "w": peso}).dropna()
    if df["v"].nunique() < 2 or df["g"].nunique() < 2 or len(df) < 10:
        return float("nan"), 0.0
    tabla = df.pivot_table(index="v", columns="g", values="w", aggfunc="sum", fill_value=0.0, observed=False)
    n_eff = _kish_ess(df["w"].to_numpy())
    total = tabla.to_numpy().sum()
    fila = tabla.sum(axis=1).to_numpy()
    col = tabla.sum(axis=0).to_numpy()
    esperado = np.outer(fila, col) / total
    chi2 = np.nansum(np.where(esperado > 0, (tabla.to_numpy() - esperado) ** 2 / esperado, 0.0))
    r, c = tabla.shape
    if n_eff <= 1:
        return 0.0, n_eff
    phi2_corr = max(0.0, chi2 / n_eff - (r - 1) * (c - 1) / (n_eff - 1))
    r_corr = r - (r - 1) ** 2 / (n_eff - 1)
    c_corr = c - (c - 1) ** 2 / (n_eff - 1)
    denom = min(r_corr - 1, c_corr - 1)
    if denom <= 0:
        return 0.0, n_eff
    return float(np.sqrt(phi2_corr / denom)), n_eff


def bootstrap_ci(valores: pd.Series, grupo: pd.Series, peso: pd.Series, tipo: str, seed: int = 0) -> tuple:
    """Bootstrap ponderado: remuestrea hogares con probabilidad proporcional
    al peso (asi el remuestreo ya representa a la poblacion ponderada) y
    recalcula el efecto SIN ponderar dentro de cada replica -- IC percentil
    95%. Solo se llama sobre el subconjunto que ya paso el umbral de efecto,
    para mantener el costo computacional acotado."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": valores, "g": grupo, "w": peso}).dropna()
    n = len(df)
    probs = (df["w"] / df["w"].sum()).to_numpy()
    unos = pd.Series(np.ones(n))
    replicas = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True, p=probs)
        muestra = df.iloc[idx].reset_index(drop=True)
        if tipo == "Numerica":
            efecto, _ = weighted_rank_eta2(muestra["v"], muestra["g"], unos)
        else:
            efecto, _ = weighted_cramers_v(muestra["v"], muestra["g"], unos)
        replicas.append(efecto)
    return float(np.nanpercentile(replicas, 2.5)), float(np.nanpercentile(replicas, 97.5))


def rankear_covariables(covariables: pd.DataFrame, tipos: dict, panel: pd.DataFrame, excluir: set) -> pd.DataFrame:
    """Ranking de covariables por tamano de efecto (eta2 de rangos para
    Numerica, V de Cramer sesgo-corregida para Categorica/Booleana), ambos
    ponderados por `peso_longitudinal` y acotados en [0,1] -- comparables
    entre si pese a la diferencia de escala/tipo. Devuelve 1 fila por
    variable candidata que paso el filtro minimo de datos."""
    df = panel.merge(covariables, left_on="consecutivo", right_index=True, how="left")
    grupo, peso = df["categoria"], df["peso_longitudinal"]
    filas = []
    for var in covariables.columns:
        if var in excluir:
            continue
        tipo = tipos.get(var, "Numerica")
        valores = df[var]
        if tipo == "Numerica":
            if not pd.api.types.is_numeric_dtype(valores):
                continue
            efecto, n_eff = weighted_rank_eta2(valores, grupo, peso)
        else:
            efecto, n_eff = weighted_cramers_v(valores.astype(object), grupo, peso)
        if pd.isna(efecto):
            continue
        filas.append({"variable": var, "tipo": tipo, "efecto": efecto, "n_efectivo": n_eff})

    ranking = pd.DataFrame(filas).sort_values("efecto", ascending=False).reset_index(drop=True)
    ranking["umbral"] = np.where(ranking["tipo"] == "Numerica", UMBRAL_ETA2, UMBRAL_V)
    ranking["pasa_umbral"] = ranking["efecto"] >= ranking["umbral"]

    ci_low, ci_high = [], []
    for _, fila in ranking.iterrows():
        if not fila["pasa_umbral"]:
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            continue
        lo, hi = bootstrap_ci(df[fila["variable"]], grupo, peso, fila["tipo"])
        ci_low.append(lo)
        ci_high.append(hi)
    ranking["ci_low"], ranking["ci_high"] = ci_low, ci_high
    umbral_ci = np.where(ranking["tipo"] == "Numerica", UMBRAL_CI_ETA2, UMBRAL_CI_V)
    ranking["robusto"] = ranking["pasa_umbral"] & (ranking["ci_low"] >= umbral_ci)
    return ranking


def deduplicar_por_correlacion(ranking_robusto: pd.DataFrame, covariables: pd.DataFrame) -> pd.DataFrame:
    """Colapsa clusters de correlacion de Spearman (|rho| > UMBRAL_REDUNDANCIA)
    entre variables Numericas Y Booleanas (estas ultimas recodificadas 0/1 --
    Spearman sobre binarias es el coeficiente phi, perfectamente valido, y es
    necesario: `pobre_ingreso`/`pobre_extremo_ingreso`/`pobre_gasto`/
    `pobre_extremo_gasto` son booleanas casi-redundantes entre si) y se queda
    con la de mayor efecto por cluster -- evita mostrar varias variantes de
    la misma señal. Categorica multinivel pasa sin cambios (redundancia
    entre categoricas de >2 niveles queda para revision manual, limitacion
    conocida)."""
    candidatas = ranking_robusto[ranking_robusto["tipo"].isin(["Numerica", "Booleana"])]["variable"].tolist()
    categoricas = ranking_robusto[~ranking_robusto["tipo"].isin(["Numerica", "Booleana"])]
    if len(candidatas) < 2:
        return ranking_robusto

    valores = covariables[candidatas].apply(
        lambda s: s.astype(float) if s.dtype != object else pd.to_numeric(s, errors="coerce")
    )
    corr = valores.corr(method="spearman").fillna(0.0)
    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)
    condensada = squareform(dist.values, checks=False)
    clusters = fcluster(linkage(condensada, method="average"), t=1 - UMBRAL_REDUNDANCIA, criterion="distance")

    ranking_cand = ranking_robusto[ranking_robusto["tipo"].isin(["Numerica", "Booleana"])].copy()
    ranking_cand["cluster"] = clusters
    representantes = ranking_cand.sort_values("efecto", ascending=False).drop_duplicates(subset="cluster")
    return pd.concat([representantes.drop(columns="cluster"), categoricas]).sort_values(
        "efecto", ascending=False
    )


def seleccionar_variables_finales(ranking: pd.DataFrame, covariables: pd.DataFrame, tipos_modulo: dict) -> list:
    """De las variables robustas y no-redundantes, arma la seleccion final:
    fuerza VARIABLES_OBLIGATORIAS (ver esa constante), y completa hasta
    N_VARIABLES_OBJETIVO priorizando efecto pero evitando repetir un mismo
    `modulo` mas de MAX_POR_MODULO veces."""
    robustas = ranking[ranking["robusto"]].copy()
    sin_redundancia = deduplicar_por_correlacion(robustas, covariables)

    seleccion = list(VARIABLES_OBLIGATORIAS)
    conteo_modulo = {}
    for _, fila in sin_redundancia.sort_values("efecto", ascending=False).iterrows():
        var = fila["variable"]
        if var in VARIABLES_OBLIGATORIAS or var in seleccion:
            continue
        modulo = tipos_modulo.get(var, "Otro")
        if conteo_modulo.get(modulo, 0) >= MAX_POR_MODULO and len(seleccion) < N_VARIABLES_OBJETIVO:
            continue  # da preferencia a otros modulos mientras haya cupo
        seleccion.append(var)
        conteo_modulo[modulo] = conteo_modulo.get(modulo, 0) + 1
        if len(seleccion) >= N_VARIABLES_OBJETIVO + len(VARIABLES_OBLIGATORIAS):
            break
    return seleccion


def construir_tabla_comparativa(
    covariables: pd.DataFrame, tipos: dict, panel: pd.DataFrame, seleccion: list, ranking: pd.DataFrame
) -> pd.DataFrame:
    """Tabla compacta final: 1 fila por variable seleccionada, con su valor
    ponderado por grupo de transicion.

    Numerica: media ponderada por `peso_longitudinal` (en unidades
    originales -- se decidio no estandarizar la tabla de presentacion, el
    z-score solo se uso internamente para el ranking).

    Categorica/Booleana: en vez de desplegar TODOS los niveles (rompe la
    compacidad), se reporta el nivel que mas separa a los 4 grupos --
    definido automaticamente como el de mayor rango (max-min) de % de fila
    entre grupos -- junto con el nombre de ese nivel, para que quede
    transparente y reproducible cual categoria se esta mostrando.
    """
    df = panel.merge(covariables, left_on="consecutivo", right_index=True, how="left")
    grupo, peso = df["categoria"], df["peso_longitudinal"]
    filas = []
    for var in seleccion:
        tipo = tipos.get(var, "Numerica")
        valores = df[var]
        match = ranking.loc[ranking["variable"] == var, "efecto"]
        efecto = float(match.iloc[0]) if len(match) else float("nan")

        if tipo == "Numerica":
            # Proporciones 0/1 mal clasificadas como Numerica en el
            # inventario (ej. tiene_deuda_hogar, pct_adultos_alfabetizados):
            # se muestran en % (x100) para que queden en la misma escala que
            # las filas Categorica/Booleana de la tabla.
            valores_no_nulos = valores.dropna()
            es_proporcion = len(valores_no_nulos) > 0 and valores_no_nulos.between(0, 1).all()
            escala = 100 if es_proporcion else 1
            etiqueta = "(media, %)" if es_proporcion else "(media)"
            medias = {}
            for cat in CATEGORIAS_ORDEN:
                mask = (grupo == cat) & valores.notna() & peso.notna()
                medias[cat] = np.average(valores[mask], weights=peso[mask]) * escala if mask.sum() else float("nan")
            filas.append({"variable": var, "tipo": tipo, "nivel_mostrado": etiqueta, "efecto": efecto, **medias})
        else:
            tmp = pd.DataFrame({"v": valores.astype(object), "g": grupo, "w": peso}).dropna()
            tabla_pesos = tmp.pivot_table(index="v", columns="g", values="w", aggfunc="sum", fill_value=0, observed=False)
            pct_col = tabla_pesos.div(tabla_pesos.sum(axis=0), axis=1) * 100  # % de fila (grupo) en cada nivel
            if tipo == "Booleana":
                # Solo 2 niveles (True/False), el rango es identico para
                # ambos -- se muestra siempre el lado "True" (la condicion
                # nombrada, ej. "% pobre por gasto"), mas natural de leer
                # que su complemento.
                nivel = True
            else:
                rango = pct_col.max(axis=1) - pct_col.min(axis=1)
                nivel = rango.idxmax()
            fila_nivel = pct_col.loc[nivel]
            filas.append({
                "variable": var, "tipo": tipo, "nivel_mostrado": str(nivel), "efecto": efecto,
                **{cat: fila_nivel.get(cat, float("nan")) for cat in CATEGORIAS_ORDEN},
            })

    tabla = pd.DataFrame(filas).sort_values("efecto", ascending=False).reset_index(drop=True)
    return tabla[["variable", "tipo", "nivel_mostrado", "efecto"] + CATEGORIAS_ORDEN]


def comparar_periodos(
    ranking_a: pd.DataFrame, seleccion_a: list, sufijo_a: str,
    ranking_b: pd.DataFrame, seleccion_b: list, sufijo_b: str,
) -> pd.DataFrame:
    """Compara el tamano de efecto y la seleccion de una misma definicion
    de pobreza entre las 2 transiciones -- responde la pregunta de la
    Parte A: ¿el perfil de "quien entra a pobreza" es estable entre
    2010->2013 y 2013->2016, o es especifico de un periodo?

    Incluye toda variable seleccionada en AL MENOS una de las dos
    transiciones (union de ambas listas de 12), con su efecto en cada
    periodo (NaN si no paso el ranking de ese periodo) y si fue
    seleccionada en cada uno."""
    efecto_a = dict(zip(ranking_a["variable"], ranking_a["efecto"]))
    efecto_b = dict(zip(ranking_b["variable"], ranking_b["efecto"]))
    variables = sorted(set(seleccion_a) | set(seleccion_b))

    filas = []
    for var in variables:
        sel_a = var in seleccion_a
        sel_b = var in seleccion_b
        filas.append({
            "variable": var,
            f"efecto_{sufijo_a}": efecto_a.get(var, float("nan")),
            f"efecto_{sufijo_b}": efecto_b.get(var, float("nan")),
            f"seleccionada_{sufijo_a}": sel_a,
            f"seleccionada_{sufijo_b}": sel_b,
            "en_ambas": sel_a and sel_b,
        })
    tabla = pd.DataFrame(filas)
    orden = tabla[[f"efecto_{sufijo_a}", f"efecto_{sufijo_b}"]].max(axis=1)
    return tabla.assign(_orden=orden).sort_values("_orden", ascending=False).drop(columns="_orden").reset_index(drop=True)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INVENTARIO_PATH)
    tipos_modulo = dict(zip(inv["variable"], inv["modulo"]))
    tipos_modulo["dmsp_stable_lights"] = "Geoespacial"
    tipos_modulo.setdefault("zona", "Vivienda")
    tipos = cargar_tipos_variables()

    # resultados[nombre_pobreza][sufijo_transicion] = {"ranking":..., "seleccion":...}
    resultados = {"monetaria": {}, "ipm": {}}

    for transicion in TRANSICIONES:
        ola_ini, ola_fin = transicion["ola_ini"], transicion["ola_fin"]
        anio_dmsp, sufijo = transicion["anio_dmsp"], transicion["sufijo"]

        region = cargar_region_por_consecutivo(ola_ini)
        dmsp = cargar_dmsp_por_consecutivo(anio_dmsp)
        covariables = cargar_covariables_ola_base(dmsp, ola_ini)

        pobreza = pd.read_parquet(POBREZA_PATH)
        llave_pobreza = _llave_compuesta(pobreza)
        cargar_pesos_muestrales(pobreza, llave_pobreza)
        resultado_monetaria = construir_matriz_transicion(
            pobreza, ola_ini, ola_fin, col_pobre="pobre_ingreso", peso_col="peso_longitudinal"
        )

        ipm = pd.read_parquet(IPM_PATH)
        cargar_peso_longitudinal_por_consecutivo(ipm, ola_fin)
        resultado_ipm = construir_matriz_transicion(
            ipm, ola_ini, ola_fin, col_pobre="pobre_ipm", peso_col="peso_longitudinal"
        )

        for nombre, resultado in [("monetaria", resultado_monetaria), ("ipm", resultado_ipm)]:
            perfil = perfilar_transicion(resultado["panel_categorias"], region, dmsp)
            print(f"\n=== {nombre.upper()} {sufijo} (n panel={perfil['n_total']}) ===")
            print(f"Sin peso_longitudinal: {perfil['n_sin_peso']} | sin region: {perfil['n_sin_region']} | sin DMSP: {perfil['n_sin_dmsp']}")
            print("\nDMSP por categoria (media ponderada / mediana / n):")
            print(perfil["dmsp_por_categoria"].reindex(CATEGORIAS_ORDEN))

            perfil["region_x_categoria_pct"].reindex(columns=CATEGORIAS_ORDEN).to_csv(
                TABLES_DIR / f"region_x_categoria_{nombre}_{sufijo}.csv"
            )
            perfil["dmsp_por_categoria"].reindex(CATEGORIAS_ORDEN).to_csv(
                TABLES_DIR / f"dmsp_por_categoria_{nombre}_{sufijo}.csv"
            )
            perfil["dmsp_por_region"].to_csv(TABLES_DIR / f"dmsp_por_region_{nombre}_{sufijo}.csv")

            excluir = COLS_LABEL_MONETARIA if nombre == "monetaria" else set()
            panel = resultado["panel_categorias"]
            ranking = rankear_covariables(covariables, tipos, panel, excluir)
            seleccion = seleccionar_variables_finales(ranking, covariables, tipos_modulo)

            print(f"\nCandidatas robustas (pasan umbral y CI): {ranking['robusto'].sum()} de {len(ranking)}")
            print(f"Seleccion final ({len(seleccion)} variables, incluye {VARIABLES_OBLIGATORIAS}): {seleccion}")

            ranking.to_csv(TABLES_DIR / f"ranking_covariables_{nombre}_{sufijo}.csv", index=False)
            pd.Series(seleccion, name="variable").to_csv(
                TABLES_DIR / f"seleccion_final_{nombre}_{sufijo}.csv", index=False
            )

            tabla_comparativa = construir_tabla_comparativa(covariables, tipos, panel, seleccion, ranking)
            tabla_comparativa.to_csv(TABLES_DIR / f"tabla_comparativa_{nombre}_{sufijo}.csv", index=False)

            resultados[nombre][sufijo] = {"ranking": ranking, "seleccion": seleccion}

    sufijo_a, sufijo_b = TRANSICIONES[0]["sufijo"], TRANSICIONES[1]["sufijo"]
    for nombre in ("monetaria", "ipm"):
        comparacion = comparar_periodos(
            resultados[nombre][sufijo_a]["ranking"], resultados[nombre][sufijo_a]["seleccion"], sufijo_a,
            resultados[nombre][sufijo_b]["ranking"], resultados[nombre][sufijo_b]["seleccion"], sufijo_b,
        )
        print(f"\n--- Comparacion entre periodos ({nombre}) ---")
        with pd.option_context("display.max_rows", 30, "display.width", 160, "display.precision", 3):
            print(comparacion.to_string(index=False))
        comparacion.to_csv(TABLES_DIR / f"comparacion_periodos_{nombre}.csv", index=False)

    print(f"\nGuardado en: {TABLES_DIR}")


if __name__ == "__main__":
    main()
