"""
diagnostico_embeddings_dmsp.py
=====================================================================
Correlacion entre los embeddings de fotos de Google Street View (GSV) y
las variables de iluminacion nocturna DMSP-OLS, ola 2013 -- y su relacion
con el hallazgo central de la tesis sobre DMSP-OLS ("sirve como proxy que
puede reemplazar informacion de encuesta, mas que aportar poder predictivo
adicional per se").

ORIGEN Y CORRECCION METODOLOGICA IMPORTANTE (leer antes de interpretar
los resultados)
---------------------------------------------------------------------
Una primera version exploratoria de este analisis cruzo TODOS los embeddings etiquetados
`ola=2013` (n=6,237 hogares) contra DMSP-OLS 2013, sin verificar la fecha
real de captura de la foto. Esto es metodologicamente invalido: la etiqueta `ola` en el pipeline de GSV indica
solo "la ola mas cercana a la que se asigno la foto", NO la fecha real en
que fue tomada -- y el propio proyecto ya habia cuantificado este problema
en `src/01_download/02_scr_GoogleStreetView/03f_analisis_calidad_imagenes_gsv.py`
(Seccion F, tabla `tab_gsv_ventana_temporal.tex`): de las 6,788 fotos
etiquetadas `ola=2013`, solo el 13.8% (939) fueron realmente tomadas en la
ventana elegible 2011-2013; el resto tiene fecha de captura posterior
(hasta 2025 en el archivo fuente), porque Google no tiene cobertura
historica de Street View en Colombia para esos anios y el pipeline asigno
el panorama disponible mas cercano geograficamente, no temporalmente.

Este script SI filtra por fecha de captura real (`fecha_pano` en
`descargas_final.csv`) antes de correlacionar -- ver `Analisis 1`. Tambien
cuantifica explicitamente (`Analisis 1b`) cuanto se degrada la correlacion
a medida que se relaja esa ventana, como base empirica para decidir si una
ventana mas laxa es defendible en trabajo futuro con mayor n.

QUE HACE
---------------------------------------------------------------------
Analisis 1 -- Correlacion embeddings <-> DMSP-OLS, SOLO fotos con fecha de
    captura real dentro de la ventana elegible de la ola (VENTANAS_OLA,
    importado de `01_analisis_cobertura_gsv.py`, no duplicado). Restringe
    ademas a hogares con exactamente 1 foto en la ola (evita ambiguedad de
    emparejamiento -- ver "Limitacion de join" mas abajo). Por cada uno de
    los 4 espacios de embedding (CLIP, VGG19, ResNet50, Places365):
    PCA (5 componentes) + RidgeCV con validacion cruzada de 5 folds,
    prediciendo cada variable DMSP-OLS (avg_vis, stable_lights,
    log_stable_lights, avg_vis_acum_media). Reporta r(prediccion, real) y
    R2 out-of-sample.

Analisis 1b -- Curva de degradacion por desfase temporal: para TODOS los
    hogares con 1 sola foto en ola=2013 (sin filtrar por ventana), se
    calcula el desfase = anio_foto - 2013 y se agrupa en bins
    (0, 1-2, 3-5, 6-10, 11+ anios). El PC1 de cada embedding se ajusta UNA
    SOLA VEZ sobre todo el universo (todos los desfases juntos) y luego se
    proyecta cada bin sobre ese mismo eje fijo -- IMPORTANTE: una primera
    version de este analisis ajustaba PCA de forma independiente dentro de
    cada bin, lo que hacia que "PC1" fuera una direccion distinta (con
    signo y rotacion arbitrarios) en cada bin y volvia invalida la
    comparacion entre bins (se detecto al revisar los primeros resultados:
    la correlacion cambiaba de signo de forma erratica entre bins). La
    version corregida (`pc1_proyectado_sobre_eje_global`) resuelve esto.
    Dentro de cada bin se mide la correlacion entre ese PC1 fijo y
    `dmsp_avg_vis`. Si |r| se mantiene alto en bins de desfase moderado
    (ej. 1-2 anios), eso da una base empirica para ampliar la ventana
    valida en trabajo futuro sin comprometer demasiado la validez temporal
    -- ver "Posibles soluciones" abajo.

Analisis 2 -- Sesgo de cobertura: compara la distribucion de
    `dmsp_avg_vis` entre tres grupos de hogares (ola 2013): (a) sin
    ninguna foto GSV, (b) con foto pero fuera de la ventana elegible,
    (c) con foto dentro de la ventana elegible (la muestra usada en el
    Analisis 1) -- Kruskal-Wallis + comparaciones pareadas, para chequear
    si Google fotografio primero sistematicamente las zonas mas
    iluminadas/urbanizadas (lo que inflaria espureamente el Analisis 1).

Analisis 3 -- Aporte marginal de DMSP-OLS controlando por embeddings:
    usa las probabilidades YA calculadas en
    `predicciones_test_dmsp_A.parquet` (proba_base = sin DMSP-OLS,
    proba_geo = con DMSP-OLS, conjunto de prueba 2013->2016) cruzadas con
    PC1 del embedding (temporalmente valido) para el subconjunto de
    hogares con overlap. Reporta: (i) correlacion entre PC1 y el
    desplazamiento marginal (proba_geo - proba_base) -- si el desplazamiento
    que provoca DMSP-OLS es explicado por la misma senal visual que ya
    esta en la foto, deberia correlacionar; (ii) regresion logistica de Y
    sobre dmsp_avg_vis + PC1 conjuntamente, para ver si el coeficiente de
    DMSP sobrevive controlando por la senal visual.

LIMITACION DE JOIN (aplica a Analisis 1, 1b y 3)
---------------------------------------------------------------------
`descargas_final.csv` (que tiene `fecha_pano`) y
`embeddings_unidos_anonimizado.parquet` (que tiene los vectores) NO
comparten una llave de foto explicita mas alla de (consecutivo, ola) --
`photo_id_uuid` se genero DESPUES de anonimizar y esa correspondencia no
esta disponible localmente (ver docstring de `03f_analisis_calidad_imagenes_gsv.py`).
Para hogares con 1 sola foto en la ola esto no es un problema (el join por
consecutivo+ola es inequivoco). Para hogares con 2+ fotos en la misma ola
(1,553 + 74 + 4 + 1 = 1,632 hogares en ola 2013, ver
`reporte_analisis_imagenes_gsv.txt` Seccion E) NO se puede saber cual de
las N filas de embedding corresponde a la fila de `fecha_pano` que cayo
en ventana -- por eso este script los EXCLUYE enteramente, no solo cuando
caen fuera de ventana. Esto reduce el n disponible pero evita contaminar
el analisis con un embedding que podria no ser el fotografiado en la
fecha valida.

CAVEATS QUE SIGUEN APLICANDO AUN DESPUES DEL FILTRO TEMPORAL
---------------------------------------------------------------------
1. n chico (~830 en Analisis 1, decenas a un par de cientos en Analisis
   3): los intervalos de confianza implicitos en cualquier r o R2
   reportado son anchos; no se calculan explicitamente aqui (fuera de
   alcance de un diagnostico exploratorio) pero deben asumirse no
   triviales antes de citar estas cifras como definitivas en el cuerpo de
   la tesis.
2. Sesgo de seleccion geografica (ver Analisis 2): los ~830 hogares con
   foto contemporanea a la ola no son necesariamente un subconjunto
   aleatorio -- Google pudo haber cubierto primero zonas mas
   accesibles/urbanizadas.
3. Los embeddings pre-entrenados (CLIP/VGG19/ResNet50 en ImageNet,
   Places365 en escenas) nunca vieron el problema de pobreza colombiana
   durante su entrenamiento -- la correlacion con DMSP-OLS es evidencia de
   que capturan textura urbana/infraestructura en general, no evidencia
   directa de que predigan pobreza mejor o peor que DMSP-OLS.
4. DMSP-OLS esta saturado en avg_vis=63 para ~25% de las observaciones de
   2013
   -- la correlacion real (sin censura) es probablemente mayor a la aqui
   reportada.
5. El Analisis 3 usa probabilidades de un modelo YA entrenado sin
   embeddings -- es un diagnostico correlacional sobre las predicciones
   del modelo existente, NO un reentrenamiento del pipeline con
   embeddings como covariable (ver "Posibles soluciones", punto 2).

POSIBLES SOLUCIONES / TRABAJO FUTURO (propuesto, NO implementado en este
script -- ver justificacion de por que se deja fuera de alcance)
---------------------------------------------------------------------
1. Ventana temporal mas laxa pero explicita y justificada por la curva de
   degradacion del Analisis 1b (ej. si la correlacion con desfase 1-2
   anios es estadisticamente indistinguible de desfase 0, ampliar la
   ventana a 2011-2015 para la ola 2013 aumentaria el n sustancialmente
   sin perder validez -- pero requiere decidir ese trade-off
   explicitamente, no asumirlo).
2. "Modelo C" -- reentrenar el pipeline de prediccion de pobreza
   (`modelo_utils.py` + `comparar_balanceo_y_tunear`) agregando PCA de los
   embeddings (10-20 componentes) como covariables geoespaciales
   adicionales, analogo a `construir_pipeline_geo_dmsp.py`. NO se hizo
   aqui porque el n temporalmente valido (~830, y menos aun cruzado con el
   panel de transicion de pobreza) es demasiado chico para una busqueda de
   hiperparametros por CV de 10 folds sin un riesgo alto de sobreajuste --
   requeriria primero resolver el punto 1 (mas n) o usar un modelo mucho
   mas regularizado (ridge/lasso puro, sin tunear).
3. Indicadores CLIP zero-shot interpretables (`clip_score_vivienda_*`,
   `clip_score_via_*`, 19 prompts) SI fueron calculados -- ver
   `reporte_embeddings_clip.txt` (Seccion 4, medias/std de los 19
   `clip_score_*`) y el Output 1 de
   `03b_extraer_embeddings_clip.py` (`embeddings/embeddings_clip.parquet`,
   con columnas `clip_score_<prompt>` ademas del embedding). El problema
   NO es que falte calcularlos, sino que `union_parquets.py` los excluyo
   deliberadamente al armar `embeddings_unidos_anonimizado.parquet`
   (selecciona explicitamente `cols_llave + col_emb`, es decir solo
   columnas de embedding, descartando toda columna `clip_score_*`), y el
   parquet original con los scores vive en una ruta de red Windows
   (`\\ECON-E420004947\...\gsv\embeddings\embeddings_clip.parquet`, ver
   `union_parquets.py`) no accesible desde esta maquina/sesion. Para
   contrastar `clip_score_vivienda_*`/`clip_score_via_*` contra los
   indicadores de privacion de vivienda del IPM (material de
   paredes/pisos, hacinamiento -- mas granular y mas interpretable que
   correlacionar contra DMSP-OLS, que mide iluminacion de vecindario, no
   de la vivienda individual) hace falta: (a) recuperar
   `embeddings_clip.parquet` desde esa maquina/ruta de red, o (b)
   re-ejecutar `03b_extraer_embeddings_clip.py` guardando esta vez
   tambien las columnas `clip_score_*` en el paso de union (modificar
   `union_parquets.py` para no descartarlas).
4. Estabilidad temporal de los embeddings (analogo a
   `diagnostico_variabilidad_temporal_dmsp.py`): correlacion test-retest
   2010->2013 de PC1 por hogar. NO se hizo aqui porque la ola 2010 tiene
   0.0% de fotos dentro de su propia ventana elegible (Seccion F del
   reporte GSV) -- no hay ninguna foto de 2010 valida contra la cual
   comparar.

OUTPUTS
---------------------------------------------------------------------
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_correlacion.csv
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_degradacion_desfase.csv
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_sesgo_cobertura.csv
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_aporte_marginal.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_embeddings_dmsp.py
"""

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import modelo_utils as mu

# Reutiliza (importa, no duplica) las ventanas temporales YA definidas y
# ya usadas para cuantificar el problema en 03f_analisis_calidad_imagenes_gsv.py
_GSV_DIR = Path(__file__).resolve().parents[1] / "01_download" / "02_scr_GoogleStreetView"
sys.path.insert(0, str(_GSV_DIR))
_cobertura = import_module("01_analisis_cobertura_gsv")
VENTANAS_OLA = _cobertura.VENTANAS_OLA

REPO_ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = REPO_ROOT / "data" / "processed" / "embeddings"
DMSP_PATH = REPO_ROOT / "data" / "processed" / "SALE_13082026" / "indicadores_derivados_dmsp.parquet"
OUT_DIR = mu.RESULTADOS_DIR

OLA_FOCAL = 2013
EMB_COLS = ["embedding_clip", "embedding_vgg19", "embedding_resnet50", "embedding_places365"]
DMSP_COLS = ["dmsp_avg_vis", "dmsp_stable_lights", "dmsp_log_stable_lights", "dmsp_avg_vis_acum_media"]
RANDOM_STATE = mu.RANDOM_STATE


def cargar_descargas() -> pd.DataFrame:
    """`descargas_final.csv` tiene 7 filas corruptas (comas sin comillar en
    mensaje_error, todas de descargas fallidas) -- on_bad_lines='skip' las
    descarta sin afectar las 20,935 descargas exitosas (ver docstring de
    03f_analisis_calidad_imagenes_gsv.py, Seccion A)."""
    d = pd.read_csv(EMB_DIR / "descargas_final.csv", on_bad_lines="skip", engine="python")
    d = d[d.exito == True].copy()
    d["anio_pano"] = d["fecha_pano"].str[:4].astype(int)
    d["consecutivo"] = d["consecutivo"].astype(str)
    return d


def hogares_una_foto(descargas: pd.DataFrame, ola: int) -> pd.DataFrame:
    """Hogares con exactamente 1 fila (1 foto) en la ola dada -- evita la
    ambiguedad de join documentada en el docstring del modulo."""
    d_ola = descargas[descargas.ola == ola].copy()
    n_por_hogar = d_ola.groupby("consecutivo").size()
    unicos = n_por_hogar[n_por_hogar == 1].index
    return d_ola[d_ola.consecutivo.isin(unicos)]


def cargar_embeddings_ola(ola: int) -> pd.DataFrame:
    emb = pd.read_parquet(EMB_DIR / "embeddings_unidos_anonimizado.parquet")
    emb = emb[emb.ola == ola].copy()
    emb["consecutivo"] = emb["consecutivo"].astype(str)
    return emb


def cargar_dmsp_ola(ola: int) -> pd.DataFrame:
    dmsp = pd.read_parquet(DMSP_PATH)
    d = dmsp[dmsp.ola == ola][["consecutivo"] + DMSP_COLS].copy()
    d["consecutivo"] = d["consecutivo"].astype(str)
    return d.drop_duplicates("consecutivo")


def matriz_embedding_por_hogar(emb: pd.DataFrame, col: str, consecutivos: set) -> pd.DataFrame:
    sub = emb[emb.consecutivo.isin(consecutivos)]
    grp = sub.groupby("consecutivo")[col].apply(lambda s: np.mean(np.stack(s.values), axis=0))
    mat = pd.DataFrame(np.stack(grp.values), index=grp.index)
    mat.columns = [f"d{i}" for i in range(mat.shape[1])]
    mat.index.name = "consecutivo"
    return mat.reset_index()


def pc1_por_hogar(emb: pd.DataFrame, col: str, consecutivos: set) -> pd.DataFrame:
    """PC1 de un espacio de embedding, restringido a `consecutivos` -- usado
    como indice resumen de una sola dimension para el Analisis 3 (un unico
    subconjunto, sin necesidad de comparar entre subgrupos -> reajustar el
    PCA sobre ese mismo subconjunto es valido aqui)."""
    mat = matriz_embedding_por_hogar(emb, col, consecutivos)
    x_cols = [c for c in mat.columns if c != "consecutivo"]
    Xs = np.nan_to_num(StandardScaler().fit_transform(mat[x_cols].values))
    pc1 = PCA(n_components=1, random_state=RANDOM_STATE).fit_transform(Xs)[:, 0]
    return pd.DataFrame({"consecutivo": mat["consecutivo"], "pc1": pc1})


def pc1_proyectado_sobre_eje_global(emb: pd.DataFrame, col: str, consecutivos_universo: set) -> pd.DataFrame:
    """A diferencia de `pc1_por_hogar`, ajusta el scaler+PCA UNA SOLA VEZ
    sobre `consecutivos_universo` (todos los hogares candidatos, sin
    importar el bin de desfase) y devuelve la proyeccion de TODOS ellos
    sobre ese mismo eje fijo. Necesario para el Analisis 1b: si se
    reajustara el PCA de forma independiente dentro de cada bin de
    desfase, "PC1" seria una direccion distinta (arbitraria en signo y
    rotacion) en cada bin, y comparar su correlacion con DMSP entre bins
    no seria una comparacion valida -- version corregida tras detectar el
    problema al inspeccionar los primeros resultados (ver docstring del
    modulo, Analisis 1b)."""
    mat = matriz_embedding_por_hogar(emb, col, consecutivos_universo)
    x_cols = [c for c in mat.columns if c != "consecutivo"]
    scaler = StandardScaler().fit(mat[x_cols].values)
    Xs_todos = np.nan_to_num(scaler.transform(mat[x_cols].values))
    pca = PCA(n_components=1, random_state=RANDOM_STATE).fit(Xs_todos)
    pc1 = pca.transform(Xs_todos)[:, 0]
    return pd.DataFrame({"consecutivo": mat["consecutivo"], "pc1": pc1})


# ──────────────────────────────────────────────────────────────────────────
# ANALISIS 1 -- correlacion, solo fotos temporalmente validas
# ──────────────────────────────────────────────────────────────────────────

def analisis_1_correlacion_valida(descargas: pd.DataFrame, emb: pd.DataFrame, dmsp: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*78}\nANALISIS 1 -- Correlacion embeddings <-> DMSP-OLS, ola {OLA_FOCAL}\n"
          f"(SOLO fotos con fecha de captura real dentro de la ventana elegible)\n{'='*78}")

    d_1foto = hogares_una_foto(descargas, OLA_FOCAL)
    lo, hi = VENTANAS_OLA[OLA_FOCAL]
    elegibles = d_1foto[(d_1foto.anio_pano >= lo) & (d_1foto.anio_pano <= hi)]
    consecutivos_validos = set(elegibles.consecutivo)
    print(f"Hogares con 1 sola foto en ola {OLA_FOCAL} Y fecha de captura real en "
          f"{lo}-{hi}: {len(consecutivos_validos)}")

    filas = []
    for col in EMB_COLS:
        mat = matriz_embedding_por_hogar(emb, col, consecutivos_validos)
        merged = mat.merge(dmsp, on="consecutivo", how="inner")
        n = merged.shape[0]
        x_cols = [c for c in mat.columns if c != "consecutivo"]
        Xs = np.nan_to_num(StandardScaler().fit_transform(merged[x_cols].values))

        primera_fila_idx = len(filas)
        for dmsp_col in DMSP_COLS:
            y = merged[dmsp_col].values
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            modelo = RidgeCV(alphas=np.logspace(-1, 5, 25))
            y_pred = cross_val_predict(modelo, Xs, y, cv=cv, n_jobs=-1)
            r, p = pearsonr(y_pred, y)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
            filas.append({"embedding": col, "dmsp_var": dmsp_col, "n": n,
                           "r_pred_vs_real": r, "r2_oos": r2, "p": p})
        f0 = filas[primera_fila_idx]
        print(f"  {col:<22s} n={n}  r(avg_vis)={f0['r_pred_vs_real']:.3f}  R2_oos(avg_vis)={f0['r2_oos']:.3f}")

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_correlacion.csv", index=False)
    print(f"\nGuardado: diagnostico_embeddings_dmsp_correlacion.csv")
    return df


# ──────────────────────────────────────────────────────────────────────────
# ANALISIS 1b -- curva de degradacion por desfase temporal
# ──────────────────────────────────────────────────────────────────────────

def analisis_1b_degradacion_desfase(descargas: pd.DataFrame, emb: pd.DataFrame, dmsp: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*78}\nANALISIS 1b -- Degradacion de la correlacion (PC1) segun desfase "
          f"temporal\nfoto vs. ola {OLA_FOCAL} (base empirica para decidir una ventana mas laxa)\n{'='*78}")

    d_1foto = hogares_una_foto(descargas, OLA_FOCAL).copy()
    d_1foto["abs_desfase"] = (d_1foto["anio_pano"] - OLA_FOCAL).abs()

    bins = [(-0.5, 0.5, "0 (mismo anio)"), (0.5, 2.5, "1-2 anios"),
            (2.5, 5.5, "3-5 anios"), (5.5, 10.5, "6-10 anios"), (10.5, np.inf, "11+ anios")]

    def bin_de(a):
        for lo, hi, nombre in bins:
            if lo < a <= hi:
                return nombre
        return "0 (mismo anio)"

    d_1foto["bin_desfase"] = d_1foto["abs_desfase"].apply(bin_de)
    consecutivos_universo = set(d_1foto.consecutivo)

    filas = []
    for col in EMB_COLS:
        # eje PC1 fijo, ajustado UNA vez sobre todo el universo (todos los
        # desfases) -- todos los bins se proyectan sobre el mismo eje, para
        # que la comparacion entre bins sea valida (ver docstring de
        # `pc1_proyectado_sobre_eje_global`).
        pc1_global = pc1_proyectado_sobre_eje_global(emb, col, consecutivos_universo)
        for _, _, nombre_bin in bins:
            consecutivos = set(d_1foto[d_1foto.bin_desfase == nombre_bin].consecutivo)
            if len(consecutivos) < 20:
                filas.append({"embedding": col, "bin_desfase": nombre_bin, "n": len(consecutivos), "r": np.nan, "p": np.nan})
                continue
            pc1_bin = pc1_global[pc1_global.consecutivo.isin(consecutivos)]
            merged = pc1_bin.merge(dmsp[["consecutivo", "dmsp_avg_vis"]], on="consecutivo", how="inner")
            r, p = pearsonr(merged["pc1"], merged["dmsp_avg_vis"])
            filas.append({"embedding": col, "bin_desfase": nombre_bin, "n": merged.shape[0], "r": r, "p": p})

    df = pd.DataFrame(filas)
    orden_bins = [b[2] for b in bins]
    tabla = df.pivot(index="bin_desfase", columns="embedding", values="r").reindex(orden_bins)
    print(tabla.round(3))
    n_tabla = df.pivot(index="bin_desfase", columns="embedding", values="n").reindex(orden_bins)
    print("\nn por bin:")
    print(n_tabla)
    print("\nNOTA: el signo de PC1 es arbitrario (artefacto de PCA), pero al proyectar TODOS"
          "\nlos bins sobre el mismo eje fijo el signo es consistente entre bins -- lo que hay"
          "\nque leer es si |r| se mantiene o cae a medida que el desfase crece. Si se mantiene"
          "\nalto incluso en bins de desfase moderado, eso da base empirica para ampliar la"
          "\nventana temporal valida en trabajo futuro (ver docstring, 'Posibles soluciones' 1).")
    df.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_degradacion_desfase.csv", index=False)
    print(f"\nGuardado: diagnostico_embeddings_dmsp_degradacion_desfase.csv")
    return df


# ──────────────────────────────────────────────────────────────────────────
# ANALISIS 2 -- sesgo de cobertura
# ──────────────────────────────────────────────────────────────────────────

def analisis_2_sesgo_cobertura(descargas: pd.DataFrame, dmsp: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*78}\nANALISIS 2 -- Sesgo de cobertura: ¿DMSP-OLS difiere entre hogares\n"
          f"sin foto / con foto fuera de ventana / con foto dentro de ventana?\n{'='*78}")

    d_ola = descargas[descargas.ola == OLA_FOCAL]
    consecutivos_con_foto = set(d_ola.consecutivo)

    d_1foto = hogares_una_foto(descargas, OLA_FOCAL)
    lo, hi = VENTANAS_OLA[OLA_FOCAL]
    consecutivos_validos = set(d_1foto[(d_1foto.anio_pano >= lo) & (d_1foto.anio_pano <= hi)].consecutivo)
    consecutivos_fuera_ventana = consecutivos_con_foto - consecutivos_validos

    dmsp_c = dmsp.copy()
    dmsp_c["consecutivo"] = dmsp_c["consecutivo"].astype(str)

    grupo_sin_foto = dmsp_c[~dmsp_c.consecutivo.isin(consecutivos_con_foto)]["dmsp_avg_vis"]
    grupo_fuera_ventana = dmsp_c[dmsp_c.consecutivo.isin(consecutivos_fuera_ventana)]["dmsp_avg_vis"]
    grupo_dentro_ventana = dmsp_c[dmsp_c.consecutivo.isin(consecutivos_validos)]["dmsp_avg_vis"]

    print(f"  Sin foto GSV:                n={len(grupo_sin_foto):>5}  media dmsp_avg_vis={grupo_sin_foto.mean():.2f}")
    print(f"  Con foto, FUERA de ventana:  n={len(grupo_fuera_ventana):>5}  media dmsp_avg_vis={grupo_fuera_ventana.mean():.2f}")
    print(f"  Con foto, DENTRO de ventana: n={len(grupo_dentro_ventana):>5}  media dmsp_avg_vis={grupo_dentro_ventana.mean():.2f}")

    stat, p_kw = kruskal(grupo_sin_foto, grupo_fuera_ventana, grupo_dentro_ventana)
    print(f"\n  Kruskal-Wallis (3 grupos): H={stat:.2f}  p={p_kw:.4g}")

    _, p_dv_vs_fv = mannwhitneyu(grupo_dentro_ventana, grupo_fuera_ventana)
    _, p_dv_vs_sf = mannwhitneyu(grupo_dentro_ventana, grupo_sin_foto)
    print(f"  Mann-Whitney dentro-ventana vs. fuera-ventana: p={p_dv_vs_fv:.4g}")
    print(f"  Mann-Whitney dentro-ventana vs. sin-foto:      p={p_dv_vs_sf:.4g}")
    if p_kw < 0.05:
        print("  -> HAY diferencia significativa entre grupos: el subconjunto temporalmente"
              "\n     valido del Analisis 1 probablemente NO es una muestra aleatoria de la"
              "\n     poblacion -- interpretar la correlacion del Analisis 1 con cautela.")
    else:
        print("  -> No se detecta diferencia significativa entre grupos: no hay evidencia,"
              "\n     con este test, de que la cobertura temporal este sesgada hacia zonas"
              "\n     con mas/menos iluminacion nocturna.")

    df = pd.DataFrame({
        "grupo": ["sin_foto", "fuera_ventana", "dentro_ventana"],
        "n": [len(grupo_sin_foto), len(grupo_fuera_ventana), len(grupo_dentro_ventana)],
        "dmsp_avg_vis_media": [grupo_sin_foto.mean(), grupo_fuera_ventana.mean(), grupo_dentro_ventana.mean()],
        "dmsp_avg_vis_mediana": [grupo_sin_foto.median(), grupo_fuera_ventana.median(), grupo_dentro_ventana.median()],
        "kruskal_wallis_p": [p_kw] * 3,
    })
    df.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_sesgo_cobertura.csv", index=False)
    print(f"\nGuardado: diagnostico_embeddings_dmsp_sesgo_cobertura.csv")
    return df


# ──────────────────────────────────────────────────────────────────────────
# ANALISIS 3 -- aporte marginal de DMSP-OLS controlando por embeddings
# ──────────────────────────────────────────────────────────────────────────

def analisis_3_aporte_marginal(descargas: pd.DataFrame, emb: pd.DataFrame, dmsp: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*78}\nANALISIS 3 -- ¿Sobrevive el aporte marginal de DMSP-OLS controlando\n"
          f"por la senal visual de los embeddings? (diagnostico correlacional,\n"
          f"NO reentrena el pipeline -- ver docstring, 'Posibles soluciones' punto 2)\n{'='*78}")

    d_1foto = hogares_una_foto(descargas, OLA_FOCAL)
    lo, hi = VENTANAS_OLA[OLA_FOCAL]
    consecutivos_validos = set(d_1foto[(d_1foto.anio_pano >= lo) & (d_1foto.anio_pano <= hi)].consecutivo)

    dmsp_c = dmsp.copy()
    dmsp_c["consecutivo"] = dmsp_c["consecutivo"].astype(str)

    resultados = []
    for espec, archivo in [("A", "predicciones_test_dmsp_A.parquet"), ("B", "predicciones_test_dmsp_B.parquet")]:
        ruta = OUT_DIR / archivo
        if not ruta.exists():
            print(f"  [{espec}] {archivo} no encontrado -- se omite")
            continue
        pred = pd.read_parquet(ruta)
        pred["consecutivo"] = pred["consecutivo"].astype(str)

        for algoritmo in pred["algoritmo"].unique():
            sub = pred[pred.algoritmo == algoritmo].copy()
            sub = sub[sub.consecutivo.isin(consecutivos_validos)]
            if sub.shape[0] < 30:
                print(f"  [{espec}/{algoritmo}] n={sub.shape[0]} -- insuficiente, se omite")
                continue

            pc1 = pc1_por_hogar(emb, "embedding_places365", set(sub.consecutivo))
            sub = sub.merge(pc1, on="consecutivo", how="inner")
            sub = sub.merge(dmsp_c[["consecutivo", "dmsp_avg_vis"]], on="consecutivo", how="inner")
            n = sub.shape[0]

            sub["delta_proba"] = sub["proba_geo"] - sub["proba_base"]
            r_delta_pc1, p_delta_pc1 = pearsonr(sub["delta_proba"], sub["pc1"])

            auc_base = roc_auc_score(sub["Y"], sub["proba_base"])
            auc_geo = roc_auc_score(sub["Y"], sub["proba_geo"])

            X = sub[["dmsp_avg_vis", "pc1"]].values
            Xs = StandardScaler().fit_transform(X)
            logit = LogisticRegression(max_iter=1000).fit(Xs, sub["Y"])
            coef_dmsp, coef_pc1 = logit.coef_[0]

            print(f"  [{espec}/{algoritmo}] n={n}  AUC base={auc_base:.3f}  AUC+DMSP={auc_geo:.3f}  "
                  f"r(delta_proba, PC1)={r_delta_pc1:.3f} (p={p_delta_pc1:.3f})  "
                  f"coef_logit(dmsp)={coef_dmsp:+.3f}  coef_logit(pc1)={coef_pc1:+.3f}")

            resultados.append({
                "especificacion": espec, "algoritmo": algoritmo, "n": n,
                "auc_base": auc_base, "auc_geo": auc_geo,
                "r_delta_proba_vs_pc1": r_delta_pc1, "p_delta_proba_vs_pc1": p_delta_pc1,
                "coef_logit_dmsp_avg_vis": coef_dmsp, "coef_logit_pc1_places365": coef_pc1,
            })

    df = pd.DataFrame(resultados)
    df.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_aporte_marginal.csv", index=False)
    print(f"\nGuardado: diagnostico_embeddings_dmsp_aporte_marginal.csv")
    print("\nNOTA DE LECTURA: n es chico (decenas a un par de cientos) -- estos coeficientes"
          "\nson exploratorios. Un coeficiente de dmsp_avg_vis que se mantiene positivo y del"
          "\norden de magnitud del de pc1 sugiere que DMSP aporta algo que la foto NO capta"
          "\n(ej. contexto de vecindario mas amplio que el radio de la foto); un coeficiente"
          "\nde dmsp_avg_vis que colapsa hacia 0 sugiere que la foto ya contiene esa senal.")
    return df


def main() -> None:
    print("Cargando descargas_final.csv (fecha real de captura)...")
    descargas = cargar_descargas()
    print("Cargando embeddings...")
    emb = cargar_embeddings_ola(OLA_FOCAL)
    print("Cargando DMSP-OLS...")
    dmsp = cargar_dmsp_ola(OLA_FOCAL)

    analisis_1_correlacion_valida(descargas, emb, dmsp)
    analisis_1b_degradacion_desfase(descargas, emb, dmsp)
    analisis_2_sesgo_cobertura(descargas, dmsp)
    analisis_3_aporte_marginal(descargas, emb, dmsp)

    print(f"\n{'='*78}\nFIN -- ver docstring del script para caveats completos y ejercicios"
          f"\npropuestos NO implementados aqui (Modelo C, CLIP interpretable -- que requiere"
          f"\nrecuperar embeddings_clip.parquet o re-correr el paso de union sin descartar"
          f"\nclip_score_*, estabilidad temporal).\n{'='*78}")


if __name__ == "__main__":
    main()
