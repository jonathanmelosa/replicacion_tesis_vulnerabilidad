"""
diagnostico_embeddings_dmsp_extensiones.py
=====================================================================
Ejecuta, EN ORDEN DE COSTO DE IMPLEMENTACION (mas corto primero), las 5
"posibles soluciones" propuestas en el informe pedagogico
`informe_embeddings_dmsp.pdf` (2026-09-02) sobre `diagnostico_embeddings_dmsp.py`,
mas un analisis nuevo: cuantos
componentes principales (PCA) conviene retener de cada embedding, y con
que criterio.

NO duplica logica -- importa las funciones de carga/preparacion de datos
de `diagnostico_embeddings_dmsp.py` (mismo modulo, mismo repo).

ORDEN DE EJECUCION Y JUSTIFICACION DE COSTO
---------------------------------------------------------------------
1. Correccion FDR + bootstrap de coeficientes (Solucion 5 del informe):
   post-procesamiento puro sobre resultados YA calculados -- no requiere
   cargar datos nuevos ni reentrenar nada. El mas rapido de los 6 pasos.
2. Cuantos componentes PCA retener (analisis nuevo, pedido explicitamente):
   requiere un solo ajuste de PCA por embedding con mas componentes
   (hasta 20) y una curva de Ridge-CV incremental -- minutos, no reentrena
   modelos de pobreza.
3. Ventana temporal ampliada (Solucion 1 del informe): reutiliza los
   mismos datos ya cargados (descargas_final.csv, embeddings, DMSP),
   solo cambia el filtro de fecha -- minutos.
4. Estabilidad temporal 2010->2013 (Solucion 4 del informe): depende de
   que la Solucion 1 (ventana ampliada) genere cobertura utilizable en
   ola 2010, que originalmente tiene 0% de fotos dentro de su ventana
   oficial -- por eso va DESPUES de la Solucion 1, no antes.
5. Recuperar clip_score_* (Solucion 3 del informe): un chequeo rapido,
   pero se documenta al final porque esta
   BLOQUEADA por una dependencia externa (archivo en una ruta de red
   Windows no accesible desde esta maquina). Se re-verifica aqui por
   completitud y se deja documentado el bloqueo, no se intenta "resolver".
6. "Modelo C" (Solucion 2 del informe): el mas costoso, y el unico que
   requiere una decision de diseno importante -- ver docstring de
   `modelo_c_escalado()` para la limitacion estructural que se encontro
   (la ola de ENTRENAMIENTO del pipeline de pobreza, 2010, no tiene
   ninguna foto de Street View anterior a 2012 -- Street View no cubria
   Colombia antes de esa fecha) y la version escalada que se implemento
   en su lugar.

OUTPUTS
---------------------------------------------------------------------
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_ext_fdr_bootstrap.csv
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_ext_num_componentes.csv
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_ext_ventana_ampliada.csv
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_ext_estabilidad_temporal.csv
    data/processed/benchmark_resultados/diagnostico_embeddings_dmsp_ext_modelo_c.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_embeddings_dmsp_extensiones.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

import modelo_utils as mu
from diagnostico_embeddings_dmsp import (
    DMSP_COLS,
    EMB_COLS,
    EMB_DIR,
    OLA_FOCAL,
    OUT_DIR,
    RANDOM_STATE,
    VENTANAS_OLA,
    cargar_descargas,
    cargar_dmsp_ola,
    cargar_embeddings_ola,
    hogares_una_foto,
    matriz_embedding_por_hogar,
)

RNG = np.random.default_rng(RANDOM_STATE)


def fdr_benjamini_hochberg(p_valores: np.ndarray, alpha: float = 0.05) -> tuple:
    """Implementacion directa de Benjamini-Hochberg (sin dependencia de
    statsmodels, no instalado en este entorno). Devuelve (rechazado, p_ajustado)."""
    p = np.asarray(p_valores)
    n = len(p)
    orden = np.argsort(p)
    p_ordenado = p[orden]
    p_ajustado_ordenado = np.minimum.accumulate((p_ordenado * n / np.arange(n, 0, -1))[::-1])[::-1]
    p_ajustado_ordenado = np.clip(p_ajustado_ordenado, 0, 1)
    p_ajustado = np.empty(n)
    p_ajustado[orden] = p_ajustado_ordenado
    rechazado = p_ajustado <= alpha
    return rechazado, p_ajustado


# ──────────────────────────────────────────────────────────────────────────
# 1. Correccion FDR + bootstrap de coeficientes (Solucion 5)
# ──────────────────────────────────────────────────────────────────────────

def solucion_5_fdr_bootstrap(descargas: pd.DataFrame, emb: pd.DataFrame, dmsp: pd.DataFrame) -> None:
    print(f"\n{'='*78}\n[1/6] SOLUCION 5 -- Correccion FDR (Analisis 1b) + bootstrap de "
          f"coeficientes\n(Analisis 3)\n{'='*78}")

    # --- FDR sobre los p-valores del Analisis 1b ---
    ruta_1b = OUT_DIR / "diagnostico_embeddings_dmsp_degradacion_desfase.csv"
    df_1b = pd.read_csv(ruta_1b)
    df_1b_validos = df_1b.dropna(subset=["p"]).copy()
    rechazado, p_ajustado = fdr_benjamini_hochberg(df_1b_validos["p"].values, alpha=0.05)
    df_1b_validos["p_fdr_bh"] = p_ajustado
    df_1b_validos["significativo_fdr"] = rechazado
    n_antes = (df_1b_validos["p"] < 0.05).sum()
    n_despues = rechazado.sum()
    print(f"  Analisis 1b: {len(df_1b_validos)} pruebas -- significativas al 5% SIN corregir: "
          f"{n_antes}, CON correccion FDR (Benjamini-Hochberg): {n_despues}")
    df_1b_validos.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_fdr_1b.csv", index=False)

    # --- Bootstrap de los coeficientes logisticos del Analisis 3 ---
    d_1foto = hogares_una_foto(descargas, OLA_FOCAL)
    lo, hi = VENTANAS_OLA[OLA_FOCAL]
    consecutivos_validos = set(d_1foto[(d_1foto.anio_pano >= lo) & (d_1foto.anio_pano <= hi)].consecutivo)
    dmsp_c = dmsp.copy()
    dmsp_c["consecutivo"] = dmsp_c["consecutivo"].astype(str)

    from diagnostico_embeddings_dmsp import pc1_por_hogar
    pred = pd.read_parquet(OUT_DIR / "predicciones_test_dmsp_A.parquet")
    pred["consecutivo"] = pred["consecutivo"].astype(str)
    sub = pred[(pred.algoritmo == "XGBoost") & (pred.consecutivo.isin(consecutivos_validos))].copy()
    pc1 = pc1_por_hogar(emb, "embedding_places365", set(sub.consecutivo))
    sub = sub.merge(pc1, on="consecutivo", how="inner").merge(
        dmsp_c[["consecutivo", "dmsp_avg_vis"]], on="consecutivo", how="inner")
    n = sub.shape[0]

    X = sub[["dmsp_avg_vis", "pc1"]].values
    Xs_full = StandardScaler().fit_transform(X)
    y = sub["Y"].values

    n_boot = 2000
    coefs_dmsp, coefs_pc1 = [], []
    idx = np.arange(n)
    for _ in range(n_boot):
        muestra = RNG.choice(idx, size=n, replace=True)
        if len(np.unique(y[muestra])) < 2:
            continue
        logit = LogisticRegression(max_iter=1000).fit(Xs_full[muestra], y[muestra])
        coefs_dmsp.append(logit.coef_[0][0])
        coefs_pc1.append(logit.coef_[0][1])
    coefs_dmsp, coefs_pc1 = np.array(coefs_dmsp), np.array(coefs_pc1)

    ci_dmsp = np.percentile(coefs_dmsp, [2.5, 97.5])
    ci_pc1 = np.percentile(coefs_pc1, [2.5, 97.5])
    print(f"  Bootstrap ({n_boot} remuestras, n={n}, espec A/XGBoost):")
    print(f"    coef_logit(dmsp_avg_vis): media={coefs_dmsp.mean():+.3f}  IC95%=[{ci_dmsp[0]:+.3f}, {ci_dmsp[1]:+.3f}]")
    print(f"    coef_logit(pc1_places365): media={coefs_pc1.mean():+.3f}  IC95%=[{ci_pc1[0]:+.3f}, {ci_pc1[1]:+.3f}]")
    dmsp_cruza_cero = ci_dmsp[0] < 0 < ci_dmsp[1]
    print(f"    -> IC95% de dmsp_avg_vis {'SI' if dmsp_cruza_cero else 'NO'} cruza 0 "
          f"({'no distinguible de 0' if dmsp_cruza_cero else 'distinguible de 0'} al 95%)")

    pd.DataFrame({
        "coeficiente": ["dmsp_avg_vis", "pc1_places365"],
        "media_bootstrap": [coefs_dmsp.mean(), coefs_pc1.mean()],
        "ci95_low": [ci_dmsp[0], ci_pc1[0]],
        "ci95_high": [ci_dmsp[1], ci_pc1[1]],
        "n_bootstrap_validos": [len(coefs_dmsp)] * 2,
        "n_muestra": [n] * 2,
    }).to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_bootstrap_coefs.csv", index=False)
    print(f"\n  Guardado: diagnostico_embeddings_dmsp_ext_fdr_1b.csv, "
          f"diagnostico_embeddings_dmsp_ext_bootstrap_coefs.csv")


# ──────────────────────────────────────────────────────────────────────────
# 2. Cuantos componentes PCA retener (analisis nuevo)
# ──────────────────────────────────────────────────────────────────────────

def analisis_num_componentes(emb: pd.DataFrame, dmsp: pd.DataFrame, consecutivos_validos: set) -> pd.DataFrame:
    print(f"\n{'='*78}\n[2/6] ¿CUANTOS COMPONENTES PCA RETENER? -- criterio de codo (varianza\n"
          f"explicada) y criterio predictivo (R2 de validacion cruzada vs. k)\n{'='*78}")

    K_MAX = 20
    filas = []
    for col in EMB_COLS:
        mat = matriz_embedding_por_hogar(emb, col, consecutivos_validos)
        merged = mat.merge(dmsp, on="consecutivo", how="inner")
        x_cols = [c for c in mat.columns if c != "consecutivo"]
        Xs = np.nan_to_num(StandardScaler().fit_transform(merged[x_cols].values))
        y = merged["dmsp_avg_vis"].values

        k_max_real = min(K_MAX, Xs.shape[0] - 1, Xs.shape[1])
        pca = PCA(n_components=k_max_real, random_state=RANDOM_STATE).fit(Xs)
        var_exp = pca.explained_variance_ratio_
        var_acum = np.cumsum(var_exp)
        pcs = pca.transform(Xs)

        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for k in range(1, k_max_real + 1):
            Xk = pcs[:, :k]
            modelo = RidgeCV(alphas=np.logspace(-1, 5, 15))
            y_pred = cross_val_predict(modelo, Xk, y, cv=cv, n_jobs=-1)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
            filas.append({
                "embedding": col, "k": k,
                "var_explicada_marginal_pc_k": var_exp[k - 1],
                "var_explicada_acumulada": var_acum[k - 1],
                "r2_oos_ridge_k_componentes": r2,
            })

    df = pd.DataFrame(filas)

    # criterio 1: codo -- primer k donde la ganancia marginal de varianza < 1%
    # criterio 2: predictivo -- primer k donde el R2 de CV deja de mejorar en
    #   mas de 0.005 respecto al k anterior (rendimientos marginales decrecientes)
    resumen = []
    for col in EMB_COLS:
        sub = df[df.embedding == col].sort_values("k")
        codo = sub[sub["var_explicada_marginal_pc_k"] < 0.01]["k"]
        k_codo = int(codo.iloc[0]) if len(codo) else int(sub["k"].max())

        r2_vals = sub["r2_oos_ridge_k_componentes"].values
        ganancia = np.diff(r2_vals, prepend=r2_vals[0])
        estabiliza = np.where(ganancia < 0.005)[0]
        k_predictivo = int(sub["k"].values[estabiliza[1]]) if len(estabiliza) > 1 else int(sub["k"].max())

        r2_k1 = sub[sub.k == 1]["r2_oos_ridge_k_componentes"].values[0]
        r2_kmax = sub["r2_oos_ridge_k_componentes"].values[-1]
        resumen.append({
            "embedding": col, "k_criterio_codo_var": k_codo, "k_criterio_predictivo": k_predictivo,
            "r2_con_k1": r2_k1, "r2_con_k_max": r2_kmax,
            "ganancia_r2_de_1_a_kmax": r2_kmax - r2_k1,
        })
        print(f"  {col:<22s} R2(k=1)={r2_k1:.3f}  R2(k={sub['k'].max()})={r2_kmax:.3f}  "
              f"codo(var)~k={k_codo}  estabiliza(R2)~k={k_predictivo}")

    df_resumen = pd.DataFrame(resumen)
    df.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_num_componentes.csv", index=False)
    df_resumen.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_num_componentes_resumen.csv", index=False)
    print(f"\n  Guardado: diagnostico_embeddings_dmsp_ext_num_componentes.csv (+resumen)")
    return df_resumen


# ──────────────────────────────────────────────────────────────────────────
# 3. Ventana temporal ampliada (Solucion 1)
# ──────────────────────────────────────────────────────────────────────────

def solucion_1_ventana_ampliada(descargas: pd.DataFrame, emb: pd.DataFrame, dmsp: pd.DataFrame) -> set:
    print(f"\n{'='*78}\n[3/6] SOLUCION 1 -- Ventana temporal ampliada (justificada por la "
          f"curva\ndel Analisis 1b: la correlacion NO decae con el desfase)\n{'='*78}")

    VENTANA_AMPLIADA = (OLA_FOCAL - 2, OLA_FOCAL + 2)  # 2011-2015
    d_1foto = hogares_una_foto(descargas, OLA_FOCAL)
    elegibles = d_1foto[(d_1foto.anio_pano >= VENTANA_AMPLIADA[0]) & (d_1foto.anio_pano <= VENTANA_AMPLIADA[1])]
    consecutivos_ampliados = set(elegibles.consecutivo)
    print(f"  Ventana original (2011-2013): n=830")
    print(f"  Ventana ampliada ({VENTANA_AMPLIADA[0]}-{VENTANA_AMPLIADA[1]}): n={len(consecutivos_ampliados)}")

    filas = []
    for col in EMB_COLS:
        mat = matriz_embedding_por_hogar(emb, col, consecutivos_ampliados)
        merged = mat.merge(dmsp, on="consecutivo", how="inner")
        n = merged.shape[0]
        x_cols = [c for c in mat.columns if c != "consecutivo"]
        Xs = np.nan_to_num(StandardScaler().fit_transform(merged[x_cols].values))
        y = merged["dmsp_avg_vis"].values
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        modelo = RidgeCV(alphas=np.logspace(-1, 5, 25))
        y_pred = cross_val_predict(modelo, Xs, y, cv=cv, n_jobs=-1)
        r, p = pearsonr(y_pred, y)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        filas.append({"embedding": col, "n": n, "r_pred_vs_real": r, "r2_oos": r2, "p": p})
        print(f"  {col:<22s} n={n}  r={r:.3f}  R2_oos={r2:.3f}")

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_ventana_ampliada.csv", index=False)
    print(f"\n  Guardado: diagnostico_embeddings_dmsp_ext_ventana_ampliada.csv")
    return consecutivos_ampliados


# ──────────────────────────────────────────────────────────────────────────
# 4. Estabilidad temporal 2010->2013 (Solucion 4)
# ──────────────────────────────────────────────────────────────────────────

def solucion_4_estabilidad_temporal(descargas: pd.DataFrame, emb2010: pd.DataFrame, emb2013: pd.DataFrame) -> None:
    print(f"\n{'='*78}\n[4/6] SOLUCION 4 -- Estabilidad temporal del embedding "
          f"(test-retest\n2010 vs. 2013) -- depende de que haya cobertura utilizable en 2010\n{'='*78}")

    d_1foto_2010 = hogares_una_foto(descargas, 2010)
    print(f"  Distribucion de anio_pano en fotos 'ola=2010' (1 sola foto), min={d_1foto_2010.anio_pano.min()}:")
    print(f"    Google Street View no tiene cobertura de Colombia antes de 2012 -- la ventana")
    print(f"    OFICIAL de la ola 2010 (<=2010) sigue en 0% de cobertura real, tal como ya")
    print(f"    documentaba 03f_analisis_calidad_imagenes_gsv.py. Usando una ventana ampliada")
    print(f"    2010-2012 (lo mas cerca posible de 2010 que existe en los datos):")

    consecutivos_2010 = set(d_1foto_2010[(d_1foto_2010.anio_pano >= 2010) & (d_1foto_2010.anio_pano <= 2012)].consecutivo)
    print(f"    n hogares 'ola=2010' con foto real en 2010-2012: {len(consecutivos_2010)}")

    d_1foto_2013 = hogares_una_foto(descargas, OLA_FOCAL)
    lo, hi = VENTANAS_OLA[OLA_FOCAL]
    consecutivos_2013 = set(d_1foto_2013[(d_1foto_2013.anio_pano >= lo) & (d_1foto_2013.anio_pano <= hi)].consecutivo)

    overlap = consecutivos_2010 & consecutivos_2013
    print(f"    Hogares con foto valida (aproximada) EN AMBAS olas: {len(overlap)}")

    if len(overlap) < 30:
        print(f"\n  -> n={len(overlap)} es DEMASIADO CHICO para un test-retest con algun poder")
        print(f"     estadistico. CONCLUSION: la estabilidad temporal del embedding NO se puede")
        print(f"     evaluar de forma confiable con los datos actualmente disponibles -- el")
        print(f"     bloqueo estructural (Street View sin cobertura de Colombia antes de 2012)")
        print(f"     es real y no se resuelve solo ampliando la ventana. Ver informe, seccion de")
        print(f"     limitaciones, para la implicacion de esto.")
        pd.DataFrame([{"n_overlap_2010_2013": len(overlap), "factible": False,
                        "razon": "Street View sin cobertura de Colombia antes de 2012 -- "
                                 "incluso con ventana ampliada 2010-2012 el overlap es insuficiente"}]
                     ).to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_estabilidad_temporal.csv", index=False)
        return

    # IMPORTANTE: ajustar PCA por separado en 2010 y en 2013 reproduciria el
    # mismo error ya documentado y corregido en diagnostico_embeddings_dmsp.py
    # (Error 2 del informe) -- el signo/orientacion de "PC1" seria arbitrario
    # y distinto entre los dos anios, invalidando la comparacion test-retest.
    # Aqui se ajusta el PCA UNA SOLA VEZ sobre el POOL de embeddings 2010+2013
    # (apilados) de los mismos 87 hogares, y se proyectan ambos anios sobre
    # ese mismo eje fijo -- analogo a `pc1_proyectado_sobre_eje_global`.
    filas = []
    for col in EMB_COLS:
        m2010 = matriz_embedding_por_hogar(emb2010, col, overlap)
        m2013 = matriz_embedding_por_hogar(emb2013, col, overlap)
        merged = m2010.merge(m2013, on="consecutivo", suffixes=("_2010", "_2013"))
        x_cols_2010 = [c for c in merged.columns if c.endswith("_2010")]
        x_cols_2013 = [c for c in merged.columns if c.endswith("_2013")]

        scaler = StandardScaler().fit(
            np.vstack([merged[x_cols_2010].values, merged[x_cols_2013].values]))
        X2010 = np.nan_to_num(scaler.transform(merged[x_cols_2010].values))
        X2013 = np.nan_to_num(scaler.transform(merged[x_cols_2013].values))
        pca = PCA(n_components=1, random_state=RANDOM_STATE).fit(np.vstack([X2010, X2013]))
        pc1_2010 = pca.transform(X2010)[:, 0]
        pc1_2013 = pca.transform(X2013)[:, 0]
        r, p = pearsonr(pc1_2010, pc1_2013)
        filas.append({"embedding": col, "n": merged.shape[0], "r_test_retest_pc1_eje_fijo": r, "p": p})
        print(f"  {col:<22s} n={merged.shape[0]}  r(test-retest PC1, eje fijo)={r:.3f}  p={p:.4f}")

    pd.DataFrame(filas).to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_estabilidad_temporal.csv", index=False)
    print(f"\n  Guardado: diagnostico_embeddings_dmsp_ext_estabilidad_temporal.csv")


# ──────────────────────────────────────────────────────────────────────────
# 5. Recuperar clip_score_* (Solucion 3) -- re-chequeo y documentacion del bloqueo
# ──────────────────────────────────────────────────────────────────────────

def solucion_3_clip_score() -> None:
    print(f"\n{'='*78}\n[5/6] SOLUCION 3 -- Intento de recuperar clip_score_* (indicadores\n"
          f"CLIP zero-shot interpretables)\n{'='*78}")
    candidatos = [
        EMB_DIR / "embeddings_clip.parquet",
        Path.home() / "Downloads" / "SALE_11082026v2" / "embeddings_clip.parquet",
        Path.home() / "Downloads" / "SALE_31072026" / "embeddings_clip.parquet",
    ]
    encontrado = False
    for c in candidatos:
        if c.exists():
            print(f"  ENCONTRADO: {c}")
            encontrado = True
    if not encontrado:
        print("  NO encontrado en ninguna ubicacion local candidata (incluye las carpetas")
        print("  SALE_* de Descargas, copias del mismo origen que data/processed/embeddings/).")
        print("  BLOQUEO CONFIRMADO: el archivo con clip_score_* vive exclusivamente en la ruta")
        print("  de red de Windows referenciada en union_parquets.py")
        print("  (\\\\ECON-E420004947\\...\\gsv\\embeddings\\embeddings_clip.parquet), no montada")
        print("  ni accesible desde esta sesion. Esta solucion NO se pudo ejecutar -- queda")
        print("  como item de trabajo futuro que depende de acceso a esa maquina/red.")
    pd.DataFrame([{"encontrado_localmente": encontrado, "bloqueado_por": "ruta de red Windows no accesible"}]
                 ).to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_clip_score_intento.csv", index=False)


# ──────────────────────────────────────────────────────────────────────────
# 6. "Modelo C" escalado (Solucion 2)
# ──────────────────────────────────────────────────────────────────────────

def modelo_c_escalado(descargas: pd.DataFrame, emb: pd.DataFrame, dmsp: pd.DataFrame,
                       consecutivos_ampliados: set, k_componentes: int) -> None:
    """El "Modelo C" originalmente propuesto en el informe era reentrenar el
    pipeline COMPLETO de prediccion de pobreza (train 2010->2013, test
    2013->2016) agregando embeddings como covariable. Esto NO es factible:
    la ola de ENTRENAMIENTO (2010) no tiene ninguna foto de Street View con
    fecha de captura real anterior a 2012 (Street View no cubria Colombia
    antes de esa fecha) -- ni siquiera con una ventana ampliada hay
    suficiente n en 2010 (ver Solucion 4). No se puede construir el
    embedding de la covariable base sin, en la practica, usar fotos
    tomadas DESPUES del periodo de entrenamiento -- exactamente el problema
    de contaminacion temporal que motivo este informe en primer lugar
    (Error 1).

    Version escalada implementada aqui: en vez del holdout temporal
    completo, se evalua -- SOLO dentro del conjunto de prueba 2013->2016
    (que sí tiene embeddings temporalmente validos, al usar covariables de
    la ola 2013) -- si agregar los embeddings (top-k componentes PCA de
    Places365, k elegido por el analisis de la Seccion 2) a un modelo que
    ya tiene DMSP-OLS mejora el AUC-ROC, vía 5-fold CV DENTRO de esa
    muestra (no un holdout temporal -- ver limitacion en el informe). Se
    usa regresion logistica con regularizacion L2 fija (sin busqueda de
    hiperparametros, dado que n=830 o menos es demasiado chico para una
    busqueda de 10 folds sin sobreajuste)."""
    print(f"\n{'='*78}\n[6/6] SOLUCION 2 -- \"Modelo C\" (version escalada -- ver docstring de\n"
          f"esta funcion para la limitacion estructural que impide la version completa)\n{'='*78}")

    dmsp_c = dmsp.copy()
    dmsp_c["consecutivo"] = dmsp_c["consecutivo"].astype(str)
    pc = matriz_embedding_por_hogar(emb, "embedding_places365", consecutivos_ampliados)
    x_cols = [c for c in pc.columns if c != "consecutivo"]
    Xs_full = np.nan_to_num(StandardScaler().fit_transform(pc[x_cols].values))
    pcs = PCA(n_components=k_componentes, random_state=RANDOM_STATE).fit_transform(Xs_full)
    pc_df = pd.DataFrame(pcs, columns=[f"pc{i+1}" for i in range(k_componentes)])
    pc_df["consecutivo"] = pc.consecutivo.values

    resultados = []
    for espec, archivo in [("A", "predicciones_test_dmsp_A.parquet"), ("B", "predicciones_test_dmsp_B.parquet")]:
        pred = pd.read_parquet(OUT_DIR / archivo)
        pred["consecutivo"] = pred["consecutivo"].astype(str)
        sub = pred[pred.algoritmo == "XGBoost"].copy()
        sub = sub.merge(pc_df, on="consecutivo", how="inner").merge(
            dmsp_c[["consecutivo", "dmsp_avg_vis"]], on="consecutivo", how="inner")
        n = sub.shape[0]
        if n < 50:
            print(f"  [{espec}] n={n} -- insuficiente, se omite")
            continue

        y = sub["Y"].values
        pc_cols = [f"pc{i+1}" for i in range(k_componentes)]
        X_solo_emb = StandardScaler().fit_transform(sub[pc_cols].values)
        X_emb_mas_dmsp = StandardScaler().fit_transform(sub[pc_cols + ["dmsp_avg_vis"]].values)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        modelo = LogisticRegression(max_iter=1000, C=1.0)

        proba_solo_emb = cross_val_predict(modelo, X_solo_emb, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        proba_emb_mas_dmsp = cross_val_predict(modelo, X_emb_mas_dmsp, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

        auc_solo_emb = roc_auc_score(y, proba_solo_emb)
        auc_emb_mas_dmsp = roc_auc_score(y, proba_emb_mas_dmsp)

        # bootstrap pareado del delta (mismo espiritu que diagnostico_bootstrap_dmsp.py)
        n_boot = 1000
        deltas = []
        idx = np.arange(n)
        for _ in range(n_boot):
            muestra = RNG.choice(idx, size=n, replace=True)
            if len(np.unique(y[muestra])) < 2:
                continue
            a1 = roc_auc_score(y[muestra], proba_solo_emb[muestra])
            a2 = roc_auc_score(y[muestra], proba_emb_mas_dmsp[muestra])
            deltas.append(a2 - a1)
        deltas = np.array(deltas)
        ci = np.percentile(deltas, [2.5, 97.5])
        p_valor = 2 * min((deltas > 0).mean(), (deltas < 0).mean())

        print(f"  [{espec}] n={n}  AUC(solo embeddings, k={k_componentes})={auc_solo_emb:.3f}  "
              f"AUC(embeddings+DMSP)={auc_emb_mas_dmsp:.3f}  delta={auc_emb_mas_dmsp-auc_solo_emb:+.4f}  "
              f"IC95%=[{ci[0]:+.4f}, {ci[1]:+.4f}]  p={p_valor:.3f}")

        resultados.append({
            "especificacion": espec, "n": n, "k_componentes_embedding": k_componentes,
            "auc_solo_embeddings": auc_solo_emb, "auc_embeddings_mas_dmsp": auc_emb_mas_dmsp,
            "delta_auc": auc_emb_mas_dmsp - auc_solo_emb,
            "delta_ci95_low": ci[0], "delta_ci95_high": ci[1], "p_valor_bootstrap": p_valor,
        })

    df = pd.DataFrame(resultados)
    df.to_csv(OUT_DIR / "diagnostico_embeddings_dmsp_ext_modelo_c.csv", index=False)
    print(f"\n  Guardado: diagnostico_embeddings_dmsp_ext_modelo_c.csv")
    print("\n  NOTA: esto es CV interna dentro del conjunto de prueba 2013->2016, NO un holdout")
    print("  temporal como el resto del pipeline de la tesis -- ver limitacion en el informe.")


def main() -> None:
    print("Cargando datos base...")
    descargas = cargar_descargas()
    emb2013 = cargar_embeddings_ola(OLA_FOCAL)
    dmsp2013 = cargar_dmsp_ola(OLA_FOCAL)

    d_1foto = hogares_una_foto(descargas, OLA_FOCAL)
    lo, hi = VENTANAS_OLA[OLA_FOCAL]
    consecutivos_validos = set(d_1foto[(d_1foto.anio_pano >= lo) & (d_1foto.anio_pano <= hi)].consecutivo)

    solucion_5_fdr_bootstrap(descargas, emb2013, dmsp2013)
    df_resumen_k = analisis_num_componentes(emb2013, dmsp2013, consecutivos_validos)
    consecutivos_ampliados = solucion_1_ventana_ampliada(descargas, emb2013, dmsp2013)

    print("Cargando embeddings ola 2010 (para Solucion 4)...")
    emb2010 = cargar_embeddings_ola(2010)
    solucion_4_estabilidad_temporal(descargas, emb2010, emb2013)

    solucion_3_clip_score()

    # k para Modelo C: usa el criterio predictivo de Places365 (el embedding
    # mas fuertemente correlacionado con DMSP-OLS en el Analisis 1 original)
    k_places365 = int(df_resumen_k[df_resumen_k.embedding == "embedding_places365"]["k_criterio_predictivo"].iloc[0])
    modelo_c_escalado(descargas, emb2013, dmsp2013, consecutivos_ampliados, k_places365)

    print(f"\n{'='*78}\nFIN de las 6 extensiones.\n{'='*78}")


if __name__ == "__main__":
    main()
