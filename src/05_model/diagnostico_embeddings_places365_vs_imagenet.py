"""
diagnostico_embeddings_places365_vs_imagenet.py
=====================================
Pregunta: ¿predicen mejor la luz nocturna las
representaciones de un modelo entrenado para reconocer ESCENAS
(Places365) que las de modelos entrenados para reconocer OBJETOS
(ImageNet) o pares imagen-texto generales (CLIP)?

Places365 y ResNet50 comparten arquitectura (ResNet-50, ver
`03_extraer_embeddings.py`): solo cambia el conjunto de entrenamiento
(Places365 = escenas, ImageNet = objetos). Esa comparacion aisla el efecto
de los datos de entrenamiento -- la de VGG19 (otra arquitectura, ImageNet)
y CLIP (otra arquitectura y otro objetivo) no lo aisla.

Las diferencias de R2 de `diagnostico_embeddings_dmsp.py` (Analisis 1) y
`diagnostico_embeddings_dmsp_extensiones.py` (Solucion 1) se reportaron
sin intervalo de confianza. Aqui:

METODOLOGIA
    Mismo modelo que esos scripts (embedding completo estandarizado +
    RidgeCV, alphas logspace(-1, 5, 25), KFold de 5 particiones) y mismas
    ventanas de fecha de la foto (original 2011-2013, ampliada 2011-2015),
    variable objetivo `dmsp_avg_vis` de 2013. Diferencias respecto a esos
    scripts, necesarias para una comparacion PAREADA:
      1. Los cuatro embeddings se evaluan sobre EXACTAMENTE los mismos
         hogares (interseccion) y las mismas particiones.
      2. IC95% de R2(Places365) - R2(otro) por bootstrap pareado de
         hogares (N_BOOT remuestreos) sobre las predicciones fuera de
         muestra -- refleja la variabilidad muestral de los hogares.
      3. La diferencia se recalcula con N_SEMILLAS particiones distintas
         de la validacion cruzada -- refleja la variabilidad por particion.

INPUTS
    data/processed/embeddings/descargas_final.csv
    data/processed/embeddings/embeddings_unidos_anonimizado.parquet
    data/processed/SALE_13082026/indicadores_derivados_dmsp.parquet

OUTPUTS
    data/processed/benchmark_resultados/diagnostico_embeddings_places365_vs_imagenet.csv
    (una fila por ventana x embedding comparado: R2 de cada uno, diferencia,
    IC95% bootstrap y rango de la diferencia entre semillas)

COMO CORRER
    cd src/05_model && python -u diagnostico_embeddings_places365_vs_imagenet.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
from diagnostico_embeddings_dmsp import (
    EMB_COLS, OLA_FOCAL, RANDOM_STATE, cargar_descargas, cargar_dmsp_ola,
    cargar_embeddings_ola, hogares_una_foto, matriz_embedding_por_hogar,
)

VENTANAS = {"original 2011-2013": (2011, 2013), "ampliada 2011-2015": (2011, 2015)}
REFERENCIA = "embedding_places365"
Y_COL = "dmsp_avg_vis"
N_BOOT = 2000
N_SEMILLAS = 5
RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_embeddings_places365_vs_imagenet.csv"


def r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)


def predicciones_oof(matrices: dict, y: np.ndarray, semilla: int) -> dict:
    cv = KFold(n_splits=5, shuffle=True, random_state=semilla)
    return {
        col: cross_val_predict(RidgeCV(alphas=np.logspace(-1, 5, 25)), X, y, cv=cv, n_jobs=-1)
        for col, X in matrices.items()
    }


def main() -> None:
    descargas = cargar_descargas()
    emb = cargar_embeddings_ola(OLA_FOCAL)
    dmsp = cargar_dmsp_ola(OLA_FOCAL)
    d_1foto = hogares_una_foto(descargas, OLA_FOCAL)
    rng = np.random.default_rng(RANDOM_STATE)

    filas = []
    for nombre_ventana, (lo, hi) in VENTANAS.items():
        consecutivos = set(d_1foto[(d_1foto.anio_pano >= lo) & (d_1foto.anio_pano <= hi)].consecutivo)

        # Interseccion: mismos hogares en los cuatro embeddings y con DMSP.
        mats = {col: matriz_embedding_por_hogar(emb, col, consecutivos).merge(
                    dmsp[["consecutivo", Y_COL]], on="consecutivo", how="inner")
                for col in EMB_COLS}
        comunes = set.intersection(*(set(m.consecutivo) for m in mats.values()))
        orden = sorted(comunes)
        matrices = {}
        y = None
        for col, m in mats.items():
            m = m.set_index("consecutivo").loc[orden]
            x_cols = [c for c in m.columns if c != Y_COL]
            matrices[col] = np.nan_to_num(StandardScaler().fit_transform(m[x_cols].values))
            y = m[Y_COL].values if y is None else y
        n = len(orden)
        print(f"\n=== Ventana {nombre_ventana}: n={n} hogares comunes ===")

        preds_por_semilla = [predicciones_oof(matrices, y, RANDOM_STATE + s) for s in range(N_SEMILLAS)]
        preds = preds_por_semilla[0]
        idx_boot = rng.integers(0, n, size=(N_BOOT, n))

        for col in EMB_COLS:
            if col == REFERENCIA:
                continue
            dif = r2(y, preds[REFERENCIA]) - r2(y, preds[col])
            boot = np.array([r2(y[i], preds[REFERENCIA][i]) - r2(y[i], preds[col][i]) for i in idx_boot])
            dif_semillas = [r2(y, p[REFERENCIA]) - r2(y, p[col]) for p in preds_por_semilla]
            filas.append({
                "ventana": nombre_ventana, "n": n, "comparado_con": col,
                "r2_places365": round(r2(y, preds[REFERENCIA]), 4),
                "r2_comparado": round(r2(y, preds[col]), 4),
                "dif_places365_menos_comparado": round(dif, 4),
                "ic95_inf": round(float(np.percentile(boot, 2.5)), 4),
                "ic95_sup": round(float(np.percentile(boot, 97.5)), 4),
                "dif_min_semillas": round(min(dif_semillas), 4),
                "dif_max_semillas": round(max(dif_semillas), 4),
            })
            f = filas[-1]
            print(f"  Places365 - {col:<20s} dif={f['dif_places365_menos_comparado']:+.4f} "
                  f"IC95%=[{f['ic95_inf']:+.4f}, {f['ic95_sup']:+.4f}]  "
                  f"semillas=[{f['dif_min_semillas']:+.4f}, {f['dif_max_semillas']:+.4f}]")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_SALIDA, index=False)
    print(f"\nGuardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
