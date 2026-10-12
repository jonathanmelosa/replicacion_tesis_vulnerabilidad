"""
Version de `generar_predicciones_test_dmsp.py` para el target IPM --
genera y persiste las probabilidades de test (con metadatos por hogar) de
los modelos ya calibrados en `registro_modelos_ipm.csv` (5 algoritmos x
Aipm/AipmgeoDMSP/Bipm/BipmgeoDMSP). Artefacto compartido para el bootstrap
con clustering por comunidad y el analisis de heterogeneidad sobre el
hallazgo de que DMSP-OLS SI aporta bajo IPM en XGBoost y LightGBM (ver
`diagnostico_bootstrap_ipm.py`).

CORREGIDO (2026-09-02): la version anterior tenia una lista
`ALGOS = {...}` hardcodeada con solo 3 de 5 algoritmos (XGBoost,
HistGradientBoosting, Logistica) -- por eso `diagnostico_heterogeneidad_ipm.py`,
que SI deriva sus algoritmos dinamicamente de este parquet
(`pred["algoritmo"].unique()`), nunca vio a Random Forest ni LightGBM: el
hueco no estaba en ese script sino uno mas arriba, en este. Se corrige
igual que `diagnostico_bootstrap_ipm.py` y `diagnostico_shap.py`: los
algoritmos a procesar se DERIVAN de `registro_modelos_ipm.csv`
(`algoritmos_presentes_en_registro`, en `algoritmos_suite.py`), no de una
lista mantenida a mano.

NO vuelve a correr RandomizedSearchCV -- reutiliza `balanceo_elegido` y
`mejores_params` ya encontrados, UN solo fit por combinacion
(semilla=RANDOM_STATE=42).

OUTPUTS

    data/processed/benchmark_resultados/predicciones_test_ipm_Aipm.parquet
    data/processed/benchmark_resultados/predicciones_test_ipm_Bipm.parquet

COMO CORRER

    cd src/05_model && python -u generar_predicciones_test_ipm.py
"""

from typing import Optional

import pandas as pd

import modelo_utils as mu
from algoritmos_suite import (
    algoritmos_presentes_en_registro,
    filtrar_params_modelo,
    preparar_x_y,
    resolver_algoritmo,
)

REGISTRO = mu.RESULTADOS_DIR / "registro_modelos_ipm.csv"
OUT_DIR = mu.RESULTADOS_DIR
PARES = [("Aipm", "AipmgeoDMSP"), ("Bipm", "BipmgeoDMSP")]

METADATA_COLS = [
    "consecutivo", "consecutivo_c", "zona", "brecha_lp_ingreso",
    "estrato_verificado_hogar", "n_servicios_publicos_hogar", "n_bienes_durables_hogar",
]


def entrenar_y_predecir(algoritmo_raw: str, espec: str, registro: pd.DataFrame, metadata_extra: Optional[pd.DataFrame] = None):
    fila = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)].iloc[0]
    balanceo = fila["balanceo_elegido"]
    params = filtrar_params_modelo(fila["hiperparametros"])

    train, test_raw = mu.cargar_datos(espec)
    faltantes = [c for c in METADATA_COLS if c not in test_raw.columns]
    if faltantes:
        assert metadata_extra is not None, f"faltan columnas {faltantes} y no hay metadata_extra para completarlas"
        test_raw = test_raw.merge(metadata_extra[["consecutivo"] + faltantes], on="consecutivo", how="left")

    x_train, y_train, x_test, y_test, _ = preparar_x_y(algoritmo_raw, train, test_raw)
    pipe = resolver_algoritmo(algoritmo_raw)["construir_pipeline_fn"](x_train, y_train, balanceo, mu.RANDOM_STATE)
    if params:
        pipe.set_params(**params)
    pipe.fit(x_train, y_train)
    proba_test = pipe.predict_proba(x_test)[:, 1]

    meta = test_raw[METADATA_COLS].reset_index(drop=True).copy()
    meta["Y"] = y_test.reset_index(drop=True).values
    meta["proba"] = proba_test
    return meta


def main() -> None:
    registro = pd.read_csv(REGISTRO)
    algoritmos_crudos = algoritmos_presentes_en_registro(REGISTRO)
    print(f"Algoritmos detectados en {REGISTRO.name}: {algoritmos_crudos}")
    _, metadata_ref = mu.cargar_datos("Aipm")

    for base, geo in PARES:
        piezas = []
        for algoritmo_raw in algoritmos_crudos:
            nombre = resolver_algoritmo(algoritmo_raw)["nombre_bonito"]
            print(f"=== {nombre} -- {base} ===")
            m_base = entrenar_y_predecir(algoritmo_raw, base, registro, metadata_ref)
            print(f"=== {nombre} -- {geo} ===")
            m_geo = entrenar_y_predecir(algoritmo_raw, geo, registro, metadata_ref)

            assert (m_base["consecutivo"].values == m_geo["consecutivo"].values).all(), "hogares desalineados entre base y geo"
            assert (m_base["Y"].values == m_geo["Y"].values).all(), "Y debe coincidir entre base y geo"

            combinado = m_base.drop(columns=["proba"]).copy()
            combinado["algoritmo"] = nombre
            combinado["proba_base"] = m_base["proba"].values
            combinado["proba_geo"] = m_geo["proba"].values
            piezas.append(combinado)

        out = pd.concat(piezas, ignore_index=True)
        ruta = OUT_DIR / f"predicciones_test_ipm_{base}.parquet"
        out.to_parquet(ruta, index=False)
        print(f"Guardado: {ruta}  ({out.shape})")


if __name__ == "__main__":
    main()
