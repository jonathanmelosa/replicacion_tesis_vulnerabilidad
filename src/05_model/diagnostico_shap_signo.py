"""
diagnostico_shap_signo.py
=====================================
Recupera el SIGNO (direccion) de la relacion variable -> SHAP que los
rankings de importancia ya publicados (`diagnostico_shap_importancia*.csv`)
no guardan: esos CSV solo tienen `shap_abs_medio`, el promedio del valor
absoluto, donde el signo se pierde. Sin signo, SHAP dice que una variable
importa pero no en que sentido (mas valor de la variable -> mas o menos
riesgo de entrar en pobreza).

Reconstruye exactamente los mismos modelos y los mismos SHAP values de
los cuatro scripts originales (misma fuente de hiperparametros, mismo
`calcular_shap`), para las 4 combinaciones definicion de pobreza x ventana
temporal, y en vez de descartar el signo con `np.abs` lo resume por
variable:

    shap_abs_medio     el mismo valor del ranking original (sirve de
                       verificacion de reproducibilidad, ver abajo)
    shap_medio         promedio SHAP CON signo (poco informativo por si
                       solo: SHAP suma ~0 sobre la poblacion)
    corr_spearman      correlacion de Spearman entre el valor de la
                       variable y su SHAP, sobre los hogares con dato.
                       > 0: valores altos de la variable empujan la
                       prediccion hacia "entra en pobreza"; < 0: hacia
                       "no entra". Es la lectura de un beeswarm en un
                       numero, monotonica (no supone linealidad).
    n_validos          hogares con dato de la variable

Tipos de fila (columna `tipo`)

    columna              variable numerica/binaria, o columna one-hot /
                         indicador de faltante de los algoritmos con
                         preprocesador clasico (Random Forest, Logistica)
    categorica_nativa    variable categorica de los algoritmos de arbol
                         nativo (XGBoost, HistGB, LightGBM): su signo NO
                         esta definido a nivel de variable (el orden de
                         las categorias es arbitrario), asi que
                         corr_spearman queda vacio
    nivel                un nivel de esa variable categorica
                         (`variable = col=nivel`): corr_spearman entre el
                         indicador (hogar tiene ese nivel) y el SHAP de la
                         columna -- equivalente a la columna one-hot de
                         los algoritmos clasicos. Solo niveles con al
                         menos MIN_N_NIVEL hogares.

Verificacion de reproducibilidad: al terminar cada combinacion se compara
`shap_abs_medio` contra el CSV de importancia original de la misma fuente
(las filas de tipo columna/categorica_nativa); el calculo no tiene
aleatoriedad, asi que debe coincidir salvo por el redondeo a 6 decimales.

La misma cautela de las fuentes originales aplica: las ventanas
2013->2016 estan ajustadas y explicadas sobre la misma muestra
(in-sample) con hiperparametros heredados de 2010->2013; el signo mide
como se apoya el ajuste en cada variable, no un efecto causal.

Reanudable: si RUTA_SALIDA ya tiene una combinacion
(fuente, algoritmo, especificacion), se omite. Para volver a calcular
todo, borrar el archivo.

INPUTS (los mismos de los scripts originales)

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_resultados/registro_modelos_ipm.csv
    data/processed/benchmark_train_test/modelo_{A,B,Aipm,Bipm}_{2010_2013,2013_2016}.parquet

OUTPUT

    data/processed/benchmark_resultados/diagnostico_shap_signo.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_signo.py
    # prueba rapida de una sola combinacion:
    python -u diagnostico_shap_signo.py --fuente Monetaria_2010-2013 --algoritmo XGBoost --espec A
"""

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

import modelo_utils as mu
from algoritmos_suite import algoritmos_presentes_en_registro, resolver_algoritmo
from diagnostico_shap import calcular_shap, entrenar
from diagnostico_shap_2013_2016 import entrenar_sobre_2013_2016

MIN_N_CORR = 30  # hogares con dato para reportar una correlacion
MIN_N_NIVEL = 30  # hogares por nivel para reportar un nivel categorico

R = mu.RESULTADOS_DIR
FUENTES = {
    "Monetaria_2010-2013": {
        "registro": R / "registro_modelos_fbeta2_cv10.csv", "especs": ["A", "B"],
        "entrenar": "test", "csv_original": R / "diagnostico_shap_importancia_ab.csv", "sufijo_espec": "",
    },
    "Monetaria_2013-2016": {
        "registro": R / "registro_modelos_fbeta2_cv10.csv", "especs": ["A", "B"],
        "entrenar": "propia", "csv_original": R / "diagnostico_shap_importancia_2013_2016.csv", "sufijo_espec": "_2013_2016",
    },
    "IPM_2010-2013": {
        "registro": R / "registro_modelos_ipm.csv", "especs": ["Aipm", "Bipm"],
        "entrenar": "test", "csv_original": R / "diagnostico_shap_importancia_ipm.csv", "sufijo_espec": "",
    },
    "IPM_2013-2016": {
        "registro": R / "registro_modelos_ipm.csv", "especs": ["Aipm", "Bipm"],
        "entrenar": "propia", "csv_original": R / "diagnostico_shap_importancia_ipm_2013_2016.csv", "sufijo_espec": "_2013_2016",
    },
}
RUTA_SALIDA = R / "diagnostico_shap_signo.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def _spearman(x: pd.Series, s: pd.Series):
    """Spearman entre x y s sobre los hogares con dato en ambos; devuelve
    (corr, n_validos), con corr vacio si hay pocos datos o alguna serie
    es constante."""
    ok = x.notna() & s.notna()
    n = int(ok.sum())
    if n < MIN_N_CORR or x[ok].nunique() < 2 or s[ok].nunique() < 2:
        return np.nan, n
    return float(x[ok].corr(s[ok], method="spearman")), n


def resumir_signo(fuente: str, algoritmo_raw: str, espec_salida: str, shap_values, nombres, x_shap, x_orig, cat_cols):
    """Una fila por columna del explainer (y una por nivel para las
    categoricas nativas) con |SHAP| medio, SHAP medio y Spearman."""
    info = resolver_algoritmo(algoritmo_raw)
    nombre_algo = info["nombre_bonito"]
    shap_df = pd.DataFrame(np.asarray(shap_values), columns=list(nombres))
    x_df = x_shap.reset_index(drop=True) if isinstance(x_shap, pd.DataFrame) else pd.DataFrame(x_shap, columns=list(nombres))
    x_orig = x_orig.reset_index(drop=True)
    cats_nativas = set(cat_cols) if info["familia"] == "arbol_nativo" else set()

    base = {"fuente": fuente, "algoritmo": nombre_algo, "especificacion": espec_salida}
    filas = []
    for col in shap_df.columns:
        s = shap_df[col]
        fila = {**base, "variable": col, "shap_abs_medio": round(float(s.abs().mean()), 6), "shap_medio": round(float(s.mean()), 6)}
        if col in cats_nativas:
            filas.append({**fila, "tipo": "categorica_nativa", "corr_spearman": np.nan, "n_validos": int(x_orig[col].notna().sum())})
            conteo = x_orig[col].value_counts()
            for nivel in conteo[conteo >= MIN_N_NIVEL].index:
                indicador = (x_orig[col] == nivel).astype(float).where(x_orig[col].notna())
                corr, n = _spearman(indicador, s)
                filas.append({**base, "variable": f"{col}={nivel}", "tipo": "nivel", "shap_abs_medio": np.nan,
                              "shap_medio": np.nan, "corr_spearman": corr, "n_validos": n})
        else:
            corr, n = _spearman(pd.to_numeric(x_df[col], errors="coerce"), s)
            filas.append({**fila, "tipo": "columna", "corr_spearman": corr, "n_validos": n})
    return filas


def verificar_contra_original(cfg: dict, espec_salida: str, nombre_algo: str, filas: list) -> None:
    """Compara shap_abs_medio contra el ranking original (mismo algoritmo,
    especificacion y variable). No hay aleatoriedad, asi que solo debe
    haber diferencia de redondeo."""
    original = pd.read_csv(cfg["csv_original"])
    original = original[(original.algoritmo == nombre_algo) & (original.especificacion == espec_salida)]
    nuevo = pd.DataFrame([f for f in filas if f["tipo"] != "nivel"])
    m = original.merge(nuevo, on="variable", suffixes=("_orig", "_nuevo"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        log(f"  ADVERTENCIA: {(m['_merge'] != 'both').sum()} variables solo en una de las dos corridas.")
        return
    dif = (m["shap_abs_medio_orig"] - m["shap_abs_medio_nuevo"]).abs().max()
    log(f"  Reproducibilidad {'OK' if dif < 1e-5 else 'ADVERTENCIA'}: diferencia maxima |SHAP| vs {cfg['csv_original'].name} = {dif:.2e} ({len(m)} variables)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default=None, help="solo esta fuente (clave de FUENTES)")
    ap.add_argument("--algoritmo", default=None, help="solo este algoritmo (nombre_bonito)")
    ap.add_argument("--espec", default=None, help="solo esta especificacion (A, B, Aipm, Bipm)")
    args = ap.parse_args()

    previo = pd.read_csv(RUTA_SALIDA) if RUTA_SALIDA.exists() else pd.DataFrame()
    hechas = set(zip(previo.fuente, previo.algoritmo, previo.especificacion)) if not previo.empty else set()
    acumulado = [previo] if not previo.empty else []

    for fuente, cfg in FUENTES.items():
        if args.fuente and args.fuente != fuente:
            continue
        registro = pd.read_csv(cfg["registro"])
        for espec in cfg["especs"]:
            if args.espec and args.espec != espec:
                continue
            for algoritmo_raw in algoritmos_presentes_en_registro(cfg["registro"]):
                nombre_algo = resolver_algoritmo(algoritmo_raw)["nombre_bonito"]
                espec_salida = espec + cfg["sufijo_espec"]
                if args.algoritmo and args.algoritmo != nombre_algo:
                    continue
                if not ((registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)).any():
                    log(f"OMITIDO: no hay fila para {algoritmo_raw!r}/{espec!r} en {cfg['registro'].name}")
                    continue
                if (fuente, nombre_algo, espec_salida) in hechas:
                    log(f"YA HECHO: {fuente} / {nombre_algo} / {espec_salida}")
                    continue

                log(f"=== {fuente} -- {nombre_algo} -- Modelo {espec_salida} ===")
                if cfg["entrenar"] == "test":
                    pipe, x_train, x_ev, _, cat_cols = entrenar(algoritmo_raw, espec, registro)
                else:
                    pipe, x_train, x_ev, cat_cols = entrenar_sobre_2013_2016(algoritmo_raw, espec, registro)
                log(f"  Calculando SHAP sobre {x_ev.shape[0]} hogares...")
                shap_values, nombres, x_shap = calcular_shap(algoritmo_raw, pipe, x_train, x_ev, cat_cols)

                filas = resumir_signo(fuente, algoritmo_raw, espec_salida, shap_values, nombres, x_shap, x_ev, cat_cols)
                verificar_contra_original(cfg, espec_salida, nombre_algo, filas)
                acumulado.append(pd.DataFrame(filas))
                pd.concat(acumulado, ignore_index=True).to_csv(RUTA_SALIDA, index=False)

    log(f"FIN. Guardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
