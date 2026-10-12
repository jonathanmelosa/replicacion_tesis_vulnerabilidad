"""
diagnostico_variacion_dmsp_comunidad.py
=======================================
Documenta DE DONDE viene la variacion de la iluminacion nocturna
(DMSP-OLS) entre hogares y la compara con la del resultado (estar en
pobreza en la ola siguiente), para responder la pregunta
"no queda claro de donde viene la variacion que captura la
luz, especialmente la espacial".

Argumento que soporta:
  - La luz de cada hogar es el promedio de un buffer de ~928 m alrededor de
    SU coordenada (prompts/prompt_dmsp_ols_pipeline.md). La ELCA encuesta
    varios hogares vecinos por comunidad (`consecutivo_c`), asi que sus
    buffers casi coinciden: la luz es, en la practica, una variable de
    comunidad.
  - El resultado, en cambio, varia sobre todo ENTRE vecinos de una misma
    comunidad.
  - Por eso la luz solo puede aportar sobre la parte comunitaria del
    riesgo (que ademas la encuesta ya recoge, ver
    `diagnostico_residual_dmsp_riqueza.py`).

Metricas por definicion de pobreza (monetaria / IPM) x ventana
(2010->2013 / 2013->2016) x zona (Total / Urbano / Rural):
  - icc_luz, icc_y: correlacion intraclase por comunidad, estimador ANOVA
    de un factor con tamaño de grupo desbalanceado (n0). Es la fraccion de
    la varianza "verdadera" que se debe a la comunidad, corregida por el
    ruido de las comunidades pequeñas (mediana de 3-4 hogares). Para Y
    (binaria) es el ICC del modelo lineal de probabilidad.
  - share_within_luz, share_within_y_bruto: fraccion de la varianza
    muestral dentro de comunidad, SIN corregir (referencia; para Y esta
    subestimada: con comunidades de 3-4 hogares, el ruido de muestreo infla
    la parte entre comunidades. Por eso la cifra del paper es el ICC).
  - pct_comunidades_luz_identica, rango_luz_p50/p90: entre comunidades con
    2+ hogares, dispersion de la luz dentro de la comunidad.
  - pct_saturado: hogares con stable_lights >= 63 (tope del sensor de 6 bits).
  - auc_luz: AUC-ROC de la luz sola (signo invertido: menos luz = mas
    riesgo), y ap_luz vs prevalencia.

Hogares con `consecutivo_c == 8888888` (comunidad no identificada, ver
`build_comunidades_hogar.py`; 0 en 2010->2013, 600 en 2013->2016 bajo
pobreza monetaria) se EXCLUYEN de las metricas por comunidad (no se sabe
con quien comparten comunidad) y se reporta cuantos son. Las metricas de
zona (saturacion, AUC) usan todos los hogares con luz observada.

Variable de luz: `dmsp_stable_lights` (la misma de
`diagnostico_residual_dmsp_riqueza.py` y de la Figura de correlacion
parcial del paper).

INPUTS

    data/processed/benchmark_train_test/modelo_AgeoDMSP_{2010_2013,2013_2016}.parquet
    data/processed/benchmark_train_test/modelo_AipmgeoDMSP_{2010_2013,2013_2016}.parquet

OUTPUT

    data/processed/benchmark_resultados/diagnostico_variacion_dmsp_comunidad.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_variacion_dmsp_comunidad.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import modelo_utils as mu

VARIABLE_LUZ = "dmsp_stable_lights"
SENTINEL_COMUNIDAD = 8888888
TOPE_DMSP = 63

ARCHIVOS = [
    ("monetaria", "2010_2013", "modelo_AgeoDMSP_2010_2013.parquet"),
    ("monetaria", "2013_2016", "modelo_AgeoDMSP_2013_2016.parquet"),
    ("ipm", "2010_2013", "modelo_AipmgeoDMSP_2010_2013.parquet"),
    ("ipm", "2013_2016", "modelo_AipmgeoDMSP_2013_2016.parquet"),
]


def icc_anova(valores: pd.Series, grupos: pd.Series) -> float:
    """ICC(1) por ANOVA de un factor, con tamaño de grupo efectivo n0 para
    grupos desbalanceados: (MSB - MSW) / (MSB + (n0 - 1) * MSW)."""
    g = valores.groupby(grupos)
    k = g.size()
    n_total, n_grupos = len(valores), len(k)
    n0 = (n_total - (k ** 2).sum() / n_total) / (n_grupos - 1)
    media = valores.mean()
    msb = (k * (g.mean() - media) ** 2).sum() / (n_grupos - 1)
    msw = ((valores - g.transform("mean")) ** 2).sum() / (n_total - n_grupos)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def share_within(valores: pd.Series, grupos: pd.Series) -> float:
    """Fraccion de la varianza muestral que es desviacion respecto de la
    media de la comunidad (sin corregir por tamaño de grupo)."""
    desvio = valores - valores.groupby(grupos).transform("mean")
    return float(desvio.var() / valores.var())


def metricas(df: pd.DataFrame) -> dict:
    luz, y = df[VARIABLE_LUZ], df["Y"]
    fila = {
        "n_hogares": len(df),
        "prevalencia": round(y.mean(), 4),
        "pct_saturado": round((luz >= TOPE_DMSP).mean(), 4),
        "auc_luz": round(roc_auc_score(y, -luz), 4),
        "ap_luz": round(average_precision_score(y, -luz), 4),
    }

    con_com = df[df["consecutivo_c"] != SENTINEL_COMUNIDAD]
    grupos = con_com["consecutivo_c"]
    tam = grupos.value_counts()
    rango = con_com[tam.reindex(grupos).values >= 2].groupby("consecutivo_c")[VARIABLE_LUZ].agg(lambda s: s.max() - s.min())
    fila.update({
        "n_sin_comunidad": int(len(df) - len(con_com)),
        "n_comunidades": int(grupos.nunique()),
        "hogares_por_comunidad_mediana": float(tam.median()),
        "icc_luz": round(icc_anova(con_com[VARIABLE_LUZ], grupos), 4),
        "icc_y": round(icc_anova(con_com["Y"].astype(float), grupos), 4),
        "share_within_luz": round(share_within(con_com[VARIABLE_LUZ], grupos), 4),
        "share_within_y_bruto": round(share_within(con_com["Y"].astype(float), grupos), 4),
        "n_comunidades_2mas": int(len(rango)),
        "pct_comunidades_luz_identica": round((rango == 0).mean(), 4),
        "rango_luz_p50": round(rango.median(), 2),
        "rango_luz_p90": round(rango.quantile(0.9), 2),
    })
    return fila


def main() -> None:
    filas = []
    for definicion, ventana, archivo in ARCHIVOS:
        df = pd.read_parquet(mu.DATA_DIR / archivo)
        n_sin_luz = int(df[VARIABLE_LUZ].isna().sum())
        df = df[df[VARIABLE_LUZ].notna()]
        print(f"\n=== {definicion} {ventana}: {len(df)} hogares con luz ({n_sin_luz} sin luz excluidos) ===")
        for zona, sub in [("Total", df)] + list(df.groupby("zona")):
            fila = {"definicion": definicion, "ventana": ventana, "zona": zona, "n_sin_luz_excluidos": n_sin_luz}
            fila.update(metricas(sub))
            filas.append(fila)
            print(f"  {zona:7s} n={fila['n_hogares']:5d} ICC luz={fila['icc_luz']:.3f} ICC Y={fila['icc_y']:.3f} "
                  f"luz identica={fila['pct_comunidades_luz_identica']:.2f} saturado={fila['pct_saturado']:.2f} "
                  f"AUC luz={fila['auc_luz']:.3f} AP luz={fila['ap_luz']:.3f} (prev {fila['prevalencia']:.3f}) "
                  f"sin comunidad={fila['n_sin_comunidad']}")

    out = pd.DataFrame(filas)
    out_path = mu.RESULTADOS_DIR / "diagnostico_variacion_dmsp_comunidad.csv"
    out.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
