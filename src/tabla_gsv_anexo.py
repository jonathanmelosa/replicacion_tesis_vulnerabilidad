"""
tabla_gsv_anexo.py
=====================================
Tablas del anexo de Google Street View:
las Conclusiones (agenda de investigacion) resumen el diagnostico
exploratorio de las fotos a nivel de calle, pero ninguna tabla del
documento mostraba sus cifras. No se recalcula nada: solo se leen los
resultados de `diagnostico_embeddings_dmsp.py` y
`diagnostico_embeddings_dmsp_extensiones.py`.

Contenido:
  1. Que tan bien predicen las representaciones (embeddings) de las fotos
     la luz nocturna de su zona, con la ventana de fechas original
     (2011-2013) y la ampliada (2011-2015), y que tan estable es su primer
     componente entre olas.
  2. Cobertura y relacion con la pobreza: luz nocturna media segun si el
     hogar tiene foto; regresion logistica de caer en pobreza sobre la luz
     nocturna y el primer componente de la foto (IC95% por bootstrap); y
     AUC-ROC de un modelo solo con las fotos, con y sin luz nocturna
     (validacion cruzada dentro del conjunto de prueba, porque no hay fotos
     anteriores a 2012 para entrenar en 2010).
  3. Panel A: correlacion entre el primer
     componente de la foto y la luz nocturna de 2013 segun cuantos anios
     despues se tomo la foto (Analisis 1b de `diagnostico_embeddings_dmsp.py`).
     Panel B: diferencia de R2 de Places365 (ResNet-50 entrenada con
     escenas) frente a los otros modelos, con IC95% por bootstrap pareado
     (`diagnostico_embeddings_places365_vs_imagenet.py`).

INPUTS (data/processed/benchmark_resultados/)

    diagnostico_embeddings_dmsp_correlacion.csv
    diagnostico_embeddings_dmsp_ext_ventana_ampliada.csv
    diagnostico_embeddings_dmsp_ext_estabilidad_temporal.csv
    diagnostico_embeddings_dmsp_sesgo_cobertura.csv
    diagnostico_embeddings_dmsp_ext_bootstrap_coefs.csv
    diagnostico_embeddings_dmsp_ext_modelo_c.csv
    diagnostico_embeddings_dmsp_degradacion_desfase.csv
    diagnostico_embeddings_places365_vs_imagenet.csv

OUTPUTS

    paper/tables/tab_gsv_prediccion_luz.tex
    paper/tables/tab_gsv_pobreza.tex
    paper/tables/tab_gsv_desfase_modelos.tex

COMO CORRER

    python src/tabla_gsv_anexo.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"
PREFIJO = "diagnostico_embeddings_dmsp_"

EMBEDDINGS = {
    "embedding_clip": "CLIP",
    "embedding_vgg19": "VGG19",
    "embedding_resnet50": "ResNet50",
    "embedding_places365": "Places365",
}
GRUPOS_COBERTURA = {
    "sin_foto": "Sin foto",
    "fuera_ventana": "Con foto fuera de la ventana de fechas",
    "dentro_ventana": "Con foto dentro de la ventana (muestra usada)",
}


def _leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(RESULTADOS / f"{PREFIJO}{nombre}.csv")


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _envolver(caption: str, label: str, cuerpo: list, columnas: str, nota: str) -> str:
    return aplicar_signo_menos("\n".join([
        r"\begin{table}[H]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{6pt}",
        f"  \\begin{{tabular}}{{{columnas}}}",
        r"    \toprule",
        *cuerpo,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        f"    \\footnotesize \\textit{{Nota:}} {nota} {FUENTE}",
        r"  \end{minipage}",
        r"\end{table}",
    ])) + "\n"


def tabla_prediccion_luz() -> str:
    corr = _leer("correlacion")
    corr = corr[corr["dmsp_var"] == "dmsp_stable_lights"].set_index("embedding")
    ampl = _leer("ext_ventana_ampliada").set_index("embedding")
    estab = _leer("ext_estabilidad_temporal").set_index("embedding")
    n_orig, n_ampl, n_estab = int(corr["n"].iloc[0]), int(ampl["n"].iloc[0]), int(estab["n"].iloc[0])
    cuerpo = [
        r"    \textbf{Modelo de imagen} & \shortstack{\textbf{$R^2$, ventana}\\\textbf{2011--2013}} & "
        r"\shortstack{\textbf{$R^2$, ventana}\\\textbf{2011--2015}} & "
        r"\shortstack{\textbf{Correlación}\\\textbf{entre olas}} \\",
        r"    \midrule",
    ]
    for clave, nombre in EMBEDDINGS.items():
        cuerpo.append(
            f"    {nombre} & {corr.loc[clave, 'r2_oos']:.3f} & {ampl.loc[clave, 'r2_oos']:.3f} & "
            f"{estab.loc[clave, 'r_test_retest_pc1_eje_fijo']:+.3f} \\\\"
        )
    nota = (
        "$R^2$ fuera de muestra de predecir el valor de DMSP-OLS "
        f"de la zona a partir de la representación de la foto ($n={_miles(n_orig)}$ "
        f"hogares con la ventana original y $n={_miles(n_ampl)}$ con la ampliada). "
        "Correlación entre olas: del primer componente de la representación "
        f"para los mismos hogares ($n={n_estab}$), comparable con 0.958 de DMSP-OLS "
        "y 0.600 de los bienes durables."
    )
    return _envolver(
        "Predicción de DMSP-OLS a partir de las fotos de Google Street View "
        "y estabilidad en el tiempo, 2013 (exploratorio).",
        "tab:gsv_prediccion_luz", cuerpo, "lccc", nota,
    )


def tabla_pobreza() -> str:
    cob = _leer("sesgo_cobertura").set_index("grupo")
    coefs = _leer("ext_bootstrap_coefs").set_index("coeficiente")
    mod_c = _leer("ext_modelo_c").iloc[0]
    n_coef = int(coefs["n_muestra"].iloc[0])
    cuerpo = [r"    \textbf{Indicador} & \textbf{Valor} & $\boldsymbol{n}$ \\", r"    \midrule",
              r"    \multicolumn{3}{l}{\textit{DMSP-OLS medio según cobertura de fotos}} \\"]
    for clave, nombre in GRUPOS_COBERTURA.items():
        cuerpo.append(f"    {nombre} & {cob.loc[clave, 'dmsp_avg_vis_media']:.1f} & {_miles(int(cob.loc[clave, 'n']))} \\\\")
    cuerpo += [r"    \addlinespace",
               r"    \multicolumn{3}{l}{\textit{Caer en pobreza: regresión logística conjunta (IC95\%)}} \\"]
    for clave, nombre in [("pc1_places365", "Primer componente de la foto (Places365)"),
                          ("dmsp_avg_vis", "DMSP-OLS")]:
        f = coefs.loc[clave]
        cuerpo.append(
            f"    {nombre} & {f['media_bootstrap']:+.3f} [{f['ci95_low']:+.2f}, {f['ci95_high']:+.2f}] & {n_coef} \\\\"
        )
    cuerpo += [r"    \addlinespace",
               r"    \multicolumn{3}{l}{\textit{Caer en pobreza: AUC-ROC con validación cruzada}} \\",
               f"    Solo fotos & {mod_c['auc_solo_embeddings']:.3f} & {int(mod_c['n'])} \\\\",
               f"    Fotos y DMSP-OLS & {mod_c['auc_embeddings_mas_dmsp']:.3f} & {int(mod_c['n'])} \\\\",
               f"    Cambio (IC95\\%) & {mod_c['delta_auc']:+.3f} [{mod_c['delta_ci95_low']:+.3f}, "
               f"{mod_c['delta_ci95_high']:+.3f}] & {int(mod_c['n'])} \\\\"]
    nota = (
        "Hogares con foto dentro de la ventana de fechas: los que tienen una foto "
        "de Street View tomada cerca de la ola de 2013. No hay fotos anteriores a "
        "2012, de modo que no es posible entrenar con 2010 y evaluar en 2016 como "
        "en el resto del documento: la regresión logística se estima sobre los "
        "hogares del conjunto de prueba con foto, y el AUC-ROC, con validación "
        "cruzada de 5 particiones dentro de esa muestra, con los primeros cinco "
        "componentes de Places365."
    )
    return _envolver(
        "Cobertura de las fotos de Google Street View y su relación con la caída en "
        "pobreza, pobreza monetaria, 2013$\\to$2016 (exploratorio).",
        "tab:gsv_pobreza", cuerpo, "lcc", nota,
    )


BINS_DESFASE = ["0 (mismo anio)", "1-2 anios", "3-5 anios", "6-10 anios", "11+ anios"]
ETIQUETA_BIN = {"0 (mismo anio)": "Mismo año", "1-2 anios": "1--2 años", "3-5 anios": "3--5 años",
                "6-10 anios": "6--10 años", "11+ anios": "11 o más años"}
VENTANA_ETIQUETA = {"original 2011-2013": "2011--2013", "ampliada 2011-2015": "2011--2015"}


def tabla_desfase_modelos() -> str:
    desf = _leer("degradacion_desfase")
    comp = pd.read_csv(RESULTADOS / "diagnostico_embeddings_places365_vs_imagenet.csv")
    cuerpo = [
        r"    \multicolumn{6}{l}{\textit{A. Correlación (valor absoluto) entre la foto y DMSP-OLS de 2013, según el desfase de la foto}} \\",
        r"    \midrule",
        r"    \textbf{Desfase} & $\boldsymbol{n}$ & "
        + " & ".join(rf"\textbf{{{n}}}" for n in EMBEDDINGS.values()) + r" \\",
        r"    \midrule",
    ]
    for b in BINS_DESFASE:
        sub = desf[desf["bin_desfase"] == b].set_index("embedding")
        vals = " & ".join(f"{abs(sub.loc[c, 'r']):.2f}" for c in EMBEDDINGS)
        cuerpo.append(f"    {ETIQUETA_BIN[b]} & {_miles(int(sub['n'].iloc[0]))} & {vals} \\\\")
    cuerpo += [
        r"    \addlinespace",
        r"    \multicolumn{6}{l}{\textit{B. $R^2$ de Places365 menos $R^2$ de cada modelo al predecir DMSP-OLS (IC95\%)}} \\",
        r"    \midrule",
        r"    \textbf{Ventana} & $\boldsymbol{n}$ & \textbf{$R^2$ Places365} & \textbf{CLIP} & \textbf{VGG19} & \textbf{ResNet50} \\",
        r"    \midrule",
    ]
    for v, etiqueta in VENTANA_ETIQUETA.items():
        sub = comp[comp["ventana"] == v].set_index("comparado_con")
        celdas = " & ".join(
            # `{}` tras `\\`: sin el, LaTeX lee `\\[...]` como espacio vertical opcional.
            rf"\shortstack{{{sub.loc[c, 'dif_places365_menos_comparado']:+.3f}\\{{}}"
            rf"[{sub.loc[c, 'ic95_inf']:+.3f}, {sub.loc[c, 'ic95_sup']:+.3f}]}}"
            for c in ["embedding_clip", "embedding_vgg19", "embedding_resnet50"]
        )
        cuerpo.append(
            f"    {etiqueta} & {_miles(int(sub['n'].iloc[0]))} & {sub['r2_places365'].iloc[0]:.3f} & {celdas} \\\\"
        )
        cuerpo.append(r"    \addlinespace")
    nota = (
        "Panel A: correlación de Pearson entre el primer componente de la "
        "representación de la foto (eje común a todos los desfases) y el valor "
        "medio de DMSP-OLS de 2013, para hogares con una sola foto, "
        "según cuántos años después de 2013 se tomó; se reporta el valor "
        "absoluto porque el signo del componente es arbitrario. Un solo "
        "componente no resume igual de bien todos los modelos: con la "
        "representación completa, VGG19 predice DMSP-OLS con un $R^2$ de 0.64 a "
        r"0.71 (Tabla~\ref{tab:gsv_prediccion_luz}). Los grupos de desfase "
        "tienen hogares distintos, por lo que no son una curva de degradación "
        "de una misma muestra. Panel B: mismos hogares y particiones para los "
        r"cuatro modelos; IC95\% por bootstrap pareado de hogares (2{,}000 "
        "remuestreos); la diferencia es positiva con las cinco semillas de "
        "validación cruzada probadas. Places365 y ResNet50 comparten "
        "arquitectura (ResNet-50) y solo difieren en el entrenamiento: escenas "
        "frente a objetos (ImageNet)."
    )
    return _envolver(
        "Relación entre las fotos de Google Street View y DMSP-OLS según el "
        "desfase de la foto y el modelo de imagen, 2013 (exploratorio).",
        "tab:gsv_desfase_modelos", cuerpo, "lcllll", nota,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, tex in {
        "tab_gsv_prediccion_luz.tex": tabla_prediccion_luz(),
        "tab_gsv_pobreza.tex": tabla_pobreza(),
        "tab_gsv_desfase_modelos.tex": tabla_desfase_modelos(),
    }.items():
        (OUTPUT_DIR / nombre).write_text(tex, encoding="utf-8")
        print(f"Tabla exportada: {OUTPUT_DIR / nombre}")


if __name__ == "__main__":
    main()
