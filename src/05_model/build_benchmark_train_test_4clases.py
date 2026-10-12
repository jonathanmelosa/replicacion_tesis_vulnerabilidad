"""
build_benchmark_train_test_4clases.py
=======================================

Extension del benchmark binario (`build_benchmark_train_test.py`, que NO
se modifica) a las 4 categorias de la matriz de transicion de pobreza
(nunca pobre / sale / entra / siempre pobre), en vez de restringir la
poblacion a hogares no pobres en la ola base y predecir solo "entra vs.
nunca".

DIFERENCIA CLAVE CON EL BENCHMARK BINARIO
------------------------------------------
El benchmark binario filtra `pobre_ingreso == False` en la ola base
(un hogar ya pobre no puede "entrar") y predice solo si ese hogar no-pobre
termina pobre. Eso dejaba a "sale de la pobreza" y "siempre pobre" FUERA
del ejercicio predictivo por completo -- no mezclados en una clase
negativa (hipotesis inicial descartada), sino ausentes. Esto es
consistente para responder "de los hogares HOY no pobres, quien esta en
riesgo" (enfoque de vulnerabilidad a la pobreza, Chaudhuri et al. 2002),
pero no permite poner a prueba el hallazgo de la Seccion 5.1 de que
"entra" se parece mas a "sale" que a "nunca pobre": ese contraste
requiere que "sale" este en los datos del modelo.

Este script NO filtra por `pobre_ingreso` en la ola base -- usa TODA la
poblacion emparejada 1 a 1, y clasifica el resultado en 4 categorias
mutuamente excluyentes:

    Y_grupo = "nunca_pobre"    si no pobre en base y no pobre en fin
              "entra"          si no pobre en base y pobre en fin
              "sale"           si pobre en base y no pobre en fin
              "siempre_pobre"  si pobre en base y pobre en fin

Logica de clasificacion IDENTICA a `clasificar()` en
`construir_matriz_transicion` (src/04_features/build_pobreza_desagregaciones.py)
-- mismas 4 etiquetas conceptuales que la Seccion 5.1 del paper (aqui en
snake_case, sin tildes, por convencion de nombres de columna). No se
importa esa funcion directamente (evita acoplar 05_model a 04_features
via import de otro directorio, no es el patron que sigue el resto de
`05_model/`) -- se replica la logica, verificada por los conteos
(n=723 "entra" en 2010->2013, igual al reportado en el paper).

Emparejamiento de hogares: identico a `build_benchmark_train_test.py`
-- match 1 a 1 por `consecutivo`, hogares divididos excluidos.

Covariables monetarias de la ola base (cambio de diseno respecto al
binario)
--------------------------------------------------------------------------
En el benchmark binario, `pobre_ingreso`/`pobre_extremo_ingreso`/
`pobre_gasto`/`pobre_extremo_gasto` de la ola base se excluian de las
covariables por ser CONSTANTES dentro de la poblacion filtrada (todos
no-pobres) -- no por fuga (`Y` es de la ola FINAL, ver docstring del
binario). Aqui, al no filtrar la poblacion, esas 4 columnas dejan de ser
constantes y se vuelven covariables legitimas y muy informativas (casi
determinan la particion {sale,siempre} vs. {entra,nunca}) -- se agregan
al bloque de variables monetarias `COLS_MONETARIAS_MODELO_A` (excluidas
en la especificacion B, igual que `ingreso_percapita_hogar_real`/
`brecha_lp_*`), para mantener la misma logica de comparacion "con
informacion monetaria de la ola base" (A) vs. "sin ella" (B).
`lp`/`li`/`concuerdan_ingreso_gasto` (umbrales administrativos, no
caracteristicas del hogar) se mantienen siempre excluidos, igual que en
el binario.

Output
------
Para cada transicion (`2010_2013`, `2013_2016`) y cada especificacion
(`A4` con info monetaria de base, `B4` sin ella): un parquet en
`data/processed/benchmark_train_test/`, con columnas de identidad
(`consecutivo`, `zona`), las covariables de la ola base, y `Y_grupo`
(categorica, 4 niveles) -- el archivo binario original
(`modelo_{A,B}_*.parquet`) NO se toca.

COMO CORRER
-----------
    cd src/05_model && python build_benchmark_train_test_4clases.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSOLIDADO_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_consolidado_elca_longitudinal.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"

# Igual que build_benchmark_train_test.py, MAS las 4 variables de pobreza
# de la ola base (ya no son constantes al no filtrar la poblacion -- ver
# docstring).
COLS_MONETARIAS_MODELO_A = [
    "ingreso_percapita_hogar_real", "gasto_percapita_hogar_real",
    "brecha_lp_ingreso", "brecha_lp_gasto",
    "pobre_ingreso", "pobre_extremo_ingreso", "pobre_gasto", "pobre_extremo_gasto",
]
# Umbrales administrativos / llaves de otras olas -- nunca son covariable,
# en ninguna especificacion (identico criterio al binario).
COLS_EXCLUIR_SIEMPRE = [
    "lp", "li", "concuerdan_ingreso_gasto",
    "ingreso_percapita_hogar", "gasto_percapita_hogar",  # nominales, se usa la version real en Modelo A4
    "llave", "llave_n16",
    "ola",
]
COLS_ID = ["consecutivo", "llave", "llave_n16", "ola", "zona"]

CATEGORIAS_Y_GRUPO = ["nunca_pobre", "entra", "sale", "siempre_pobre"]


def _clasificar(pobre_ini: bool, pobre_fin: bool) -> str:
    """Identico a `clasificar()` de construir_matriz_transicion en
    build_pobreza_desagregaciones.py (no pobre->no pobre = nunca_pobre,
    pobre->pobre = siempre_pobre, pobre->no pobre = sale, no pobre->pobre
    = entra) -- ver docstring del modulo para por que no se importa
    directamente."""
    if not pobre_ini and not pobre_fin:
        return "nunca_pobre"
    if pobre_ini and pobre_fin:
        return "siempre_pobre"
    if pobre_ini and not pobre_fin:
        return "sale"
    return "entra"


def construir_transicion_4clases(consolidado: pd.DataFrame, ola_ini: int, ola_fin: int) -> tuple[pd.DataFrame, dict]:
    ini = consolidado[consolidado["ola"] == ola_ini].copy()
    fin = consolidado[consolidado["ola"] == ola_fin][["consecutivo", "pobre_ingreso"]].dropna(subset=["pobre_ingreso"])

    ini_unicos = ini[~ini["consecutivo"].duplicated(keep=False)]
    fin_unicos = fin[~fin["consecutivo"].duplicated(keep=False)]

    n_excluidos = (
        ini["consecutivo"].nunique() - ini_unicos["consecutivo"].nunique()
        + fin["consecutivo"].nunique() - fin_unicos["consecutivo"].nunique()
    )

    panel = ini_unicos.merge(fin_unicos, on="consecutivo", suffixes=("", "_fin"))
    panel = panel.dropna(subset=["pobre_ingreso", "pobre_ingreso_fin"])

    panel["Y_grupo"] = [
        _clasificar(bool(pi), bool(pf))
        for pi, pf in zip(panel["pobre_ingreso"], panel["pobre_ingreso_fin"])
    ]
    panel["Y_grupo"] = pd.Categorical(panel["Y_grupo"], categories=CATEGORIAS_Y_GRUPO)

    stats = {
        "ola_ini": ola_ini, "ola_fin": ola_fin,
        "n_hogares_ola_ini": len(ini), "n_hogares_ola_fin": len(fin),
        "n_excluidos_por_division": n_excluidos,
        "n_panel_1a1": len(panel),
        "distribucion_Y_grupo": panel["Y_grupo"].value_counts().to_dict(),
    }
    return panel, stats


def separar_covariables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_base = [c for c in panel.columns if c not in ("pobre_ingreso_fin", "Y_grupo")]
    covariables = panel[cols_base].drop(columns=[c for c in COLS_EXCLUIR_SIEMPRE if c in cols_base])

    cols_modelo_b4 = [c for c in covariables.columns if c not in COLS_MONETARIAS_MODELO_A]
    modelo_a4 = covariables.copy()
    modelo_a4["Y_grupo"] = panel["Y_grupo"].values
    modelo_b4 = covariables[cols_modelo_b4].copy()
    modelo_b4["Y_grupo"] = panel["Y_grupo"].values
    return modelo_a4, modelo_b4


def main() -> None:
    consolidado = pd.read_parquet(CONSOLIDADO_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transiciones = {"2010_2013": (1, 2), "2013_2016": (2, 3)}
    for nombre, (ola_ini, ola_fin) in transiciones.items():
        panel, stats = construir_transicion_4clases(consolidado, ola_ini, ola_fin)
        print(f"=== Transicion {nombre} ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        modelo_a4, modelo_b4 = separar_covariables(panel)
        modelo_a4.to_parquet(OUTPUT_DIR / f"modelo_A4_{nombre}.parquet", index=False)
        modelo_b4.to_parquet(OUTPUT_DIR / f"modelo_B4_{nombre}.parquet", index=False)
        print(f"  Modelo A4: {modelo_a4.shape}  Modelo B4: {modelo_b4.shape}")


if __name__ == "__main__":
    main()
