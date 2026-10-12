"""
diagnostico_shap_nucleo_perfil.py
=====================================
Delimita el "nucleo" de variables que sostiene el poder predictivo de los
5 algoritmos (Seccion~5.3 de `main.tex`, Hallazgo Primero) -- tras notar que comparar cual variable "domina" en
Random Forest vs. Logistica (enfoque de la version anterior de esta
seccion) es menos informativo que preguntar cual es el PERFIL de
variables que se repite de forma robusta entre los 5 algoritmos, siguiendo
la misma logica de "perfil" que ya usa la Seccion~\\ref{subsec:caracterizacion_grupos}
(nucleo de 53 variables robustas por bootstrap de eta^2/Cramer's V).

Post-procesa `diagnostico_shap_importancia_ab.csv` (NO recalcula SHAP) y
reutiliza la agregacion one-hot->variable original de
`diagnostico_shap_variables_necesarias.py` (logica duplicada
deliberadamente, script pequeno y autocontenido, ver ese modulo para el
razonamiento completo de por que hay que agregar antes de comparar).

Definicion de "nucleo" (umbral de mayoria, mismo espiritu que el umbral
de robustez del perfil univariado de la Seccion 4.2, pero aplicado aqui a
consenso entre algoritmos en vez de bootstrap de muestra)
---------------------------------------------------------------------
Para cada una de las 10 combinaciones algoritmo/especificacion, se marca
que variables (ya agregadas a nivel de variable original) estan dentro
del conjunto que acumula el 80% del SHAP total de esa combinacion (mismo
umbral de `diagnostico_shap_variables_necesarias.py`). Una variable entra
al nucleo si aparece en ese conjunto en AL MENOS 5 de las 10
combinaciones (mayoria simple) -- da 22 variables, un tamano manejable
para una tabla de perfil categorizada.

INPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia_ab.csv
    data/processed/benchmark_train_test/modelo_A_2010_2013.parquet
    src/etiquetas_variables.py (`CATEGORIA`: categoria tematica de las
    variables que la tienen asignada; las demas se categorizan a mano, ver
    CATEGORIA_MANUAL)

Cambio 2026-09-30: la categoria ya no se lee de la tabla del perfil de 53 variables
(`perfil_completo_monetaria_2010_2013.csv`), sino del diccionario
`CATEGORIA` del que esa tabla la tomaba, y se elimina la columna
`en_perfil_53`. La lista del nucleo y sus categorias no cambian.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_nucleo_perfil.csv
    (columnas: variable, categoria, n_combinaciones,
    shap_promedio_normalizado)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_nucleo_perfil.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelo_utils as mu
from etiquetas_variables import CATEGORIA

RUTA_SHAP = mu.PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_shap_importancia_ab.csv"
RUTA_DATOS_A = mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_A_2010_2013.parquet"
RUTA_SALIDA = mu.PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_shap_nucleo_perfil.csv"

UMBRAL_MASA_ACUMULADA = 0.80
UMBRAL_COMBINACIONES = 5  # de 10 (5 algoritmos x 2 especificaciones)

# Variables del nucleo sin categoria en `CATEGORIA` (etiquetas_variables.py)
# -- categorizadas a mano usando el mismo esquema de 10 categorias.
# Cambio 2026-09-24: se alinean con la tabla de
# temas del Anexo (src/tabla_temas_variables.py), que agrego temas que el
# esquema de 10 no tenia -- edad_jefe pasa a Composicion del hogar,
# tvip_puntaje_directo_hogar a Desarrollo infantil,
# tasa_control_preventivo_hogar a Salud y tuvo_choque_economico_hogar a
# Choques y afrontamiento.
CATEGORIA_MANUAL = {
    "pct_ninos_madre_viva": "Composición del hogar",
    "pct_ninos_padre_vivo": "Composición del hogar",
    "edad_jefe": "Composición del hogar",
    "grado_educ_jefe": "Educación, empleo y seguridad social",
    "tvip_puntaje_directo_hogar": "Desarrollo infantil (6--9 años)",
    "tuvo_choque_economico_hogar": "Choques y afrontamiento",
    "tasa_control_preventivo_hogar": "Salud",
    "n_desplazados_comunidad": "Comunidad",
}


def _variables_categoricas_originales() -> list:
    df = pd.read_parquet(RUTA_DATOS_A)
    return df.select_dtypes(include="object").columns.tolist()


def _construir_mapa_variable_base(categoricas: list):
    categoricas_por_largo = sorted(categoricas, key=len, reverse=True)

    def variable_base(columna: str) -> str:
        columna = re.sub(r"^(num__|cat__)", "", columna)
        columna = re.sub(r"^missingindicator_", "", columna)
        for c in categoricas_por_largo:
            if columna == c or columna.startswith(c + "_"):
                return c
        return columna

    return variable_base


def main() -> None:
    shap_raw = pd.read_csv(RUTA_SHAP)
    categoricas = _variables_categoricas_originales()
    variable_base = _construir_mapa_variable_base(categoricas)
    shap_raw = shap_raw.copy()
    shap_raw["base"] = shap_raw["variable"].map(variable_base)

    en_80_por_combinacion = {}
    shap_normalizado = {}
    for (algoritmo, espec), g in shap_raw.groupby(["algoritmo", "especificacion"]):
        agregado = g.groupby("base")["shap_abs_medio"].sum().sort_values(ascending=False)
        normalizado = agregado / agregado.sum()
        masa_acumulada = agregado.cumsum() / agregado.sum()
        n_para_umbral = int((masa_acumulada < UMBRAL_MASA_ACUMULADA).sum() + 1)
        conjunto_80 = set(agregado.index[:n_para_umbral])

        clave = f"{algoritmo}_{espec}"
        en_80_por_combinacion[clave] = conjunto_80
        shap_normalizado[clave] = normalizado

    todas_las_variables = set().union(*en_80_por_combinacion.values())
    conteo = pd.Series({
        v: sum(v in conjunto for conjunto in en_80_por_combinacion.values())
        for v in todas_las_variables
    })
    matriz_normalizada = pd.concat(shap_normalizado, axis=1).fillna(0.0)
    promedio_normalizado = matriz_normalizada.mean(axis=1)

    nucleo = conteo[conteo >= UMBRAL_COMBINACIONES].sort_values(ascending=False)

    filas = []
    for variable, n_comb in nucleo.items():
        categoria = CATEGORIA[variable][0] if variable in CATEGORIA else CATEGORIA_MANUAL.get(variable)
        if categoria is None:
            raise ValueError(f"Variable del núcleo sin categoría asignada: {variable} -- agregar a CATEGORIA_MANUAL")
        filas.append({
            "variable": variable,
            "categoria": categoria,
            "n_combinaciones": int(n_comb),
            "shap_promedio_normalizado": round(float(promedio_normalizado.get(variable, 0.0)), 5),
        })

    resultado = pd.DataFrame(filas).sort_values(["categoria", "n_combinaciones"], ascending=[True, False])
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(RUTA_SALIDA, index=False)
    print(resultado.to_string(index=False))
    print(f"\nGuardado: {RUTA_SALIDA} ({len(resultado)} variables en el núcleo, "
          f"umbral >= {UMBRAL_COMBINACIONES}/10 combinaciones)")


if __name__ == "__main__":
    main()
