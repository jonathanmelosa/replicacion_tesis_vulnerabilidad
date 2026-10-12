"""
Encuesta Longitudinal Colombiana (ELCA)
Consolidación y armonización de las bases de datos de choques – olas 2010, 2013 y 2016.

Autor: Jonathan Melo Sarta
Fecha: Junio 2026

Lógica general
--------------
1. 2010 (U y R): archivos sin encabezado, formato "ancho" por hogar con bloques
   posicionales que enumeran hasta 7 (U) o 6 (R) choques por fila. Se convierten
   a formato largo, se corrigen caracteres, se armoniza nomenclatura y se pivotea
   a hogar × choque.

2. 2013 (U y R): archivos con encabezado, ya en formato largo hogar × choque.
   Se filtra tuvo_choque == 'SI', el conteo se deriva del número de pares mes/año
   no nulos (mínimo 1) y la respuesta es hizo_princ.

3. 2016 (U y R): igual que 2013 pero el conteo viene en columnas veces_YYYY y se
   suma (mínimo 1). Se corrige 'S???' → 'SI' en UChoques (codificación de 'Sí').

4. Las seis bases procesadas se concatenan en un único dataset hogar–ola cuyas
   columnas son la unión de todas las variables; las celdas vacías quedan como NaN.

Unidad de análisis final: hogar–ola (una fila por cada hogar que participó en esa
ola, haya reportado o no algún choque).

CORRECCIÓN (2026-08-09): antes de este fix, las tres funciones `procesar_*`
descartaban las filas/bloques sin choque reportado ANTES de pivotear, de modo
que un hogar sin ningún choque desaparecía del panel en vez de quedar con una
fila de choque_*=0. Esto reducía la cobertura real (9.854/9.854 hogares en
2010, ~4.653 sub-hogares en 2013, ~4.804 en 2016 según los.tab crudos) a solo
los hogares con al menos un choque. Se verificó contra los
.tab crudos que SÍ traen una fila (2010, formato ancho con celdas vacías) o un
bloque completo de filas con tuvo_choque='No' (2013/2016, formato largo) por
cada hogar, incluidos los que no tuvieron ningún choque -- el archivo fuente
nunca fue el problema. El fix reindexa cada pivote contra el universo completo
de hogares del archivo crudo (antes del filtro `tuvo_choque=='SI'`/`pd.isna`) y
rellena `choque_*`/`total_choques` con 0 para los hogares sin eventos;
`imp_econ_*`/`resp_*` se dejan en NaN para esos hogares porque no aplica
(no hubo choque del que reportar impacto o respuesta de afrontamiento).

Identificadores incluidos
-------------------------
- 2010: consecutivo
- 2013: consecutivo, hogar, llave (llave = consecutivo + hogar)
- 2016: consecutivo, llave, hogar_n16, llave_n16

Columnas de variables
---------------------
- choque_{*}: conteo de ocurrencias de cada tipo de choque
- imp_econ_{*}: impacto económico declarado (solo 2013 y 2016; NaN en 2010)
- resp_{*}: conteo de veces que se usó cada tipo de respuesta
"""

import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = str(PROJECT_ROOT / "data" / "interim" / "raw")
OUTPUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "choques_elca_longitudinal.parquet")

# ─── Estructura de bloques 2010 (índices de columna, base 0) ─────────────────

BLOQUES_2010 = {
    "U": [
        {"choque": 2,  "conteo": 3,  "respuestas": [4, 5, 6, 7]},
        {"choque": 8,  "conteo": 9,  "respuestas": [10, 11, 12]},
        {"choque": 13, "conteo": 14, "respuestas": [15, 16, 17, 18]},
        {"choque": 19, "conteo": 20, "respuestas": [21, 22]},
        {"choque": 23, "conteo": 24, "respuestas": [25, 26]},
        {"choque": 27, "conteo": 28, "respuestas": [29]},
        {"choque": 30, "conteo": 31, "respuestas": [32]},
    ],
    "R": [
        {"choque": 2,  "conteo": 3,  "respuestas": [4, 5, 6]},
        {"choque": 7,  "conteo": 8,  "respuestas": [9, 10]},
        {"choque": 11, "conteo": 12, "respuestas": [13, 14]},
        {"choque": 15, "conteo": 16, "respuestas": [17, 18]},
        {"choque": 19, "conteo": 20, "respuestas": [21]},
        {"choque": 22, "conteo": 23, "respuestas": [24]},
    ],
}

# ─── Identificadores por ola ─────────────────────────────────────────────────
# 'key'   : columna que identifica unívocamente cada sub-hogar en esa ola.
#           Es el índice del pivot y define una fila en el panel final.
# 'links' : columnas que enlazan el sub-hogar con el hogar padre de olas anteriores.
#           Se conservan como atributos de la fila para permitir joins longitudinales.
#
# Jerarquía de identidad entre olas:
#   2010 → consecutivo        (hogar base, sin escisiones)
#   2013 → llave = consecutivo + hogar  (01 = original, 02/03… = hogares escindidos)
#   2016 → llave_n16                    (escisiones adicionales dentro de 2016)
# Enlace entre olas: consecutivo (2010↔2013↔2016) y llave (2013↔2016).

IDS_POR_OLA = {
    2010: {"key": "consecutivo",  "links": []},
    2013: {"key": "llave",        "links": ["consecutivo", "hogar"]},
    2016: {"key": "llave_n16",    "links": ["consecutivo", "llave", "hogar_n16"]},
}

# ─── Correcciones de codificación ─────────────────────────────────────────────
# Los archivos contienen '???' donde deberían aparecer caracteres especiales del
# español (ó, é, ú, í, ñ, etc.).  Los diccionarios cubren todas las olas.

CORRECCIONES_CHOQUES = {
    "Abandono del hogar por parte de un menor de 18 a???os":
        "Abandono del hogar por parte de un menor de 18 años",
    "Abandono del que era jefe del hogar o del c???nyuge":
        "Abandono del que era jefe del hogar o del cónyuge",
    "Accidente de alg???n miembro del hogar que le impidi??? realizar sus actividades cotidianas":
        "Accidente de algún miembro del hogar que le impidió realizar sus actividades cotidianas",
    "Accidente o enfermedad de alg???n miembro del hogar que le impidi??? realizar sus actividades cotidianas":
        "Accidente o enfermedad de algún miembro del hogar que le impidió realizar sus actividades cotidianas",
    "El c???nyuge perdi??? su empleo":
        "El cónyuge perdió su empleo",
    "El jefe del hogar perdi??? su empleo":
        "El jefe del hogar perdió su empleo",
    "Enfermedad de alg???n miembro del hogar que le impidi??? realizar sus actividades cotidianas":
        "Enfermedad de algún miembro del hogar que le impidió realizar sus actividades cotidianas",
    "Fueron v???ctimas de la violencia":
        "Fueron víctimas de la violencia",
    "Muerte de alg???n(os) otro(s) miembro(s) del hogar":
        "Muerte de algún(os) otro(s) miembro(s) del hogar",
    "Muerte del que era jefe del hogar o del c???nyuge":
        "Muerte del que era jefe del hogar o del cónyuge",
    "Otro miembro del hogar perdi??? su empleo":
        "Otro miembro del hogar perdió su empleo",
    "P???rdida de fincas, lotes, terrenos o pedazos de tierra":
        "Pérdida de fincas, lotes, terrenos o pedazos de tierra",
    # variante con coma extra que aparece en 2013 R
    "P???rdida de fincas, lotes, terrenos, o pedazos de tierra":
        "Pérdida de fincas, lotes, terrenos o pedazos de tierra",
    "Pérdida de fincas, lotes, terrenos, o pedazos de tierra":
        "Pérdida de fincas, lotes, terrenos o pedazos de tierra",
    "P???rdida de la vivienda":
        "Pérdida de la vivienda",
    "P???rdida o muerte de animales":
        "Pérdida o muerte de animales",
    "P???rdida o recorte de remesas":
        "Pérdida o recorte de remesas",
    "Plagas o p???rdida de cosechas":
        "Plagas o pérdida de cosechas",
    "Robo, incendio o destrucci???n de bienes del hogar":
        "Robo, incendio o destrucción de bienes del hogar",
    "Robo, incendio o destrucci???n de bienes del hogar (en casa o raponeo)":
        "Robo, incendio o destrucción de bienes del hogar (en casa o raponeo)",
    "Separaci???n de los c???nyuges":
        "Separación de los cónyuges",
    "Sufrieron sequ???as":
        "Sufrieron sequías",
}

CORRECCIONES_RESPUESTAS = {
    "Alg???n miembro del hogar o todos se fueron a vivir con familiares":
        "Algún miembro del hogar o todos se fueron a vivir con familiares",
    "Arrendaron alg???n activo (casa, carro, finca, etc.)":
        "Arrendaron algún activo (casa, carro, finca, etc.)",
    "Hipotecaron alg???n activo (casa, carro, finca, etc.)":
        "Hipotecaron algún activo (casa, carro, finca, etc.)",
    "Hipotecaron o arrendaron alg???n activo (casa, carro, finca, etc.)":
        "Hipotecaron o arrendaron algún activo (casa, carro, finca, etc.)",
    "Pasaron los hijos a un colegio o universidad m???s barata":
        "Pasaron los hijos a un colegio o universidad más barata",
    "Quer???an hacer algo, pero no pudieron por no tener recursos o posibilidades":
        "Querían hacer algo, pero no pudieron por no tener recursos o posibilidades",
    "Se cambiaron a una vivienda m???s econ???mica":
        "Se cambiaron a una vivienda más económica",
    "Uno o m???s miembros del hogar cambiaron de residencia":
        "Uno o más miembros del hogar cambiaron de residencia",
    "Uno o m???s miembros del hogar salieron del pa???s":
        "Uno o más miembros del hogar salieron del país",
    "Usaron alg???n seguro":
        "Usaron algún seguro",
}

# En UChoques 2016, 'Sí' aparece codificado como 'S???'
CORRECCIONES_TUVO_CHOQUE_2016U = {"S???": "SI"}

# ─── Armonización de nombres de choques entre olas ────────────────────────────
# En 2010 'Enfermedad...' y 'Accidente...' son choques separados;
# a partir de 2013 se fusionaron en 'Accidente o enfermedad...'.
# Se mapean los dos nombres de 2010 al nombre canónico de 2013/2016.

ARMONIZACION_CHOQUES = {
    # Enfermedad y Accidente separados en 2010 → nombre unificado de 2013/2016
    "Enfermedad de algún miembro del hogar que le impidió realizar sus actividades cotidianas":
        "Accidente o enfermedad de algún miembro del hogar que le impidió realizar sus actividades cotidianas",
    "Accidente de algún miembro del hogar que le impidió realizar sus actividades cotidianas":
        "Accidente o enfermedad de algún miembro del hogar que le impidió realizar sus actividades cotidianas",

    # Eliminar paréntesis plurales para obtener nombres de columna limpios
    "Muerte de algún(os) otro(s) miembro(s) del hogar":
        "Muerte de algunos otros miembros del hogar",
    "Quiebra(s) y/o cierre(s) del(los) negocio(s) familiar(es)":
        "Quiebras y/o cierres del(los) negocios familiares",

    # Robo/incendio: 2010 y 2013 R usan nombre corto; 2013 U y 2016 usan nombre largo.
    # Se unifica todo bajo el nombre largo (con "en casa o raponeo").
    "Robo, incendio o destrucción de bienes del hogar":
        "Robo, incendio o destrucción de bienes del hogar (en casa o raponeo)",
}


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def normalizar_nombre(nombre: str) -> str:
    """Convierte una cadena al formato snake_case sin acentos para usar como nombre de columna."""
    nfkd = unicodedata.normalize("NFKD", str(nombre))
    sin_acentos = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", sin_acentos.lower()).strip("_")


def raw_2010_a_long(df_raw: pd.DataFrame, bloques: list) -> pd.DataFrame:
    """
    Transforma el archivo crudo de 2010 (sin encabezado, bloques posicionales) a
    formato largo: una fila por (hogar, choque) con hasta 4 columnas de respuesta.

    Parámetros
    ----------
    df_raw  : DataFrame leído con header=None.
    bloques : lista de dicts {'choque': col, 'conteo': col, 'respuestas': [cols]}.
    """
    registros = []
    for _, fila in df_raw.iterrows():
        consecutivo = fila.iloc[1]
        for bloque in bloques:
            choque = fila.iloc[bloque["choque"]]
            if pd.isna(choque):
                continue
            conteo = fila.iloc[bloque["conteo"]]
            respuestas = [
                fila.iloc[col] if pd.notna(fila.iloc[col]) else float("nan")
                for col in bloque["respuestas"]
            ]
            while len(respuestas) < 4:
                respuestas.append(float("nan"))
            registros.append({
                "consecutivo": consecutivo,
                "choque":      choque,
                "conteo":      conteo,
                "respuesta_1": respuestas[0],
                "respuesta_2": respuestas[1],
                "respuesta_3": respuestas[2],
                "respuesta_4": respuestas[3],
            })
    return pd.DataFrame(registros)


def long_a_wide_2010(
    df_long: pd.DataFrame, ola: int, zona: str, hogares_universo: pd.Series
) -> pd.DataFrame:
    """
    Pivotea el formato largo de 2010 a hogar × variables.

    - choque_{nombre} : suma de conteos por hogar y tipo de choque.
    - resp_{nombre}   : cantidad de choques en que se usó cada respuesta.
    - imp_econ no existe en 2010; la columna se añade con NaN al concatenar.

    `hogares_universo` es la lista de TODOS los hogares del archivo crudo
    (incluidos los sin ningún choque) contra la cual se reindexa el pivote de
    conteo, para que esos hogares queden con choque_*=0 en vez de desaparecer.
    """
    df_conteo = (
        df_long
        .assign(choque_norm=lambda d: d["choque"].apply(normalizar_nombre))
        .pivot_table(
            index="consecutivo",
            columns="choque_norm",
            values="conteo",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(hogares_universo, fill_value=0)
        .reset_index()
    )
    df_conteo.columns.name = None
    df_conteo.columns = (
        ["consecutivo"] +
        [f"choque_{c}" for c in df_conteo.columns[1:]]
    )

    df_resp = (
        df_long[["consecutivo", "respuesta_1", "respuesta_2", "respuesta_3", "respuesta_4"]]
        .melt(id_vars="consecutivo", value_name="respuesta")
        .dropna(subset=["respuesta"])[["consecutivo", "respuesta"]]
        .groupby(["consecutivo", "respuesta"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    df_resp.columns.name = None
    df_resp.columns = (
        ["consecutivo"] +
        [f"resp_{normalizar_nombre(c)}" for c in df_resp.columns[1:]]
    )

    wide = df_conteo.merge(df_resp, on="consecutivo", how="left")
    wide["ola"]  = ola
    wide["zona"] = zona
    return wide


def long_a_wide_post2010(
    df: pd.DataFrame,
    ola: int,
    zona: str,
    key_col: str,
    id_extra: list,
    hogares_universo: pd.Series,
    df_ids_universo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivotea el formato largo de 2013/2016 a sub-hogar × variables.

    Parámetros
    ----------
    df       : DataFrame largo con una fila por (sub-hogar, choque), YA
               filtrado a tuvo_choque=='SI' (solo eventos efectivamente
               ocurridos: es la base correcta para conteo/imp_econ/resp).
    ola      : número de ola (2 = 2013, 3 = 2016).
    zona     : 'Urbano' o 'Rural'.
    key_col  : columna que identifica unívocamente cada sub-hogar en esta ola.
               2013 → 'llave', 2016 → 'llave_n16'.
               Cada valor único genera una fila en el panel.
    id_extra : columnas de enlace hacia olas anteriores (ej. consecutivo, hogar).
               Se conservan como atributos pero no determinan la unicidad de la fila.
    hogares_universo : TODOS los valores de key_col del archivo crudo (antes del
               filtro tuvo_choque=='SI'), para reindexar el conteo y no perder
               los hogares sin ningún choque.
    df_ids_universo  : id_extra por key_col calculado sobre el archivo SIN
               filtrar, para que los hogares sin choque tambien conserven sus
               columnas de enlace (consecutivo, hogar, etc.).

    Columnas generadas
    ------------------
    - choque_{nombre}  : suma de conteos por sub-hogar y tipo de choque.
    - imp_econ_{nombre}: impacto económico (primer valor registrado).
    - resp_{nombre}    : cantidad de choques en que se usó cada respuesta.
    """
    df = df.assign(choque_norm=lambda d: d["choque"].apply(normalizar_nombre))

    # Conteo de choques por sub-hogar, reindexado contra el universo completo
    # (incluye hogares sin ningun choque, que quedan en 0 en vez de desaparecer)
    df_conteo = (
        df.pivot_table(
            index=key_col,
            columns="choque_norm",
            values="conteo",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(hogares_universo, fill_value=0)
        .reset_index()
    )
    df_conteo.columns.name = None
    df_conteo.columns = (
        [key_col] +
        [f"choque_{c}" for c in df_conteo.columns[1:]]
    )

    # Impacto económico por choque (categoría: Alta / Media / Baja)
    df_imp = (
        df.pivot_table(
            index=key_col,
            columns="choque_norm",
            values="imp_econ",
            aggfunc="first",
        )
        .reset_index()
    )
    df_imp.columns.name = None
    df_imp.columns = (
        [key_col] +
        [f"imp_econ_{c}" for c in df_imp.columns[1:]]
    )

    # Conteo de respuestas por sub-hogar (una respuesta por choque en 2013/2016)
    df_resp = (
        df[[key_col, "hizo_princ"]]
        .dropna(subset=["hizo_princ"])
        .groupby([key_col, "hizo_princ"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    df_resp.columns.name = None
    df_resp.columns = (
        [key_col] +
        [f"resp_{normalizar_nombre(c)}" for c in df_resp.columns[1:]]
    )

    wide = (
        df_conteo
        .merge(df_ids_universo, on=key_col, how="left")
        .merge(df_imp,  on=key_col, how="left")
        .merge(df_resp, on=key_col, how="left")
    )
    wide["ola"]  = ola
    wide["zona"] = zona
    return wide


# ─── Procesamiento por ola ────────────────────────────────────────────────────

def procesar_2010(tipo: str) -> pd.DataFrame:
    """
    Procesa UChoques o RChoques de 2010.

    Pasos:
    1. Carga sin encabezado.
    2. Convierte bloques posicionales a formato largo.
    3. Corrige codificación de choques y respuestas.
    4. Armoniza nombres de choques al estándar 2013/2016.
    5. Pivotea a formato hogar × variables.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2010", f"{tipo}Choques-csv.tab")

    df_raw = pd.read_csv(path, sep="\t", header=None)
    hogares_universo = df_raw.iloc[:, 1].drop_duplicates()

    df_long = raw_2010_a_long(df_raw, BLOQUES_2010[tipo])

    df_long["choque"] = df_long["choque"].replace(CORRECCIONES_CHOQUES)
    for col in ["respuesta_1", "respuesta_2", "respuesta_3", "respuesta_4"]:
        df_long[col] = df_long[col].replace(CORRECCIONES_RESPUESTAS)

    # Fusionar Enfermedad/Accidente al nombre canónico unificado
    df_long["choque"] = df_long["choque"].replace(ARMONIZACION_CHOQUES)

    return long_a_wide_2010(df_long, ola=1, zona=zona, hogares_universo=hogares_universo)


def procesar_2013(tipo: str) -> pd.DataFrame:
    """
    Procesa UChoques o RChoques de 2013.

    Pasos:
    1. Carga con encabezado (datos ya en formato largo hogar × choque).
    2. Corrige codificación de choques y respuestas.
    3. Filtra filas donde tuvo_choque == 'SI'.
    4. Calcula conteo = número de columnas mes_* no nulas (mínimo 1).
    5. Pivotea a formato hogar × variables, conservando hogar y llave.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2013", f"{tipo}Choques-csv.tab")

    df = pd.read_csv(path, sep="\t", header=0, low_memory=False)

    df["choque"]     = df["choque"].replace(CORRECCIONES_CHOQUES).replace(ARMONIZACION_CHOQUES)
    df["hizo_princ"] = df["hizo_princ"].replace(CORRECCIONES_RESPUESTAS)

    key_col, id_extra = "llave", ["consecutivo", "hogar"]
    hogares_universo = df[key_col].drop_duplicates()
    cols_disponibles = [c for c in id_extra if c in df.columns]
    df_ids_universo = df[[key_col] + cols_disponibles].groupby(key_col, as_index=False).first()

    df = df[df["tuvo_choque"] == "SI"].copy()

    mes_cols = [c for c in df.columns if c.startswith("mes_")]
    df["conteo"] = df[mes_cols].notna().sum(axis=1).clip(lower=1)

    return long_a_wide_post2010(
        df, ola=2, zona=zona, key_col=key_col, id_extra=id_extra,
        hogares_universo=hogares_universo, df_ids_universo=df_ids_universo,
    )


def procesar_2016(tipo: str) -> pd.DataFrame:
    """
    Procesa UChoques o RChoques de 2016.

    Pasos:
    1. Carga con encabezado (datos ya en formato largo hogar × choque).
    2. Corrige codificación de choques, respuestas y tuvo_choque.
       UChoques: 'S???' → 'SI'  (codificación de 'Sí')
    3. Filtra filas donde tuvo_choque == 'SI'.
    4. Calcula conteo = suma de veces_{año} (mínimo 1).
    5. Pivotea a formato hogar × variables, conservando hogar_n16, llave, llave_n16.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2016", f"{tipo}Choques-csv.tab")

    df = pd.read_csv(path, sep="\t", header=0, low_memory=False)

    df["choque"]      = df["choque"].replace(CORRECCIONES_CHOQUES).replace(ARMONIZACION_CHOQUES)
    df["hizo_princ"]  = df["hizo_princ"].replace(CORRECCIONES_RESPUESTAS)
    df["tuvo_choque"] = df["tuvo_choque"].replace(CORRECCIONES_TUVO_CHOQUE_2016U)

    key_col, id_extra = "llave_n16", ["consecutivo", "llave", "hogar_n16"]
    hogares_universo = df[key_col].drop_duplicates()
    cols_disponibles = [c for c in id_extra if c in df.columns]
    df_ids_universo = df[[key_col] + cols_disponibles].groupby(key_col, as_index=False).first()

    df = df[df["tuvo_choque"] == "SI"].copy()

    veces_cols = [c for c in df.columns if c.startswith("veces_")]
    for col in veces_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["conteo"] = df[veces_cols].sum(axis=1).clip(lower=1)

    return long_a_wide_post2010(
        df, ola=3, zona=zona, key_col=key_col, id_extra=id_extra,
        hogares_universo=hogares_universo, df_ids_universo=df_ids_universo,
    )


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Procesa las seis bases (U y R × 3 olas), las concatena y guarda el resultado.

    Retorna el DataFrame final.
    """
    frames = []
    for tipo in ["U", "R"]:
        print(f"  Procesando 2010 {tipo}Choques...")
        frames.append(procesar_2010(tipo))

        print(f"  Procesando 2013 {tipo}Choques...")
        frames.append(procesar_2013(tipo))

        print(f"  Procesando 2016 {tipo}Choques...")
        frames.append(procesar_2016(tipo))

    # Concatenar: la unión de columnas llena con NaN las celdas ausentes
    base_final = pd.concat(frames, axis=0, ignore_index=True)

    # Total de choques sufridos por hogar-ola (NaN en columnas ausentes se trata como 0)
    choque_cols_all = [c for c in base_final.columns if c.startswith("choque_")]
    base_final["total_choques"] = base_final[choque_cols_all].fillna(0).sum(axis=1)

    # Ordenar columnas: identificadores → total_choques → choques → imp_econ → respuestas
    # llave_n16 y llave van primero porque son las claves de fila de 2016 y 2013;
    # consecutivo queda como columna de enlace hacia 2010.
    all_id_cols   = ["llave_n16", "llave", "consecutivo", "hogar_n16", "hogar", "ola", "zona"]
    id_cols       = [c for c in all_id_cols if c in base_final.columns]
    choque_cols   = sorted(c for c in base_final.columns if c.startswith("choque_"))
    imp_cols      = sorted(c for c in base_final.columns if c.startswith("imp_econ_"))
    resp_cols     = sorted(c for c in base_final.columns if c.startswith("resp_"))
    base_final    = base_final[id_cols + ["total_choques"] + choque_cols + imp_cols + resp_cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    base_final.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    print(f"\nBase final guardada en: {OUTPUT_PATH}")
    print(f"Dimensiones         : {base_final.shape[0]} filas × {base_final.shape[1]} columnas")
    print(f"Olas presentes      : {sorted(base_final['ola'].unique())}")
    print(f"Zonas presentes     : {sorted(base_final['zona'].unique())}")
    print(f"Hogares únicos      : {base_final['consecutivo'].nunique()}")
    print(f"Columnas choque     : {len(choque_cols)}")
    print(f"Columnas imp_econ   : {len(imp_cols)}")
    print(f"Columnas resp       : {len(resp_cols)}")

    return base_final


if __name__ == "__main__":
    base_final = main()
