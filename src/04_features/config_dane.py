"""
Parametros oficiales del DANE para pobreza monetaria (ELCA 2010, 2013, 2016).

Unica fuente de verdad para las lineas de pobreza (LP), lineas de pobreza
extrema (LI) y los insumos del deflactor de precios usados en todo el
pipeline de 04_features. Ningun valor de este archivo fue estimado,
interpolado ni tomado de memoria: todos vienen copiados literalmente de los
documentos oficiales guardados en docs/fuentes_dane/ (ver README.md de esa
carpeta para el enlace exacto y el numero de tabla de cada cifra).

Cobertura metodologica
-----------------------
Los tres anos usan la MISMA serie continua del DANE: linea base ENIG
2006-2007, actualizada anualmente por IPC (total de ingresos bajos para LP;
alimentos de ingresos bajos para LI). Esto es lo que hace comparables los
pesos de 2010, 2013 y 2016 entre si. Deliberadamente NO se usan las lineas
recalculadas con la actualizacion metodologica ENPH 2016-2017 (esa serie solo
cubre 2012-2018 y no tiene un valor comparable para 2010).

Dominio geografico
-------------------
La ELCA reporta la columna `zona` como "Urbano" / "Rural" en las tres olas.
Esto coincide exactamente con el dominio "Urbano/Rural" que el DANE reporta
para 2010, y es la misma particion conceptual que el DANE llama
"Cabecera/Resto" en 2013 y "Cabeceras/Centros poblados y rural disperso" en
2016 (cambia el nombre de la categoria, no la definicion). Por eso todas las
filas de LINEAS_POBREZA usan las etiquetas "Urbano"/"Rural" tal como vienen
en la ELCA, para que el merge con `zona` sea directo.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUENTES_DANE_DIR = PROJECT_ROOT / "docs" / "fuentes_dane"

# ─── Lineas de pobreza (LP) y pobreza extrema (LI) ────────────────────────
# Pesos corrientes mensuales por persona. Fuente y tabla exacta por ano:
#
#   2010 -> docs/fuentes_dane/Pobreza_nuevametodologia.pdf (MESEP)
#           Anexo J, Cuadro J1 (p.92), dominios Nacional/Urbano/Rural
#   2013 -> docs/fuentes_dane/bol_pobreza_13.pdf
#           Tabla 3 y Tabla 4 (p.6), dominios Nacional/Cabecera/Resto
#   2016 -> docs/fuentes_dane/bol_pobreza_16.pdf
#           Tabla 1 y Tabla 2 (p.3-4), dominios Total Nacional/Cabeceras/
#           Centros poblados y rural disperso
#
# La columna `zona` usa las etiquetas de la ELCA ("Urbano"/"Rural"); ver
# docstring del modulo para la correspondencia de dominios.

LINEAS_POBREZA = pd.DataFrame(
    [
        {"ola": 1, "ano": 2010, "zona": "Urbano", "lp": 207_005, "li": 87_401},
        {"ola": 1, "ano": 2010, "zona": "Rural",  "lp": 123_502, "li": 71_392},
        {"ola": 2, "ano": 2013, "zona": "Urbano", "lp": 227_367, "li": 95_884},
        {"ola": 2, "ano": 2013, "zona": "Rural",  "lp": 136_192, "li": 77_947},
        {"ola": 3, "ano": 2016, "zona": "Urbano", "lp": 266_043, "li": 119_685},
        {"ola": 3, "ano": 2016, "zona": "Rural",  "lp": 159_543, "li": 97_867},
    ]
)

# Linea de pobreza y de pobreza extrema a nivel nacional (para chequeos de
# consistencia y para derivar el deflactor "IPC ingresos bajos", ver
# build_deflactor_ipc.py). Mismas fuentes que LINEAS_POBREZA.
LP_NACIONAL = {2010: 187_079, 2013: 206_091, 2016: 241_673}
LI_NACIONAL = {2010: 83_581, 2013: 91_698, 2016: 114_692}

OLA_A_ANO = {1: 2010, 2: 2013, 3: 2016}

# ─── Lineas de pobreza ELCO (2019, 2022) -- SERIE METODOLOGICA DISTINTA ──
# NO combinar ni comparar directamente con LINEAS_POBREZA (2010/2013/2016):
# el DANE recalculo las lineas de pobreza a partir de 2019 con una base de
# canasta nueva (Encuesta Nacional de Presupuesto de los Hogares -- ENPH --
# 2016-2017), reemplazando la serie ENIG 2006-2007 (MESEP) que se usa arriba.
# El boletin oficial de 2019 lo dice explicitamente: "Estas cifras no son
# comparables con las cifras de la serie MESEP." Verificado tambien que 2019
# y 2022 SI estan en la misma serie ENPH entre si (sin otro rebasing
# intermedio): el boletin de 2023 (que reporta 2022 en su tabla comparativa)
# cita la misma fuente "líneas base ENPH 2016-2017, actualizadas con el
# deflactor especial de las lineas de pobreza" que el de 2019.
#
# Fuente y tabla exacta por ano:
#   2019 -> docs/fuentes_dane/Boletin-pobreza-monetaria_2019.pdf
#           Tabla 1 (p.4) y Tabla 6 (p.13), dominios Total Nacional/
#           Cabeceras/Centros poblados y rural disperso
#   2022 -> docs/fuentes_dane/bol-PM-2023.pdf (boletin de 2023, reporta 2022
#           en su tabla comparativa 2022-2023)
#           Tabla 1 (p.4) y Tabla 4 (p.pobreza extrema), dominios Total
#           Nacional/Cabeceras/Centros poblados y rural disperso
#
# Dominio: "Cabeceras" -> zona "Urbano", "Centros poblados y rural disperso"
# -> zona "Rural" (misma correspondencia que ya usa LINEAS_POBREZA para
# 2013/2016 -- ver docstring del modulo).
LINEAS_POBREZA_ELCO = pd.DataFrame(
    [
        {"ola": 2019, "ano": 2019, "zona": "Urbano", "lp": 361_574, "li": 146_189},
        {"ola": 2019, "ano": 2019, "zona": "Rural",  "lp": 210_969, "li": 106_924},
        {"ola": 2022, "ano": 2022, "zona": "Urbano", "lp": 440_047, "li": 213_624},
        {"ola": 2022, "ano": 2022, "zona": "Rural",  "lp": 253_150, "li": 149_024},
    ]
)

LP_NACIONAL_ELCO = {2019: 327_674, 2022: 396_864}
LI_NACIONAL_ELCO = {2019: 137_350, 2022: 198_698}

# ─── Linea de pobreza 2016 bajo metodologia ENPH (para transicion 2016->2019) ──
# Necesaria porque LINEAS_POBREZA (arriba) trae el 2016 en metodologia MESEP,
# que el propio DANE dice explicitamente que NO es comparable con la serie ENPH
# de LINEAS_POBREZA_ELCO. Para construir una transicion pobre/no-pobre
# 2016->2019 valida, ambos anos deben estar en la MISMA metodologia -- se
# necesita el 2016 recalculado bajo ENPH, no el de LINEAS_POBREZA.
#
# Fuente: DANE, dataset oficial "Lineas de Pobreza Monetaria y Pobreza
# Monetaria Extrema -- Actualizacion Metodologica. Serie 2012-2018"
# (microdatos.dane.gov.co/index.php/catalog/689, archivo
# lineas_20122018.csv -- copia en
# docs/fuentes_dane/lineas_pobreza_2012_2018_enph/lineas_20122018.csv,
# descargado 2026-08-25). Trae LP/LI mensual por dominio (24 ciudades +
# "Resto Urbano" + "Rural"), sin agregados Urbano/Nacional ponderados.
#
# CONFIRMADO -- dominio RURAL, diciembre 2016 (mes base de la ENPH,
# ano=2016, semestre=2, mes=12 en el csv), corresponde 1:1 a zona="Rural":
LP_2016_ENPH_RURAL = 196_225
LI_2016_ENPH_RURAL = 102_020

# PENDIENTE -- dominio "Urbano" (agregado tipo "Cabeceras"): el csv fuente
# solo trae LP/LI por ciudad individual + "Resto Urbano", sin los pesos
# poblacionales de la GEIH necesarios para agregarlos en un unico valor
# nacional urbano ponderado. No se debe promediar sin ponderar -- eso
# subestimaria el peso de Bogota/las 13 ciudades grandes. Sin resolver.
LP_2016_ENPH_URBANO = None
LI_2016_ENPH_URBANO = None

# ─── Insumo del deflactor IPC total nacional ──────────────────────────────
# Variaciones porcentuales mensuales del IPC total nacional, metodologia
# IPC-08 (vigente desde enero de 2009, sin quiebre entre 2010 y 2016).
# Fuente: docs/fuentes_dane/IPC_Variacion.xls, hoja "VariaNal"
# (https://www.dane.gov.co/files/investigaciones/ipc/dic18/IPC_Variacion.xls).
IPC_VARIACION_XLS = FUENTES_DANE_DIR / "IPC_Variacion.xls"
