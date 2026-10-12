# Fuentes oficiales DANE — pobreza monetaria e IPC

Documentos oficiales descargados para construir las líneas de pobreza (LP), líneas de
pobreza extrema (LI) y los deflactores de precios usados en
`src/04_features/build_deflactor_ipc.py` y `src/04_features/build_pobreza_monetaria.py`.
Todos los valores de pesos usados en el código están tomados literalmente de las tablas
citadas abajo — no se estimó ni interpoló ningún valor de línea de pobreza.

## 1. Líneas de pobreza (LP) y pobreza extrema (LI)

| Año | Documento local | Fuente oficial | Tabla usada |
|---|---|---|---|
| 2010 | `Pobreza_nuevametodologia.pdf` | [MESEP — Misión para el Empalme de las Series de Empleo, Pobreza y Desigualdad](https://www.dane.gov.co/files/noticias/Pobreza_nuevametodologia.pdf) | Anexo J, Cuadro J1 (p. 92): "Valor mensual promedio por persona de la línea de pobreza extrema y pobreza para Colombia 2002-2010", dominios Nacional/Urbano/Rural |
| 2013 | `bol_pobreza_13.pdf` | [DANE — Boletín técnico, Pobreza Monetaria y Multidimensional en Colombia 2013](https://www.dane.gov.co/files/investigaciones/condiciones_vida/pobreza/bol_pobreza_13.pdf) | Tabla 3 y Tabla 4 (p. 6): "Comportamiento de la Línea de Pobreza / Pobreza Extrema 2012-2013", dominios Nacional/Cabecera/Resto |
| 2016 | `bol_pobreza_16.pdf` | [DANE — Boletín técnico, Pobreza Monetaria y Multidimensional en Colombia 2016](https://www.dane.gov.co/files/investigaciones/condiciones_vida/pobreza/bol_pobreza_16.pdf) | Tabla 1 y Tabla 2 (p. 3-4): "Comportamiento de la Línea de Pobreza / Pobreza Extrema 2015-2016", dominios Total Nacional/Cabeceras/Centros poblados y rural disperso |

Las tres tablas citan la misma metodología en su pie de página: **"Fuente: DANE, línea
base ENIG 2006-2007, actualizadas por IPC total de ingresos bajos"** (LP) y **"... por IPC
de alimentos por ingresos bajos"** (LI). Es decir, los tres años están en la misma serie
metodológica continua (pre actualización ENPH 2016-2017), lo que las hace comparables
entre sí — condición necesaria para clasificar pobreza en un panel 2010-2013-2016.

`cp_pobreza_2011.pdf` (comunicado de prensa, cifras 2010-2011) se conservó como respaldo:
confirma el quiebre metodológico "anterior/nueva metodología" y los valores de incidencia
de pobreza 2010 usados como chequeo cruzado, aunque no se usó para tomar ningún valor de
línea de pobreza.

### Dominios: correspondencia con la variable `zona` de la ELCA

La ELCA reporta `zona` como `"Urbano"` / `"Rural"` en las tres olas (ver
`data/processed/hogar_elca_longitudinal_clean.parquet`). Esto coincide exactamente con el
dominio "Urbano/Rural" del DANE en 2010, y se mapea de forma directa a "Cabecera/Resto"
(2013) y "Cabeceras/Centros poblados y rural disperso" (2016), que son la misma partición
conceptual con nombres distintos según el año de publicación.

## 1.b Líneas de pobreza ELCO (2019, 2022) — serie metodológica distinta

| Año | Documento local | Fuente oficial | Tabla usada |
|---|---|---|---|
| 2019 | `Boletin-pobreza-monetaria_2019.pdf` | [DANE — Boletín Técnico, Pobreza Monetaria en Colombia 2019](https://www.dane.gov.co/files/investigaciones/condiciones_vida/pobreza/2019/Boletin-pobreza-monetaria_2019.pdf) | Tabla 1 (p. 4) y Tabla 6 (p. 13): "Comportamiento de la línea de pobreza / pobreza extrema", dominios Total Nacional/Cabeceras/Centros poblados y rural disperso |
| 2022 | `bol-PM-2023.pdf` | [DANE — Boletín técnico, Pobreza Monetaria (PM) Año 2023](https://www.dane.gov.co/files/operaciones/PM/bol-PM-2023.pdf) | Tabla 1 (p. 4) y Tabla 4 (p. 14): tablas comparativas "Años 2022 a 2023" — se toma la columna 2022, no 2023 |

**Este bloque NO es continuación de la tabla de la sección 1.** El boletín de 2019 dice
textualmente: *"Estas cifras no son comparables con las cifras de la serie MESEP."* A partir
de 2019 el DANE recalculó la línea de pobreza usando una canasta base nueva (Encuesta
Nacional de Presupuesto de los Hogares — ENPH — 2016-2017), reemplazando la serie ENIG
2006-2007/MESEP que sustenta la sección 1. Ambas tablas (2019 y 2023, que reporta 2022)
citan la misma fuente — *"líneas base ENPH 2016-2017, actualizadas con el deflactor
especial de las líneas de pobreza"* — lo que confirma que **2019 y 2022 sí están en una
serie interna consistente entre sí**, sin otro rebasing intermedio detectado.

Implicación práctica: 2019 y 2022 se pueden comparar entre sí para clasificar pobreza, pero
**no se debe mezclar esta tabla con `LINEAS_POBREZA` (2010/2013/2016) como si fuera una
serie continua** — son dos definiciones de canasta distintas. Ver `config_dane.py`,
`LINEAS_POBREZA_ELCO`.

Dominio: mismo mapeo que ya usa la sección 1 para 2013/2016 — "Cabeceras" → zona "Urbano",
"Centros poblados y rural disperso" → zona "Rural".

## 1.c Línea de pobreza 2016 bajo metodología ENPH (para transición 2016→2019)

Necesaria porque el 2016 de la sección 1 está en metodología MESEP, incompatible con el
2019/2022 ENPH de la sección 1.b — comparar pobreza 2016→2019 exige que ambos años estén en
la misma metodología.

| Documento local | Fuente oficial | Contenido |
|---|---|---|
| `lineas_pobreza_2012_2018_enph/lineas_20122018.csv` | DANE, catálogo de microdatos — [Líneas de Pobreza Monetaria y Pobreza Monetaria Extrema, Actualización Metodológica, Serie 2012-2018](https://microdatos.dane.gov.co/index.php/catalog/689) (archivo `lineas_20122018.zip`, descargado vía el flujo "Obtener Microdatos" de esa página) | LP/LI mensual, 2012-2018, por dominio: 24 ciudades individuales + "Resto Urbano" + "Rural" |

**Confirmado:** dominio `RURAL`, diciembre 2016 (fila `año=2016;semestre=2;mes=12;dominio=RURAL`
del csv) — corresponde 1:1 a `zona="Rural"` del proyecto, sin necesidad de agregación:
- LP = $196.225, LI = $102.020

Para comparar: el valor MESEP que usa `LINEAS_POBREZA` para 2016/Rural es LP=$159.543,
LI=$97.867 — una diferencia de ~23% en LP, que confirma que efectivamente no son series
intercambiables (ver `config_dane.py`, `LP_2016_ENPH_RURAL`).

**Pendiente:** el dominio `zona="Urbano"` (equivalente al agregado "Cabeceras" que publican
los boletines) no viene directamente en este csv — solo trae ciudades individuales + "Resto
Urbano" por separado, sin los pesos poblacionales de la GEIH necesarios para agregarlos
correctamente (promediar sin ponderar subestimaría el peso de Bogotá y las 13 ciudades
grandes frente a las ciudades pequeñas). Ver `config_dane.py`, `LP_2016_ENPH_URBANO = None`.

## 2. Índice de Precios al Consumidor (IPC)

| Documento local | Fuente oficial | Contenido usado |
|---|---|---|
| `IPC_Variacion.xls` | [DANE — IPC histórico, "Variaciones porcentuales (IPC) 2003-2018 (diciembre)"](https://www.dane.gov.co/files/investigaciones/ipc/dic18/IPC_Variacion.xls) | Hoja `VariaNal`: variación mensual (%) del IPC total nacional, base metodológica IPC-08 (vigente desde enero 2009, canasta actualizada con la Encuesta de Ingresos y Gastos 2006-2007) |

El archivo trae variaciones porcentuales mes a mes, no el índice en niveles. El índice se
reconstruye encadenando esas variaciones desde una base arbitraria (100 en diciembre de
2008, mes de referencia de la metodología IPC-08 vigente en todo el período 2010-2016) —
ver `build_deflactor_ipc.py`. El promedio anual de ese índice encadenado para 2010, 2013 y
2016 es el deflactor **"IPC total nacional"**.

No se encontró un archivo histórico único y descargable con el IPC específico para el
grupo de "ingresos bajos" (el que el DANE usa para actualizar la LP/LI). Como ese vínculo
está documentado explícitamente en los boletines de pobreza ("actualizadas por IPC total
de ingresos bajos"), el deflactor **"IPC ingresos bajos"** se deriva directamente de la
razón entre los valores oficiales de LP nacional de cada año (ver tabla de la sección 1),
que por construcción DANE ya incorporan ese índice. Esto se documenta explícitamente en
`build_deflactor_ipc.py` para que quede claro que es un valor derivado, no una serie de
IPC descargada de forma independiente.

## 3. Notas de alcance

- No se usaron las líneas de pobreza recalculadas con la actualización metodológica ENPH
  2016-2017 (que solo cubre 2012-2018 y no es comparable con 2010).
- Todos los valores están en pesos corrientes mensuales por persona; la comparación de
  pobreza (pobre/no pobre) se hace en términos **nominales** (ingreso nominal del hogar del
  año X vs. LP nominal de ese mismo año y zona), tal como lo hace el DANE oficialmente. Los
  deflactores de esta carpeta solo se usan para construir series de ingreso/gasto en
  términos reales para análisis descriptivo y de tendencia, no para la clasificación de
  pobreza.
- Fecha de descarga de los documentos de la sección 1 y 2: 2026-08-06. Fecha de descarga
  de los documentos de la sección 1.b (2019/2022): 2026-08-25.
- Pendiente: no se ha descargado la serie de IPC con la nueva base (diciembre 2018 = 100)
  que el DANE usa desde enero de 2019 — `IPC_Variacion.xls` (sección 2) solo cubre hasta
  esa fecha. Necesaria solo para series descriptivas en pesos reales de 2019/2022, no para
  la clasificación de pobreza (que es nominal-contra-nominal, ver arriba).
