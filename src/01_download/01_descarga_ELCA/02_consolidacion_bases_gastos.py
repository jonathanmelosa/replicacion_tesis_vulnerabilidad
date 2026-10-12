"""
Encuesta Longitudinal Colombiana (ELCA)
Consolidación y armonización de las bases de datos de gastos – olas 2010, 2013 y 2016.

Autor: Jonathan Melo Sarta
Fecha: Junio 2026

Lógica general
--------------
1. 2010 (U y R): archivos con encabezado en formato largo (una fila por hogar × artículo).
   El indicador de compra se construye combinando dos columnas:
     - Si 'compro' es "Si"/"No" → 1/0 directamente (ítems de compra esporádica).
     - Si 'compro' es NaN → se infiere de 'per_compra': cualquier valor distinto de
       "No compra" y no nulo se trata como compra (1), lo demás como (0).
       (Ítems rutinarios cuya pregunta era de frecuencia, no de Sí/No.)
   U tiene 35 categorías agregadas; R tiene 72 categorías detalladas.
   Solo cuenta con 'consecutivo' como identificador de hogar.

2. 2013 (U y R): archivos ya en formato largo, 72 categorías.
   'compro' = "SI" → 1, "NO" → 0.
   Agrega 'hogar' y 'llave' como identificadores.
   'vr_compra_2013' → se renombra a 'vr_compra'.

3. 2016 (U y R): igual a 2013 con estas diferencias:
   'compro' en UGastos usa 'S???' en lugar de 'SI' (corregido → 'SI').
   UGastos tiene nombres de artículo en MAYÚSCULAS con '-' y '+' en lugar de
   vocales acentuadas (corregidos vía CORRECCIONES_ARTICULOS_2016U).
   Agrega 'hogar_n16' y 'llave_n16' como identificadores de ola 3.
   'vr_compra_2016' → se renombra a 'vr_compra'.

4. Las seis bases procesadas se pivotean a formato ancho (sub-hogar × variables) y
   se concatenan en un único dataset longitudinal; las celdas ausentes quedan NaN.

Unidad de análisis
------------------
Sub-hogar × ola. Cuando un hogar se divide entre olas, cada rama resultante
es una fila independiente identificada por su propia llave/llave_n16.
En 2010 no hay división de hogares, por lo que la unidad es simplemente hogar.

Identificadores incluidos
-------------------------
- 2010 : consecutivo
- 2013 : consecutivo, hogar, llave              (llave = consecutivo + hogar)
- 2016 : consecutivo, llave, hogar_n16, llave_n16

Columnas de variables – 8 grupos por artículo
----------------------------------------------
Por cada artículo {a} se generan hasta 8 columnas:

  gasto_{a}   : 1 si el hogar compró ese artículo en la ola, 0 si no.
                Equivalente al conteo de choque_{tipo} en la base de choques.
  vr_{a}      : valor pagado en la compra (NaN si no compró).
  per_{a}     : periodicidad de compra cuando compró; NaN si no compró o si el
                artículo es de compra esporádica (ver nota más abajo).
                Categorías: Diariamente | Semanalmente | Quincenalmente |
                            Mensualmente | Bimestralmente | Trimestralmente |
                            Semestralmente | Anualmente.
  adq_{a}     : 1 si el hogar obtuvo el artículo SIN comprarlo, 0 si no.
  vr_obt_{a}  : valor estimado del artículo obtenido sin comprar (NaN si no aplica).
  finca_{a}   : 1 si la fuente de obtención fue finca/huerta/negocio propio/mar/río.
  pago_{a}    : 1 si la fuente fue pago en especie.
  regalo_{a}  : 1 si la fuente fue regalo.

  adq_, finca_, pago_ y regalo_ son mutuamente excluyentes entre sí:
  adq_=1 si y solo si al menos una de las tres fuentes es 1.

Artículos sin columna per_ (35)
---------------------------------
La encuesta no recoge periodicidad para bienes duraderos, servicios puntuales
ni gastos de tipo excepcional. Los 35 artículos afectados son:

  anillos_relojes_y_otros_articulos_de_joyeria_artesanias_porcelanas_etc
  colchones_cobijas_manteles_y_ropa_de_cama
  compra_de_bienes_raices_diferentes_a_la_vivienda_que_ocupa_el_hogar
  compra_de_computador_personal
  compra_y_sostenimiento_de_mascotas
  conexion_o_pago_por_uso_de_internet
  cuadros_y_obras_originales_de_arte
  cuotas_extraordinarias_de_administracion_o_comunitarias
  diversiones_y_entretenimiento_espectaculos_discotecas_cines_deportes_etc
  educacion_pagos_de_matriculas_escolares_universitarias_uniformes_diferentes_a_las_reportadas_en_403_m
  gastos_en_educacion
  gastos_en_educacion_pensiones_libros_utiles_fotocopias_etc
  gastos_en_salud_medicamentos_examenes_consultas_etc
  gastos_en_vacaciones_hospedaje_transporte_pasajes_de_avion_otros
  impuesto_a_la_renta_complementarios_y_predial
  impuesto_a_la_renta_y_complementarios
  lavado_y_planchado_de_ropa_fuera_del_hogar
  libros_discos_y_cds
  medias_veladas_para_mujer
  muebles_para_el_hogar_sala_comedor_camas_etc
  nevera_estufa_tv_lavadora_brilladora_horno_y_otros_electrodomesticos_y_gasodomesticos
  ollas_vajillas_cubiertos_y_otros_utensilios_domesticos
  pago_de_hoteles
  pago_de_impuestos_de_vehiculos_de_uso_del_hogar_seguro_obligatorio_soat
  pago_del_servicio_de_celular_compra_de_minutos_o_recargas
  pasajes_de_avion
  polizas_o_seguro_de_salud
  primas_o_polizas_de_seguros
  reparacion_repuestos_y_mantenimiento_de_vehiculo_de_uso_del_hogar
  reparaciones_y_mejoras_de_la_vivienda_plomeria_electricidad_pintura_resane_y_panetes
  salud_medicamentos_examenes_consultas_etc
  salud_terapias_hospitalizaciones_etc_diferentes_a_las_reportadas_en_403_l
  seguro_contra_incendios_o_contra_robo_de_la_vivienda_o_vehiculos_de_uso_del_hogar
  tela_para_vestuario_u_otros_usos
  vehiculo_moto_para_uso_del_hogar

Artículos sin columna vr_obt_ (12)
------------------------------------
Ningún hogar de la muestra reportó haber obtenido estos artículos sin comprarlos,
por lo que pivot_table no generó la columna:

  cuotas_extraordinarias_de_administracion_o_comunitarias
  educacion_pagos_de_matriculas_escolares_universitarias_uniformes_diferentes_a_las_reportadas_en_403_m
  gas_en_pipeta
  gastos_en_educacion_pensiones_libros_utiles_fotocopias_etc
  gastos_en_vacaciones_hospedaje_transporte_pasajes_de_avion_otros
  impuesto_a_la_renta_complementarios_y_predial
  medias_veladas_para_mujer
  pago_del_servicio_de_celular_compra_de_minutos_o_recargas
  polizas_o_seguro_de_salud
  salud_medicamentos_examenes_consultas_etc
  salud_terapias_hospitalizaciones_etc_diferentes_a_las_reportadas_en_403_l
  seguro_contra_incendios_o_contra_robo_de_la_vivienda_o_vehiculos_de_uso_del_hogar

Nota sobre per_ en 2010
------------------------
En 2010, los ítems de compra esporádica usan 'compro'=Si/No y dejan 'per_compra'=NaN.
Solo los ítems de compra rutinaria (alimentos, servicios básicos) tienen frecuencia
registrada en per_compra. Por eso, incluso hogares que compraron ciertos artículos
en 2010 pueden tener per_{a}=NaN para esos artículos.

Armonización de nombres entre olas
------------------------------------
Los nombres de artículos difieren entre olas de tres formas:
(a) Prefijo de letra: 'a. ', 'B. ' → eliminado antes de normalizar.
(b) Codificación: '???', '??', '-', '+' → corregidos por archivo.
(c) Redacción: cambios de ola a ola → cubiertos por ARMONIZACION_ARTICULOS.
Después de (a)-(c) se aplica normalizar_snake que genera snake_case sin acentos.

Totales de resumen incluidos
-----------------------------
  total_items_comprados : número de artículos distintos comprados por el sub-hogar en la ola.
  total_gasto           : suma de vr_{a} para todos los artículos (gasto monetario total).
  total_gasto_obt       : suma de vr_obt_{a} (valor de bienes obtenidos sin comprar).
"""

import os
import re
import unicodedata

from pathlib import Path

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = str(PROJECT_ROOT / "data" / "interim" / "raw")
OUTPUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "gastos_elca_longitudinal.parquet")

# ─── Identificadores por ola ─────────────────────────────────────────────────

IDS_POR_OLA = {
    2010: {"key": "consecutivo",  "links": []},
    2013: {"key": "llave",        "links": ["consecutivo", "hogar"]},
    2016: {"key": "llave_n16",    "links": ["consecutivo", "llave", "hogar_n16"]},
}

# ─── Correcciones de codificación: nivel de palabra (patrones ???) ───────────
# Cubre RGastos 2010, UGastos/RGastos 2013 y RGastos 2016.

PALABRAS_CORRECCIONES_ARTICULO = {
    "???AME":            "ÑAME",
    "???ame":            "ñame",
    "???tiles":          "útiles",
    "ALCOH???LICAS":     "ALCOHÓLICAS",
    "Algod???n":         "Algodón",
    "Art???culos":       "Artículos",
    "Az???car":          "Azúcar",
    "Conexi???n":        "Conexión",
    "Educaci???n":       "Educación",
    "F???SFOROS":        "FÓSFOROS",
    "F???sforos":        "Fósforos",
    "Fr???jol":          "Fríjol",
    "JAM???N":           "JAMÓN",
    "LIM???N":           "LIMÓN",
    "P???lizas":         "Pólizas",
    "PERI???DICOS":      "PERIÓDICOS",
    "PI???A":            "PIÑA",
    "Peri???dicos":      "Periódicos",
    "Pl???tano":         "Plátano",
    "Reparaci???n":      "Reparación",
    "SALCHICH???N":      "SALCHICHÓN",
    "Veh???culo":        "Vehículo",
    "administraci???n":  "administración",
    "alcoh???licas":     "alcohólicas",
    "almoj???banas":     "almojábanas",
    "art???culos":       "artículos",
    "artesan???as":      "artesanías",
    "at???n":            "atún",
    "avi???n":           "avión",
    "botiqu???n":        "botiquín",
    "caf???":            "café",
    "champ???":          "champú",
    "com???n":           "común",
    "d???as":            "días",
    "dom???stico":       "doméstico",
    "dom???sticos":      "domésticos",
    "educaci???n":       "educación",
    "el???ctrica":       "eléctrica",
    "el???ctricos":      "eléctricos",
    "electrodom???sticos": "electrodomésticos",
    "energ???a":         "energía",
    "espect???culos":    "espectáculos",
    "ex???menes":        "exámenes",
    "fr???as":           "frías",
    "fr???jol":          "fríjol",
    "gasodom???sticos":  "gasodomésticos",
    "higi???nicas":      "higiénicas",
    "higi???nico":       "higiénico",
    "jab???n":           "jabón",
    "jam???n":           "jamón",
    "joyer???a":         "joyería",
    "l???cteos":         "lácteos",
    "lim???n":           "limón",
    "loter???as":        "loterías",
    "m???quinas":        "máquinas",
    "ni???a":            "niña",
    "ni???o":            "niño",
    "p???blicos":        "públicos",
    "p???lizas":         "pólizas",
    "pa???ales":         "pañales",
    "pa???etes":         "pañetes",
    "pi???a":            "piña",
    "plomer???a":        "plomería",
    "r???o":             "río",
    "ra???ces":          "raíces",
    "salchich???n":      "salchichón",
    "t???":              "té",
    "tel???fono":        "teléfono",
    "u???as":            "uñas",
    "v???sceras":        "vísceras",
    "veh???culo":        "vehículo",
    "veh???culos":       "vehículos",
}

# ─── Correcciones de codificación: UGastos 2010 (patrones ??) ────────────────

PALABRAS_CORRECCIONES_ARTICULO_2010U = {
    "??ame":             "ñame",
    "ART??CULOS":        "ARTÍCULOS",
    "Algod??n":          "Algodón",
    "Educaci??n":        "Educación",
    "L??CTEOS":          "LÁCTEOS",
    "P??BLICOS":         "PÚBLICOS",
    "PERI??DICOS":       "PERIÓDICOS",
    "Reparaci??n":       "Reparación",
    "Veh??culo":         "Vehículo",
    "aj??":              "ají",
    "art??culos":        "artículos",
    "avi??n":            "avión",
    "az??car":           "azúcar",
    "botiqu??n":         "botiquín",
    "caf??":             "café",
    "champ??":           "champú",
    "dom??stico":        "doméstico",
    "dom??sticos":       "domésticos",
    "el??ctrica":        "eléctrica",
    "el??ctricos":       "eléctricos",
    "electrodom??sticos": "electrodomésticos",
    "energ??a":          "energía",
    "fr??jol":           "fríjol",
    "gasodom??sticos":   "gasodomésticos",
    "higi??nicas":       "higiénicas",
    "higi??nico":        "higiénico",
    "jab??n":            "jabón",
    "jam??n":            "jamón",
    "m??quinas":         "máquinas",
    "ma??z":             "maíz",
    "ni??a":             "niña",
    "ni??o":             "niño",
    "p??lizas":          "pólizas",
    "pa??ales":          "pañales",
    "pa??etes":          "pañetes",
    "parab??lica":       "parabólica",
    "plomer??a":         "plomería",
    "ra??ces":           "raíces",
    "tel??fono":         "teléfono",
    "v??sceras":         "vísceras",
    "veh??culo":         "vehículo",
}

# ─── Correcciones de codificación: artículos MAYÚSCULAS en UGastos 2016 ──────

CORRECCIONES_ARTICULOS_2016U = {
    "A. PAN, AREPAS, BOLLOS, ALMOJ-BANAS":
        "A. PAN, AREPAS, BOLLOS, ALMOJÁBANAS",
    "D. CARNE DE RES, CERDO O CORDERO, HUESO Y V-SCERAS":
        "D. CARNE DE RES, CERDO O CORDERO, HUESO Y VÍSCERAS",
    "D. COMBUSTIBLE PARA VEH-CULO DE USO DEL HOGAR":
        "D. COMBUSTIBLE PARA VEHÍCULO DE USO DEL HOGAR",
    "E. PARQUEO DE VEH-CULO DE USO DEL HOGAR":
        "E. PARQUEO DE VEHÍCULO DE USO DEL HOGAR",
    "F. PESCADO DE R-O O DE MAR U OTROS PRODUCTOS DE MAR FRESCOS  O CONGELADOS":
        "F. PESCADO DE RÍO O DE MAR U OTROS PRODUCTOS DE MAR FRESCOS O CONGELADOS",
    "G. SALCHICHAS, JAM???N, MORTADELA, SALCHICH???N Y OTRAS CARNES FR-AS PREPARADAS":
        "G. SALCHICHAS, JAMÓN, MORTADELA, SALCHICHÓN Y OTRAS CARNES FRÍAS PREPARADAS",
    "H. PAPA COM+N, CRIOLLA, ARRACACHA, YUCA, ???AME":
        "H. PAPA COMÚN, CRIOLLA, ARRACACHA, YUCA, ÑAME",
    "I. APUESTAS Y LOTER-AS":
        "I. APUESTAS Y LOTERÍAS",
    "J. FR-JOL SECO, ARVEJA SECA, LENTEJAS, GARBANZOS Y OTROS GRANOS":
        "J. FRÍJOL SECO, ARVEJA SECA, LENTEJAS, GARBANZOS Y OTROS GRANOS",
    "J. SERVICIOS P+BLICOS (ACUEDUCTO, ALCANTARILLADO Y ASEO, ENERG-A, EL+CTRICA, TEL+FONO FIJO, GAS ETC.)":
        "J. SERVICIOS PÚBLICOS (ACUEDUCTO, ALCANTARILLADO Y ASEO, ENERGÍA, ELÉCTRICA, TELÉFONO FIJO, GAS ETC.)",
    "K. PL-TANO VERDE Y MADURO.":
        "K. PLÁTANO VERDE Y MADURO.",
    "K. SERVICIO DOM+STICO POR D-AS":
        "K. SERVICIO DOMÉSTICO POR DÍAS",
    "L. ARVEJA VERDE, FR-JOL VERDE, HABICHUELA, ZANAHORIA, TOMATE, LECHUGA, PEPINO, CEBOLLA LARGA, CABEZONA Y OTRAS VERDURAS":
        "L. ARVEJA VERDE, FRÍJOL VERDE, HABICHUELA, ZANAHORIA, TOMATE, LECHUGA, PEPINO, CEBOLLA LARGA, CABEZONA Y OTRAS VERDURAS",
    "O. AZ+CAR, SAL, CONDIMENTOS Y SALSAS":
        "O. AZÚCAR, SAL, CONDIMENTOS Y SALSAS",
    "P. PANELA, CAF+, CHOCOLATE, T+":
        "P. PANELA, CAFÉ, CHOCOLATE, TÉ",
    "R. ENLATADOS (ARVEJAS, AT+N, SARDINAS, SALCHICHAS)":
        "R. ENLATADOS (ARVEJAS, ATÚN, SARDINAS, SALCHICHAS)",
}

CORRECCION_SI = {"S???": "SI"}

# ─── Armonización de nombres de artículos entre olas ─────────────────────────
# Se aplica sobre el texto ya: (1) corregido de codificación y (2) despojado del
# prefijo de letra (vía strip_prefijo_articulo).  Los valores son la forma
# canónica sin prefijo a la que también apuntarán los datos de 2013/2016 tras
# strip_prefijo.
#
# Cubre:
#  A) Diferencias de redacción entre 2010 R y 2013/2016.
#  B) Diferencias de redacción entre 2010 U y 2013/2016.
#  C) Diferencias entre 2013 U y 2013 R / 2016 para el mismo artículo.

ARMONIZACION_ARTICULOS = {

    # ── A) 2010 R vs 2013/2016 ─────────────────────────────────────────────

    # "lácteos" en 2010 R, ausente en 2013/2016
    "Leche y derivados lácteos (queso, cuajada, kumis, yogur, crema de leche, mantequilla)":
        "Leche y derivados (queso, cuajada, kumis, yogur, crema de leche, mantequilla)",

    # "huevo" (errata) vs "hueso" en 2013/2016
    "Carne de res, cerdo o cordero, huevo y vísceras":
        "Carne de res, cerdo o cordero, hueso y vísceras",

    # "marinos" y "frescos congelados" vs "de mar" y "frescos o congelados"
    "Pescado de río o de mar u otros productos marinos frescos congelados":
        "Pescado de río o de mar u otros productos de mar frescos o congelados",

    # "papa criolla" explícito y orden distinto
    "Papa común, papa criolla, yuca, arracacha, ñame":
        "Papa común, criolla, arracacha, yuca, ñame",

    # doble "y" en la cebolla vs coma
    "Arveja verde, fríjol verde, habichuela, zanahoria, tomate, lechuga, pepino, cebolla larga y cabezona y otras verduras":
        "Arveja verde, fríjol verde, habichuela, zanahoria, tomate, lechuga, pepino, cebolla larga, cabezona y otras verduras",

    # "Pago de agua..." vs "Agua diferente a la del acueducto"
    "Pago de agua diferente a acueducto (carro tanque, aguatero, agua embotellada)":
        "Agua diferente a la del acueducto (carro tanque, aguatero, agua embotellada)",

    # "de dulce" vs "dulce"
    "Galletas de sal y de dulce":
        "Galletas de sal y dulce",

    # "de por días" vs "por días"
    "Servicio doméstico de por días":
        "Servicio doméstico por días",

    # "y/o manicure" vs "arreglo de uñas"
    "Corte de pelo y/o manicure":
        "Corte de pelo, arreglo de uñas",

    # 2010 R versión corta de servicios públicos
    "Servicios públicos (acueducto, alcantarillado y aseo, energía eléctrica, teléfono, etc.)":
        "Servicios públicos (acueducto, alcantarillado y aseo, energía, eléctrica, teléfono fijo, gas etc.)",

    # 2010 R: sin "de uso" antes de "del hogar"
    "Seguro contra incendios o contra robo de la vivienda o vehículos del hogar":
        "Seguro contra incendios o contra robo de la vivienda o vehículos de uso del hogar",

    # 2010 R: "gasas desinfectantes", sin "condones"; "del botiquín"
    "Algodón, gasas desinfectantes, alcohol, curitas, anticonceptivos, aspirinas y otros elementos del botiquín":
        "Algodón, gasas, alcohol, curitas, anticonceptivos, condones, aspirinas y otros elementos de botiquín",

    # 2010 R: "Gastos en salud" sin "consultas"
    "Gastos en salud (medicamentos, exámenes, etc.)":
        "Gastos en salud (medicamentos, exámenes, consultas, etc.)",

    # 2010 R: "aparatos" extra
    "Nevera, estufa, TV, lavadora, brilladora, horno y otros aparatos electrodomésticos y gasodomésticos":
        "Nevera, estufa, TV, lavadora, brilladora, horno y otros electrodomésticos y gasodomésticos",

    # 2010 R: sin "(compra de minutos o recargas)"
    "Pago del servicio de celular":
        "Pago del servicio de celular (compra de minutos o recargas)",

    # 2010 R: sin "bicitaxis"
    "Pasajes urbanos en bus, buseta, colectivo, ejecutivo, taxi, metro, transmilenio, pasajes intermunicipales":
        "Pasajes urbanos en bus, buseta, colectivo, ejecutivo, taxi, metro, Transmilenio, pasajes intermunicipales, bicitaxis",

    # 2010 R: "de vestuario" vs "vestuario"
    "Reparación de calzado y/o de vestuario":
        "Reparación de calzado y/o vestuario",

    # 2010 R: "resane, pañetes" vs "resane y pañetes"
    "Reparaciones y mejoras de la vivienda (plomería, electricidad, pintura, resane, pañetes)":
        "Reparaciones y mejoras de la vivienda (plomería, electricidad, pintura, resane y pañetes)",

    # ── B) 2010 U vs 2013/2016 ─────────────────────────────────────────────

    # 2010 U: "gasas desinfectantes"; "aspirinas," (coma extra); "de botiquín" (sin l)
    "Algodón, gasas desinfectantes, alcohol, curitas, anticonceptivos, aspirinas, y otros elementos de botiquín":
        "Algodón, gasas, alcohol, curitas, anticonceptivos, condones, aspirinas y otros elementos de botiquín",

    # 2010 U: "pañales" sin "desechables"
    "ARTÍCULOS DE ASEO PERSONAL (crema dental, jabón, champú, papel higiénico, desodorantes, toallas higiénicas, pañales, máquinas y cuchillas de afeitar desechables)":
        "Artículos de aseo personal (crema dental, jabón, champú, papel higiénico, desodorantes, toallas higiénicas, pañales desechables, máquinas y cuchillas de afeitar desechables)",

    # 2010 U: "DE" en vez de "PARA EL" y sin coma en "detergentes desinfectantes"
    "ARTÍCULOS DE ASEO DEL HOGAR (detergentes desinfectantes, escobas, ceras, servilletas, etc.)":
        "Artículos para el aseo del hogar (detergentes, desinfectantes, escobas, ceras, servilletas, etc.)",

    # 2010 U: "T.V." con puntos y coma extra tras "horno,"
    "Nevera, estufa, T.V., lavadora, brilladora, horno, y otros aparatos electrodomésticos y gasodomésticos":
        "Nevera, estufa, TV, lavadora, brilladora, horno y otros electrodomésticos y gasodomésticos",

    # 2010 U: nombre agregado con "PAGO DE" al inicio
    "PAGO DE SERVICIOS PÚBLICOS (acueducto, alcantarillado y aseo, energía eléctrica, teléfono, internet, TV por cable, parabólica, gas natural)":
        "Servicios públicos (acueducto, alcantarillado y aseo, energía, eléctrica, teléfono fijo, gas etc.)",

    # 2010 U: formato diferente de pasajes urbanos
    "PASAJES URBANOS (buseta, colectivo, bus, taxi, metro, transmilenio, pasajes intermunicipales)":
        "Pasajes urbanos en bus, buseta, colectivo, ejecutivo, taxi, metro, Transmilenio, pasajes intermunicipales, bicitaxis",

    # ── B-extra) Más diferencias 2010 U ──────────────────────────────────

    # 2010 U: "manicure" en lugar de "arreglo de uñas"
    "Corte de pelo, manicure":
        "Corte de pelo, arreglo de uñas",

    # 2010 U: forma corta en mayúsculas
    "COMIDAS FUERA DEL HOGAR":
        "Comidas fuera del hogar y alimentos preparados fuera para consumo del hogar",

    # ── Typos en 2010 R ────────────────────────────────────────────────────

    # 2010 R: "hombres" (plural erróneo) vs "hombre" en 2013 y 2010 U
    "Calzado para hombres, mujer, niño y niña":
        "Calzado para hombre, mujer, niño y niña",

    # 2010 R: "respuestos" (typo) vs "repuestos" en 2013
    "Reparación, respuestos y mantenimiento de vehículo de uso del hogar":
        "Reparación, repuestos y mantenimiento de vehículo de uso del hogar",

    # 2010 R: "cerveza, aguardiente" (orden diferente) vs "aguardiente, cerveza" en 2013
    "Bebidas alcohólicas (cerveza, aguardiente, ron, etc.)":
        "Bebidas alcohólicas (aguardiente, cerveza, ron, etc.)",

    # ── C) Diferencias internas 2013 / 2016 ────────────────────────────────

    # 2013 R y 2016: coma en lugar de "o contra"
    "Seguro contra incendios,contra robo de la vivienda o vehículos de uso del hogar":
        "Seguro contra incendios o contra robo de la vivienda o vehículos de uso del hogar",
}


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def strip_prefijo_articulo(texto: object) -> str:
    """Elimina prefijo de letra tipo 'a. ', 'B. ', 'aa. ' y punto final."""
    s = re.sub(r"^[A-Za-z]+\.\s*", "", str(texto).strip())
    return s.rstrip(".")


def normalizar_snake(texto: str) -> str:
    """Convierte texto a snake_case sin acentos (sin tocar prefijos)."""
    nfkd = unicodedata.normalize("NFKD", texto)
    sin_acentos = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", sin_acentos.lower()).strip("_")


def normalizar_articulo(texto: object) -> str:
    """Pipeline completo: strip prefijo → armonizar → snake_case."""
    stripped = strip_prefijo_articulo(texto)
    harmonized = ARMONIZACION_ARTICULOS.get(stripped, stripped)
    return normalizar_snake(harmonized)


def corregir_texto(texto: object, correcciones: dict) -> object:
    """Aplica reemplazos exactos de subcadena."""
    if pd.isna(texto):
        return texto
    texto = str(texto)
    for patron, correcto in correcciones.items():
        texto = texto.replace(patron, correcto)
    return texto


def compro_a_bin_2010(compro, per_compra) -> int:
    """
    Indicador de compra para 2010.

    - 'compro' Si/No: ítems de compra esporádica.
    - Si 'compro' es NaN: 'per_compra' ≠ 'No compra' y no nulo → compró.
    """
    if pd.notna(compro):
        return 1 if str(compro).strip().upper() == "SI" else 0
    if pd.isna(per_compra) or str(per_compra).strip() == "No compra":
        return 0
    return 1


def compro_a_bin(compro) -> int:
    """Indicador de compra para 2013/2016: 'SI' → 1."""
    return 1 if pd.notna(compro) and str(compro).strip().upper() == "SI" else 0


def bin_si(valor) -> int:
    """Convierte 'SI' (cualquier capitalización) → 1; todo lo demás → 0."""
    return 1 if pd.notna(valor) and str(valor).strip().upper() == "SI" else 0


def _pivotar(df, index_col, value_col, prefix, aggfunc, fill_value=None):
    """Pivotea una columna y prefija los nombres de columna resultantes."""
    pt = (
        df.pivot_table(
            index=index_col,
            columns="art_norm",
            values=value_col,
            aggfunc=aggfunc,
            fill_value=fill_value,
        )
        .reset_index()
    )
    pt.columns.name = None
    pt.columns = [index_col] + [f"{prefix}{c}" for c in pt.columns[1:]]
    return pt


# ─── Pivoteo largo → ancho ───────────────────────────────────────────────────

def long_a_wide_2010(df_long: pd.DataFrame, ola: int, zona: str) -> pd.DataFrame:
    """
    Pivotea el formato largo de 2010 a hogar × variables.

    Columnas generadas por artículo:
    - gasto_{a} : 1 si el hogar compró ese artículo, 0 si no.
    - vr_{a}    : valor de compra declarado (NaN si no compró).
    - per_{a}   : periodicidad de compra (NaN si no compró o no informada).
    - adq_{a}   : 1 si lo obtuvo sin comprarlo.
    - vr_obt_{a}: valor estimado de lo obtenido sin comprar.
    - finca_{a} : 1 si la fuente fue finca/huerta/negocio propio.
    - pago_{a}  : 1 si la fuente fue pago en especie.
    - regalo_{a}: 1 si la fuente fue regalo.

    Nota: en 2010 'per_compra' solo registra frecuencia para ítems rutinarios;
    los ítems esporádicos usan 'compro'=Si/No y dejan per_compra=NaN.
    """
    df = df_long.dropna(subset=["articulo"]).copy()

    df["art_norm"]  = df["articulo"].apply(normalizar_articulo)
    df["gasto_bin"] = [
        compro_a_bin_2010(c, p)
        for c, p in zip(df["compro"], df["per_compra"])
    ]
    df["vr_num"]    = pd.to_numeric(df["vr_compra"], errors="coerce")

    # per_compra: "No compra" → NaN (no se compró; el gasto_bin ya lo indica)
    df["per_str"]   = df["per_compra"].where(
        df["per_compra"].notna() & (df["per_compra"] != "No compra")
    )

    df["adq_bin"]   = df["adq_sincompra"].apply(bin_si)
    df["vr_obt_num"]= pd.to_numeric(df["vr_obtenido"], errors="coerce")

    # 2010 usa una sola columna 'donde_obtuvo' con tres posibles valores
    df["finca_bin"] = df["donde_obtuvo"].apply(
        lambda x: 1 if pd.notna(x) and "Finca" in str(x) else 0
    )
    df["pago_bin"]  = df["donde_obtuvo"].apply(
        lambda x: 1 if pd.notna(x) and "especie" in str(x).lower() else 0
    )
    df["regalo_bin"]= df["donde_obtuvo"].apply(
        lambda x: 1 if pd.notna(x) and "Regalo" in str(x) else 0
    )

    idx = "consecutivo"
    parts = [
        _pivotar(df, idx, "gasto_bin",  "gasto_",   "max",   fill_value=0),
        _pivotar(df, idx, "vr_num",     "vr_",      "first"),
        _pivotar(df, idx, "per_str",    "per_",     "first"),
        _pivotar(df, idx, "adq_bin",    "adq_",     "max",   fill_value=0),
        _pivotar(df, idx, "vr_obt_num", "vr_obt_",  "first"),
        _pivotar(df, idx, "finca_bin",  "finca_",   "max",   fill_value=0),
        _pivotar(df, idx, "pago_bin",   "pago_",    "max",   fill_value=0),
        _pivotar(df, idx, "regalo_bin", "regalo_",  "max",   fill_value=0),
    ]

    wide = parts[0]
    for p in parts[1:]:
        wide = wide.merge(p, on=idx, how="left")

    wide["ola"]  = ola
    wide["zona"] = zona
    return wide


def long_a_wide_post2010(
    df: pd.DataFrame,
    ola: int,
    zona: str,
    key_col: str,
    id_extra: list,
) -> pd.DataFrame:
    """
    Pivotea el formato largo de 2013/2016 a sub-hogar × variables.

    Antes de llamar esta función, procesar_2013/procesar_2016 ya normalizan:
    - obtuvo_finca/pago/regalo → "SI"/"No" (uniforme entre olas)
    - adq_sincompra            → "SI"/"NO" (uniforme entre olas)

    Columnas generadas por artículo: gasto_, vr_, per_, adq_, vr_obt_,
    finca_, pago_, regalo_  (misma semántica que en 2010).
    """
    df = df.copy()
    df["art_norm"]   = df["articulo"].apply(normalizar_articulo)
    df["gasto_bin"]  = df["compro"].apply(compro_a_bin)
    df["vr_num"]     = pd.to_numeric(df["vr_compra"], errors="coerce")
    df["per_str"]    = df["per_compra"].where(df["per_compra"].notna())
    df["adq_bin"]    = df["adq_sincompra"].apply(bin_si)
    df["vr_obt_num"] = pd.to_numeric(df["vr_obtenido"], errors="coerce")
    df["finca_bin"]  = df["obtuvo_finca"].apply(bin_si)
    df["pago_bin"]   = df["obtuvo_pago"].apply(bin_si)
    df["regalo_bin"] = df["obtuvo_regalo"].apply(bin_si)

    cols_disponibles = [c for c in id_extra if c in df.columns]
    df_ids = df[[key_col] + cols_disponibles].groupby(key_col, as_index=False).first()

    parts = [
        _pivotar(df, key_col, "gasto_bin",  "gasto_",   "max",   fill_value=0),
        _pivotar(df, key_col, "vr_num",     "vr_",      "first"),
        _pivotar(df, key_col, "per_str",    "per_",     "first"),
        _pivotar(df, key_col, "adq_bin",    "adq_",     "max",   fill_value=0),
        _pivotar(df, key_col, "vr_obt_num", "vr_obt_",  "first"),
        _pivotar(df, key_col, "finca_bin",  "finca_",   "max",   fill_value=0),
        _pivotar(df, key_col, "pago_bin",   "pago_",    "max",   fill_value=0),
        _pivotar(df, key_col, "regalo_bin", "regalo_",  "max",   fill_value=0),
    ]

    wide = parts[0]
    for p in parts[1:]:
        wide = wide.merge(p, on=key_col, how="left")

    wide = wide.merge(df_ids, on=key_col, how="left")
    wide["ola"]  = ola
    wide["zona"] = zona
    return wide


# ─── Procesamiento por ola ────────────────────────────────────────────────────

def procesar_2010(tipo: str) -> pd.DataFrame:
    """
    Procesa UGastos o RGastos de 2010.

    UGastos → correcciones '??' (PALABRAS_CORRECCIONES_ARTICULO_2010U)
    RGastos → correcciones '???' (PALABRAS_CORRECCIONES_ARTICULO)
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2010", f"{tipo}Gastos-csv.tab")

    df = pd.read_csv(path, sep="\t", low_memory=False)

    corr = (
        PALABRAS_CORRECCIONES_ARTICULO_2010U if tipo == "U"
        else PALABRAS_CORRECCIONES_ARTICULO
    )
    df["articulo"] = df["articulo"].apply(lambda x: corregir_texto(x, corr))

    return long_a_wide_2010(df, ola=1, zona=zona)


def procesar_2013(tipo: str) -> pd.DataFrame:
    """
    Procesa UGastos o RGastos de 2013.

    Renombra 'vr_compra_2013' → 'vr_compra'.
    Corrige 'S???' → 'SI' en compro y en las tres columnas obtuvo_*.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2013", f"{tipo}Gastos-csv.tab")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={"vr_compra_2013": "vr_compra"})

    df["articulo"] = df["articulo"].apply(
        lambda x: corregir_texto(x, PALABRAS_CORRECCIONES_ARTICULO)
    )
    for col in ["compro", "obtuvo_finca", "obtuvo_pago", "obtuvo_regalo"]:
        df[col] = df[col].apply(lambda x: corregir_texto(x, CORRECCION_SI))

    return long_a_wide_post2010(
        df, ola=2, zona=zona,
        key_col="llave",
        id_extra=["consecutivo", "hogar"],
    )


def procesar_2016(tipo: str) -> pd.DataFrame:
    """
    Procesa UGastos o RGastos de 2016.

    Renombra 'vr_compra_2016' → 'vr_compra'.
    UGastos: aplica CORRECCIONES_ARTICULOS_2016U; corrige 'S???' → 'SI' en
             compro y adq_sincompra (que usan esa codificación en la zona U).
    RGastos: aplica solo correcciones '???'.
    Ambas zonas: normaliza obtuvo_* de "Aplica"/"No aplica" → "SI"/"No"
                 para uniformizar con la codificación de 2013.
    """
    zona = "Urbano" if tipo == "U" else "Rural"
    path = os.path.join(DATA_ROOT, "elca_2016", f"{tipo}Gastos-csv.tab")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={"vr_compra_2016": "vr_compra"})
    df = df.drop(columns=["zona_2016"], errors="ignore")

    if tipo == "U":
        df["articulo"] = df["articulo"].replace(CORRECCIONES_ARTICULOS_2016U)
        for col in ["compro", "adq_sincompra"]:
            df[col] = df[col].apply(lambda x: corregir_texto(x, CORRECCION_SI))

    df["articulo"] = df["articulo"].apply(
        lambda x: corregir_texto(x, PALABRAS_CORRECCIONES_ARTICULO)
    )

    # obtuvo_* en 2016 usa "Aplica"/"No aplica" en ambas zonas → normalizar a "SI"/"No"
    for col in ["obtuvo_finca", "obtuvo_pago", "obtuvo_regalo"]:
        df[col] = df[col].map({"Aplica": "SI", "No aplica": "No"}).fillna(df[col])

    return long_a_wide_post2010(
        df, ola=3, zona=zona,
        key_col="llave_n16",
        id_extra=["consecutivo", "llave", "hogar_n16"],
    )


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Procesa las seis bases (U y R × 3 olas), las concatena y guarda el resultado.

    Retorna el DataFrame final en formato ancho (hogar × ola).
    """
    frames = []
    for tipo in ["U", "R"]:
        print(f"  Procesando 2010 {tipo}Gastos...")
        frames.append(procesar_2010(tipo))

        print(f"  Procesando 2013 {tipo}Gastos...")
        frames.append(procesar_2013(tipo))

        print(f"  Procesando 2016 {tipo}Gastos...")
        frames.append(procesar_2016(tipo))

    # Concatenar: la unión de columnas llena con NaN las celdas ausentes.
    # .copy() desfragmenta el DataFrame antes de asignar columnas derivadas.
    base_final = pd.concat(frames, axis=0, ignore_index=True).copy()

    # Recopilar columnas por prefijo ANTES de agregar los totales.
    # vr_ debe excluir vr_obt_ (ambos empiezan con "vr_").
    prefijos = ["gasto_", "vr_", "per_", "adq_", "vr_obt_", "finca_", "pago_", "regalo_"]
    cols_por_prefijo = {
        "gasto_":  sorted(c for c in base_final.columns if c.startswith("gasto_")),
        "vr_":     sorted(c for c in base_final.columns if c.startswith("vr_") and not c.startswith("vr_obt_")),
        "per_":    sorted(c for c in base_final.columns if c.startswith("per_")),
        "adq_":    sorted(c for c in base_final.columns if c.startswith("adq_")),
        "vr_obt_": sorted(c for c in base_final.columns if c.startswith("vr_obt_")),
        "finca_":  sorted(c for c in base_final.columns if c.startswith("finca_")),
        "pago_":   sorted(c for c in base_final.columns if c.startswith("pago_")),
        "regalo_": sorted(c for c in base_final.columns if c.startswith("regalo_")),
    }
    g_cols      = cols_por_prefijo["gasto_"]
    v_cols      = cols_por_prefijo["vr_"]
    v_obt_cols  = cols_por_prefijo["vr_obt_"]

    # Totales de resumen
    base_final["total_items_comprados"] = base_final[g_cols].fillna(0).sum(axis=1)
    base_final["total_gasto"]           = base_final[v_cols].fillna(0).sum(axis=1)
    base_final["total_gasto_obt"]       = base_final[v_obt_cols].fillna(0).sum(axis=1)

    # Ordenar columnas: ids → totales → [gasto_ vr_ per_ adq_ vr_obt_ finca_ pago_ regalo_]
    all_id_cols = ["llave_n16", "llave", "consecutivo", "hogar_n16", "hogar", "ola", "zona"]
    id_cols     = [c for c in all_id_cols if c in base_final.columns]
    totales     = ["total_items_comprados", "total_gasto", "total_gasto_obt"]
    var_cols    = [c for p in prefijos for c in cols_por_prefijo[p]]
    base_final  = base_final[id_cols + totales + var_cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    base_final.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    print(f"\nBase final guardada en: {OUTPUT_PATH}")
    print(f"Dimensiones           : {base_final.shape[0]} filas × {base_final.shape[1]} columnas")
    print(f"Olas presentes        : {sorted(base_final['ola'].unique())}")
    print(f"Zonas presentes       : {sorted(base_final['zona'].unique())}")
    print(f"Hogares únicos        : {base_final['consecutivo'].nunique()}")
    for p in prefijos:
        print(f"Columnas {p:<10}: {len(cols_por_prefijo[p])}")

    return base_final


if __name__ == "__main__":
    base_final = main()
