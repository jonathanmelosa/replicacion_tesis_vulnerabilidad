"""
Construccion de covariables del modulo Hogar (ELCA 2010, 2013, 2016) para
el modelo benchmark de prediccion de transicion a la pobreza. Mismo
nivel de auditoria que Personas/Comunidades/Niños/Choques -- ver
`03_clean/05_limpieza_corrupcion_hogar.py` para la limpieza de corrupcion (117 columnas -> 0
residual en cerradas, `05_limpieza_corrupcion_hogar.py`), clasificacion
de las 827 columnas, y busqueda de renombrados entre olas (sin hallazgos
de valor).

A diferencia de Personas/Niños, Hogar YA esta a nivel de hogar -- no
requiere agregacion, solo normalizacion de categorias y construccion de
composites tematicos.

Alcance: 129 columnas candidatas. De
estas, 5 (`ing_arriendos`, `ing_intereses_div`, `ing_otros_nrem`,
`ing_pensiones`, `ing_trabajo`) son EXCLUIDAS de este bloque por ser
componentes que YA alimentan `ingreso_total_hogar` en
`build_ingreso_hogar.py` -- incluirlas de nuevo aqui duplicaria la misma
informacion ya capturada en la covariable de ingreso del benchmark.
`vr_gtos_mens_alim`/`vr_gtos_mensuales` (auto-reporte agregado de gasto
mensual) tambien se EXCLUYEN por ser conceptualmente redundantes con
`gasto_percapita_hogar` (que usa el enfoque mas granular de 88 articulos,
ya preferido en `build_gasto_hogar.py`).

Hallazgos de calidad de dato encontrados al verificar antes de construir
--------------------------------------------------------------------------
  - **Bateria `act_*` (activos financieros)**: ola 1 usa CODIGOS NUMERICOS
    ("1"/"2") sin texto, ola 2/3 usan texto ("Sí"/"No", con variantes de
    mayuscula). Verificado cruzando con la proporcion de "No" en ola 2/3
    (~99%) contra la frecuencia de "2" en ola 1 (~99% de los casos no
    nulos): confirma "1"="Sí", "2"="No" (mismo codigo 1=Sí/2=No usado en
    el resto de ELCA). Normalizado antes de construir.
  - **`n_internet`**: mezcla CONTEOS numericos ("0") con texto Sí/No en
    la misma columna -- se colapsa a binario (0 o "No" -> No; >0 o "Sí"
    -> Sí).
  - **`con_quien_1`/`con_quien_2`/`con_quien_3`**: tenian corrupcion
    U+FFFD/"???" residual NO detectada en el escaneo original de
    `05_limpieza_corrupcion_hogar.py` porque su cardinalidad (25/26/26)
    cae justo en el limite del umbral automatico de vocabulario cerrado
    -- corregido directamente en ese script (ver su docstring,
    correccion 2026-08-09).
  - **`credito_financiera`/`credito_cooperativa`/`credito_fna`/
    `otra_financiacion`/`recursos_propios`/`prestamo_familiar`/
    `subsidios`/`dcto_vivienda`**: el nombre sugiere "acceso a credito"
    en general, pero el diccionario (HU55-56, pregunta 20: "¿Cuáles de
    las siguientes fuentes de financiación utilizaron para la COMPRA O
    CONSTRUCCIÓN DE ESTA VIVIENDA?") confirma que es especificamente
    sobre financiacion de VIVIENDA, no credito de consumo general. Se
    documenta con el nombre correcto para no confundir con
    `con_quien_1/2` (que SI es endeudamiento general: bancos, amigos,
    prestamistas, tenderos, etc., verificado con las categorias reales
    de la columna).
  - **`eay_*` vs `ayu_*`**: el diccionario (HU178-179, pregunta 36:
    "Durante los últimos 12 meses, ¿algún miembro de este hogar ENVIÓ
    ayuda...?") confirma que `eay_*` es ayuda ENVIADA por el hogar (a
    otros), NO "ayuda esperada" como sugeriria el nombre -- distinto de
    `ayu_*` (ayuda RECIBIDA por el hogar). Son conceptos complementarios,
    no redundantes: enviar ayuda a otros es señal de capacidad economica
    relativa, recibir ayuda es señal de vulnerabilidad. Se construyen
    ambos.
  - **`uay_*`**: el diccionario (pregunta 35: "Las ayudas en dinero o en
    especie que recibió este hogar, fueron utilizadas para...") confirma
    que es el USO de la ayuda RECIBIDA (condicional a `ayu_*`), no una
    pregunta independiente -- coherente con su cobertura mas baja
    (16%-62%, filtrada a quienes recibieron algun tipo de ayuda).

Variables construidas (nivel hogar, directo de columnas normalizadas)
--------------------------------------------------------------------------
Vivienda:
  tenencia_vivienda_hogar, tipo_vivienda_hogar, material_paredes_hogar,
  material_pisos_hogar, servicio_sanitario_hogar, obtencion_agua_hogar,
  energia_cocinan_hogar, eliminan_basura_hogar (categoricas, directas).
  personas_por_cuarto_hogar, personas_por_dormitorio_hogar (hacinamiento,
  t_personas/t_cuartos_hogar y t_personas/t_cuartos_dormir).
  n_servicios_publicos_hogar (conteo 0-6: acueducto, alcantarillado,
  energia, gas natural, telefono, recoleccion de basura).
  estrato_hogar (auto-reporte temprano) y estrato_verificado_hogar
  (verificado contra recibo de energia, `sp_estrato`) -- AMBAS variables
  se mantienen por pedido explicito, con la limitacion metodologica
  documentada arriba.
  n_hogares_comparte_vivienda_hogar (de `t_hogares`).

Activos/riqueza:
  riqueza_pca_hogar.
  n_bienes_durables_hogar (conteo 0-18 de tipos de bienes duraderos con
  al menos 1 unidad: neveras, lavadoras, television, computadores,
  internet, aire acondicionado, etc.).
  tiene_vehiculo_hogar (automoviles>0 O motocicletas>0).
  tiene_internet_hogar.
  n_activos_financieros_hogar (conteo 0-15 de tipos de activos
  financieros/seguros con Sí: cesantias, dinero, fondos, herencias,
  otros ingresos, polizas, roscas, seguros de cosechas/maquinaria/otros/
  vehiculo/vida/vivienda, venta de inmueble/negocio).

Programas sociales:
  beneficiario_familias_accion_hogar, beneficiario_red_juntos_hogar
  (directas -- los 2 programas de proteccion social mas relevantes
  individualmente).
  n_programas_sociales_hogar (conteo 0-7: familias_accion, red_juntos,
  sena, icbf, prg_adultomayor, prg_tierras, otro_programa).
  beneficiario_algun_programa_hogar (OR de los 7).

Choques/desastres a nivel hogar:
  tuvo_desastre_natural_hogar (OR de avalancha, creciente, hundimiento,
  inundacion, terremoto -- distinto del modulo Choques ya construido,
  que es auto-reportado por categoria en una pregunta separada).

Financiacion de vivienda (compra/construccion, ver hallazgo arriba):
  financio_credito_formal_vivienda_hogar (OR credito_financiera/
  cooperativa/fna).
  financio_recursos_propios_vivienda_hogar, financio_subsidio_vivienda_hogar.
  tiene_escritura_vivienda_hogar (dcto_vivienda = escritura publica
  registrada, el unico documento con pleno valor legal).

Endeudamiento general (con_quien_1/2, distinto de financiacion de vivienda):
  tiene_deuda_hogar, deuda_formal_hogar (banco/cooperativa/ICETEX/fondo
  de empleados), deuda_informal_hogar (amigos/familiares/prestamistas/
  tenderos/casas de empeño).

Ayudas recibidas y enviadas:
  recibio_ayuda_alimentos_hogar, recibio_ayuda_fam_colombia_hogar,
  recibio_ayuda_fam_exterior_hogar, recibio_ayuda_ong_hogar,
  recibio_ayuda_org_internacional_hogar, recibio_ayuda_religiosa_hogar,
  recibio_ayuda_desplazados_hogar, n_tipos_ayuda_recibida_hogar (conteo).
  envio_ayuda_alimentos_hogar, envio_ayuda_fam_colombia_hogar,
  envio_ayuda_fam_exterior_hogar, envio_ayuda_otras_hogar -- señal de
  capacidad economica relativa (el hogar tiene margen para ayudar a
  otros).
  uso_ayuda_alimentos_hogar, uso_ayuda_salud_hogar,
  uso_ayuda_educacion_hogar, uso_ayuda_vivienda_hogar (condicional a
  haber recibido ayuda).

Religion:
  practica_religion_hogar (de `religion`, normalizado).

Caveat encontrado en la validacion (no bloquea, queda como observacion)
--------------------------------------------------------------------------
`n_activos_financieros_hogar` sube de 0.37 (ola 1) a 0.64 (ola 2) --
investigado: 5 de los 15 `act_*` (herencias, otrosing, polizas, vtainm,
vtaneg) se preguntaron SOLO a zona URBANA en ola 1 (0% cobertura rural,
100% urbana, verificado) y se agregaron para ambas zonas en ola 2; a la
inversa, otros 5 (segmaq, segotros, segveh, segvida, segviv) se
preguntaron a ambas zonas en ola 1 pero SOLO a zona urbana en ola 2. El
composite (conteo de tipos con Sí) se calcula correctamente sobre los
items disponibles para cada hogar (`tiene_dato_af.any(axis=1)` como
denominador), pero el NUMERO de items "preguntables" varia por ola/zona,
por lo que el conteo no es perfectamente comparable en magnitud entre
olas -- mismo tipo de limitacion ya documentada para variables similares
en otros modulos (ej. TVIP en Niños). Se deja como covariable de
intensidad relativa, no como conteo estandarizado.

Verificacion final de cobertura completa de las 129 candidatas
--------------------------------------------------------------------------
De las 129, 103 se usan (directas o en composites); 26 se EXCLUYEN con
razon documentada:
  - 5 redundantes con `ingreso_total_hogar` ya construido en
    `build_ingreso_hogar.py`: `ing_trabajo`, `ing_pensiones`,
    `ing_arriendos`, `ing_intereses_div`, `ing_otros_nrem`.
  - 2 redundantes con `gasto_percapita_hogar` (enfoque granular de 88
    articulos, ya preferido): `vr_gtos_mens_alim`, `vr_gtos_mensuales`.
  - 12 detalle de prestamo #1/#2 (`cuota_1/2`, `meses_plazo_1/2`,
    `period_cuota_1/2`, `fechai_mes_1/2`, `fechai_ano_1/2`,
    `vr_inicial_1/2`, `vr_saldo_1/2`): demasiado granular, ya capturado a
    nivel de presencia/formalidad de la deuda en `tiene_deuda_hogar`/
    `deuda_formal_hogar`/`deuda_informal_hogar` (de `con_quien_1`).
  - 2 valores condicionales (`ayu_fc_vr`, `eay_fc_vr`): montos de ayuda
    recibida/enviada de/a familiares en el exterior -- la variable-puerta
    (`ayu_fam_ext`/`eay_fam_exte`) ya esta capturada, mismo patron de
    "variable puerta ya en las candidatas" verificado en la auditoria
    previa.
  - 3 categorias de uso de ayuda con prevalencia negligible/degenerada
    (`uay_noagropec` 8 casos "Sí" en total, `uay_tierras` 1 caso,
    `uay_otros` cobertura muy baja): no aportan señal util como variable
    individual.

Output: data/processed/hogar_features_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOGAR_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_features_elca_longitudinal.parquet"

COLUMNAS_DURABLES = [
    "n_neveras", "n_lavadoras", "n_secadoras", "n_licuadoras", "n_hornos",
    "n_microondas", "n_calentadores", "n_duchas", "n_radios",
    "n_equipos_sonido", "n_equipos_video", "n_televisores",
    "n_television_cable", "n_computadores", "n_internet", "n_internet",
    "n_aire_acondicionado", "n_bicicletas", "n_otros_bienes",
]
COLUMNAS_DURABLES = sorted(set(COLUMNAS_DURABLES))

COLUMNAS_ACTIVOS_FINANCIEROS = [
    "act_cesantias", "act_dinero", "act_fondos", "act_herencias",
    "act_otrosing", "act_polizas", "act_roscas", "act_segcosechas",
    "act_segmaq", "act_segotros", "act_segveh", "act_segvida",
    "act_segviv", "act_vtainm", "act_vtaneg",
]
COLUMNAS_SP = ["sp_acueducto", "sp_alcantarillado", "sp_energia", "sp_gasnatural", "sp_telefono", "sp_recoleccion_basura"]
COLUMNAS_PROGRAMAS = ["familias_accion", "red_juntos", "sena", "icbf", "prg_adultomayor", "prg_tierras", "otro_programa"]
COLUMNAS_DESASTRE = ["avalancha", "creciente", "hundimiento", "inundacion", "terremoto"]
COLUMNAS_AYUDA_RECIBIDA = [
    "ayu_alimentos", "ayu_fam_colom", "ayu_fam_ext", "ayu_ong",
    "ayu_orgintern", "ayu_religiosas", "ayu_desplazados",
]

DEUDA_FORMAL_TOKENS = {
    "bancos o entidades financieras", "bancos o entidades financieras en colombia",
    "bancos o entidades financieras en el exterior", "cajas de compensación",
    "fondos de empleados o cooperativas", "icetex", "gremios o asociaciones",
    "almacenes de cadena, hipermercados", "almacenes de cadena, hipermercados o codensa",
    "empleador", "empleadores",
}
DEUDA_INFORMAL_TOKENS = {
    "amigos", "familiares (de otros hogares)", "prestamistas", "tenderos",
    "casas de empeño o casas comerciales", "compras por catálogo",
}


def normalizar_espacios(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalizar_si_no(serie: pd.Series) -> pd.Series:
    """Colapsa Sí/Si/SI/1 -> Sí, No/NO/2 -> No (bateria act_* usa codigos
    numericos 1/2 en ola 1, texto en ola 2/3 -- ver hallazgo en docstring)."""
    s = normalizar_espacios(serie).str.lower()
    resultado = pd.Series(np.nan, index=serie.index, dtype=object)
    resultado[s.isin({"si", "sí", "1"})] = "Sí"
    resultado[s.isin({"no", "2"})] = "No"
    return resultado


def normalizar_conteo_binario(serie: pd.Series) -> pd.Series:
    """Para columnas que mezclan conteo numerico ('0','1',...) con texto
    Sí/No (ej. n_internet): colapsa a binario Sí (>0 o 'Sí') / No (0 o 'No')."""
    s = normalizar_espacios(serie).str.lower()
    numerico = pd.to_numeric(s, errors="coerce")
    resultado = pd.Series(np.nan, index=serie.index, dtype=object)
    resultado[s.isin({"si", "sí"})] = "Sí"
    resultado[s.isin({"no"})] = "No"
    resultado[numerico == 0] = "No"
    resultado[numerico > 0] = "Sí"
    return resultado


def cargar_hogar() -> pd.DataFrame:
    h = pd.read_parquet(HOGAR_PATH)
    h["ola"] = pd.to_numeric(h["ola"], errors="coerce")

    for col in COLUMNAS_ACTIVOS_FINANCIEROS + COLUMNAS_SP + COLUMNAS_PROGRAMAS + COLUMNAS_DESASTRE + COLUMNAS_AYUDA_RECIBIDA:
        h[col] = normalizar_si_no(h[col])

    for col in ["n_neveras", "n_lavadoras", "n_secadoras", "n_licuadoras", "n_hornos",
                "n_microondas", "n_calentadores", "n_duchas", "n_radios",
                "n_equipos_sonido", "n_equipos_video", "n_televisores",
                "n_television_cable", "n_computadores",
                "n_aire_acondicionado", "n_bicicletas", "n_otros_bienes",
                "automoviles", "motocicletas", "casas", "lotes", "semovientes",
                "transporte", "valor_arriendo_pagado"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")
    h["n_internet"] = normalizar_conteo_binario(h["n_internet"])

    h["agro_ingresos"] = normalizar_si_no(h["agro_ingresos"])
    h["tit_baldios"] = normalizar_si_no(h["tit_baldios"])
    h["otra_financiacion"] = normalizar_si_no(h["otra_financiacion"])
    h["prestamo_familiar"] = normalizar_si_no(h["prestamo_familiar"])
    h["uay_agropec"] = normalizar_si_no(h["uay_agropec"])
    h["uay_ahorrar"] = normalizar_si_no(h["uay_ahorrar"])

    h["t_personas"] = pd.to_numeric(h["t_personas"], errors="coerce")
    h["t_cuartos_hogar"] = pd.to_numeric(h["t_cuartos_hogar"], errors="coerce")
    h["t_cuartos_dormir"] = pd.to_numeric(h["t_cuartos_dormir"], errors="coerce")
    h["t_hogares"] = pd.to_numeric(h["t_hogares"], errors="coerce")
    h["estrato"] = pd.to_numeric(h["estrato"], errors="coerce")
    h["sp_estrato"] = pd.to_numeric(h["sp_estrato"], errors="coerce")

    con_quien_norm = normalizar_espacios(h["con_quien_1"]).str.lower()
    h["deuda_formal"] = np.where(h["con_quien_1"].notna(), con_quien_norm.isin(DEUDA_FORMAL_TOKENS), np.nan)
    h["deuda_informal"] = np.where(h["con_quien_1"].notna(), con_quien_norm.isin(DEUDA_INFORMAL_TOKENS), np.nan)

    dcto_norm = normalizar_espacios(h["dcto_vivienda"])
    h["tiene_escritura"] = np.where(
        h["dcto_vivienda"].notna(),
        (dcto_norm == "Escritura pública registrada en la Oficina de Instrumentos Públicos").astype(float),
        np.nan,
    )

    for col in ["credito_financiera", "credito_cooperativa", "credito_fna",
                "recursos_propios", "subsidios"]:
        h[col] = normalizar_si_no(h[col])

    religion_norm = normalizar_espacios(h["religion"]).str.lower()
    h["practica_religion"] = np.where(h["religion"].notna(), religion_norm.isin({"si", "sí"}).astype(float), np.nan)

    for col in ["uay_comida", "uay_salud", "uay_educacion", "uay_vivienda"]:
        h[col] = normalizar_si_no(h[col])
    for col in ["eay_alimentos", "eay_fam_colom", "eay_fam_exte", "eay_otras"]:
        h[col] = normalizar_si_no(h[col])

    return h


def construir_composites(h: pd.DataFrame) -> pd.DataFrame:
    ind = pd.DataFrame(index=h.index)

    cols_durables_numericas = [c for c in COLUMNAS_DURABLES if c != "n_internet"]
    es_si_durable = pd.DataFrame({c: (h[c] > 0) for c in cols_durables_numericas})
    tiene_dato_durable = pd.DataFrame({c: h[c].notna() for c in cols_durables_numericas})
    es_si_durable["n_internet"] = h["n_internet"].eq("Sí")
    tiene_dato_durable["n_internet"] = h["n_internet"].notna()
    ind["n_bienes_durables"] = np.where(tiene_dato_durable.any(axis=1), es_si_durable.sum(axis=1), np.nan)

    es_si_af = h[COLUMNAS_ACTIVOS_FINANCIEROS].eq("Sí")
    tiene_dato_af = h[COLUMNAS_ACTIVOS_FINANCIEROS].notna()
    ind["n_activos_financieros"] = np.where(tiene_dato_af.any(axis=1), es_si_af.sum(axis=1), np.nan)

    es_si_sp = h[COLUMNAS_SP].eq("Sí")
    tiene_dato_sp = h[COLUMNAS_SP].notna()
    ind["n_servicios_publicos"] = np.where(tiene_dato_sp.any(axis=1), es_si_sp.sum(axis=1), np.nan)

    es_si_prog = h[COLUMNAS_PROGRAMAS].eq("Sí")
    tiene_dato_prog = h[COLUMNAS_PROGRAMAS].notna()
    ind["n_programas_sociales"] = np.where(tiene_dato_prog.any(axis=1), es_si_prog.sum(axis=1), np.nan)
    ind["beneficiario_algun_programa"] = np.where(tiene_dato_prog.any(axis=1), es_si_prog.any(axis=1).astype(float), np.nan)

    es_si_desastre = h[COLUMNAS_DESASTRE].eq("Sí")
    tiene_dato_desastre = h[COLUMNAS_DESASTRE].notna()
    ind["tuvo_desastre_natural"] = np.where(tiene_dato_desastre.any(axis=1), es_si_desastre.any(axis=1).astype(float), np.nan)

    es_si_ayuda = h[COLUMNAS_AYUDA_RECIBIDA].eq("Sí")
    tiene_dato_ayuda = h[COLUMNAS_AYUDA_RECIBIDA].notna()
    ind["n_tipos_ayuda_recibida"] = np.where(tiene_dato_ayuda.any(axis=1), es_si_ayuda.sum(axis=1), np.nan)

    ind["tiene_vehiculo"] = np.where(
        h["automoviles"].notna() | h["motocicletas"].notna(),
        ((h["automoviles"].fillna(0) > 0) | (h["motocicletas"].fillna(0) > 0)).astype(float),
        np.nan,
    )

    ind["financio_credito_formal_vivienda"] = np.where(
        h["credito_financiera"].notna() | h["credito_cooperativa"].notna() | h["credito_fna"].notna(),
        (h["credito_financiera"].eq("Sí") | h["credito_cooperativa"].eq("Sí") | h["credito_fna"].eq("Sí")).astype(float),
        np.nan,
    )

    ind["personas_por_cuarto"] = h["t_personas"] / h["t_cuartos_hogar"].replace(0, np.nan)
    ind["personas_por_dormitorio"] = h["t_personas"] / h["t_cuartos_dormir"].replace(0, np.nan)

    ind["tiene_propiedad_rural"] = np.where(
        h["casas"].notna() | h["lotes"].notna() | h["semovientes"].notna(),
        ((h["casas"].fillna(0) > 0) | (h["lotes"].fillna(0) > 0) | (h["semovientes"].fillna(0) > 0)).astype(float),
        np.nan,
    )
    ind["tiene_transporte_carga"] = np.where(h["transporte"].notna(), (h["transporte"] > 0).astype(float), np.nan)

    ind["financio_otra_fuente_vivienda"] = np.where(
        h["otra_financiacion"].notna() | h["prestamo_familiar"].notna(),
        (h["otra_financiacion"].eq("Sí") | h["prestamo_familiar"].eq("Sí")).astype(float),
        np.nan,
    )

    return ind


def main() -> None:
    h = cargar_hogar()
    ind = construir_composites(h)

    salida = h[["consecutivo", "llave", "llave_n16", "ola", "zona"]].copy()

    directas = {
        "tenencia_vivienda_hogar": "tenencia_vivienda", "tipo_vivienda_hogar": "tipo_vivienda",
        "material_paredes_hogar": "material_paredes", "material_pisos_hogar": "material_pisos",
        "servicio_sanitario_hogar": "servicio_sanitario", "obtencion_agua_hogar": "obtencion_agua",
        "energia_cocinan_hogar": "energia_cocinan", "eliminan_basura_hogar": "eliminan_basura",
        "estrato_hogar": "estrato", "estrato_verificado_hogar": "sp_estrato",
        "n_hogares_comparte_vivienda_hogar": "t_hogares",
        "riqueza_pca_hogar": "riqueza_pca",
    }
    for nuevo, original in directas.items():
        salida[nuevo] = h[original]

    salida["n_servicios_publicos_hogar"] = ind["n_servicios_publicos"]
    salida["personas_por_cuarto_hogar"] = ind["personas_por_cuarto"]
    salida["personas_por_dormitorio_hogar"] = ind["personas_por_dormitorio"]

    salida["n_bienes_durables_hogar"] = ind["n_bienes_durables"]
    salida["tiene_vehiculo_hogar"] = ind["tiene_vehiculo"]
    salida["tiene_internet_hogar"] = (h["n_internet"] == "Sí").astype(float)
    salida.loc[h["n_internet"].isna(), "tiene_internet_hogar"] = np.nan
    salida["n_activos_financieros_hogar"] = ind["n_activos_financieros"]
    salida["tiene_propiedad_rural_hogar"] = ind["tiene_propiedad_rural"]
    salida["tiene_transporte_carga_hogar"] = ind["tiene_transporte_carga"]
    salida["tiene_ingreso_agropecuario_hogar"] = (h["agro_ingresos"] == "Sí").astype(float)
    salida.loc[h["agro_ingresos"].isna(), "tiene_ingreso_agropecuario_hogar"] = np.nan

    salida["beneficiario_familias_accion_hogar"] = (h["familias_accion"] == "Sí").astype(float)
    salida.loc[h["familias_accion"].isna(), "beneficiario_familias_accion_hogar"] = np.nan
    salida["beneficiario_red_juntos_hogar"] = (h["red_juntos"] == "Sí").astype(float)
    salida.loc[h["red_juntos"].isna(), "beneficiario_red_juntos_hogar"] = np.nan
    salida["n_programas_sociales_hogar"] = ind["n_programas_sociales"]
    salida["beneficiario_algun_programa_hogar"] = ind["beneficiario_algun_programa"]

    salida["tuvo_desastre_natural_hogar"] = ind["tuvo_desastre_natural"]

    salida["financio_credito_formal_vivienda_hogar"] = ind["financio_credito_formal_vivienda"]
    salida["financio_recursos_propios_vivienda_hogar"] = (h["recursos_propios"] == "Sí").astype(float)
    salida.loc[h["recursos_propios"].isna(), "financio_recursos_propios_vivienda_hogar"] = np.nan
    salida["financio_subsidio_vivienda_hogar"] = (h["subsidios"] == "Sí").astype(float)
    salida.loc[h["subsidios"].isna(), "financio_subsidio_vivienda_hogar"] = np.nan
    salida["financio_otra_fuente_vivienda_hogar"] = ind["financio_otra_fuente_vivienda"]
    salida["tiene_escritura_vivienda_hogar"] = h["tiene_escritura"]
    salida["tiene_titulo_baldio_hogar"] = (h["tit_baldios"] == "Sí").astype(float)
    salida.loc[h["tit_baldios"].isna(), "tiene_titulo_baldio_hogar"] = np.nan
    salida["valor_arriendo_pagado_hogar"] = h["valor_arriendo_pagado"]

    salida["tiene_deuda_hogar"] = h["con_quien_1"].notna().astype(float)
    salida["deuda_formal_hogar"] = h["deuda_formal"]
    salida["deuda_informal_hogar"] = h["deuda_informal"]

    for nuevo, original in [
        ("recibio_ayuda_alimentos_hogar", "ayu_alimentos"),
        ("recibio_ayuda_fam_colombia_hogar", "ayu_fam_colom"),
        ("recibio_ayuda_fam_exterior_hogar", "ayu_fam_ext"),
        ("recibio_ayuda_ong_hogar", "ayu_ong"),
        ("recibio_ayuda_org_internacional_hogar", "ayu_orgintern"),
        ("recibio_ayuda_religiosa_hogar", "ayu_religiosas"),
        ("recibio_ayuda_desplazados_hogar", "ayu_desplazados"),
        ("envio_ayuda_alimentos_hogar", "eay_alimentos"),
        ("envio_ayuda_fam_colombia_hogar", "eay_fam_colom"),
        ("envio_ayuda_fam_exterior_hogar", "eay_fam_exte"),
        ("envio_ayuda_otras_hogar", "eay_otras"),
        ("uso_ayuda_alimentos_hogar", "uay_comida"),
        ("uso_ayuda_salud_hogar", "uay_salud"),
        ("uso_ayuda_educacion_hogar", "uay_educacion"),
        ("uso_ayuda_vivienda_hogar", "uay_vivienda"),
    ]:
        salida[nuevo] = (h[original] == "Sí").astype(float)
        salida.loc[h[original].isna(), nuevo] = np.nan
    salida["n_tipos_ayuda_recibida_hogar"] = ind["n_tipos_ayuda_recibida"]

    salida["uso_ayuda_agropecuario_hogar"] = (h["uay_agropec"] == "Sí").astype(float)
    salida.loc[h["uay_agropec"].isna(), "uso_ayuda_agropecuario_hogar"] = np.nan
    salida["uso_ayuda_ahorrar_hogar"] = (h["uay_ahorrar"] == "Sí").astype(float)
    salida.loc[h["uay_ahorrar"].isna(), "uso_ayuda_ahorrar_hogar"] = np.nan

    salida["practica_religion_hogar"] = h["practica_religion"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas, {salida.shape[1]} columnas)")
    print()
    cols_resumen = [c for c in salida.columns if pd.api.types.is_numeric_dtype(salida[c]) and c not in ("consecutivo", "llave", "llave_n16")]
    with pd.option_context("display.max_rows", 100, "display.width", 160):
        print(salida.groupby("ola")[cols_resumen].mean().T)


if __name__ == "__main__":
    main()
