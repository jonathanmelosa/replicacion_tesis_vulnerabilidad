"""
etiquetas_variables.py
=====================================
Etiquetas legibles y categorias tematicas de las covariables, compartidas
por las tablas y figuras del paper.

Creado 2026-09-30: al retirar el perfil univariado del
paper, las tablas vigentes no deben depender de los scripts del perfil.
Estos diccionarios vivian en `src/02_build/eda_perfil_completo.py`
(`CATEGORIA`), `src/tabla_perfil_completo.py` (`ETIQUETAS`) y
`src/tabla_shap_nucleo_perfil.py` (etiquetas del nucleo SHAP y
`etiqueta()`); se movieron aqui sin cambios y esos scripts ahora los
importan de este modulo.
"""

# Categoria tematica y sub-panel por unidad -- asignado a mano a partir
# del `modulo` del inventario y de la unidad real de cada variable
# (revisado con `construir_tabla_comparativa`: "(media)" vs "(media, %)"
# vs categorica -- ver columna nivel_mostrado del output).
# Cambio 2026-09-24: "Educación y empleo del jefe" se
# renombra "Educación, empleo y seguridad social" (incluye variables del
# hogar, no solo del jefe, y afiliación/cotización); etnia_jefe y
# estado_civil_jefe pasan a "Composición del hogar", y
# pct_ninos_apoyo_alimentario_escolar a "Programas sociales y deuda".
CATEGORIA = {
    "dmsp_stable_lights": ("Geoespacial", "Luces nocturnas (0-63)"),
    "zona": ("Zona de residencia", "% del grupo"),
    "brecha_lp_ingreso": ("Ingreso y gasto", "Veces la línea de pobreza"),
    "brecha_lp_gasto": ("Ingreso y gasto", "Veces la línea de pobreza"),
    "ingreso_percapita_hogar_real": ("Ingreso y gasto", "Miles $ por mes"),
    "gasto_percapita_hogar_real": ("Ingreso y gasto", "Miles $ por mes"),
    "material_pisos_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "energia_cocinan_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "servicio_sanitario_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "eliminan_basura_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "obtencion_agua_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "tipo_vivienda_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "material_paredes_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "tenencia_vivienda_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "personas_por_cuarto_hogar": ("Vivienda: hacinamiento", "Personas por habitación/dormitorio"),
    "personas_por_dormitorio_hogar": ("Vivienda: hacinamiento", "Personas por habitación/dormitorio"),
    "valor_arriendo_pagado_hogar": ("Vivienda: hacinamiento", "Miles $ de arriendo/mes"),
    "n_bienes_durables_hogar": ("Activos del hogar", "Número (conteo)"),
    "n_servicios_publicos_hogar": ("Activos del hogar", "Número (conteo)"),
    "n_activos_financieros_hogar": ("Activos del hogar", "Número (conteo)"),
    "estrato_hogar": ("Activos del hogar", "Estrato (1-6)"),
    "estrato_verificado_hogar": ("Activos del hogar", "Estrato (1-6)"),
    "riqueza_pca_hogar": ("Activos del hogar", "Índice de riqueza (PCA)"),
    "tiene_internet_hogar": ("Activos del hogar", "% del grupo"),
    "n_programas_sociales_hogar": ("Programas sociales y deuda", "Número (conteo)"),
    "beneficiario_familias_accion_hogar": ("Programas sociales y deuda", "% del grupo"),
    "beneficiario_algun_programa_hogar": ("Programas sociales y deuda", "% del grupo"),
    "nivel_educ_jefe": ("Educación, empleo y seguridad social", "% del grupo"),
    "categoria_ocupacional_jefe": ("Educación, empleo y seguridad social", "% del grupo"),
    "medio_consiguio_jefe": ("Educación, empleo y seguridad social", "% del grupo"),
    "registro_mercantil_jefe": ("Educación, empleo y seguridad social", "% del grupo"),
    "n_empleados_jefe": ("Educación, empleo y seguridad social", "% del grupo"),
    "etnia_jefe": ("Composición del hogar", "% del grupo"),
    "estado_civil_jefe": ("Composición del hogar", "% del grupo"),
    "nivel_educ_max_hogar": ("Educación, empleo y seguridad social", "Años/nivel (escala propia)"),
    "nivel_educ_ordinal_jefe": ("Educación, empleo y seguridad social", "Años/nivel (escala propia)"),
    "tasa_cotizacion_pension_hogar": ("Educación, empleo y seguridad social", "% del grupo"),
    "cotiza_pension_jefe": ("Educación, empleo y seguridad social", "% del grupo"),
    "n_ninos_12": ("Composición del hogar", "Número (conteo)"),
    "razon_dependencia_demografica": ("Composición del hogar", "Razón de dependencia"),
    # -- Extensión 2026-09-15 (VARIABLES_ESTABLES_AMPLIADAS, ver docstring) --
    "tasa_afiliacion_pension_hogar": ("Educación, empleo y seguridad social", "% del grupo"),
    "tasa_afiliacion_salud_laboral_hogar": ("Educación, empleo y seguridad social", "% del grupo"),
    "afiliado_pension_jefe": ("Educación, empleo y seguridad social", "% del grupo"),
    "afiliado_salud_laboral_jefe": ("Educación, empleo y seguridad social", "% del grupo"),
    "deuda_formal_hogar": ("Programas sociales y deuda", "% del grupo"),
    "deuda_informal_hogar": ("Programas sociales y deuda", "% del grupo"),
    "tiene_escritura_vivienda_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "financio_credito_formal_vivienda_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "tiene_vehiculo_hogar": ("Activos del hogar", "% del grupo"),
    "pct_ninos_cuidado_terceros_hogar": ("Composición del hogar", "% del grupo"),
    "pct_ninos_apoyo_alimentario_escolar": ("Programas sociales y deuda", "% del grupo"),
    "n_espacios_publicos_comunidad": ("Comunidad", "Número (conteo)"),
    "tiene_transporte_publico_comunidad": ("Comunidad", "% del grupo"),
}


ETIQUETAS = {
    "brecha_lp_ingreso": "Ingreso (veces la línea de pobreza)",
    "brecha_lp_gasto": "Gasto (veces la línea de pobreza)",
    "ingreso_percapita_hogar_real": "Ingreso per cápita",
    "gasto_percapita_hogar_real": "Gasto per cápita",
    "lp": "Línea de pobreza (referencia)",
    "li": "Línea de indigencia (referencia)",
    "zona": "Vive en zona rural",
    "dmsp_stable_lights": "Luces nocturnas (DMSP-OLS)",
    "material_pisos_hogar": "Material de piso: {nivel}",
    "energia_cocinan_hogar": "Combustible de cocina: {nivel}",
    "servicio_sanitario_hogar": "Servicio sanitario: {nivel}",
    "eliminan_basura_hogar": "Recolección de basura: {nivel}",
    "obtencion_agua_hogar": "Obtención de agua: {nivel}",
    "tipo_vivienda_hogar": "Tipo de vivienda: {nivel}",
    "material_paredes_hogar": "Material de paredes: {nivel}",
    "tenencia_vivienda_hogar": "Tenencia de vivienda: {nivel}",
    "personas_por_cuarto_hogar": "Personas por habitación",
    "personas_por_dormitorio_hogar": "Personas por dormitorio",
    "valor_arriendo_pagado_hogar": "Valor del arriendo pagado",
    "n_bienes_durables_hogar": "N.\\textsuperscript{o} de bienes durables",
    "n_servicios_publicos_hogar": "N.\\textsuperscript{o} de servicios públicos",
    "n_activos_financieros_hogar": "N.\\textsuperscript{o} de activos financieros",
    "estrato_hogar": "Estrato (autorreportado)",
    "estrato_verificado_hogar": "Estrato (verificado)",
    "riqueza_pca_hogar": "Índice de riqueza (PCA)",
    "tiene_internet_hogar": "Tiene internet en el hogar",
    "n_programas_sociales_hogar": "N.\\textsuperscript{o} de programas sociales",
    "beneficiario_familias_accion_hogar": "Beneficiario de Familias en Acción",
    "beneficiario_algun_programa_hogar": "Beneficiario de algún programa social",
    "nivel_educ_jefe": "Educación del jefe: {nivel}",
    "categoria_ocupacional_jefe": "Ocupación del jefe: {nivel}",
    "medio_consiguio_jefe": "Consiguió trabajo: {nivel}",
    "registro_mercantil_jefe": "Registro mercantil del jefe: {nivel}",
    "n_empleados_jefe": "Negocio del jefe: {nivel}",
    "etnia_jefe": "Etnia del jefe: {nivel}",
    "estado_civil_jefe": "Estado civil del jefe: {nivel}",
    "sexo_jefe": "Sexo del jefe: {nivel}",
    "tasa_cotizacion_pension_hogar": "Cotización a pensión (hogar)",
    "cotiza_pension_jefe": "Jefe cotiza a pensión",
    "nivel_educ_max_hogar": "Máx. nivel educativo del hogar",
    "nivel_educ_ordinal_jefe": "Nivel educativo del jefe (ordinal)",
    "pct_adultos_alfabetizados": "Adultos alfabetizados en el hogar",
    "n_ninos_12": "N.\\textsuperscript{o} de niños en el hogar",
    "pct_ninos_cuidado_terceros_hogar": "Niños al cuidado de terceros",
    "razon_dependencia_demografica": "Razón de dependencia demográfica",
    "pobre_ingreso": "Pobre por ingreso (2010)",
    "pobre_extremo_ingreso": "Pobre extremo por ingreso (2010)",
    "pobre_gasto": "Pobre por gasto (2010)",
    "pobre_extremo_gasto": "Pobre extremo por gasto (2010)",
    # -- Extensión 2026-09-15 (VARIABLES_ESTABLES_AMPLIADAS) --
    "tasa_afiliacion_pension_hogar": "Afiliación a pensión (hogar)",
    "tasa_afiliacion_salud_laboral_hogar": "Afiliación a salud laboral (hogar)",
    "afiliado_pension_jefe": "Jefe afiliado a pensión",
    "afiliado_salud_laboral_jefe": "Jefe afiliado a salud laboral",
    "deuda_formal_hogar": "Tiene deuda formal",
    "deuda_informal_hogar": "Tiene deuda informal",
    "tiene_escritura_vivienda_hogar": "Tiene escritura de la vivienda",
    "financio_credito_formal_vivienda_hogar": "Financió la vivienda con crédito formal",
    "tiene_vehiculo_hogar": "Tiene vehículo",
    "pct_ninos_apoyo_alimentario_escolar": "Niños con apoyo alimentario escolar",
    "n_espacios_publicos_comunidad": "N.\\textsuperscript{o} de espacios públicos en la comunidad",
    "tiene_transporte_publico_comunidad": "Comunidad con transporte público",
}


# Etiquetas de las 8 variables del núcleo que no están en el perfil
# univariado de 53 variables de la Sección 5.1 (y por tanto no tienen
# entrada en ETIQUETAS).
ETIQUETAS_NUEVAS = {
    # \char`\%{} en vez de \%: babel-spanish redefine \% con un \unskip
    # que borra el \quad de sangria que va justo antes (la fila quedaba
    # sin sangria); \char imprime el mismo glifo sin pasar por babel.
    "pct_ninos_madre_viva": "\\char`\\%{} de niños con madre viva",
    "pct_ninos_padre_vivo": "\\char`\\%{} de niños con padre vivo",
    "edad_jefe": "Edad del jefe de hogar",
    "grado_educ_jefe": "Último grado aprobado por el jefe",  # no son años de escolaridad: el grado se reinicia en cada nivel (2026-09-28)
    "tvip_puntaje_directo_hogar": "Puntaje de vocabulario infantil (test TVIP)",
    "tuvo_choque_economico_hogar": "Tuvo un choque económico (hogar)",
    "tasa_control_preventivo_hogar": "Tasa de controles médicos preventivos (hogar)",
    "n_desplazados_comunidad": "N.\\textsuperscript{o} de desplazados en la comunidad",
}

ORDEN_CATEGORIAS = [
    "Activos del hogar", "Educación, empleo y seguridad social",
    "Vivienda: materiales y servicios", "Vivienda: hacinamiento",
    "Composición del hogar", "Desarrollo infantil (6--9 años)", "Salud",
    "Programas sociales y deuda", "Choques y afrontamiento", "Comunidad",
]


# Variables categóricas cuya etiqueta en `tabla_perfil_completo.py` es una
# plantilla por nivel ("Material de piso: {nivel}"), pensada para la tabla
# de perfil de la Sección 5.1 (una fila por categoría). Aquí el SHAP ya
# está agregado a nivel de VARIABLE completa (todas las categorías
# sumadas), así que se usa el nombre de la variable sin nivel.
ETIQUETAS_SIN_NIVEL = {
    "material_pisos_hogar": "Material de piso del hogar",
    "estado_civil_jefe": "Estado civil del jefe",
}


def etiqueta(variable: str) -> str:
    if variable in ETIQUETAS_SIN_NIVEL:
        return ETIQUETAS_SIN_NIVEL[variable]
    if variable in ETIQUETAS:
        return ETIQUETAS[variable]
    if variable in ETIQUETAS_NUEVAS:
        return ETIQUETAS_NUEVAS[variable]
    raise ValueError(f"Variable del núcleo sin etiqueta legible: {variable}")
