"""
Construccion de covariables de choques (shocks) a partir de
choques_elca_longitudinal.parquet (ELCA 2010, 2013, 2016), para el modelo
benchmark de prediccion de transicion a la pobreza.

A diferencia de Personas/Niños, esta base YA esta a nivel de hogar (una
fila por hogar-ola, generada por
`01_download/01_descarga_ELCA/01_consolidacion_bases_choques.py`) -- no
requiere agregacion desde individuos, solo join directo por llave de hogar
(`consecutivo`/`llave`/`llave_n16` segun la ola, verificado 1:1 exacto
contra `hogar_elca_longitudinal_clean.parquet`: 9.853/9.261/8.818 hogares
por ola en ambas bases).

Precondicion: el HALLAZGO CRITICO de cobertura (35%/70%/76% de los
hogares, hogares sin ningun choque desaparecian del panel en vez de
quedar en 0) ya fue corregido en `01_consolidacion_bases_choques.py` -- la base
ahora tiene 27.932 filas, cobertura 100% en las 3 olas.

Auditoria de columnas (82 en total: 23 `choque_*`, 21 `imp_econ_*`, 30
`resp_*`, mas 8 identificadores/total)
--------------------------------------------------------------------------
Esta base ya es un resumen consolidado (no datos crudos de encuesta), asi
que la auditoria de corrupcion/renombrados de Personas/Comunidades/Niños
no aplica igual -- se verifico que 0 columnas tienen corrupcion U+FFFD o
"???" residual (la correccion ya se aplico en la consolidacion). Lo que
si se audito a fondo, a, fue la COBERTURA de cada
`choque_*` y `resp_*` por ola para decidir que construir:

**Cluster de desastres naturales -- 0% en ola 1, confirmado con el
diccionario oficial, NO es un bug**: `sufrieron_inundaciones_avalanchas_
derrumbes_desbordamientos_o_deslizamientos_vendavales` (2 variantes de
nombre, una con "temblores o terremotos" fusionado y otra sin fusionar --
inconsistencia de nomenclatura entre 2013 y 2016 que el script de
consolidacion no armonizo), `sufrieron_sequias`, `sufrieron_temblores_o_
terremotos`: las 4 columnas dan exactamente 0% en ola 1. Se
verifico explicitamente (resultaba extraño dado que 2010
coincidio con la Ola Invernal de Colombia) -- se extrajo el diccionario
PDF oficial de 2010 (`UChoques.pdf`) y se confirmo que el cuestionario de
2010 enumera exactamente 18 categorias de choque posibles (todas
listadas explicitamente en el diccionario) y NINGUNA es un desastre
natural -- la categoria se agrego recien en el cuestionario de 2013. Es
una limitacion real del instrumento de ELCA, no un artefacto de esta
consolidacion. Excluidas de los composites porque fallan Eje 1
(comparabilidad ola1-ola2).

**Dos categorias de "abandono" desaparecen despues de ola 1**:
`abandono_del_hogar_por_parte_de_un_menor_de_18_anos` (0.28% ola1, 0%
ola2/3) y `abandono_del_que_era_jefe_del_hogar_o_del_conyuge` (0.69%
ola1, 0% ola2/3) -- preguntas eliminadas del cuestionario despues de
2010. Excluidas de los composites por la misma razon (fallan Eje 1).

**`imp_econ_*` (severidad del impacto: Alta/Media/Baja) confirmado 0% en
ola 1** (ya documentado en la auditoria de modulos previa) -- no se
construye como feature del benchmark (train/test simetrico requiere
ola1+ola2), queda como candidato solo para una especificacion futura que
use unicamente ola 2 como fuente (no la actual).

**`resp_*` (estrategias de afrontamiento) tienen una brecha de cobertura
adicional, investigada**: la cobertura de las columnas `resp_*` (34.6%/
54.0%/60.9%) es MENOR que la proporcion de hogares con `total_choques>0`
(34.6%/68.9%/75.0%) -- brecha de ~15 puntos en ola 2/3. Se verifico
contra el archivo crudo de 2013 (`UChoques-csv.tab`+`RChoques-csv.tab`):
de 12.439 choques con `tuvo_choque=='SI'`, 3.903 (31.4%) tienen
`hizo_princ` (la respuesta principal) en blanco -- es una no-respuesta
real de ELCA a esa pregunta de seguimiento, no un error de esta
consolidacion ni de esta auditoria. Los hogares cuyos choques TODOS
tienen `hizo_princ` en blanco quedan correctamente en NaN para los
`resp_*` (no se imputa).

**Nombres de `resp_*` inconsistentes entre olas para 2 conceptos**
(mismo patron de re-nombrado ya visto en Personas/Comunidades, aqui dentro
de una sola base ya consolidada): `hipotecaron_algun_activo` +
`arrendaron_algun_activo` (ola 1, dos respuestas separadas) se fusionaron
en `hipotecaron_o_arrendaron_algun_activo` (ola 2/3, una sola respuesta)
-- se arma un indicador armonizado con OR. El resto de columnas usadas en
los composites de afrontamiento estan presentes con el MISMO nombre en
las 3 olas (verificado uno por uno antes de incluir), asi que no
necesitan armonizacion.

Variables construidas (nivel hogar, directo de la base ya consolidada)
--------------------------------------------------------------------------
Incidencia de choques:
  total_choques_hogar: pass-through de `total_choques` (ya
                                 corregido para incluir 0 en hogares sin
                                 choques).
  tuvo_algun_choque_hogar: 1 si `total_choques>0`.
  n_tipos_choque_hogar: numero de TIPOS DISTINTOS de choque
                                 sufridos (no la suma de `n_veces`, que
                                 puede repetir el mismo tipo) -- mide
                                 amplitud de exposicion, no solo
                                 frecuencia. Excluye el cluster de
                                 desastres naturales y las 2 categorias
                                 de "abandono" (fallan Eje 1).
  tuvo_choque_salud_hogar: OR(accidente o enfermedad, muerte de
                                 otros miembros, muerte del jefe/cónyuge).
  tuvo_choque_economico_hogar: OR(cónyuge/jefe/otro miembro perdió
                                 empleo, quiebra de negocio familiar).
  tuvo_choque_patrimonial_hogar: OR(pérdida de fincas, pérdida de
                                 vivienda, pérdida/recorte de remesas,
                                 robo/incendio/destrucción de bienes).
  tuvo_choque_agropecuario_hogar: OR(pérdida o muerte de animales, plagas
                                 o pérdida de cosechas). Verificado: estas
                                 2 columnas se preguntan EXCLUSIVAMENTE a
                                 hogares rurales (100% cobertura rural, 0%
                                 urbana, por diseño del cuestionario) --
                                 el ~54% de NaN en el agregado nacional es
                                 la proporcion urbana del panel, no un
                                 error; dentro de hogares rurales la
                                 cobertura es completa.
  tuvo_choque_familiar_hogar: OR(víctimas de violencia, llegada de un
                                 familiar al hogar, separación de
                                 cónyuges, tuvieron que abandonar su lugar
                                 de residencia).
  tuvo_choque_severo_hogar: OR(muerte del jefe/cónyuge, pérdida de
                                 vivienda, pérdida de fincas) -- subgrupo
                                 de choques con pérdida patrimonial o de
                                 capital humano dificil de revertir.

Estrategias de afrontamiento (nivel hogar, condicional a haber tenido
algun choque con respuesta registrada):
  afrontamiento_erosivo_hogar: 1 si el hogar uso CUALQUIERA de: vendieron
                                 bienes/activos, retiraron hijos del
                                 colegio, hipotecaron/arrendaron un
                                 activo, sacrificaron animales,
                                 disminuyeron gastos en alimentos, algun
                                 miembro salió del país -- estrategias que
                                 comprometen capital productivo o humano
                                 futuro.
  afrontamiento_protector_hogar: 1 si el hogar uso CUALQUIERA de: gastaron
                                 ahorros, usaron algún seguro, pidieron
                                 ayuda a familiares/comunidad, pidieron
                                 ayuda a instituciones -- estrategias que
                                 usan amortiguadores sin comprometer
                                 capital productivo.
  intensifico_trabajo_hogar: 1 si miembros que trabajaban aumentaron
                                 horas O miembros que no trabajaban
                                 salieron a buscar/trabajar -- respuesta
                                 de oferta laboral, categoria propia (ni
                                 claramente erosiva ni protectora).
  retiro_hijos_colegio_choque_hogar: 1 si el hogar retiró hijos del
                                 colegio/universidad como respuesta a
                                 algún choque -- señal directa de erosión
                                 de capital humano.
  redujo_alimentos_choque_hogar: 1 si el hogar disminuyó gastos en
                                 alimentos como respuesta a algún choque.
  se_endeudo_formal_choque_hogar: 1 si el hogar se endeudó con un banco o
                                 entidad financiera.
  se_endeudo_informal_choque_hogar: 1 si el hogar se endeudó con
                                 familiares o amigos -- informalidad del
                                 endeudamiento como proxy de exclusion
                                 financiera (comparar con el formal).
  no_ajusto_choque_hogar: 1 si el hogar reporta que NO fue
                                 necesario alterar sus costumbres --
                                 señal inversa, choque absorbido sin
                                 ajuste visible.

Output: data/processed/choques_hogar_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHOQUES_PATH = PROJECT_ROOT / "data" / "processed" / "choques_elca_longitudinal.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "choques_hogar_elca_longitudinal.parquet"

COLUMNAS_CHOQUE_EXCLUIDAS_EJE1 = [
    "choque_abandono_del_hogar_por_parte_de_un_menor_de_18_anos",
    "choque_abandono_del_que_era_jefe_del_hogar_o_del_conyuge",
    "choque_sufrieron_inundaciones_avalanchas_derrumbes_desbordamientos_o_deslizamientos_vendavales",
    "choque_sufrieron_inundaciones_avalanchas_derrumbes_desbordamientos_o_deslizamientos_vendavales_temblores_o_terremotos",
    "choque_sufrieron_sequias",
    "choque_sufrieron_temblores_o_terremotos",
]

CHOQUE_SALUD = [
    "choque_accidente_o_enfermedad_de_algun_miembro_del_hogar_que_le_impidio_realizar_sus_actividades_cotidianas",
    "choque_muerte_de_algunos_otros_miembros_del_hogar",
    "choque_muerte_del_que_era_jefe_del_hogar_o_del_conyuge",
]
CHOQUE_ECONOMICO = [
    "choque_el_conyuge_perdio_su_empleo",
    "choque_el_jefe_del_hogar_perdio_su_empleo",
    "choque_otro_miembro_del_hogar_perdio_su_empleo",
    "choque_quiebras_y_o_cierres_del_los_negocios_familiares",
]
CHOQUE_PATRIMONIAL = [
    "choque_perdida_de_fincas_lotes_terrenos_o_pedazos_de_tierra",
    "choque_perdida_de_la_vivienda",
    "choque_perdida_o_recorte_de_remesas",
    "choque_robo_incendio_o_destruccion_de_bienes_del_hogar_en_casa_o_raponeo",
]
CHOQUE_AGROPECUARIO = [
    "choque_perdida_o_muerte_de_animales",
    "choque_plagas_o_perdida_de_cosechas",
]
CHOQUE_FAMILIAR = [
    "choque_fueron_victimas_de_la_violencia",
    "choque_llegada_o_acogida_de_un_familiar_en_el_hogar",
    "choque_separacion_de_los_conyuges",
    "choque_tuvieron_que_abandonar_su_lugar_de_residencia_habitual",
]
CHOQUE_SEVERO = [
    "choque_muerte_del_que_era_jefe_del_hogar_o_del_conyuge",
    "choque_perdida_de_la_vivienda",
    "choque_perdida_de_fincas_lotes_terrenos_o_pedazos_de_tierra",
]

RESP_EROSIVO_DIRECTO = [
    "resp_vendieron_bienes_o_activos",
    "resp_retiraron_a_los_hijos_del_colegio_o_la_universidad",
    "resp_sacrificaron_animales",
    "resp_disminuyeron_los_gastos_en_alimentos",
    "resp_uno_o_mas_miembros_del_hogar_salieron_del_pais",
]
# armonizado: ola 1 tiene 2 respuestas separadas, ola 2/3 las fusiono en 1
RESP_ACTIVO_HIPOTECADO_OLA1 = ["resp_hipotecaron_algun_activo_casa_carro_finca_etc", "resp_arrendaron_algun_activo_casa_carro_finca_etc"]
RESP_ACTIVO_HIPOTECADO_OLA23 = "resp_hipotecaron_o_arrendaron_algun_activo_casa_carro_finca_etc"

RESP_PROTECTOR = [
    "resp_gastaron_los_ahorros",
    "resp_usaron_algun_seguro",
    "resp_pidieron_ayuda_a_familiares_amigos_u_otras_personas_de_la_comunidad",
    "resp_pidieron_ayuda_a_instituciones_nacionales_o_internacionales",
]
RESP_LABORAL = [
    "resp_los_miembros_del_hogar_que_trabajaban_aumentaron_las_horas_de_trabajo",
    "resp_miembros_del_hogar_que_no_trabajaban_salieron_a_buscar_trabajo_o_trabajar",
]


def cualquiera_positivo(df: pd.DataFrame, columnas: list) -> pd.Series:
    """1 si cualquiera de las columnas (conteos) es > 0, 0 si todas son 0,
    NaN si todas son NaN (hogar sin dato valido en ninguna)."""
    sub = df[columnas]
    tiene_dato = sub.notna().any(axis=1)
    es_positivo = (sub.fillna(0) > 0).any(axis=1)
    return np.where(tiene_dato, es_positivo.astype(float), np.nan)


def main() -> None:
    c = pd.read_parquet(CHOQUES_PATH)

    choque_cols_todas = [col for col in c.columns if col.startswith("choque_")]
    choque_cols_eje1 = [col for col in choque_cols_todas if col not in COLUMNAS_CHOQUE_EXCLUIDAS_EJE1]

    salida = c[["consecutivo", "llave", "llave_n16", "ola", "zona"]].copy()
    salida["total_choques_hogar"] = c["total_choques"]
    salida["tuvo_algun_choque_hogar"] = (c["total_choques"] > 0).astype(float)

    tiene_dato_eje1 = c[choque_cols_eje1].notna().any(axis=1)
    n_tipos = (c[choque_cols_eje1].fillna(0) > 0).sum(axis=1)
    salida["n_tipos_choque_hogar"] = np.where(tiene_dato_eje1, n_tipos, np.nan)

    salida["tuvo_choque_salud_hogar"] = cualquiera_positivo(c, CHOQUE_SALUD)
    salida["tuvo_choque_economico_hogar"] = cualquiera_positivo(c, CHOQUE_ECONOMICO)
    salida["tuvo_choque_patrimonial_hogar"] = cualquiera_positivo(c, CHOQUE_PATRIMONIAL)
    salida["tuvo_choque_agropecuario_hogar"] = cualquiera_positivo(c, CHOQUE_AGROPECUARIO)
    salida["tuvo_choque_familiar_hogar"] = cualquiera_positivo(c, CHOQUE_FAMILIAR)
    salida["tuvo_choque_severo_hogar"] = cualquiera_positivo(c, CHOQUE_SEVERO)

    activo_hipotecado = pd.Series(np.nan, index=c.index)
    es_ola1 = c["ola"] == 1
    activo_hipotecado[es_ola1] = cualquiera_positivo(c[es_ola1], RESP_ACTIVO_HIPOTECADO_OLA1)
    activo_hipotecado[~es_ola1] = np.where(
        c.loc[~es_ola1, RESP_ACTIVO_HIPOTECADO_OLA23].notna(),
        (c.loc[~es_ola1, RESP_ACTIVO_HIPOTECADO_OLA23].fillna(0) > 0).astype(float),
        np.nan,
    )

    erosivo_cols_presentes = cualquiera_positivo(c, RESP_EROSIVO_DIRECTO)
    tiene_dato_erosivo = c[RESP_EROSIVO_DIRECTO].notna().any(axis=1) | activo_hipotecado.notna()
    es_erosivo = (c[RESP_EROSIVO_DIRECTO].fillna(0) > 0).any(axis=1) | (activo_hipotecado.fillna(0) > 0)
    salida["afrontamiento_erosivo_hogar"] = np.where(tiene_dato_erosivo, es_erosivo.astype(float), np.nan)

    salida["afrontamiento_protector_hogar"] = cualquiera_positivo(c, RESP_PROTECTOR)
    salida["intensifico_trabajo_hogar"] = cualquiera_positivo(c, RESP_LABORAL)

    salida["retiro_hijos_colegio_choque_hogar"] = np.where(
        c["resp_retiraron_a_los_hijos_del_colegio_o_la_universidad"].notna(),
        (c["resp_retiraron_a_los_hijos_del_colegio_o_la_universidad"] > 0).astype(float), np.nan,
    )
    salida["redujo_alimentos_choque_hogar"] = np.where(
        c["resp_disminuyeron_los_gastos_en_alimentos"].notna(),
        (c["resp_disminuyeron_los_gastos_en_alimentos"] > 0).astype(float), np.nan,
    )
    salida["se_endeudo_formal_choque_hogar"] = np.where(
        c["resp_se_endeudaron_con_un_banco_o_entidad_financiera"].notna(),
        (c["resp_se_endeudaron_con_un_banco_o_entidad_financiera"] > 0).astype(float), np.nan,
    )
    salida["se_endeudo_informal_choque_hogar"] = np.where(
        c["resp_se_endeudaron_con_familiares_o_amigos"].notna(),
        (c["resp_se_endeudaron_con_familiares_o_amigos"] > 0).astype(float), np.nan,
    )
    salida["no_ajusto_choque_hogar"] = np.where(
        c["resp_no_fue_necesario_hacer_algo_que_alterara_las_costumbres_del_hogar"].notna(),
        (c["resp_no_fue_necesario_hacer_algo_que_alterara_las_costumbres_del_hogar"] > 0).astype(float), np.nan,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUTPUT_PATH, index=False)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print()
    cols_resumen = [
        "total_choques_hogar", "tuvo_algun_choque_hogar", "n_tipos_choque_hogar",
        "tuvo_choque_salud_hogar", "tuvo_choque_economico_hogar", "tuvo_choque_patrimonial_hogar",
        "tuvo_choque_agropecuario_hogar", "tuvo_choque_familiar_hogar", "tuvo_choque_severo_hogar",
        "afrontamiento_erosivo_hogar", "afrontamiento_protector_hogar", "intensifico_trabajo_hogar",
        "retiro_hijos_colegio_choque_hogar", "redujo_alimentos_choque_hogar",
        "se_endeudo_formal_choque_hogar", "se_endeudo_informal_choque_hogar", "no_ajusto_choque_hogar",
    ]
    print(salida.groupby("ola")[cols_resumen].mean().T)
    print()
    print("Nulos por columna:")
    print(salida[cols_resumen].isna().mean())


if __name__ == "__main__":
    main()
