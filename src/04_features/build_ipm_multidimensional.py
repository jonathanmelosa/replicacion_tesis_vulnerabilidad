"""
Construccion del Indice de Pobreza Multidimensional (IPM-Colombia,
metodologia DANE -- Angulo, Diaz y Pardo 2011, Alkire-Foster) sobre el
panel ELCA (2010/2013/2016)
probar si el hallazgo de "DMSP-OLS no aporta" es especifico de la
definicion de pobreza monetaria o se sostiene tambien con una definicion
de vulnerabilidad multidimensional.

NO existe ningun indicador de pobreza multidimensional en el repositorio
antes de este script (confirmado por busqueda exhaustiva) -- se
construye desde cero siguiendo el glosario oficial del Boletin Tecnico
DANE "Pobreza Multidimensional en Colombia" (2018), que define los 15
indicadores palabra por palabra (extraido via pdftotext, no de memoria).

METODOLOGIA: 5 dimensiones DANE (educacion, niñez y juventud, trabajo,
salud, vivienda y servicios publicos), 15 indicadores originales,
estructura de ponderacion anidada Alkire-Foster (cada dimension 20%,
cada indicador igual peso dentro de su dimension), 33.3% como punto de
corte de pobreza. VERSION FINAL (2026-08-28, ver seccion LIMITACION mas
abajo): 5 de los 15 indicadores se EXCLUYEN del score por huecos de
cobertura irresolubles en ola 1 -- quedan 10 indicadores activos. La
dimension "Trabajo" pierde sus 2 unicos indicadores y queda vacia; su
20% de peso se redistribuye entre las 4 dimensiones restantes (25% cada
una) -- ver `PESO_INDICADOR` y `COLS_EXCLUIDAS_SCORE` mas abajo. Se
evaluo excluir tambien `priv_rezago_escolar` (cobertura imperfecta pero
mucho menos grave en ola 1, 26.6% vs. ~92%) y se REVIRTIO: con menos
indicadores en el score, el sesgo ya conocido de `priv_barreras_
acceso_salud` (proxy mas agresivo en ola 1) pasa a dominar el agregado y
genera una inestabilidad peor que el problema que se queria resolver --
ver nota junto a `PESO_INDICADOR`. El punto de corte de pobreza sigue
siendo 33.3% del score ponderado (no depende del numero de indicadores).

APROXIMACIONES DOCUMENTADAS:

  1. Bajo logro educativo: DANE define "educacion promedio de personas
     >=15 anos < 9 anos". La ELCA no tiene anos de educacion directos,
     solo `nivel_educ` (escala ordinal de NIVEL, ya usada en
     build_educacion_ocupacion_hogar.py) + `grado_educ` (grado dentro del
     nivel). Se construye una tabla de conversion nivel->anos base +
     grado_educ (documentada en `AÑOS_BASE_NIVEL` abajo) -- aproximacion
     estandar en la literatura de educacion colombiana, NO la tabla
     interna exacta de DANE (no publica). Primaria/secundaria: grado_educ
     ya es acumulativo (1-11), se usa tal cual. Tecnica/tecnologica/
     universitaria/posgrado: base (11 o 16) + grado_educ (o un valor
     tipico de duracion si "con titulo" y grado_educ falta).

  2. Rezago escolar (ninos 7-17): DANE define "anos aprobados < norma
     nacional para su edad". La norma exacta de DANE no es publica en
     detalle -- se usa la aproximacion estandar grado_esperado = edad-6
     (un nino que entra a grado 1 a los 6 anos deberia estar en grado
     edad-6 en cualquier año posterior), documentada como aproximacion.
     BUG corregido (2026-08-28): la version inicial uso `grado_educ`
     (grado YA COMPLETADO, cobertura 0.4-7.6% entre ninos 7-17 -- la
     mayoria todavia esta cursando, no aplica) en vez de
     `grado_educ_cursa` (grado que cursa actualmente, cobertura
     26.6%/91.9%/92.5% ola1/2/3) -- corregido a esta ultima.

  3. Barreras de acceso a servicios de salud: DANE define "tuvo problema
     de salud en 30 dias, SIN hospitalizacion, y NO acudio a atencion
     medica/odontologica". CORREGIDO: `enfermedad_p` (proxy inicial,
     evento de salud) solo existe en ola 1 -- para ola 2/3 se encontro la
     variable correcta, `tratar_problema` ("¿que hizo para tratar el
     problema?"), que responde CASI EXACTAMENTE la pregunta DANE (ya
     filtrada a quienes reportaron un problema de salud, por diseno de
     skip-pattern de la encuesta): privado si la respuesta es "Se auto
     recetó", "Nada", "Usó remedios caseros" o "Acudió al boticario,
     farmaceuta, droguista" (ninguna es atencion medica/odontologica
     profesional); NO privado si "Acudió a un hospital/médico
     general/especialista/odontólogo" o medicina alternativa (jucio
     propio: no esta en la lista textual de DANE pero se trata como
     busqueda de atencion). Ola 1 mantiene el proxy original (evento de
     salud + sin hospitalizacion) por no tener una pregunta equivalente
     -- diferencia de metodologia documentada entre ola 1 y olas 2/3.

  4. Barreras de acceso a cuidado primera infancia (ninos 0-5): DANE
     exige falta de TODOS los servicios integrales (salud+nutricion+
     cuidado). Se aproximan los 3 sub-componentes con lo disponible en
     `ninos_elca_longitudinal_clean.parquet`: salud = tiene esquema de
     vacunacion completo (`carne_vacunas`/vacunas especificas);
     nutricion = no reporta razones de privacion alimentaria
     (`razon_nofrutas`/`razon_noverduras`/`razon_nocarnes`/`razon_noleche`
     todas vacias/None); cuidado = asiste a algun tipo de cuidado
     (`asiste`='Si' o tipo_hogar indica cuidador). Privacion si falta
     CUALQUIERA de los 3 (igual que la definicion oficial).

  5. Desempleo de larga duracion: DANE define "PEA desempleada >12
     meses". La ELCA no separa desocupado de inactivo de forma confiable
     via `actividad_ppal` (ver limitacion ya documentada en
     build_educacion_ocupacion_hogar.py) -- se usa en cambio
     `busco_trabajo`='Sí' (pregunta directa de busqueda activa) +
     `t_busco_trab` >= 12 meses como definicion operativa de "desempleado
     de larga duracion", independiente de la limitacion de
     `actividad_ppal`.

  6. Empleo informal: DANE define "ocupado sin afiliacion a pensiones".
     Se usa `cotizando` (Sí/No, ola 2/3) con fallback a
     `normalizar_cotiza_fp(cotiza_fp)` (ola 1, reutilizando el helper ya
     existente en build_ahorro_capital_social_hogar.py) para personas
     `ocupado`=1.

  7. Hacinamiento critico y agua/saneamiento: se aplica la definicion
     diferenciada urbano/rural de DANE (umbrales distintos), usando
     `zona` de la ELCA.

  8. Sin aseguramiento en salud: `segsoc_salud` (usado en la primera
     version de este script) solo existe en ola 2/3 (0 en ola 1) --
     CORREGIDO: se usa `afiliacion`, con cobertura completa en las 3
     olas y semantica equivalente (afiliacion a salud, Sí/No).

  9. Bajo logro educativo -- arrastre de nivel educativo entre olas:
     `nivel_educ`/`grado_educ` tienen 73% de datos faltantes entre
     adultos en ola 3 (verificado: afecta directamente a jefes de hogar,
     no es un patron de "no aplica a menores" -- 7.389 de los 19.102
     adultos con el dato faltante son jefes de hogar). Patron consistente
     con que la ELCA solo vuelve a preguntar nivel educativo si cambio
     desde la ola anterior, y no repite la pregunta a quien ya la
     respondio antes. CORREGIDO: se arrastra el nivel educativo mas
     reciente conocido de la MISMA persona (identificada por
     `llave_ID_lb`, valida para personas presentes desde 2010) desde una
     ola anterior donde si estaba disponible -- recupera 13.783 de 23.380
     casos faltantes (59%). El resto (miembros nuevos del hogar sin
     registro previo, o sin `llave_ID_lb` valida) sigue faltante y se
     excluye del promedio del hogar (no se fuerza un valor).

Todas las demas privaciones (analfabetismo, sin aseguramiento en salud,
pisos/paredes inadecuados) tienen match directo o casi directo con
preguntas de la ELCA.

LIMITACION SIN ARREGLO POSIBLE -- cobertura de `actividad_ppal` y
`estudia` en OLA 1 (2010): investigado a fondo (2026-08-28) tras detectar
tasas de privacion anomalas en 2010 para varios indicadores. Confirmado:

  - `actividad_ppal` (de donde sale `ocupado`) tiene 0.4% de cobertura en
    ola 1 entre ninos 12-17 (vs. ~100% en ola 2/3) -- afecta
    `priv_trabajo_infantil` (sale artificialmente ~0% en 2010).
  - `estudia` tiene 35.3% de cobertura en ola 1 entre ninos 6-16 (vs.
    99.6% en ola 2/3) -- afecta `priv_inasistencia_escolar` (tambien
    artificialmente bajo en 2010).
  - Lo mismo afecta `actividad_ppal` a nivel adulto (cobertura ~20% en
    ola 1 vs. ~75-80% en ola 2/3), sesgando `priv_empleo_informal` y
    `priv_desempleo_larga_duracion` en 2010 hacia abajo (denominador de
    "ocupados"/PEA mas chico y no representativo).

  A diferencia del arreglo de nivel educativo (Seccion 9, que arrastra el
  valor de una OLA ANTERIOR de la misma persona), aca no hay arrastre
  posible: 2010 es la primera ola del panel, no existe un "antes" del
  cual traer el dato. Se investigaron variables alternativas (misma
  logica que resolvio salud/educacion) sin encontrar una pregunta
  equivalente con mejor cobertura en ola 1 para actividad economica o
  asistencia escolar.

  DECISION FINAL: en vez de dejar
  que `priv_trabajo_infantil`, `priv_inasistencia_escolar`,
  `priv_empleo_informal` y `priv_desempleo_larga_duracion` sesguen el
  score con datos subestimados en ola 2010, se EXCLUYEN los 4 del
  `ipm_score` (mismo tratamiento que `priv_primera_infancia`, ver punto
  4 -- lista completa en `COLS_EXCLUIDAS_SCORE`). Las 4 columnas se
  siguen calculando y guardando en el parquet de salida para auditoria,
  pero no entran en el puntaje ni en la clasificacion `pobre_ipm`.9%/20.8%/19.2% en 2010/2013/2016, vs. 17.4%/27.8%/25.3% con los
  indicadores problematicos todavia incluidos).

UNIDAD DE ANALISIS: hogar x ola (2010, 2013, 2016) -- mismo hogar
identificado por `consecutivo`, igual que el resto del pipeline de
pobreza monetaria (`build_pobreza_monetaria.py`).

OUTPUTS

    data/processed/ipm_multidimensional_elca_longitudinal.parquet
    (columnas: consecutivo, ola, zona, ipm_score, pobre_ipm, y las 15
    columnas de privacion individuales _priv_<indicador> para auditoria)

COMO CORRER

    cd src/04_features && python -u build_ipm_multidimensional.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_educacion_ocupacion_hogar import (  # noqa: E402
    ACTIVIDAD_NO_OCUPADO,
    ACTIVIDAD_OCUPADO,
    ESCALA_NIVEL_EDUC,
    normalizar_espacios,
)
from build_ahorro_capital_social_hogar import normalizar_cotiza_fp  # noqa: E402


# --- 1. Anos de educacion (aproximacion, ver docstring punto 1) ---
AÑOS_BASE_NIVEL = {
    "Ninguno": 0.0, "Preescolar": 0.0,
    "Básica primaria (1 a 5)": 0.0,
    "Básica secundaria (6 a 13)": 0.0, "Básica secundaria y media (6 a 13)": 0.0,
    "Técnico sin título": 11.0, "Técnico con título": 11.0,
    "Tecnológico sin título": 11.0, "Tecnológico con título": 11.0,
    "Universitario sin título": 11.0, "Universitario con título": 11.0,
    "Posgrado sin título": 16.0, "Posgrado con título": 16.0,
}
GRADO_TIPICO_SI_TITULO = {
    "Técnico con título": 2.0, "Tecnológico con título": 3.0,
    "Universitario con título": 5.0, "Posgrado con título": 2.0,
}


def arrastrar_nivel_educ(personas: pd.DataFrame) -> pd.DataFrame:
    """Arrastra nivel_educ/grado_educ de la ola mas reciente conocida de
    la MISMA persona (via llave_ID_lb) hacia adelante -- ver docstring
    punto 9. Solo aplica cuando el valor actual falta y hay un valor
    valido en una ola anterior de esa persona."""
    p = personas.sort_values(["llave_ID_lb", "ola"]).copy()
    con_id = p["llave_ID_lb"].notna()

    nivel_valido = normalizar_espacios(p["nivel_educ"])
    nivel_valido = nivel_valido.where(~nivel_valido.isin(["None", "No informa"]))
    p.loc[con_id, "nivel_educ"] = (
        nivel_valido[con_id].groupby(p.loc[con_id, "llave_ID_lb"]).ffill()
    )
    grado_valido = pd.to_numeric(p["grado_educ"], errors="coerce")
    p.loc[con_id, "grado_educ"] = (
        grado_valido[con_id].groupby(p.loc[con_id, "llave_ID_lb"]).ffill()
    )
    return p.sort_index()


def calcular_anos_educacion(personas: pd.DataFrame) -> pd.Series:
    nivel = normalizar_espacios(personas["nivel_educ"])
    grado = pd.to_numeric(personas["grado_educ"], errors="coerce")

    base = nivel.map(AÑOS_BASE_NIVEL)
    grado_relleno = grado.copy()
    for niv, tipico in GRADO_TIPICO_SI_TITULO.items():
        mask = (nivel == niv) & grado.isna()
        grado_relleno[mask] = tipico
    grado_relleno = grado_relleno.fillna(0.0)

    anos = base + grado_relleno
    anos[nivel.isin(["Ninguno", "Preescolar"])] = 0.0
    anos[nivel.isin(["None", "No informa"]) | nivel.isna()] = np.nan
    return anos.clip(lower=0, upper=21)


def calcular_ocupado(personas: pd.DataFrame) -> pd.Series:
    actividad = normalizar_espacios(personas["actividad_ppal"])
    ocupado = pd.Series(np.nan, index=personas.index)
    ocupado[actividad.isin(ACTIVIDAD_OCUPADO)] = 1
    ocupado[actividad.isin(ACTIVIDAD_NO_OCUPADO)] = 0
    return ocupado


def calcular_cotizando(personas: pd.DataFrame) -> pd.Series:
    directo = normalizar_espacios(personas["cotizando"]).replace({"Si": "Sí"})
    directo = directo.where(directo.isin(["Sí", "No"]))
    fallback = normalizar_cotiza_fp(personas["cotiza_fp"])
    return directo.where(directo.notna(), fallback)


def construir_privaciones_personas(personas: pd.DataFrame) -> pd.DataFrame:
    """Retorna un dataframe a nivel PERSONA con una columna booleana (o NaN
    si no aplica) por cada indicador que se agrega desde ese nivel."""
    p = arrastrar_nivel_educ(personas)
    p["edad"] = pd.to_numeric(p["edad"], errors="coerce")
    p["anos_educacion"] = calcular_anos_educacion(p)
    p["ocupado"] = calcular_ocupado(p)
    p["cotizando_norm"] = calcular_cotizando(p)

    estudia = normalizar_espacios(p["estudia"]).replace({"Si": "Sí"})
    p["estudia_norm"] = estudia.where(estudia.isin(["Sí", "No"]))
    lee_escribe = normalizar_espacios(p["lee_escribe"]).replace({"Si": "Sí"})
    p["lee_escribe_norm"] = lee_escribe.where(lee_escribe.isin(["Sí", "No"]))
    # `busco_trabajo` (Sí/No) y `t_busco_trab` (duracion de busqueda) NUNCA
    # estan pobladas en las mismas filas (verificado) -- `t_busco_trab` se
    # responde junto con `actividad_ppal`="Ninguna de las anteriores" (el
    # residual ya documentado en build_educacion_ocupacion_hogar.py como
    # mezcla de inactivos/desocupados). Se usa esa combinacion como PEA-
    # desempleado: no ocupado, con historial de busqueda (t_busco_trab !=
    # "Nunca ha buscado trabajo"), duracion >=12 meses = privacion.
    duracion_larga = {
        "Entre 1 y menos de 2 años", "Entre 2 y menos de 3 años",
        "Entre 2 y menos de 5 años", "Hace 3 años o más", "Hace más de 5 años",
    }
    t_busco = normalizar_espacios(p["t_busco_trab"])
    p["tiene_historial_busqueda"] = t_busco.notna() & (t_busco != "None") & (t_busco != "Nunca ha buscado trabajo")
    p["desempleo_largo"] = t_busco.isin(duracion_larga)

    p["salud_evento"] = normalizar_espacios(p["enfermedad_p"]).replace({"Si": "Sí"})
    p["hospitalizado_norm"] = normalizar_espacios(p["hospitalizado"]).replace({"Si": "Sí"})
    p["afiliacion_norm"] = normalizar_espacios(p["afiliacion"]).replace({"Si": "Sí"})

    tratar = normalizar_espacios(p["tratar_problema"])
    no_atencion = {
        "Se auto recetó", "Nada", "Usó remedios caseros", "Us??? remedios caseros",
        "Acudió al boticario, farmaceuta, droguista",
    }
    p["barrera_ola23"] = tratar.isin(no_atencion)

    # BUG corregido (2026-08-28): `grado_educ` es el grado ya COMPLETADO
    # (cobertura pesima en ninos, 0.4-7.6% -- la mayoria de ninos en edad
    # escolar todavia esta cursando, no "completo" nada reciente). El
    # grado ACTUAL que cursa es `grado_educ_cursa` (cobertura 26.6%/
    # 91.9%/92.5% en 7-17 anos, ola1/2/3) -- esa es la variable correcta
    # para comparar contra la norma de edad esperada.
    grado_cursa = pd.to_numeric(p["grado_educ_cursa"], errors="coerce")
    grado_esperado = p["edad"] - 6
    p["rezago"] = (p["edad"].between(7, 17)) & grado_cursa.notna() & (grado_cursa < grado_esperado - 1)

    return p


def agregar_privaciones_hogar_ola(p: pd.DataFrame) -> pd.DataFrame:
    """Colapsa las privaciones de nivel-persona a nivel hogar x ola --
    'privado' si AL MENOS UNA persona relevante del hogar cumple la
    condicion (definicion estandar Alkire-Foster: la privacion de
    cualquier miembro se atribuye a todo el hogar)."""
    g = p.groupby(["consecutivo", "ola"])

    adultos15 = p[p["edad"] >= 15]
    bajo_logro = adultos15.groupby(["consecutivo", "ola"])["anos_educacion"].mean()
    priv_bajo_logro = (bajo_logro < 9)

    analfabetismo = adultos15.assign(no_lee=lambda d: d["lee_escribe_norm"] == "No") \
        .groupby(["consecutivo", "ola"])["no_lee"].max()

    ninos_6_16 = p[p["edad"].between(6, 16)]
    inasistencia = ninos_6_16.assign(no_asiste=lambda d: d["estudia_norm"] == "No") \
        .groupby(["consecutivo", "ola"])["no_asiste"].max()

    rezago = p.groupby(["consecutivo", "ola"])["rezago"].max()

    ninos_12_17 = p[p["edad"].between(12, 17)]
    trabajo_infantil = ninos_12_17.assign(trabaja=lambda d: d["ocupado"] == 1) \
        .groupby(["consecutivo", "ola"])["trabaja"].max()

    pea = p[(p["ocupado"] == 1) | p["tiene_historial_busqueda"]]
    desempleo_largo = pea.assign(
        desemp=lambda d: (d["ocupado"] != 1) & d["tiene_historial_busqueda"] & d["desempleo_largo"]
    ).groupby(["consecutivo", "ola"])["desemp"].max()

    ocupados = p[p["ocupado"] == 1]
    empleo_informal = ocupados.assign(informal=lambda d: d["cotizando_norm"] == "No") \
        .groupby(["consecutivo", "ola"])["informal"].max()

    mayores5 = p[p["edad"] > 5]
    sin_aseguramiento = mayores5.assign(sin_seg=lambda d: d["afiliacion_norm"] == "No") \
        .groupby(["consecutivo", "ola"])["sin_seg"].max()

    # Ola 1: proxy evento de salud + sin hospitalizacion. Ola 2/3: pregunta
    # directa tratar_problema (ver docstring punto 3).
    barrera_ola1 = (p["ola"] == 1) & (p["salud_evento"] == "Sí") & (p["hospitalizado_norm"] == "No")
    barrera_ola23 = (p["ola"] != 1) & p["barrera_ola23"].fillna(False)
    barreras_salud = p.assign(barrera=barrera_ola1 | barrera_ola23) \
        .groupby(["consecutivo", "ola"])["barrera"].max()

    out = pd.DataFrame({
        "priv_bajo_logro_educativo": priv_bajo_logro,
        "priv_analfabetismo": analfabetismo,
        "priv_inasistencia_escolar": inasistencia,
        "priv_rezago_escolar": rezago,
        "priv_trabajo_infantil": trabajo_infantil,
        "priv_desempleo_larga_duracion": desempleo_largo,
        "priv_empleo_informal": empleo_informal,
        "priv_sin_aseguramiento_salud": sin_aseguramiento,
        "priv_barreras_acceso_salud": barreras_salud,
    }).reset_index()
    return out


def construir_privacion_primera_infancia(ninos: pd.DataFrame) -> pd.DataFrame:
    """Ninos 0-5: privado si falta CUALQUIERA de los 3 sub-componentes
    (salud, nutricion, cuidado) -- ver docstring punto 4."""
    n = ninos[ninos["edad_nino"].between(0, 5)].copy()

    vacunas_cols = [c for c in ["antituberculosa", "vop3", "pentavalente3", "dtf3", "hepatitis3", "triplev"] if c in n.columns]
    for c in vacunas_cols:
        n[c] = normalizar_espacios(n[c]).replace({"Si": "Sí"})
    n["salud_completa"] = n[vacunas_cols].apply(lambda row: (row == "Sí").all(), axis=1) if vacunas_cols else False

    razones_priv_nutricion = ["razon_nofrutas", "razon_noverduras", "razon_nocarnes", "razon_noleche"]
    razones_presentes = [c for c in razones_priv_nutricion if c in n.columns]
    n["nutricion_ok"] = ~n[razones_presentes].apply(lambda row: row.notna().any() & (row.astype(str) != "None").any(), axis=1) if razones_presentes else True

    asiste = normalizar_espacios(n["asiste"]).replace({"Si": "Sí"}) if "asiste" in n.columns else pd.Series("No", index=n.index)
    n["cuidado_ok"] = asiste == "Sí"

    n["privado_primera_infancia"] = ~(n["salud_completa"] & n["nutricion_ok"] & n["cuidado_ok"])

    out = n.groupby(["consecutivo", "ola"])["privado_primera_infancia"].max().rename("priv_primera_infancia").reset_index()
    return out


def construir_privaciones_vivienda(hogar: pd.DataFrame) -> pd.DataFrame:
    h = hogar.copy()
    zona_rural = normalizar_espacios(h["zona"]) == "Rural"

    piso = normalizar_espacios(h["material_pisos"])
    priv_pisos = piso == "Tierra o arena"

    pared = normalizar_espacios(h["material_paredes"])
    paredes_malas = {"Bahareque", "Tapia pisada, adobe", "Madera burda, tabla, tablón",
                      "Guadua, caña, esterilla, otro vegetal", "Zinc, tela, cartón, latas, desechos, plásticos",
                      "Sin paredes"}
    priv_paredes = pared.isin(paredes_malas)

    acueducto = normalizar_espacios(h["sp_acueducto"]).str.upper().replace({"SÍ": "SI"})
    obtencion = normalizar_espacios(h["obtencion_agua"])
    fuente_rural_mala = {"Pozo sin bomba, jagüey", "Pozo sin bomba, jaguey", "Río, quebrada, manantial, nacimiento",
                          "Agua lluvia", "Otra fuente (botella, bolsa, etc.)", "Otra fuente (botella,bolsa,etc)",
                          "Aguatero", "Carrotanque", "Pila pública"}
    priv_agua = np.where(zona_rural, obtencion.isin(fuente_rural_mala), acueducto == "NO")

    alcantarillado = normalizar_espacios(h["sp_alcantarillado"]).str.upper().replace({"SÍ": "SI"})
    sanitario = normalizar_espacios(h["servicio_sanitario"])
    sanitario_rural_malo = {"No tiene servicio sanitario", "Inodoro sin conexión", "Bajamar", "Letrina"}
    priv_excretas = np.where(zona_rural, sanitario.isin(sanitario_rural_malo), alcantarillado == "NO")

    personas_hogar = pd.to_numeric(h.get("t_personas", h.get("t_hogar")), errors="coerce")
    cuartos_dormir = pd.to_numeric(h["t_cuartos_dormir"], errors="coerce")
    personas_por_cuarto = personas_hogar / cuartos_dormir.replace(0, np.nan)
    umbral = np.where(zona_rural, 3, 3)  # DANE: >3 rural, >=3 urbano -- ver nota abajo
    priv_hacinamiento = np.where(zona_rural, personas_por_cuarto > 3, personas_por_cuarto >= 3)

    out = pd.DataFrame({
        "consecutivo": h["consecutivo"], "ola": h["ola"],
        "priv_pisos_inadecuados": priv_pisos.values,
        "priv_paredes_inadecuadas": priv_paredes.values,
        "priv_agua_mejorada": priv_agua,
        "priv_eliminacion_excretas": priv_excretas,
        "priv_hacinamiento_critico": priv_hacinamiento,
    })
    return out


## Indicadores EXCLUIDOS del score final (columnas se calculan y se
## guardan igual, para auditoria/transparencia, pero no entran en
## `ipm_score`)
##
##   - priv_primera_infancia: no medible de forma consistente en ninguna
##     ola (asiste/vacunas = 0 filas en ola 3, nutricion = 0 filas en
##     ola 1). Ver docstring del modulo, punto 4.
##   - priv_inasistencia_escolar, priv_trabajo_infantil: dependen de
##     `estudia`/`actividad_ppal` en ninos, con 0.4%-35.3% de cobertura en
##     OLA 1 (2010) vs. ~100% en ola 2/3 -- sin arrastre posible (2010 es
##     la primera ola del panel, no hay "antes" del cual traer el dato).
##     Ver seccion "LIMITACION SIN ARREGLO POSIBLE" del docstring.
##   - priv_empleo_informal, priv_desempleo_larga_duracion: dependen de
##     `actividad_ppal` en adultos, con ~20% de cobertura en ola 1 vs.
##     ~75-80% en ola 2/3 -- mismo problema, misma seccion del docstring.
##
## priv_rezago_escolar: se probo excluirla tambien (2026-08-28, segunda
## ronda) pero se REVIRTIO -- con solo 9 indicadores, cada uno pesa mas,
## y el sesgo YA CONOCIDO de `priv_barreras_acceso_salud` (proxy mas
## agresivo en ola 1, ver punto 3 del docstring) paso a dominar el score
## agregado: la tasa de pobreza IPM saltaba a 49.6% en ola 1 vs. 36.1%/
## 34.9% en ola 2/3 -- una inestabilidad IMPLANTADA por la exclusion
## misma (menos indicadores = mas peso por indicador = un sesgo conocido
## pesa mas, no menos), peor que el problema que se intentaba resolver.
## Rezago_escolar SI tiene cobertura imperfecta en ola 1 (26.6% vs. ~92%
## en ola 2/3) pero es un problema mucho menor en magnitud que los otros
## 4 ya excluidos (que estaban en 0.4%-35%) -- se acepta esa imperfeccion
## y se mantiene en el score.
PESO_INDICADOR = {
    "priv_bajo_logro_educativo": 0.25 / 2, "priv_analfabetismo": 0.25 / 2,
    "priv_rezago_escolar": 0.25 / 1,  # unico indicador de niñez que sobrevive
    "priv_sin_aseguramiento_salud": 0.25 / 2, "priv_barreras_acceso_salud": 0.25 / 2,
    "priv_agua_mejorada": 0.25 / 5, "priv_eliminacion_excretas": 0.25 / 5,
    "priv_pisos_inadecuados": 0.25 / 5, "priv_paredes_inadecuadas": 0.25 / 5,
    "priv_hacinamiento_critico": 0.25 / 5,
}
COLS_EXCLUIDAS_SCORE = [
    "priv_primera_infancia", "priv_inasistencia_escolar", "priv_trabajo_infantil",
    "priv_empleo_informal", "priv_desempleo_larga_duracion",
]
N_INDICADORES_SCORE = len(PESO_INDICADOR)  # 10 (15 originales - 5 excluidas)


def main() -> None:
    print("Cargando personas, niños y hogares...")
    personas = pd.read_parquet(DATA_DIR / "personas_elca_longitudinal_clean.parquet")
    ninos = pd.read_parquet(DATA_DIR / "ninos_elca_longitudinal_clean.parquet")
    hogar = pd.read_parquet(DATA_DIR / "hogar_elca_longitudinal_clean.parquet")

    print("Construyendo privaciones a nivel persona...")
    p = construir_privaciones_personas(personas)
    priv_personas = agregar_privaciones_hogar_ola(p)

    print("Construyendo privación de primera infancia (niños 0-5)...")
    priv_infancia = construir_privacion_primera_infancia(ninos)

    print("Construyendo privaciones de vivienda...")
    priv_vivienda = construir_privaciones_vivienda(hogar)

    base = hogar[["consecutivo", "ola", "zona"]].drop_duplicates(subset=["consecutivo", "ola"])
    out = base.merge(priv_personas, on=["consecutivo", "ola"], how="left") \
        .merge(priv_infancia, on=["consecutivo", "ola"], how="left") \
        .merge(priv_vivienda, on=["consecutivo", "ola"], how="left")

    cols_priv = list(PESO_INDICADOR.keys())
    cols_priv_auditoria = cols_priv + COLS_EXCLUIDAS_SCORE  # excluidas del score, se reportan igual
    for c in cols_priv_auditoria:
        out[c] = out[c].astype("boolean")

    # priv_primera_infancia y priv_trabajo_infantil son NaN para hogares
    # sin niños en el rango de edad -- se tratan como NO privado en ese
    # indicador (no aplica = no cuenta como privacion), estandar Alkire-Foster.
    for c in ["priv_primera_infancia", "priv_trabajo_infantil", "priv_rezago_escolar",
              "priv_inasistencia_escolar", "priv_desempleo_larga_duracion",
              "priv_empleo_informal", "priv_analfabetismo", "priv_bajo_logro_educativo",
              "priv_sin_aseguramiento_salud", "priv_barreras_acceso_salud"]:
        out[c] = out[c].fillna(False)

    out["ipm_score"] = sum(out[c].astype(float) * peso for c, peso in PESO_INDICADOR.items())
    out["pobre_ipm"] = out["ipm_score"] >= (1 / 3)

    out_path = DATA_DIR / "ipm_multidimensional_elca_longitudinal.parquet"
    out.to_parquet(out_path, index=False)

    print(f"\nGuardado: {out_path}  ({out.shape})")
    print("\n=== Tasa de pobreza IPM por ola ===")
    print(out.groupby("ola")["pobre_ipm"].mean())
    print(f"\n=== Tasa de privación por indicador y ola ({len(COLS_EXCLUIDAS_SCORE)} excluidas del score: {COLS_EXCLUIDAS_SCORE}) ===")
    tabla = out.groupby("ola")[cols_priv_auditoria].mean().T
    tabla.columns = [f"Ola {c}" for c in tabla.columns]
    print((tabla * 100).round(1))


if __name__ == "__main__":
    main()
