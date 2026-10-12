"""
tabla_temas_variables.py
=====================================
Clasifica las 177 variables finales del consolidado de la ELCA por TEMA
(no por modulo de la encuesta) y genera la tabla del Anexo
(Tabla~\\ref{tab:temas_variables}): numero de variables por tema y
ejemplos.

Reutiliza las categorias tematicas ya asignadas a las 53 variables del
perfil (`CATEGORIA` de `src/etiquetas_variables.py`, antes en
`src/02_build/eda_perfil_completo.py`), para que un
mismo tema tenga el mismo nombre en esta tabla y en las tablas de perfil.
Las 124 variables restantes se asignan en `TEMA_EXTRA` (a mano, por
variable) o, para modulos enteramente tematicos, por modulo
(`TEMA_POR_MODULO`). Cuatro temas no existian en `CATEGORIA` porque
ninguna de sus variables entro al perfil: Salud, Choques y afrontamiento,
Ayudas y redes de apoyo, Participacion y capital social.

INPUTS
    outputs/tables/eda_variables_modelo/01_inventario_variables.csv

OUTPUTS
    paper/tables/tab_temas_variables.tex

COMO CORRER
    python src/tabla_temas_variables.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from etiquetas_variables import CATEGORIA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402


INVENTARIO_PATH = REPO_ROOT / "outputs" / "tables" / "eda_variables_modelo" / "01_inventario_variables.csv"
OUTPUT_PATH = REPO_ROOT / "paper" / "tables" / "tab_temas_variables.tex"
N_ESPERADO = 177


def _asignar(tema: str, variables: list) -> dict:
    return {v: tema for v in variables}


TEMA_EXTRA = {
    **_asignar("Ingreso y gasto", [
        "ingreso_percapita_hogar", "gasto_percapita_hogar", "lp", "li", "pobre_ingreso",
        "pobre_extremo_ingreso", "pobre_gasto", "pobre_extremo_gasto", "concuerdan_ingreso_gasto",
    ]),
    **_asignar("Composición del hogar", [
        "n_ninos_5", "n_adultos_mayores", "pct_mujeres_hogar", "sexo_jefe", "edad_jefe",
        "tiene_conyuge_jefe", "edad_union_jefe", "pct_ninos_padre_vivo", "pct_ninos_madre_viva",
    ]),
    **_asignar("Educación, empleo y seguridad social", [
        "ocupado_jefe", "horas_trabajo_jefe", "jornalero_jefe", "grado_educ_jefe",
        "categoria_laboral_oit_jefe", "pct_adultos_alfabetizados", "tasa_asistencia_escolar",
        "tasa_ocupacion_hogar", "pct_adultos_fue_jornalero", "pct_ninos_trabaja_otro_hogar",
        "pct_ninos_no_estudia_razon_economica", "pct_ninos_recibio_beca_subsidio",
        "pct_ninos_credito_estudiar", "pct_ninos_apoyo_material_escolar",
    ]),
    **_asignar("Salud", [
        "tuvo_evento_salud_jefe", "pct_ninos_con_discapacidad", "n_eventos_salud_hogar",
        "tuvo_hospitalizacion_hogar", "tasa_afiliacion_salud_hogar", "tiene_prepagada_hogar",
        "tasa_control_preventivo_hogar", "pct_ninos_control_pediatrico",
        "tasa_planificacion_familiar", "pct_ninos_beneficiario_sss",
    ]),
    **_asignar("Activos del hogar", [
        "ahorra_jefe", "tasa_ahorro_hogar", "tiene_propiedad_rural_hogar",
        "tiene_transporte_carga_hogar", "tiene_ingreso_agropecuario_hogar",
    ]),
    **_asignar("Vivienda: materiales y servicios", [
        "n_hogares_comparte_vivienda_hogar", "financio_recursos_propios_vivienda_hogar",
        "financio_subsidio_vivienda_hogar", "financio_otra_fuente_vivienda_hogar",
        "tiene_titulo_baldio_hogar",
    ]),
    **_asignar("Programas sociales y deuda", ["beneficiario_red_juntos_hogar", "tiene_deuda_hogar"]),
    **_asignar("Ayudas y redes de apoyo", [
        "recibio_ayuda_alimentos_hogar", "recibio_ayuda_fam_colombia_hogar",
        "recibio_ayuda_fam_exterior_hogar", "recibio_ayuda_ong_hogar",
        "recibio_ayuda_org_internacional_hogar", "recibio_ayuda_religiosa_hogar",
        "recibio_ayuda_desplazados_hogar", "envio_ayuda_alimentos_hogar",
        "envio_ayuda_fam_colombia_hogar", "envio_ayuda_fam_exterior_hogar", "envio_ayuda_otras_hogar",
        "uso_ayuda_alimentos_hogar", "uso_ayuda_salud_hogar", "uso_ayuda_educacion_hogar",
        "uso_ayuda_vivienda_hogar", "uso_ayuda_agropecuario_hogar", "uso_ayuda_ahorrar_hogar",
        "n_tipos_ayuda_recibida_hogar",
    ]),
    **_asignar("Participación y capital social", [
        "participa_organizacion_jefe", "n_tipos_organizacion_jefe", "pct_hogar_participa_organizacion",
        "tasa_participacion_civica_hogar", "practica_religion_hogar",
    ]),
    **_asignar("Choques y afrontamiento", ["tuvo_desastre_natural_hogar"]),
}

# Modulos cuyas variables restantes caen todas en un mismo tema.
TEMA_POR_MODULO = {
    "Choques": "Choques y afrontamiento",
    "Comunidades": "Comunidad",
    "Ninos (6-9 anios)": "Desarrollo infantil (6--9 años)",
}

EJEMPLOS = {
    "Educación, empleo y seguridad social": "nivel educativo y categoría ocupacional del jefe, cotización a pensión, tasa de ocupación del hogar",
    "Comunidad": "percepción de inseguridad, acceso a agua, puesto de salud, acciones del conflicto armado",
    "Ayudas y redes de apoyo": "ayudas de familiares en Colombia y el exterior, de ONG o de organizaciones religiosas; envío de ayudas",
    "Choques y afrontamiento": "choques económicos, de salud o agropecuarios; afrontamiento erosivo; desastres naturales",
    "Vivienda: materiales y servicios": "material de pisos y paredes, servicio sanitario, tenencia, escritura de la vivienda",
    "Composición del hogar": "niños menores de 12 años, adultos mayores, razón de dependencia, sexo, edad, etnia y estado civil del jefe",
    "Activos del hogar": "bienes durables, índice de riqueza (PCA), estrato, vehículo, ahorro",
    "Ingreso y gasto": "ingreso y gasto per cápita real, brechas frente a la LP, indicadores de pobreza",
    "Desarrollo infantil (6--9 años)": "vacunación, talla y peso, oficios y trabajo infantil, puntaje TVIP",
    "Salud": "afiliación a salud, hospitalización, discapacidad, controles preventivos",
    "Programas sociales y deuda": "Familias en Acción, Red Juntos, apoyo alimentario escolar, deuda formal e informal",
    "Participación y capital social": "participación en organizaciones, participación cívica, práctica religiosa",
    "Vivienda: hacinamiento": "personas por habitación y por dormitorio, valor del arriendo",
}


def asignar_temas(inv: pd.DataFrame) -> pd.Series:
    def tema(fila) -> str:
        v = fila["variable"]
        if v in CATEGORIA:
            return CATEGORIA[v][0]
        if v in TEMA_EXTRA:
            return TEMA_EXTRA[v]
        if fila["modulo"] in TEMA_POR_MODULO:
            return TEMA_POR_MODULO[fila["modulo"]]
        raise ValueError(f"Variable sin tema asignado: {v} (modulo {fila['modulo']})")

    return inv.apply(tema, axis=1)


def main() -> None:
    inv = pd.read_csv(INVENTARIO_PATH)
    if len(inv) != N_ESPERADO:
        raise ValueError(f"Se esperaban {N_ESPERADO} variables en el inventario, hay {len(inv)}.")
    inv["tema"] = asignar_temas(inv)

    conteo = inv["tema"].value_counts()
    sin_ejemplo = set(conteo.index) - set(EJEMPLOS)
    if sin_ejemplo:
        raise ValueError(f"Temas sin ejemplos en EJEMPLOS: {sin_ejemplo}")

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Temas cubiertos por las 177 variables de la ELCA usadas en los modelos.}",
        r"  \label{tab:temas_variables}",
        r"  \footnotesize",
        r"  \begin{tabular}{>{\raggedright\arraybackslash}p{4.2cm}r>{\raggedright\arraybackslash}p{8.6cm}}",
        r"    \toprule",
        r"    \textbf{Tema} & \textbf{N.º} & \textbf{Ejemplos} \\",
        r"    \midrule",
    ]
    for tema, n in conteo.items():
        lineas.append(f"    {tema} & {n} & {EJEMPLOS[tema]} \\\\")
    lineas += [
        r"    \midrule",
        f"    \\textbf{{Total}} & {conteo.sum()} & \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Clasificación temática propia, independiente del",
        r"    módulo de la ELCA del que proviene cada variable; los temas son los mismos que",
        r"    agrupan el núcleo SHAP (Tabla~\ref{tab:shap_signo_nucleo}). No incluye las variables geoespaciales",
        r"    (Anexo~\ref{apx:variables_geo}). " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    tex = aplicar_signo_menos("\n".join(lineas))
    OUTPUT_PATH.write_text(tex + "\n", encoding="utf-8")
    print(f"Guardado: {OUTPUT_PATH}\n")
    print(conteo.to_string())


if __name__ == "__main__":
    main()
