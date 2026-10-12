"""
Limpieza sistematica de corrupcion de codificacion residual en
hogar_elca_longitudinal_clean.parquet (ELCA 2010, 2013, 2016), mismo
tratamiento de 4 capas que Personas/Comunidades/Niños
(`02_limpieza_base_personas.py`, `03_limpieza_base_comunidades.py`,
`04_limpieza_base_ninos.py`).

Por que un script nuevo, separado de `01_limpieza_base_hogar.py`
------------------------------------------------------------------------
`01_limpieza_base_hogar.py` corrige inconsistencias ESTRUCTURALES entre
olas (armonizacion de `region`/`RegionLb`). La correccion de codificacion
('???'/U+FFFD) para Hogar se habia hecho de forma AD-HOC en
`01_download/01_descarga_ELCA/04_consolidacion_bases_hogar.py` (algunas
columnas conocidas: region, RegionLb, algunas categoricas estructuradas),
sin el barrido sistematico de 4 capas (automatico + diccionario + manual +
documentacion de residual) que se aplico a Personas/Comunidades/Niños. Al
escanear la base completa se encontraron 117 columnas con corrupcion
residual sin resolver (92 con U+FFFD, 68 con "???", con solape) -- este
script cierra esa brecha con el mismo estandar de rigor.

Resultado del escaneo
-------------------------
117 columnas afectadas: 97 de vocabulario cerrado (<=25 categorias), 20
de texto libre (sin tocar). La correccion automatica (match unico dentro
de la misma columna, dos marcadores '�' y '???') resuelve 86 valores.
Quedan 65 columnas con 150 valores residuales.

Rescate via diccionario PDF (`{U,R}Hogar.pdf` de 2010 y 2013, busqueda con
limites de palabra `\\b...\\b`, cada caracter de marcador = comodin de 1
caracter): resuelve 50 de los 150 valores restantes con match unico
(`servicio_sanitario`, `obtencion_agua`, `con_quien_4/5/6/7`,
`pacto_meses_1..9`, etc.).

Los 100 valores restantes se dividen en dos grupos:
  - **Genuino texto libre "accidentalmente cerrado"** (cardinalidad chica
    por tamaño de muestra, no por diseño): `con_quien_2/4/5/6/7` (categoria
    "Otro. ¿Cuál?:_____", una plantilla de respuesta abierta, no una
    categoria fija), `aquien_cual_6`, `otro_desastre_cual`,
    `bef_otro_cual`, `destino_cual_4/5/6/8`, `conquien_cual_5/7`,
    `recibio_alim_cual`, `relacion_cual`, `no_seguros` (categoria "Otro.
    ¿Cuál?" mezclada con una categoria fija "No saben qué es un seguro" --
    se corrige la fija a mano, el resto queda libre). Se re-clasifican
    como texto libre y se documentan en el residual sin forzar correccion.
  - **Reconstrucciones de acento inequivocas** (un solo caracter faltante
    por palabra, sin ambiguedad de significado en español, mismo criterio
    ya aplicado en Personas/Niños): `compro_vivcuando`, `destino1_1/2`,
    `destino2_1/2`, `destino2013_1..10`, `destino2016_1/2`,
    `destino_cual_4/5/6/8`, `no_credito_sf1..5`, `rechazo_credito`,
    `religion1/2`, `religion1_2013/religion2_2013`,
    `no_seguros` (categoria fija), `choquec_1/2`, `hizo1c_1/2/3`,
    `hizo2c_1` -- corregidas a mano.

Validacion: 0 columnas con "�" o "???" fuera de las columnas de texto
libre documentadas, tras las 3 capas de correccion (verificado con
assert).

CORRECCION (2026-08-09, al preparar el bloque de features de Hogar):
`con_quien_1`/`con_quien_2`/`con_quien_3` (fuente de financiacion del
prestamo #1/2/3) tenian corrupcion residual NO detectada en el escaneo
original porque su cardinalidad (25/26/26 categorias) cae justo en el
limite o por encima de `CARDINALIDAD_MAXIMA_CERRADA=25` -- el barrido
automatico de vocabulario cerrado las trata como texto libre y nunca las
evalua. Se agregaron manualmente las mismas correcciones ya validadas
para `con_quien_4/5/6/7` ("Cajas de compensaci�n", "Casas de empe�o...",
"Compras por cat�logo") a estas 3 columnas. Leccion: el umbral de
cardinalidad es una heuristica, no una garantia -- columnas cerca del
limite deben revisarse a mano antes de asumir que "es texto libre y no
tiene corrupcion por resolver".
"""

from pathlib import Path

import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
DOC_PATH = PROJECT_ROOT / "docs" / "variable_audit" / "hogar_corrupcion_residual.csv"

MARCADORES = ["�", "???"]
CARDINALIDAD_MAXIMA_CERRADA = 25

CORRECCIONES_DICCIONARIO_PDF = {
    "servicio_sanitario": {"Inodoro sin conexi�n": "Inodoro sin conexión", "Inodoro sin conexi???n": "Inodoro sin conexión"},
    "obtencion_agua": {"Pozo sin bomba, jag�ey": "Pozo sin bomba, jagüey"},
    "con_quien_4": {"Cajas de compensaci�n": "Cajas de compensación"},
    "con_quien_5": {"Cajas de compensaci�n": "Cajas de compensación"},
    "con_quien_6": {"Cajas de compensaci�n": "Cajas de compensación"},
    "con_quien_7": {"Cajas de compensaci�n": "Cajas de compensación"},
    "con_quien_1": {"Cajas de compensaci�n": "Cajas de compensación"},
    "con_quien_2": {"Cajas de compensaci�n": "Cajas de compensación"},
}
for i in range(1, 13):
    CORRECCIONES_DICCIONARIO_PDF[f"pacto_meses_{i}"] = {
        "Se estableci� plazo": "Se estableció plazo", "No se estableci� plazo": "No se estableció plazo",
        "Se estableci???plazo": "Se estableció plazo", "No se estableci???plazo": "No se estableció plazo",
        "Se estableci??? plazo": "Se estableció plazo", "No se estableci??? plazo": "No se estableció plazo",
    }

# Genuino texto libre "accidentalmente cerrado" -- NO se corrigen, se
# documentan como residual (misma logica que dejo_lactar_cual/observ_antrop
# en Niños).
TEXTO_LIBRE_ADICIONAL = [
    "con_quien_2", "con_quien_4", "con_quien_5", "con_quien_6", "con_quien_7",
    "aquien_cual_6", "otro_desastre_cual", "bef_otro_cual",
    "destino_cual_4", "destino_cual_5", "destino_cual_6", "destino_cual_8",
    "conquien_cual_5", "conquien_cual_7", "recibio_alim_cual", "relacion_cual",
    "no_seguros",
]

CORRECCIONES_MANUALES = {
    "compro_vivcuando": {
        "Ocup??? un terreno de hecho y construyó esta vivienda por sus propios medios":
            "Ocupó un terreno de hecho y construyó esta vivienda por sus propios medios",
        "Hered??? o recibi??? como cesi???n esta vivienda": "Heredó o recibió como cesión esta vivienda",
    },
    "destino1_1": {"Inversiones en conservaci???n de suelos y reservas de agua": "Inversiones en conservación de suelos y reservas de agua"},
    "destino1_2": {"Inversiones en conservaci???n de suelos y reservas de agua": "Inversiones en conservación de suelos y reservas de agua"},
    "destino2_2": {"Inversiones en conservaci???n de suelos y reservas de agua": "Inversiones en conservación de suelos y reservas de agua"},
    "con_quien_1": {
        "Casas de empe�o o casas comerciales": "Casas de empeño o casas comerciales",
        "Compras por cat�logo": "Compras por catálogo",
        "Otro. Cu�l:_____": "Otro. ¿Cuál?:_____",
        "Otro. Cu???l:_____": "Otro. ¿Cuál?:_____",
    },
    "con_quien_2": {
        "Casas de empe�o o casas comerciales": "Casas de empeño o casas comerciales",
        "Compras por cat�logo": "Compras por catálogo",
        "Otro. Cu�l:_____": "Otro. ¿Cuál?:_____",
        "Otro. Cu???l:_____": "Otro. ¿Cuál?:_____",
    },
    "con_quien_3": {
        "Cajas de compensaci�n": "Cajas de compensación",
        "Casas de empe�o o casas comerciales": "Casas de empeño o casas comerciales",
        "Compras por cat�logo": "Compras por catálogo",
        "Otro. Cu�l:_____": "Otro. ¿Cuál?:_____",
        "Otro. Cu???l:_____": "Otro. ¿Cuál?:_____",
    },
    "con_quien_5": {"Casas de empe�o o casas comerciales": "Casas de empeño o casas comerciales"},
    "no_seguros": {"No saben qu??? es un seguro": "No saben qué es un seguro"},
    "religion1_2013": {
        "Iglesia de los Santos de los Ultimos D�as - mormones": "Iglesia de los Santos de los Últimos Días - mormones",
        "Ateo (agn�stico o ateo, cree en un ser superior pero no pertenece a ninguna religi�n)":
            "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religión)",
        "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religi???n)":
            "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religión)",
    },
    "religion2_2013": {
        "Ateo (agn�stico o ateo, cree en un ser superior pero no pertenece a ninguna religi�n)":
            "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religión)",
        "Iglesia de los Santos de los Ultimos D�as - mormones": "Iglesia de los Santos de los Últimos Días - mormones",
        "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religi???n)":
            "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religión)",
    },
    "religion1": {
        "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religi???n)":
            "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religión)",
        "Iglesia de los Santos de los Ultimos D???as - mormones": "Iglesia de los Santos de los Últimos Días - mormones",
    },
    "religion2": {
        "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religi???n)":
            "Ateo (agnóstico o ateo, cree en un ser superior pero no pertenece a ninguna religión)",
        "Iglesia de los Santos de los Ultimos D???as - mormones": "Iglesia de los Santos de los Últimos Días - mormones",
    },
    "recibio_ayu_alim": {
        "Acuerdo extrajudicial con intervenci�n del ICBF": "Acuerdo extrajudicial con intervención del ICBF",
        "Acuerdo extrajudicial con intervenci???n del ICBF": "Acuerdo extrajudicial con intervención del ICBF",
    },
    "envio_ayu_alim": {
        "Acuerdo extrajudicial con intervenci�n del ICBF": "Acuerdo extrajudicial con intervención del ICBF",
        "Acuerdo extrajudicial con intervenci???n del ICBF": "Acuerdo extrajudicial con intervención del ICBF",
    },
    "rechazo_credito": {
        "No ten�a suficiente ingreso": "No tenía suficiente ingreso",
        "No ten???a suficiente ingreso": "No tenía suficiente ingreso",
    },
    "choquec_1": {"Pandillas o delincuencia com???n": "Pandillas o delincuencia común"},
    "choquec_2": {"Pandillas o delincuencia com???n": "Pandillas o delincuencia común"},
}
for i in [1, 2, 3]:
    CORRECCIONES_MANUALES[f"hizo1c_{i}"] = {
        "Aumentaron la cooperaci???n con las autoridades": "Aumentaron la cooperación con las autoridades",
        "Miembros del hogar salieron del pa???s": "Miembros del hogar salieron del país",
        "Hipotecaron alg???n activo (casa, carro, finca, etc.)": "Hipotecaron algún activo (casa, carro, finca, etc.)",
    }
CORRECCIONES_MANUALES["hizo2c_1"] = {
    "Hipotecaron alg???n activo (casa, carro, finca, etc.)": "Hipotecaron algún activo (casa, carro, finca, etc.)",
}
for i in range(1, 11):
    CORRECCIONES_MANUALES[f"destino2013_{i}"] = {
        "Comprar muebles o  electrodom�sticos": "Comprar muebles o electrodomésticos",
        "Recreaci�n, celebraciones viajes y entretenimiento": "Recreación, celebraciones viajes y entretenimiento",
        "Cubrir los da�os de desastres naturales": "Cubrir los daños de desastres naturales",
        "Cubrir los da???os de desastres naturales": "Cubrir los daños de desastres naturales",
        "Pagar la educaci�n de los hijos o propia": "Pagar la educación de los hijos o propia",
        "Otro: �cu�l?": "Otro: ¿cuál?",
    }
for i in [1, 2]:
    CORRECCIONES_MANUALES[f"destino2016_{i}"] = {
        "Cubrir los da???os de desastres naturales": "Cubrir los daños de desastres naturales",
    }
for i in [1, 2]:
    CORRECCIONES_MANUALES[f"no_credito_sf{i}"] = {
        "Est� reportado en centrales de riesgo": "Está reportado en centrales de riesgo",
        "Usted o alg�n conocido tuvo una mala experiencia en el sector financiero":
            "Usted o algún conocido tuvo una mala experiencia en el sector financiero",
        "Ya tiene crédito, est??? demasiado endeudado": "Ya tiene crédito, está demasiado endeudado",
        "Est??? reportado en centrales de riesgo": "Está reportado en centrales de riesgo",
        "Usted o alg???n conocido tuvo una mala experiencia en el sector financiero":
            "Usted o algún conocido tuvo una mala experiencia en el sector financiero",
        "No sabe c�mo hacerlo": "No sabe cómo hacerlo",
        "No sabe c???mo hacerlo": "No sabe cómo hacerlo",
    }
for i in [3, 4, 5]:
    CORRECCIONES_MANUALES[f"no_credito_sf{i}"] = {
        "Tiene acceso a cr�ditos de otras fuentes": "Tiene acceso a créditos de otras fuentes",
        "Usted o alg�n conocido tuvo una mala experiencia en el sector financiero":
            "Usted o algún conocido tuvo una mala experiencia en el sector financiero",
    }


def contiene_marcador(serie: pd.Series) -> pd.Series:
    s = serie.astype(str)
    mask = pd.Series(False, index=serie.index)
    for m in MARCADORES:
        mask |= s.str.contains(m, regex=False, na=False)
    return mask


def identificar_columnas_afectadas(df: pd.DataFrame) -> tuple[list, list]:
    afectadas = [
        c for c in df.select_dtypes(include="object").columns
        if contiene_marcador(df[c]).any()
    ]
    afectadas = [c for c in afectadas if c not in TEXTO_LIBRE_ADICIONAL]
    cardinalidad = df[afectadas].nunique()
    cerradas = cardinalidad[cardinalidad <= CARDINALIDAD_MAXIMA_CERRADA].index.tolist()
    abiertas = cardinalidad[cardinalidad > CARDINALIDAD_MAXIMA_CERRADA].index.tolist()
    return cerradas, abiertas + TEXTO_LIBRE_ADICIONAL


def _patron_desde_corrupto(valor_corrupto: str, marcador: str) -> re.Pattern:
    partes = valor_corrupto.split(marcador)
    return re.compile("^" + ".".join(re.escape(p) for p in partes) + "$")


def corregir_vocabulario_cerrado_automatico(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    df = df.copy()
    for c in columnas:
        for marcador in MARCADORES:
            serie = df[c].astype(str)
            mask_corrupto = serie.str.contains(marcador, regex=False, na=False)
            if not mask_corrupto.any():
                continue
            limpios = set(serie[~contiene_marcador(df[c]) & (serie != "nan")].unique())
            corruptos = serie[mask_corrupto].unique()
            mapa_col = {}
            for corr in corruptos:
                candidatos = [l for l in limpios if _patron_desde_corrupto(corr, marcador).match(l)]
                if len(candidatos) == 1:
                    mapa_col[corr] = candidatos[0]
                elif set(candidatos) <= {"Si", "Sí"} and candidatos:
                    mapa_col[corr] = "Sí"
            if mapa_col:
                df[c] = df[c].replace(mapa_col)
    return df


def aplicar_correcciones_diccionario_pdf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapa in CORRECCIONES_DICCIONARIO_PDF.items():
        df[col] = df[col].replace(mapa)
    return df


def aplicar_correcciones_manuales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapa in CORRECCIONES_MANUALES.items():
        df[col] = df[col].replace(mapa)
    return df


def documentar_residual(df_final: pd.DataFrame, columnas_texto_libre: list) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    for col in df_final.select_dtypes(include="object").columns:
        mask = contiene_marcador(df_final[col])
        n = mask.sum()
        if n == 0:
            continue
        tipo = "texto_libre_no_tocado" if col in columnas_texto_libre else "vocabulario_cerrado_sin_match"
        filas.append({
            "columna": col, "tipo": tipo, "n_valores_residuales": n,
            "ejemplo": df_final.loc[mask, col].astype(str).iloc[0] if tipo == "vocabulario_cerrado_sin_match" else "",
        })
    pd.DataFrame(filas).sort_values("n_valores_residuales", ascending=False).to_csv(DOC_PATH, index=False)


def main() -> None:
    print(f"Leyendo: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Dimensiones de entrada: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")

    cerradas, abiertas = identificar_columnas_afectadas(df)
    print(f"\nColumnas afectadas por corrupcion ('�' o '???'): {len(cerradas) + len(abiertas)}")
    print(f"  Vocabulario cerrado (<= {CARDINALIDAD_MAXIMA_CERRADA} categorias): {len(cerradas)}")
    print(f"  Texto libre (sin tocar): {len(abiertas)}")

    df = corregir_vocabulario_cerrado_automatico(df, cerradas)
    df = aplicar_correcciones_diccionario_pdf(df)
    df = aplicar_correcciones_manuales(df)

    residual_cerradas = sum(contiene_marcador(df[c]).sum() for c in cerradas)
    print(f"\nValores residuales en columnas cerradas tras las 3 capas: {residual_cerradas}")

    documentar_residual(df, abiertas)
    print(f"Backlog de columnas no resueltas en: {DOC_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nBase limpia sobre-escrita en: {OUTPUT_PATH}")
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")


if __name__ == "__main__":
    main()
