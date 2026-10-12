"""
Limpieza de corrupcion de codificacion en ninos_elca_longitudinal.parquet
(ELCA 2010, 2013, 2016), mismo tratamiento (4 capas) que
`02_limpieza_base_personas.py`: correccion automatica de vocabulario
cerrado, rescate via diccionarios PDF, correccion manual, documentacion
del residual.

DOS marcadores de corrupcion distintos (mismo patron ya documentado para
Personas: U+FFFD en unas columnas, "???" literal en otras -- aqui
Niños tiene AMBOS, a diferencia de Personas donde "???" solo aparecia en
ola 3)
--------------------------------------------------------------------------
1. **U+FFFD ("�")**: 80 de 433 columnas de texto (79 de vocabulario
   cerrado <=25 categorias, 1 de texto libre: `descrip_oficio`, sin
   tocar). Toda la corrupcion residual (tras la correccion automatica de
   131 valores) esta en ola 2 (2013) -- mismo origen que Personas.
   Quedan 16 columnas con residual (25 valores): `resultado_m`,
   `razon_no_asiste`, `quien_cuida`, `freq_vetv`, `rz_dejo_lactar`
   rescatados via los diccionarios PDF de 2013 (`{U,R}Ninos0a13.pdf`,
   busqueda `\\b...\\b` con cada "�" como comodin de 1 caracter, match
   unico); `adq_icbf`/`curso_otro`/`curso_otro_cual` ("S�" con candidatos
   ambiguos {Si, Sí, Su} en el diccionario -- "Su" es ruido de texto no
   relacionado a una categoria -- se aplica la regla de Personas: si el
   candidato de VOCABULARIO real es {Si, Sí}, se usa "Sí"); el resto
   corregido a mano por ser reconstrucciones de acento inequivocas.

2. **"???" literal**: encontrado DESPUES, al validar `quien_cuida` contra
   el join de cuidadores (categoria "Una ni???era" seguia sin resolver
   tras la capa 1 porque en ESA fila especifica el marcador era "???", no
   "�" -- la misma columna tiene ambos tipos de corrupcion en filas
   distintas). Afecta 12 columnas en las 3 olas (a diferencia de Personas,
   donde "???" era exclusivo de ola 3): `educ_padre`, `educ_madre`,
   `razon_no_asiste`, `razon_no_asiste_cual`, `quien_cuida`,
   `pcuida_niveledu`, `actividad`, `nivel_edu_cuidador`, `rz_dejo_lactar`,
   `dejo_lactar_cual`, `descrip_oficio`, `observ_antrop`. De estas, 9 son
   de vocabulario cerrado por cardinalidad pero varias resultaron ser
   texto libre "accidentalmente cerrado" por tamaño de muestra chico
   (`dejo_lactar_cual`, `observ_antrop` -- frases completas, no
   categorias) -- se tratan como texto libre y se documentan en el
   residual sin forzar correccion. El resto (`educ_padre`, `educ_madre`,
   `pcuida_niveledu`, `razon_no_asiste`, `quien_cuida`,
   `nivel_edu_cuidador`, `rz_dejo_lactar`) se corrige a mano, mismo
   criterio de reconstruccion de acento inequivoca.

De las 21 columnas con corrupcion residual identificadas entre los dos
marcadores, SOLO `quien_cuida` se usa en `build_ninos_hogar.py`
(`pct_ninos_cuidado_terceros_hogar`) -- las demas (`educ_padre`,
`educ_madre`, `pcuida_niveledu`, etc.) se corrigen igual por completitud
de la auditoria, aunque no entren al benchmark (redundantes con Personas,
ver docstring de `build_ninos_hogar.py`).

Validacion: 0 columnas con "�" o "???" fuera de las columnas de texto
libre documentadas, tras las 3 capas de correccion (verificado con
assert).
"""

from pathlib import Path

import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ninos_elca_longitudinal.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ninos_elca_longitudinal_clean.parquet"
DOC_PATH = PROJECT_ROOT / "docs" / "variable_audit" / "ninos_corrupcion_residual.csv"

MARCADORES = ["�", "???"]
CARDINALIDAD_MAXIMA_CERRADA = 25

CORRECCIONES_DICCIONARIO_PDF = {
    "resultado_m": {"S�lo se tom� talla": "Sólo se tomó talla"},
    "razon_no_asiste": {"No hay instituci�n cerca": "No hay institución cerca"},
    "quien_cuida": {"Una ni�era": "Una niñera"},
    "freq_vetv": {"Cada 15 d�as": "Cada 15 días"},
    "rz_dejo_lactar": {
        "Qued� embarazada": "Quedó embarazada",
        "Presi�n de su pareja": "Presión de su pareja",
    },
}

# Columnas que resultaron ser texto libre "accidentalmente cerrado" (frases
# completas, no categorias) -- se excluyen del barrido automatico y se
# documentan como residual sin forzar correccion.
TEXTO_LIBRE_ADICIONAL = ["dejo_lactar_cual", "observ_antrop"]

CORRECCIONES_MANUALES = {
    "adq_icbf": {"S�": "Sí"},
    "curso_otro": {"S�": "Sí"},
    "curso_otro_cual": {"S�": "Sí", "Peque�in": "Pequeñín"},
    "educ_padre": {
        "Uno o m�s a�os de t�cnica o tecnol�gica": "Uno o más años de técnica o tecnológica",
        "Uno o más años de técnica o tecnol???gica": "Uno o más años de técnica o tecnológica",
    },
    "educ_madre": {
        "Uno o m�s a�os de t�cnica o tecnol�gica": "Uno o más años de técnica o tecnológica",
        "Uno o más años de técnica o tecnol???gica": "Uno o más años de técnica o tecnológica",
    },
    "pcuida_niveledu": {
        "Uno o m�s a�os de t�cnica o tecnol�gica": "Uno o más años de técnica o tecnológica",
        "Uno o más años de técnica o tecnol???gica": "Uno o más años de técnica o tecnológica",
    },
    "resultado_m": {"Ni�o(a)/Madre rehus�": "Niño(a)/Madre rehusó"},
    "cuidado_prefiere": {"Otro, cu�l?": "Otro, cuál?"},
    "parent_inform": {
        "Servicio dom�stico, cuidandero y sus parientes": "Servicio doméstico, cuidandero y sus parientes",
    },
    "rz_dejo_lactar": {
        "Por rechazo del beb� a amamantar": "Por rechazo del bebé a amamantar",
        "Por rechazo del beb??? a amamantar": "Por rechazo del bebé a amamantar",
        "Qued??? embarazada": "Quedó embarazada",
    },
    "nofrutas_cual": {
        "Le hacen da�o algunas frutas como el mango y la manzana":
            "Le hacen daño algunas frutas como el mango y la manzana",
    },
    "nocarnes_cual": {"La ni�a no puede prepararla sola": "La niña no puede prepararla sola"},
    "razon_no_asiste": {
        "Requiere atenci�n o educaci�n especial": "Requiere atención o educación especial",
        "No hay instituci???n cerca": "No hay institución cerca",
        "Requiere atenci???n o educaci???n especial": "Requiere atención o educación especial",
    },
    "quien_cuida": {"Una ni???era": "Una niñera"},
    "nivel_edu_cuidador": {
        "Tecnol???gico sin título": "Tecnológico sin título",
        "Tecnol???gico con título": "Tecnológico con título",
    },
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
        residual = contiene_marcador(df[col])
        assert not residual.any(), f"'{col}' sigue con {residual.sum()} valores corruptos."
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
    print(f"\nCorreccion via diccionario PDF aplicada en: {list(CORRECCIONES_DICCIONARIO_PDF.keys())}")
    print(f"Correccion manual aplicada en: {list(CORRECCIONES_MANUALES.keys())}")

    documentar_residual(df, abiertas)
    print(f"\nBacklog de columnas no resueltas en: {DOC_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nBase limpia guardada en: {OUTPUT_PATH}")
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")


if __name__ == "__main__":
    main()
