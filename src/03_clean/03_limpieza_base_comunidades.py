"""
Limpieza de corrupcion U+FFFD ("�") en comunidades_elca_longitudinal.parquet
(ELCA 2010, 2013, 2016), mismo tratamiento que
`02_limpieza_base_personas.py` pero para el modulo de Comunidades.

Alcance del problema (mucho mas acotado que Personas: 542 columnas
afectadas alla vs. 26 aca)
------------------------------------------------------------------------
De 558 columnas, 26 tienen al menos un valor con "�": 22 de vocabulario
cerrado (<=25 categorias) y 4 de texto libre (`proy_prioritario1/2/3`,
`otros_problemas_cual` -- no se tocan, mismo criterio que el texto libre de
Personas). De las 22 cerradas, la correccion automatica (match unico de
categoria limpia dentro de la MISMA columna, con "^...$" anclado) resuelve
28 valores sin ambiguedad. Queda 1 residual: `region` con 3 valores
corruptos (Bogotá/Atlántica/Pacífica corruptos como "Bogot�"/"Atl�ntica"/
"Pac�fica") -- sin candidato limpio en la misma columna porque esas 3
categorias SIEMPRE aparecen corruptas en el archivo (mismo patron que
`sindicato`/`cotiza_fp`/`estaba_sss`/`estaba_fp` en Personas: sin candidato
limpio local, se corrige a mano porque el valor correcto es inequivoco
-- son los 3 nombres de region de Colombia en el diseño muestral de ELCA).

Validacion: tras la correccion automatica + manual, 0 columnas con "�"
fuera de las 4 de texto libre (verificado con assert, igual que en
Personas).
"""

from pathlib import Path

import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "comunidades_elca_longitudinal.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "comunidades_elca_longitudinal_clean.parquet"
DOC_PATH = PROJECT_ROOT / "docs" / "variable_audit" / "comunidades_corrupcion_residual.csv"

MARCADOR = "�"
CARDINALIDAD_MAXIMA_CERRADA = 25

CORRECCIONES_MANUALES = {
    "region": {
        "Bogot�": "Bogotá",
        "Atl�ntica": "Atlántica",
        "Pac�fica": "Pacífica",
    },
}


def identificar_columnas_afectadas(df: pd.DataFrame) -> tuple[list, list]:
    afectadas = [
        c for c in df.select_dtypes(include="object").columns
        if df[c].astype(str).str.contains(MARCADOR, regex=False, na=False).any()
    ]
    cardinalidad = df[afectadas].nunique()
    cerradas = cardinalidad[cardinalidad <= CARDINALIDAD_MAXIMA_CERRADA].index.tolist()
    abiertas = cardinalidad[cardinalidad > CARDINALIDAD_MAXIMA_CERRADA].index.tolist()
    return cerradas, abiertas


def _patron_desde_corrupto(valor_corrupto: str) -> re.Pattern:
    partes = valor_corrupto.split(MARCADOR)
    return re.compile("^" + ".".join(re.escape(p) for p in partes) + "$")


def corregir_vocabulario_cerrado_automatico(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    df = df.copy()
    for c in columnas:
        serie = df[c].astype(str)
        mask_corrupto = serie.str.contains(MARCADOR, regex=False, na=False)
        if not mask_corrupto.any():
            continue
        limpios = set(serie[~mask_corrupto & (serie != "nan")].unique())
        corruptos = serie[mask_corrupto].unique()
        mapa_col = {}
        for corr in corruptos:
            candidatos = [l for l in limpios if _patron_desde_corrupto(corr).match(l)]
            if len(candidatos) == 1:
                mapa_col[corr] = candidatos[0]
            elif set(candidatos) <= {"Si", "Sí"} and candidatos:
                mapa_col[corr] = "Sí"
        if mapa_col:
            df[c] = df[c].replace(mapa_col)
    return df


def aplicar_correcciones_manuales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapa in CORRECCIONES_MANUALES.items():
        df[col] = df[col].replace(mapa)
        residual = df[col].astype(str).str.contains(MARCADOR, regex=False, na=False)
        assert not residual.any(), f"'{col}' sigue con {residual.sum()} valores corruptos."
    return df


def documentar_residual(df_final: pd.DataFrame, columnas_texto_libre: list) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    for col in df_final.select_dtypes(include="object").columns:
        serie = df_final[col].astype(str)
        mask = serie.str.contains(MARCADOR, regex=False, na=False)
        n = mask.sum()
        if n == 0:
            continue
        tipo = "texto_libre_no_tocado" if col in columnas_texto_libre else "vocabulario_cerrado_sin_match"
        filas.append({
            "columna": col, "tipo": tipo, "n_valores_residuales": n,
            "ejemplo": serie[mask].iloc[0] if tipo == "vocabulario_cerrado_sin_match" else "",
        })
    pd.DataFrame(filas).sort_values("n_valores_residuales", ascending=False).to_csv(DOC_PATH, index=False)


def main() -> None:
    print(f"Leyendo: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Dimensiones de entrada: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")

    cerradas, abiertas = identificar_columnas_afectadas(df)
    print(f"\nColumnas afectadas por '�': {len(cerradas) + len(abiertas)}")
    print(f"  Vocabulario cerrado (<= {CARDINALIDAD_MAXIMA_CERRADA} categorias): {len(cerradas)}")
    print(f"  Texto libre (sin tocar): {len(abiertas)}")

    df = corregir_vocabulario_cerrado_automatico(df, cerradas)
    df = aplicar_correcciones_manuales(df)
    print(f"\nCorreccion manual aplicada en: {list(CORRECCIONES_MANUALES.keys())}")

    documentar_residual(df, abiertas)
    print(f"\nBacklog de columnas no resueltas en: {DOC_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nBase limpia guardada en: {OUTPUT_PATH}")
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")


if __name__ == "__main__":
    main()
