"""
02_unir_panel_elca.py
=======================

Pega variables_geoespaciales_unificadas.parquet (output de
01_unificar_variables.py) sobre el panel ELCA longitudinal
(hogar_elca_longitudinal_clean.parquet), sin exponer ninguna columna de
coordenadas en el resultado.

QUÉ HACE

    1. Carga el panel ELCA y el archivo geoespacial unificado.
    2. Construye id_ola en el panel ELCA con la MISMA lógica que
       00_preparar_coordenadas.py (sentinel1_pipeline): por fila, según
       su 'ola', toma consecutivo (2010), llave (2013) o llave_n16 (2016)
       -- necesario porque 'consecutivo' se repite dentro de una ola
       cuando el hogar se dividió, y el archivo geoespacial ya está
       indexado por id_ola, no por consecutivo.
    3. Verifica que id_ola sea único en el archivo geoespacial (debe
       serlo, viene de 01_unificar_variables.py).
    4. Pega con LEFT JOIN por (id_ola, ola) -- el panel ELCA nunca pierde
       filas, aunque una fila no tenga variables geoespaciales (fuera de
       2010/2013, o sin coordenada válida).
    5. Verifica de nuevo, de forma bloqueante, que no haya quedado
       ninguna columna de coordenadas en el resultado.
    6. Exporta el panel final y un reporte de cobertura.

INPUTS

    ../../../tesis_vulnerabilidad/data/processed/hogar_elca_longitudinal_clean.parquet
    data/processed/variables_geoespaciales_unificadas.parquet (output de 01_unificar_variables.py)

OUTPUTS

    data/processed/hogar_elca_longitudinal_con_geoespacial.parquet
    logs/reporte_union_panel_elca.txt

CÓMO CORRER

    python 02_unir_panel_elca.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import re
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent            # .../unificacion_geoespacial/scripts

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Inputs ──────────────────────────────────────────────────────────────────
    "panel_elca": (
        _HERE.parent.parent.parent
        / "tesis_vulnerabilidad" / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
    ),
    "variables_geoespaciales": _HERE.parent / "data" / "processed" / "variables_geoespaciales_unificadas.parquet",

    # ── Identificador único de hogar dentro de cada ola (igual que ────────────
    # sentinel1_pipeline/scripts/00_preparar_coordenadas.py::estandarizar_columnas) ─
    "id_unico_por_ola": {
        2010: "consecutivo",
        2013: "llave",
        2016: "llave_n16",
    },

    # ── Llave de unión con el archivo geoespacial ─────────────────────────────
    "llave_union": ["id_ola", "ola"],

    # ── Patrón para verificar, de forma bloqueante, que no quede ninguna ─────
    # columna de coordenadas en el resultado final.
    "patron_columnas_coordenadas": re.compile(r"(lat|lon|latitud|longitud)", re.IGNORECASE),

    # ── Carpetas de salida ─────────────────────────────────────────────────────
    "output_dir": _HERE.parent / "data" / "processed",
    "logs_dir": _HERE.parent / "logs",
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: CARGA
# ──────────────────────────────────────────────────────────────────────────────

def cargar_parquet(ruta: Path, nombre: str) -> pd.DataFrame:
    """Lee un archivo parquet. Termina el proceso con diagnóstico si no existe."""
    ruta = Path(ruta)
    if not ruta.exists():
        log.error("-" * 62)
        log.error(f"ERROR: no se encontró {nombre}.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"  {nombre}: {len(df):,} filas x {df.shape[1]} columnas — {ruta.name}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: CONSTRUCCIÓN DE id_ola EN EL PANEL ELCA
# ──────────────────────────────────────────────────────────────────────────────

def construir_id_ola(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Agrega id_ola al panel ELCA: por fila, según su 'ola', toma el valor
    de la columna correspondiente (consecutivo/llave/llave_n16) y lo
    castea a str -- mismo criterio que 00_preparar_coordenadas.py, para
    que id_ola calce exactamente con el del archivo geoespacial.

    llave/llave_n16 suelen venir como float64 (por los NaN de las filas
    de otras olas) -- se convierten a entero antes de castear a str para
    no dejar '213000101.0' en vez de '213000101'.
    """
    panel = panel.copy()
    id_por_ola = cfg["id_unico_por_ola"]
    panel["id_ola"] = pd.NA

    for ola, col_id in id_por_ola.items():
        mask_ola = panel["ola"] == ola
        if col_id not in panel.columns:
            log.warning(f"  Ola {ola}: columna '{col_id}' no encontrada en el panel ELCA. id_ola quedará NA para estas filas.")
            continue
        valores = panel.loc[mask_ola, col_id]
        valores_str = valores.apply(
            lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer() else (str(v) if pd.notna(v) else pd.NA)
        )
        panel.loc[mask_ola, "id_ola"] = valores_str

    n_sin_id = int(panel["id_ola"].isna().sum())
    if n_sin_id > 0:
        log.warning(f"  {n_sin_id:,} filas del panel ELCA quedaron sin id_ola (no es necesariamente un error -- revisar si son olas sin coordenadas, p. ej. 2019/2022).")

    return panel


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: VALIDACIÓN Y UNIÓN
# ──────────────────────────────────────────────────────────────────────────────

def verificar_unicidad(df: pd.DataFrame, llave: list, nombre: str) -> None:
    """Verifica que `llave` identifique una única fila en `df`. Termina el proceso si encuentra duplicados."""
    n_dup = df.dropna(subset=llave).duplicated(subset=llave).sum()
    if n_dup > 0:
        log.error("-" * 62)
        log.error(f"ERROR: {nombre} tiene {n_dup:,} filas duplicadas en la llave {llave}.")
        log.error("-" * 62)
        sys.exit(1)


def unir_geoespacial(panel: pd.DataFrame, geoespacial: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Pega `geoespacial` sobre `panel` usando LEFT JOIN por CONFIG['llave_union'].

    Solo se pegan la llave y las columnas de variables geoespaciales
    (dmsp_*, alos_*, l5_*) -- 'consecutivo', 'es_split' y
    'coord_valida_ola' existen en ambas tablas (el panel ELCA ya trae las
    suyas) y quedarían duplicadas con sufijos _x/_y si se pegaran tal
    cual.
    """
    llave = cfg["llave_union"]
    prefijos_variables = ("dmsp_", "alos_", "l5_")
    columnas_a_pegar = list(llave) + [c for c in geoespacial.columns if c.startswith(prefijos_variables)]

    antes = len(panel)
    resultado = panel.merge(geoespacial[columnas_a_pegar], on=llave, how="left", validate="many_to_one")
    assert len(resultado) == antes, "El merge cambió el número de filas -- revisar unicidad de la llave en el archivo geoespacial."
    return resultado


def verificar_sin_coordenadas(df: pd.DataFrame, cfg: dict) -> None:
    """Verificación final bloqueante: termina el proceso si queda alguna columna de coordenadas."""
    patron = cfg["patron_columnas_coordenadas"]
    quedan = [c for c in df.columns if patron.search(c)]
    if quedan:
        log.error("-" * 62)
        log.error(f"ERROR: quedaron columnas de coordenadas en el panel final: {quedan}")
        log.error("  No se exporta el archivo.")
        log.error("-" * 62)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(df: pd.DataFrame, cfg: dict) -> str:
    """Genera el texto del reporte de calidad de la unión con el panel ELCA."""
    sep = "=" * 65
    prefijos = ["dmsp_", "alos_", "l5_"]
    lines = [
        sep,
        "REPORTE — UNIÓN DE VARIABLES GEOESPACIALES AL PANEL ELCA",
        sep,
        "",
        "── COBERTURA POR OLA ──────────────────────────────────────",
        "",
    ]
    cols_geo = [c for c in df.columns if any(c.startswith(p) for p in prefijos)]
    for ola, grupo in df.groupby("ola"):
        n_con_geo = int(grupo[cols_geo].notna().any(axis=1).sum()) if cols_geo else 0
        lines.append(f"  Ola {ola}: {len(grupo):,} hogares totales — {n_con_geo:,} con al menos una variable geoespacial")

    lines += [
        "",
        f"  Filas totales: {len(df):,}",
        f"  Columnas totales: {df.shape[1]}",
        f"  Columnas geoespaciales agregadas: {len(cols_geo)}",
        "",
        "  Verificación: 0 columnas de coordenadas en el archivo final (bloqueante si no).",
        "",
        sep,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  02_unir_panel_elca.py")
    log.info("  Si ves ModuleNotFoundError, el entorno virtual no esta")
    log.info("  activado o las dependencias no estan instaladas.")
    log.info("  Solucion:")
    log.info("    Mac/Linux : source .venv/bin/activate")
    log.info("    Windows   : .venv\\Scripts\\activate")
    log.info("    Luego     : pip install -r requirements.txt")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando panel ELCA y archivo geoespacial unificado ───")
    panel = cargar_parquet(cfg["panel_elca"], "hogar_elca_longitudinal_clean.parquet")
    geoespacial = cargar_parquet(cfg["variables_geoespaciales"], "variables_geoespaciales_unificadas.parquet")

    log.info("\n── Construyendo id_ola en el panel ELCA ──────────────────")
    panel = construir_id_ola(panel, cfg)

    log.info("\n── Verificando unicidad de la llave en el archivo geoespacial ─")
    verificar_unicidad(geoespacial, cfg["llave_union"], "variables_geoespaciales_unificadas")
    log.info("  OK -- sin duplicados.")

    log.info("\n── Uniendo variables geoespaciales al panel ELCA ─────────")
    resultado = unir_geoespacial(panel, geoespacial, cfg)
    log.info(f"  Panel final: {len(resultado):,} filas x {resultado.shape[1]} columnas")

    log.info("\n── Verificación final: sin rastro de coordenadas ─────────")
    verificar_sin_coordenadas(resultado, cfg)
    log.info("  OK -- confirmado, 0 columnas de coordenadas en el panel final.")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "hogar_elca_longitudinal_con_geoespacial.parquet"
    resultado.to_parquet(out_path, index=False)
    log.info(f"  Panel exportado: {out_path}")

    reporte = generar_reporte(resultado, cfg)
    txt_path = Path(cfg["logs_dir"]) / "reporte_union_panel_elca.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
