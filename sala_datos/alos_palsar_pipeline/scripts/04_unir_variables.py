"""
04_unir_variables.py
======================

Pega series_temporales_alos.parquet sobre el "espinazo" de TODOS los
hogares × ola (coordenadas_hogares_s1.parquet), para que ningún hogar
desaparezca del panel final. Deja lista la tabla final de variables ALOS
PALSAR, con su llave de unión documentada.

QUÉ HACE

    1. Carga coordenadas_hogares_s1.parquet como TABLA BASE.
    2. Carga series_temporales_alos.parquet (03).
    3. Verifica unicidad de (id_ola, ola) antes de mergear.
    4. Pega con LEFT JOIN por (id_ola, ola) — nunca (consecutivo, ola).
    5. Reporta cobertura: solo la ola 2010 tendrá variables no-nulas; 0
       para cualquier otra ola es esperado por diseño (ver
       01_construir_coleccion_alos.py, "Por qué únicamente la ola 2010").
    6. Exporta variables_alos.parquet y un reporte de calidad.

INPUTS

    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet
    data/processed/series_temporales_alos.parquet

OUTPUTS

    data/processed/variables_alos.parquet
    logs/reporte_union_variables.txt

CÓMO CORRER

    python 04_unir_variables.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from utils.io_utils import cargar_hogares  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "coordenadas_hogares": (
        _HERE.parent.parent / "sentinel1_pipeline" / "data" / "processed" / "coordenadas_hogares_s1.parquet"
    ),
    "series_temporales": _HERE.parent / "data" / "processed" / "series_temporales_alos.parquet",
    "llave_union": ["id_ola", "ola"],
    "ronda_cubierta": 2010,
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


def cargar_series_temporales(cfg: dict) -> pd.DataFrame:
    """Lee series_temporales_alos.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["series_temporales"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró el panel de series temporales.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO: corre primero 03_series_temporales.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"  Series temporales cargadas: {len(df):,} filas × {df.shape[1]} columnas")
    return df


def verificar_unicidad(df: pd.DataFrame, llave: list, nombre: str) -> None:
    """Verifica que `llave` identifique una única fila en `df`."""
    n_dup = df.duplicated(subset=llave).sum()
    if n_dup > 0:
        log.error(f"ERROR: {nombre} tiene {n_dup:,} filas duplicadas en la llave {llave}.")
        sys.exit(1)


def unir_variables(espinazo: pd.DataFrame, variables: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pega `variables` sobre `espinazo` usando LEFT JOIN por CONFIG['llave_union']."""
    llave = cfg["llave_union"]
    columnas_a_pegar = [c for c in variables.columns if c not in ("consecutivo",) or c in llave]
    antes = len(espinazo)
    resultado = espinazo.merge(variables[columnas_a_pegar], on=llave, how="left", validate="one_to_one")
    assert len(resultado) == antes, "El merge cambió el número de filas — revisar unicidad de la llave."
    return resultado


def generar_reporte(variables: pd.DataFrame, cfg: dict) -> str:
    """Genera el texto del reporte de calidad de la unión de variables."""
    sep = "=" * 65
    n_cols = len([c for c in variables.columns if c.startswith("alos_")])
    lines = [
        sep, "REPORTE DE CALIDAD — UNIÓN DE VARIABLES ALOS PALSAR", sep, "",
        "── 1. COBERTURA POR OLA ──────────────────────────────────", "",
        "  Este pipeline SOLO cubre la ola 2010 por diseño (ver",
        "  01_construir_coleccion_alos.py) — 0 variables en cualquier",
        "  otra ola es esperado, no un error.", "",
    ]
    for ola, grupo in variables.groupby("ola"):
        n_con_datos = int(grupo["alos_hh_db"].notna().sum()) if "alos_hh_db" in grupo.columns else 0
        etiqueta = "" if ola == cfg["ronda_cubierta"] else " [fuera de alcance]"
        lines.append(f"  Ola {ola}{etiqueta}: {len(grupo):,} hogares totales — {n_con_datos:,} con variables ALOS")

    lines += [
        "", "── 2. ESTRUCTURA FINAL DE LA TABLA ───────────────────────", "",
        f"  Filas (hogar × ola): {len(variables):,}",
        f"  Columnas alos_*: {n_cols}",
        "  Llave de unión con el panel de la encuesta: ['id_ola', 'ola']",
        "", sep,
    ]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  04_unir_variables.py")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando tablas ────────────────────────────────────────")
    espinazo = cargar_hogares(cfg["coordenadas_hogares"])
    variables = cargar_series_temporales(cfg)

    log.info("\n── Verificando unicidad de la llave de unión ─────────────")
    verificar_unicidad(variables, cfg["llave_union"], "series_temporales_alos")
    log.info("  OK — sin duplicados.")

    log.info("\n── Uniendo variables sobre el espinazo de hogares ────────")
    resultado = unir_variables(espinazo, variables, cfg)
    log.info(f"  Tabla final: {len(resultado):,} filas × {resultado.shape[1]} columnas")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "variables_alos.parquet"
    resultado.to_parquet(out_path, index=False)
    log.info(f"  Tabla exportada: {out_path}")

    reporte = generar_reporte(resultado, cfg)
    (Path(cfg["logs_dir"]) / "reporte_union_variables.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
