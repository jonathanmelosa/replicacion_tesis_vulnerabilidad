"""
05_unir_variables.py
======================

Pega indicadores_derivados_landsat.parquet sobre el "espinazo" de TODOS
los hogares × ola (coordenadas_hogares_s1.parquet). Deja lista la tabla
final de variables Landsat 5, con su llave de unión documentada.

QUÉ HACE

    1. Carga coordenadas_hogares_s1.parquet como TABLA BASE.
    2. Carga indicadores_derivados_landsat.parquet (04).
    3. Verifica unicidad de (id_ola, ola) antes de mergear.
    4. Pega con LEFT JOIN por (id_ola, ola).
    5. Reporta cobertura por ola (2010/2013 cubiertas por diseño; 0 en
       cualquier otra ola es esperado).
    6. Exporta variables_landsat.parquet y un reporte de calidad.

INPUTS

    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet
    data/processed/indicadores_derivados_landsat.parquet

OUTPUTS

    data/processed/variables_landsat.parquet
    logs/reporte_union_variables.txt

CÓMO CORRER

    python 05_unir_variables.py
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
    "indicadores_derivados": _HERE.parent / "data" / "processed" / "indicadores_derivados_landsat.parquet",
    "llave_union": ["id_ola", "ola"],
    "rondas_cubiertas": [2010, 2013],
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


def cargar_indicadores_derivados(cfg: dict) -> pd.DataFrame:
    """Lee indicadores_derivados_landsat.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["indicadores_derivados"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró la tabla de indicadores derivados.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO: corre primero 04_indicadores_derivados.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"  Indicadores derivados cargados: {len(df):,} filas × {df.shape[1]} columnas")
    return df


def verificar_unicidad(df: pd.DataFrame, llave: list, nombre: str) -> None:
    """Verifica que `llave` identifique una única fila en `df`."""
    n_dup = df.duplicated(subset=llave).sum()
    if n_dup > 0:
        log.error(f"ERROR: {nombre} tiene {n_dup:,} filas duplicadas en la llave {llave}.")
        sys.exit(1)


def unir_variables(espinazo: pd.DataFrame, indicadores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pega `indicadores` sobre `espinazo` usando LEFT JOIN por CONFIG['llave_union']."""
    llave = cfg["llave_union"]
    columnas_a_pegar = [c for c in indicadores.columns if c not in ("consecutivo",) or c in llave]
    antes = len(espinazo)
    resultado = espinazo.merge(indicadores[columnas_a_pegar], on=llave, how="left", validate="one_to_one")
    assert len(resultado) == antes, "El merge cambió el número de filas — revisar unicidad de la llave."
    return resultado


def generar_reporte(variables: pd.DataFrame, cfg: dict) -> str:
    """Genera el texto del reporte de calidad de la unión de variables."""
    sep = "=" * 65
    n_cols = len([c for c in variables.columns if c.startswith("l5_")])
    lines = [sep, "REPORTE DE CALIDAD — UNIÓN DE VARIABLES LANDSAT 5 TM", sep, "",
             "── 1. COBERTURA POR OLA ──────────────────────────────────", ""]
    for ola, grupo in variables.groupby("ola"):
        n_con_datos = int(grupo["l5_ndvi"].notna().sum()) if "l5_ndvi" in grupo.columns else 0
        etiqueta = "" if ola in cfg["rondas_cubiertas"] else " [fuera de alcance]"
        lines.append(f"  Ola {ola}{etiqueta}: {len(grupo):,} hogares totales — {n_con_datos:,} con variables Landsat 5")

    lines += ["", "── 2. ESTRUCTURA FINAL DE LA TABLA ───────────────────────", "",
              f"  Filas (hogar × ola): {len(variables):,}",
              f"  Columnas l5_*: {n_cols}",
              "  Llave de unión con el panel de la encuesta: ['id_ola', 'ola']", "", sep]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  05_unir_variables.py")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando tablas ────────────────────────────────────────")
    espinazo = cargar_hogares(cfg["coordenadas_hogares"])
    indicadores = cargar_indicadores_derivados(cfg)

    log.info("\n── Verificando unicidad de la llave de unión ─────────────")
    verificar_unicidad(indicadores, cfg["llave_union"], "indicadores_derivados_landsat")
    log.info("  OK — sin duplicados.")

    log.info("\n── Uniendo variables sobre el espinazo de hogares ────────")
    variables = unir_variables(espinazo, indicadores, cfg)
    log.info(f"  Tabla final: {len(variables):,} filas × {variables.shape[1]} columnas")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "variables_landsat.parquet"
    variables.to_parquet(out_path, index=False)
    log.info(f"  Tabla exportada: {out_path}")

    reporte = generar_reporte(variables, cfg)
    (Path(cfg["logs_dir"]) / "reporte_union_variables.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
