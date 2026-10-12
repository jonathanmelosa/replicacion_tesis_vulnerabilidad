"""
05_unir_variables.py
======================

Pega indicadores_derivados_dmsp.parquet (output de 03) sobre el
"espinazo" de TODOS los hogares × ola (coordenadas_hogares_s1.parquet),
para que ningún hogar desaparezca del panel final aunque no tenga
coordenada válida. Deja lista la tabla final de variables DMSP-OLS, con
su llave de unión documentada, para que el panel general del proyecto la
pegue al panel de la encuesta ELCA.

QUÉ HACE

    1. Carga coordenadas_hogares_s1.parquet como TABLA BASE — mismo
       criterio que sentinel1_pipeline/ y viirs_pipeline/.
    2. Carga indicadores_derivados_dmsp.parquet (03).
    3. Verifica que (id_ola, ola) sea único en la tabla DMSP ANTES de
       mergear.
    4. Pega la tabla DMSP sobre el espinazo con LEFT JOIN por (id_ola,
       ola) — NUNCA (consecutivo, ola), por la misma razón ya documentada
       en sentinel1_pipeline/viirs_pipeline (hogares divididos).
    5. Reporta cobertura por ola: hogares con coordenada válida vs.
       hogares con variables DMSP calculadas. A diferencia de VIIRS
       (compuesto global sin huecos), aquí SÍ puede haber diferencia por
       diseño: este pipeline solo cubre 2010 y 2013 — cualquier otra ola
       presente en el espinazo (2016, etc.) queda con NaN en todas las
       columnas dmsp_*, y eso es esperado, no un error.
    6. Exporta variables_dmsp.parquet y un reporte de calidad.

    Este script NO realiza:

    · el merge final con el panel de la encuesta ELCA/ELCO.
    · control de calidad estadístico exhaustivo (ver 06_control_calidad.py).

    ADVERTENCIA PARA EL MERGE CON EL PANEL DE LA ENCUESTA
        Usa 'id_ola' (no solo 'consecutivo'). Las variables dmsp_* solo
        existen para las olas 2010 y 2013 — para 2016/2019/2022 son NaN
        por diseño (usar viirs_pipeline/ para esas rondas).

INPUTS

    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet
    data/processed/indicadores_derivados_dmsp.parquet (output de 04_indicadores_derivados.py)

OUTPUTS

    data/processed/variables_dmsp.parquet → tabla final de variables DMSP-OLS por hogar × ola
    logs/reporte_union_variables.txt → reporte de cobertura y calidad

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
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Inputs ──────────────────────────────────────────────────────────────────
    "coordenadas_hogares": (
        _HERE.parent.parent / "sentinel1_pipeline" / "data" / "processed" / "coordenadas_hogares_s1.parquet"
    ),
    "indicadores_derivados": _HERE.parent / "data" / "processed" / "indicadores_derivados_dmsp.parquet",

    # ── Llave de unión ──────────────────────────────────────────────────────────
    "llave_union": ["id_ola", "ola"],

    # ── Rondas que este pipeline puede cubrir (para el reporte) ──────────────
    "rondas_cubiertas": [2010, 2013],

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
# MÓDULO 1: CARGA DE TABLAS
# ──────────────────────────────────────────────────────────────────────────────

def cargar_indicadores_derivados(cfg: dict) -> pd.DataFrame:
    """Lee indicadores_derivados_dmsp.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["indicadores_derivados"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró la tabla de indicadores derivados.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO:")
        log.error("    Corre primero 04_indicadores_derivados.py:")
        log.error("      python 04_indicadores_derivados.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"  Indicadores derivados cargados: {len(df):,} filas × {df.shape[1]} columnas — {ruta.name}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: VALIDACIÓN DE UNICIDAD ANTES DEL MERGE
# ──────────────────────────────────────────────────────────────────────────────

def verificar_unicidad(df: pd.DataFrame, llave: list, nombre: str) -> None:
    """Verifica que `llave` identifique una única fila en `df`. Termina el proceso si encuentra duplicados."""
    n_dup = df.duplicated(subset=llave).sum()
    if n_dup > 0:
        log.error("-" * 62)
        log.error(f"ERROR: {nombre} tiene {n_dup:,} filas duplicadas en la llave {llave}.")
        log.error("  CÓMO SOLUCIONARLO:")
        log.error(f"    Revisa por qué {nombre} tiene más de una fila por {llave}.")
        log.error("    Si viene de 02, revisa si corriste el script dos veces con")
        log.error("    CONFIG['limite_hogares_prueba'] distinto sin limpiar")
        log.error("    data/processed/checkpoints/ entre corridas.")
        log.error("-" * 62)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: UNIÓN DE TABLAS
# ──────────────────────────────────────────────────────────────────────────────

def unir_variables(espinazo: pd.DataFrame, indicadores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pega `indicadores` sobre `espinazo` usando LEFT JOIN por CONFIG['llave_union']."""
    llave = cfg["llave_union"]
    columnas_a_pegar = [c for c in indicadores.columns if c not in ("consecutivo",) or c in llave]

    antes = len(espinazo)
    resultado = espinazo.merge(indicadores[columnas_a_pegar], on=llave, how="left", validate="one_to_one")
    assert len(resultado) == antes, "El merge con indicadores_derivados cambió el número de filas — revisar unicidad de la llave."
    return resultado


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: REPORTE DE COBERTURA Y CALIDAD
# ──────────────────────────────────────────────────────────────────────────────

def calcular_cobertura(variables: pd.DataFrame, cfg: dict) -> dict:
    """
    Calcula, por ola, cuántos hogares tienen coordenada válida y cuántos
    tienen variables DMSP calculadas. A diferencia de VIIRS, una
    diferencia AQUÍ puede ser esperada: solo 2010 y 2013 tienen cobertura
    por diseño (ver CONFIG['rondas_cubiertas']).
    """
    cobertura = {}
    for ola, grupo in variables.groupby("ola"):
        n_total = len(grupo)
        n_coord_valida = int((grupo["coord_valida_ola"] == 1).sum()) if "coord_valida_ola" in grupo.columns else None
        n_con_dmsp = int(grupo["dmsp_stable_lights"].notna().sum()) if "dmsp_stable_lights" in grupo.columns else 0
        cobertura[int(ola)] = {
            "n_total": n_total, "n_coord_valida": n_coord_valida, "n_con_dmsp": n_con_dmsp,
            "cubierta_por_diseno": int(ola) in cfg["rondas_cubiertas"],
        }
    return cobertura


def generar_reporte(variables: pd.DataFrame, cobertura: dict) -> str:
    """Genera el texto del reporte de calidad de la unión de variables."""
    sep = "=" * 65
    n_cols_dmsp = len([c for c in variables.columns if c.startswith("dmsp_")])
    lines = [
        sep,
        "REPORTE DE CALIDAD — UNIÓN DE VARIABLES DMSP-OLS",
        sep,
        "",
        "── 1. COBERTURA POR OLA ──────────────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Compara coordenada válida vs. variables DMSP calculadas. Este",
        "  pipeline SOLO cubre 2010 y 2013 por diseño (DMSP-OLS no tiene",
        "  datos después de 2014) — cualquier otra ola con 0 variables",
        "  DMSP es esperado, no un error. Usa viirs_pipeline/ para 2016",
        "  en adelante.",
        "",
    ]
    for ola, c in sorted(cobertura.items()):
        etiqueta = "" if c["cubierta_por_diseno"] else " [FUERA DE ALCANCE — usar VIIRS]"
        linea = f"  Ola {ola}{etiqueta}: {c['n_total']:,} hogares totales"
        if c["n_coord_valida"] is not None:
            linea += f" — {c['n_coord_valida']:,} con coordenada válida"
        linea += f" — {c['n_con_dmsp']:,} con variables DMSP"
        lines.append(linea)
        if c["cubierta_por_diseno"] and c["n_coord_valida"] is not None and c["n_coord_valida"] != c["n_con_dmsp"]:
            lines.append(
                f"    ADVERTENCIA: {c['n_coord_valida'] - c['n_con_dmsp']:,} hogares con coordenada "
                "válida SIN variables DMSP en una ola cubierta por diseño — revisar 02/03."
            )

    lines += [
        "",
        "── 2. ESTRUCTURA FINAL DE LA TABLA ───────────────────────",
        "",
        f"  Filas (hogar × ola): {len(variables):,}",
        f"  Columnas dmsp_*: {n_cols_dmsp}",
        f"  Columnas totales: {variables.shape[1]}",
        "",
        "  Llave de unión con el panel de la encuesta: ['id_ola', 'ola']",
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
    log.info("  05_unir_variables.py")
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

    log.info("\n── Cargando tablas ────────────────────────────────────────")
    espinazo = cargar_hogares(cfg["coordenadas_hogares"])
    indicadores = cargar_indicadores_derivados(cfg)

    log.info("\n── Verificando unicidad de la llave de unión ─────────────")
    verificar_unicidad(indicadores, cfg["llave_union"], "indicadores_derivados_dmsp")
    log.info("  OK — sin duplicados.")

    log.info("\n── Uniendo variables sobre el espinazo de hogares ────────")
    variables = unir_variables(espinazo, indicadores, cfg)
    log.info(f"  Tabla final: {len(variables):,} filas × {variables.shape[1]} columnas")

    log.info("\n── Calculando cobertura ──────────────────────────────────")
    cobertura = calcular_cobertura(variables, cfg)

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "variables_dmsp.parquet"
    variables.to_parquet(out_path, index=False)
    log.info(f"  Tabla exportada: {out_path}")

    reporte = generar_reporte(variables, cobertura)
    txt_path = Path(cfg["logs_dir"]) / "reporte_union_variables.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
