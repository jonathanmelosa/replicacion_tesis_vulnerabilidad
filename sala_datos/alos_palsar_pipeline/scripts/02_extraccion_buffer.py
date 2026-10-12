"""
02_extraccion_buffer.py
=========================

Extrae, para cada hogar de la ola 2010, el promedio espacial de las
bandas HH_db/HV_db y derivadas dentro del buffer de 250 m, para cada uno
de los 3 años de la ventana acumulada (2008, 2009, 2010).

QUÉ HACE

    1. Se conecta a Earth Engine y carga la tabla de hogares.
    2. Para cada uno de los 3 años: construye la imagen anual calibrada
       (utils/alos_utils.py::construir_imagen_anual), procesa los hogares
       en LOTES, y extrae los valores promedio por buffer con
       reduceRegions.
    3. Guarda cada lote×año en checkpoints, consolida en un solo Parquet.

    Este script NO calcula estado ni ventana acumulada (03), ni prepara
    coordenadas (reutiliza sentinel1_pipeline/).

    NOTA: igual que en dmsp_ols_pipeline, la coordenada del hogar usada
    para los 3 años es SIEMPRE la de la ronda 2010 — se asume la misma
    dirección para 2008-2010 (no hay coordenada previa a la encuesta).

INPUTS

    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet
    credenciales/gee_service_account.json

OUTPUTS

    data/processed/checkpoints/extraccion_anio_<anio>_lote_<n>.parquet
    data/processed/extraccion_buffer_alos.parquet
    logs/reporte_extraccion_buffer.txt

CÓMO CORRER

    python 02_extraccion_buffer.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from utils.gee_utils import verificar_archivo_credenciales, inicializar_gee  # noqa: E402
from utils.io_utils import cargar_hogares  # noqa: E402
from utils.alos_utils import (  # noqa: E402
    construir_imagen_anual,
    construir_geometria_buffer,
    construir_feature_collection_hogares,
    extraer_valores_lote,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "gee_service_account": "mi-servidor-osm@plasma-minutia-492021-k7.iam.gserviceaccount.com",
    "gee_key_file": _HERE.parent / "credenciales" / "gee_service_account.json",
    "gee_project": "plasma-minutia-492021-k7",

    "coordenadas_hogares": (
        _HERE.parent.parent
        / "sentinel1_pipeline" / "data" / "processed" / "coordenadas_hogares_s1.parquet"
    ),

    "ronda_objetivo": 2010,
    "anio_estado": 2010,
    "anios_acumulado": [2008, 2009, 2010],

    "radio_buffer_m": 250,
    "escala_reduccion_m": 25,
    "tile_scale": 4,

    "tamano_lote": 2000,
    "limite_hogares_prueba": None,

    "checkpoints_dir": _HERE.parent / "data" / "processed" / "checkpoints",
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


def conectar_gee(cfg: dict):
    """Verifica credenciales e inicializa GEE. Termina el proceso si falla."""
    ruta_key = Path(cfg["gee_key_file"])
    try:
        credenciales_json = verificar_archivo_credenciales(ruta_key)
        ee = inicializar_gee(credenciales_json["client_email"], ruta_key, project=cfg.get("gee_project"))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)
    return ee


def procesar_lote(ee_module, imagen, hogares_lote: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Extrae los valores crudos (una fila por hogar) para un solo lote de hogares y un solo año."""
    fc_hogares = construir_feature_collection_hogares(
        ee_module, hogares_lote, lambda lat, lon: construir_geometria_buffer(ee_module, lat, lon, cfg["radio_buffer_m"])
    )
    fc_resultado = extraer_valores_lote(ee_module, imagen, fc_hogares, cfg)
    info = fc_resultado.getInfo()
    filas = [feat["properties"] for feat in info["features"]]
    return pd.DataFrame(filas)


def extraer_anio(ee_module, anio: int, hogares_ronda: pd.DataFrame, cfg: dict) -> list:
    """Orquesta la extracción de un año: construye su imagen anual una sola vez, la reutiliza para todos los lotes."""
    imagen = construir_imagen_anual(ee_module, anio)

    checkpoints_dir = Path(cfg["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tamano_lote = cfg["tamano_lote"]
    n_lotes = (len(hogares_ronda) + tamano_lote - 1) // tamano_lote
    sufijo_prueba = "_prueba" if cfg.get("limite_hogares_prueba") else ""

    rutas_checkpoint = []
    for i in range(n_lotes):
        lote = hogares_ronda.iloc[i * tamano_lote: (i + 1) * tamano_lote]
        ruta_checkpoint = checkpoints_dir / f"extraccion_anio_{anio}_lote_{i}{sufijo_prueba}.parquet"
        rutas_checkpoint.append(ruta_checkpoint)

        if ruta_checkpoint.exists():
            log.info(f"    Lote {i + 1}/{n_lotes}: ya existe — se omite (idempotencia).")
            continue

        log.info(f"    Lote {i + 1}/{n_lotes}: {len(lote):,} hogares — consultando GEE...")
        try:
            df_lote = procesar_lote(ee_module, imagen, lote, cfg)
        except Exception as e:
            log.error(f"    ERROR en el lote {i + 1}/{n_lotes} (año {anio}): {e}")
            if "accumulating over" in str(e) or "5000 elements" in str(e):
                log.error("    CAUSA: tope duro de Earth Engine. SOLUCIÓN: baja CONFIG['tamano_lote'].")
            else:
                log.error("    CAUSA PROBABLE: timeout de red. SOLUCIÓN: vuelve a correr el script.")
            continue

        df_lote["anio"] = anio
        df_lote.to_parquet(ruta_checkpoint, index=False)
        log.info(f"    Lote {i + 1}/{n_lotes}: {len(df_lote):,} filas guardadas.")

    return rutas_checkpoint


def consolidar_checkpoints(rutas_checkpoint: list) -> pd.DataFrame:
    """Lee todos los checkpoints existentes y los concatena."""
    partes = [pd.read_parquet(r) for r in rutas_checkpoint if r.exists()]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def generar_reporte(resumen: dict) -> str:
    """Genera el texto del reporte de diagnóstico de la extracción."""
    sep = "=" * 65
    lines = [sep, "REPORTE DE DIAGNÓSTICO — EXTRACCIÓN ALOS PALSAR (OLA 2010)", sep, ""]
    for anio, r in sorted(resumen.items()):
        estado = "OK" if r["lotes_completados"] == r["lotes_esperados"] else "INCOMPLETO — reintentar"
        lines.append(f"  Año {anio}: {r['lotes_completados']}/{r['lotes_esperados']} lotes — {r['n_filas']:,} filas — {estado}")
    lines += ["", sep]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  02_extraccion_buffer.py")
    log.info("  Si ves ModuleNotFoundError, el entorno virtual no esta")
    log.info("  activado o las dependencias no estan instaladas.")
    log.info("  Solucion: source .venv/bin/activate (Mac/Linux) o .venv\\Scripts\\activate (Windows)")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Conectando a Earth Engine ──────────────────────────────")
    ee = conectar_gee(cfg)

    log.info("\n── Cargando tabla de hogares ──────────────────────────────")
    hogares = cargar_hogares(cfg["coordenadas_hogares"])
    hogares = hogares[hogares["coord_valida_ola"] == 1]

    if cfg["ronda_objetivo"] not in hogares["ola"].unique():
        log.error(f"ERROR: la ronda {cfg['ronda_objetivo']} no está en la tabla de hogares.")
        sys.exit(1)
    hogares_ronda = hogares[hogares["ola"] == cfg["ronda_objetivo"]]
    if cfg["limite_hogares_prueba"]:
        hogares_ronda = hogares_ronda.head(cfg["limite_hogares_prueba"])
        log.warning(f"  MODO PRUEBA: limitando a {len(hogares_ronda)} hogares.")

    resumen = {}
    todos_los_checkpoints = []
    for anio in cfg["anios_acumulado"]:
        etiqueta = "ESTADO" if anio == cfg["anio_estado"] else "acumulado"
        log.info(f"\n── Año {anio} [{etiqueta}] ─────────────────────────────")
        rutas_checkpoint = extraer_anio(ee, anio, hogares_ronda, cfg)
        todos_los_checkpoints.extend(rutas_checkpoint)
        df_anio = consolidar_checkpoints(rutas_checkpoint)
        resumen[anio] = {
            "lotes_esperados": len(rutas_checkpoint),
            "lotes_completados": sum(1 for r in rutas_checkpoint if r.exists()),
            "n_filas": len(df_anio),
        }

    log.info("\n── Consolidando checkpoints ──────────────────────────────")
    df_final = consolidar_checkpoints(todos_los_checkpoints)
    if df_final.empty:
        log.error("ERROR: no se extrajo ninguna fila. Revisa los mensajes anteriores.")
        sys.exit(1)

    out_path = Path(cfg["output_dir"]) / "extraccion_buffer_alos.parquet"
    df_final.to_parquet(out_path, index=False)
    log.info(f"  Consolidado exportado: {out_path} ({len(df_final):,} filas)")

    reporte = generar_reporte(resumen)
    (Path(cfg["logs_dir"]) / "reporte_extraccion_buffer.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
