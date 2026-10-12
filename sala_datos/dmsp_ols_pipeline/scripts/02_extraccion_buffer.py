"""
02_extraccion_buffer.py
=========================

Extrae, para cada hogar de las rondas 2010 y 2013, el promedio espacial de
avg_vis/stable_lights/cf_cvg dentro del buffer de ~928 m alrededor de su
coordenada, sobre CADA UNO de los 3 años de la ventana acumulada de su
ronda (ver "Estado vs. ventana acumulada" en
01_construir_coleccion_dmsp.py): 2008-2010 para la ronda 2010, 2011-2013
para la ronda 2013. El año de estado de cada ronda es simplemente el
último año de su ventana — no se extrae por separado.

QUÉ HACE

    1. Se conecta a Earth Engine y carga la tabla de hogares (reutilizada
       de sentinel1_pipeline/).
    2. Para cada ronda (2010, 2013) y cada uno de los 3 años de su ventana
       acumulada:
         a. Construye la imagen anual real de ese año.
         b. Procesa los hogares de esa ronda en LOTES (CONFIG['tamano_lote']).
         c. Por cada lote: construye una ee.FeatureCollection con el
            buffer de cada hogar, y usa reduceRegions (una sola llamada
            server-side por lote×año) para obtener avg_vis/stable_lights/
            cf_cvg promedio por hogar.
         d. Guarda cada lote×año como Parquet en data/processed/checkpoints/.
            Si ya existe, lo salta (idempotencia).
    3. Consolida todos los checkpoints en un solo archivo, con una fila
       por (hogar, ronda, año): extraccion_buffer_dmsp.parquet.

    Este script NO realiza:

    · separación entre variables de "estado" y "ventana acumulada", ni
      cómputo de estadísticos (03_series_temporales.py).
    · preparación de coordenadas — reutiliza el output de
      sentinel1_pipeline/scripts/00_preparar_coordenadas.py.

    NOTA IMPORTANTE: la coordenada del hogar usada para los 3 años de una
    ronda es SIEMPRE la coordenada registrada para esa ronda (p. ej. la
    coordenada de la ronda 2010 se usa para extraer 2008, 2009 y 2010) —
    no existe una coordenada distinta para años anteriores a la encuesta,
    así que se asume la misma dirección durante la ventana acumulada.
    Ver la discusión de esta limitación en la tesis, Sección 3.3.7.

INPUTS

    ../../sentinel1_pipeline/data/processed/coordenadas_hogares_s1.parquet
    credenciales/gee_service_account.json

OUTPUTS

    data/processed/checkpoints/extraccion_ronda_<ronda>_anio_<anio>_lote_<n>.parquet → por lote
    data/processed/extraccion_buffer_dmsp.parquet → consolidado (hogar × ronda × año)
    logs/reporte_extraccion_buffer.txt → diagnóstico de la extracción

CÓMO CORRER

    python 02_extraccion_buffer.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

_HERE = Path(__file__).resolve().parent            # .../dmsp_ols_pipeline/scripts
sys.path.insert(0, str(_HERE.parent))               # .../dmsp_ols_pipeline

from utils.gee_utils import verificar_archivo_credenciales, inicializar_gee  # noqa: E402
from utils.io_utils import cargar_hogares  # noqa: E402
from utils.dmsp_utils import (  # noqa: E402
    construir_imagen_anual,
    construir_geometria_buffer,
    construir_feature_collection_hogares,
    extraer_valores_lote,
    ESCALA_NATIVA_M,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Credenciales de GEE (mismas que 00 y 01) ───────────────────────────────
    "gee_service_account": "mi-servidor-osm@plasma-minutia-492021-k7.iam.gserviceaccount.com",
    "gee_key_file": _HERE.parent / "credenciales" / "gee_service_account.json",
    "gee_project": "plasma-minutia-492021-k7",

    # ── Input: tabla de hogares (reutilizada del pipeline Sentinel-1) ────────
    "coordenadas_hogares": (
        _HERE.parent.parent
        / "sentinel1_pipeline" / "data" / "processed" / "coordenadas_hogares_s1.parquet"
    ),

    # ── Ventanas por ronda (idéntico a 01_construir_coleccion_dmsp.py) ───────
    "ventanas_por_ronda": {
        2010: {"anio_estado": 2010, "anios_acumulado": [2008, 2009, 2010]},
        2013: {"anio_estado": 2013, "anios_acumulado": [2011, 2012, 2013]},
    },

    # ── Geometría de extracción ────────────────────────────────────────────────
    "radio_buffer_m": ESCALA_NATIVA_M,
    "escala_reduccion_m": ESCALA_NATIVA_M,
    "tile_scale": 4,

    # ── Procesamiento por lotes ─────────────────────────────────────────────────
    # Una sola imagen por (ronda, año) — igual que la versión anterior de
    # este script, el límite de 5,000 filas de GEE es poco restrictivo
    # aquí (cada lote produce como máximo tamano_lote filas).
    "tamano_lote": 2000,

    # Límite de hogares a procesar por ronda, SOLO para pruebas rápidas.
    "limite_hogares_prueba": None,

    # ── Carpetas de salida ──────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: CONEXIÓN A GEE
# ──────────────────────────────────────────────────────────────────────────────

def conectar_gee(cfg: dict):
    """Verifica credenciales e inicializa GEE. Termina el proceso si falla."""
    ruta_key = Path(cfg["gee_key_file"])
    try:
        credenciales_json = verificar_archivo_credenciales(ruta_key)
        ee = inicializar_gee(
            credenciales_json["client_email"], ruta_key, project=cfg.get("gee_project")
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)
    return ee


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: PROCESAMIENTO POR LOTES CON CHECKPOINTING
# ──────────────────────────────────────────────────────────────────────────────

def procesar_lote(ee_module, imagen, hogares_lote: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Extrae los valores crudos (una fila por hogar) para un solo lote de hogares y un solo año."""
    fc_hogares = construir_feature_collection_hogares(
        ee_module, hogares_lote, lambda lat, lon: construir_geometria_buffer(ee_module, lat, lon, cfg["radio_buffer_m"])
    )
    fc_resultado = extraer_valores_lote(ee_module, imagen, fc_hogares, cfg)
    info = fc_resultado.getInfo()
    filas = [feat["properties"] for feat in info["features"]]
    return pd.DataFrame(filas)


def extraer_ronda_anio(ee_module, ronda: int, anio: int, hogares_ronda: pd.DataFrame, cfg: dict) -> list:
    """Orquesta la extracción de un (ronda, año): construye su imagen anual una sola vez, y la reutiliza para todos los lotes."""
    imagen = construir_imagen_anual(ee_module, anio)

    checkpoints_dir = Path(cfg["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tamano_lote = cfg["tamano_lote"]
    n_lotes = (len(hogares_ronda) + tamano_lote - 1) // tamano_lote
    sufijo_prueba = "_prueba" if cfg.get("limite_hogares_prueba") else ""

    rutas_checkpoint = []
    for i in range(n_lotes):
        lote = hogares_ronda.iloc[i * tamano_lote: (i + 1) * tamano_lote]
        ruta_checkpoint = checkpoints_dir / f"extraccion_ronda_{ronda}_anio_{anio}_lote_{i}{sufijo_prueba}.parquet"
        rutas_checkpoint.append(ruta_checkpoint)

        if ruta_checkpoint.exists():
            log.info(f"    Lote {i + 1}/{n_lotes}: ya existe ({ruta_checkpoint.name}) — se omite (idempotencia).")
            continue

        log.info(f"    Lote {i + 1}/{n_lotes}: {len(lote):,} hogares — consultando GEE...")
        try:
            df_lote = procesar_lote(ee_module, imagen, lote, cfg)
        except Exception as e:
            log.error(f"    ERROR en el lote {i + 1}/{n_lotes} (ronda {ronda}, año {anio}): {e}")
            if "accumulating over" in str(e) or "5000 elements" in str(e):
                log.error("    CAUSA: tope duro de Earth Engine (5000 filas por consulta).")
                log.error("    SOLUCIÓN: baja CONFIG['tamano_lote'] y vuelve a correr el script.")
            elif "memory limit" in str(e).lower():
                log.error("    CAUSA: límite de memoria del usuario en GEE excedido.")
                log.error("    SOLUCIÓN: baja CONFIG['tamano_lote'] y vuelve a correr el script.")
            else:
                log.error("    CAUSA PROBABLE: timeout de red u otro error transitorio.")
                log.error("    SOLUCIÓN: vuelve a correr el script; los lotes completados se omiten.")
            log.error(f"    El lote {i + 1} de (ronda {ronda}, año {anio}) quedó pendiente.")
            continue

        df_lote["anio"] = anio
        df_lote.to_parquet(ruta_checkpoint, index=False)
        log.info(f"    Lote {i + 1}/{n_lotes}: {len(df_lote):,} filas (hogar) guardadas.")

    return rutas_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: CONSOLIDACIÓN Y REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def consolidar_checkpoints(rutas_checkpoint: list) -> pd.DataFrame:
    """Lee todos los archivos de checkpoint existentes y los concatena en un solo DataFrame."""
    partes = [pd.read_parquet(r) for r in rutas_checkpoint if r.exists()]
    if not partes:
        return pd.DataFrame()
    return pd.concat(partes, ignore_index=True)


def generar_reporte(resumen: dict) -> str:
    """Genera el texto del reporte de diagnóstico de la extracción por buffer."""
    sep = "=" * 65
    lines = [
        sep,
        "REPORTE DE DIAGNÓSTICO — EXTRACCIÓN DMSP-OLS (ESTADO + VENTANA ACUMULADA)",
        sep,
        "",
        "── 1. COBERTURA DE LA EXTRACCIÓN POR RONDA × AÑO ─────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Cuántos lotes se completaron frente a los esperados, y cuántas",
        "  filas (una por hogar) se extrajeron por (ronda, año). Un lote",
        "  pendiente significa que volver a correr este script es",
        "  necesario antes de continuar con 03_series_temporales.py.",
        "",
    ]
    for (ronda, anio), r in sorted(resumen.items()):
        estado = "OK" if r["lotes_completados"] == r["lotes_esperados"] else "INCOMPLETO — reintentar"
        lines.append(
            f"  Ronda {ronda}, año {anio}: {r['lotes_completados']}/{r['lotes_esperados']} lotes — "
            f"{r['n_filas']:,} filas (hogar) — {estado}"
        )
    lines += ["", sep]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  02_extraccion_buffer.py")
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

    log.info("\n── Conectando a Earth Engine ──────────────────────────────")
    ee = conectar_gee(cfg)

    log.info("\n── Cargando tabla de hogares ──────────────────────────────")
    hogares = cargar_hogares(cfg["coordenadas_hogares"])
    hogares = hogares[hogares["coord_valida_ola"] == 1]  # solo coordenadas válidas

    resumen = {}
    todos_los_checkpoints = []

    for ronda, info_ronda in sorted(cfg["ventanas_por_ronda"].items()):
        log.info(f"\n── Ronda {ronda} ───────────────────────────────────────")

        if ronda not in hogares["ola"].unique():
            log.warning(f"  Ronda {ronda} no está presente en la tabla de hogares. Se omite.")
            continue

        hogares_ronda = hogares[hogares["ola"] == ronda]
        if cfg["limite_hogares_prueba"]:
            hogares_ronda = hogares_ronda.head(cfg["limite_hogares_prueba"])
            log.warning(
                f"  MODO PRUEBA: limitando a {len(hogares_ronda)} hogares de la ronda {ronda} "
                "(CONFIG['limite_hogares_prueba'])."
            )

        for anio in info_ronda["anios_acumulado"]:
            etiqueta = "ESTADO" if anio == info_ronda["anio_estado"] else "acumulado"
            log.info(f"  Año {anio} [{etiqueta}]")

            rutas_checkpoint = extraer_ronda_anio(ee, ronda, anio, hogares_ronda, cfg)
            todos_los_checkpoints.extend(rutas_checkpoint)

            df_anio = consolidar_checkpoints(rutas_checkpoint)
            resumen[(ronda, anio)] = {
                "lotes_esperados": len(rutas_checkpoint),
                "lotes_completados": sum(1 for r in rutas_checkpoint if r.exists()),
                "n_filas": len(df_anio),
            }

    log.info("\n── Consolidando checkpoints ──────────────────────────────")
    df_final = consolidar_checkpoints(todos_los_checkpoints)

    if df_final.empty:
        log.error("ERROR: no se extrajo ninguna fila. Revisa los mensajes anteriores.")
        sys.exit(1)

    out_path = Path(cfg["output_dir"]) / "extraccion_buffer_dmsp.parquet"
    df_final.to_parquet(out_path, index=False)
    log.info(f"  Consolidado exportado: {out_path} ({len(df_final):,} filas)")

    reporte = generar_reporte(resumen)
    txt_path = Path(cfg["logs_dir"]) / "reporte_extraccion_buffer.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
