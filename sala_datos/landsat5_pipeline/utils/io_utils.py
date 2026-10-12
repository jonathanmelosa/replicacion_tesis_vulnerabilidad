"""
io_utils.py
===========
Funciones reutilizables de lectura de archivos compartidas entre los
scripts del pipeline VIIRS (01 en adelante).
"""
from __future__ import annotations

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

log = logging.getLogger(__name__)


def cargar_hogares(ruta) -> pd.DataFrame:
    """
    Lee la tabla de hogares × ola (consecutivo, id_ola, ola, lat, lon,
    es_split, coord_valida_ola). Igual que sentinel2_pipeline, este
    pipeline NO tiene su propio script de preparación de coordenadas:
    reutiliza directamente coordenadas_hogares_s1.parquet, el output de
    sentinel1_pipeline/scripts/00_preparar_coordenadas.py (decisión de
    arquitectura: los hogares y sus coordenadas son los mismos sin
    importar el satélite, así que se preparan una sola vez).

    `ruta` acepta un solo Path o una lista de Path candidatos (se prueban
    en orden y se usa el primero que exista) — la ubicación real de
    sentinel1_pipeline/ en disco ha variado entre el equipo de desarrollo
    y la sala de cómputo, así que no se asume una sola jerarquía de
    carpetas (mismo criterio que sentinel1_pipeline/00_preparar_coordenadas.py).

    Termina el proceso (sys.exit) con un mensaje diagnóstico si el
    archivo no existe en ninguna de las rutas candidatas — ningún script
    de extracción puede continuar sin esta tabla.
    """
    candidatos = ruta if isinstance(ruta, (list, tuple)) else [ruta]
    candidatos = [Path(c) for c in candidatos]

    for candidato in candidatos:
        if candidato.exists():
            df = pd.read_parquet(candidato)
            log.info(f"Hogares cargados: {len(df):,} filas — {candidato.resolve()}")
            return df

    log.error("-" * 62)
    log.error("ERROR: no se encontró la tabla de hogares en ninguna ruta candidata.")
    for candidato in candidatos:
        log.error(f"  Se buscó en: {candidato.resolve()}")
    log.error("  CÓMO SOLUCIONARLO:")
    log.error("    Este pipeline reutiliza el output del pipeline Sentinel-1.")
    log.error("    Corre primero, desde sentinel1_pipeline/scripts/:")
    log.error("      python 00_preparar_coordenadas.py")
    log.error("    Si ya lo corriste, agrega la ruta correcta a la lista de")
    log.error("    candidatos de CONFIG['coordenadas_hogares'] en este script.")
    log.error("-" * 62)
    sys.exit(1)
