"""
01_unificar_variables.py
==========================

Une los tres outputs finales de los pipelines geoespaciales nuevos
(variables_dmsp.parquet, variables_alos.parquet, variables_landsat.parquet)
en una sola tabla hogar x ola, y elimina explícitamente cualquier columna
de latitud/longitud del resultado final.

QUÉ HACE

    1. Carga los tres archivos variables_*.parquet.
    2. Verifica que (id_ola, ola) sea único en cada uno ANTES de unir.
    3. Verifica que los tres compartan el mismo universo de hogares x ola
       (mismo "espinazo" — todos vienen de coordenadas_hogares_s1.parquet).
    4. Usa DMSP-OLS como tabla base (trae el espinazo completo: consecutivo,
       id_ola, ola, lat, lon, es_split, coord_valida_ola) y le pega SOLO
       las columnas nuevas (alos_*, l5_*) de los otros dos archivos, para
       no duplicar columnas de espinazo con sufijos _x/_y.
    5. Elimina toda columna cuyo nombre contenga 'lat' o 'lon' (insensible
       a mayúsculas), sin importar el nombre exacto que traiga cada fuente.
    6. Verifica DE FORMA EXPLÍCITA que no quedó ninguna columna de
       coordenadas antes de exportar — si queda alguna, el script termina
       con error en vez de exportar silenciosamente un archivo con
       coordenadas.
    7. Exporta la tabla unificada y un reporte de cobertura por fuente x ola.

    Este script NO realiza:

    · merge con el panel de la encuesta ELCA/ELCO (responsabilidad de un
      script posterior, fuera de este alcance).
    · ningún cálculo de variables nuevas — solo une lo que cada pipeline
      ya calculó.

INPUTS

    ../../dmsp_ols_pipeline/data/processed/variables_dmsp.parquet
    ../../alos_palsar_pipeline/data/processed/variables_alos.parquet
    ../../landsat5_pipeline/data/processed/variables_landsat.parquet

OUTPUTS

    data/processed/variables_geoespaciales_unificadas.parquet
    logs/reporte_unificacion.txt

CÓMO CORRER

    python 01_unificar_variables.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import re
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

_HERE = Path(__file__).resolve().parent            # .../unificacion_geoespacial/scripts
sys.path.insert(0, str(_HERE.parent))               # .../unificacion_geoespacial

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Inputs: outputs finales de cada pipeline ──────────────────────────────
    "variables_dmsp": _HERE.parent.parent / "dmsp_ols_pipeline" / "data" / "processed" / "variables_dmsp.parquet",
    "variables_alos": _HERE.parent.parent / "alos_palsar_pipeline" / "data" / "processed" / "variables_alos.parquet",
    "variables_landsat": _HERE.parent.parent / "landsat5_pipeline" / "data" / "processed" / "variables_landsat.parquet",

    # ── Llave de unión (misma que usa cada pipeline en su propio 05/06) ──────
    "llave_union": ["id_ola", "ola"],

    # ── Columnas del "espinazo" compartido (vienen de coordenadas_hogares_s1) ─
    # Se toman SOLO de la tabla base (DMSP-OLS) para no duplicarlas.
    "columnas_espinazo": ["consecutivo", "id_ola", "ola", "lat", "lon", "es_split", "coord_valida_ola"],

    # ── Patrón para detectar y eliminar cualquier columna de coordenadas ─────
    # Cubre 'lat', 'lon', 'lat_decimal', 'lon_decimal', 'latitud', 'longitud',
    # sin importar mayúsculas/minúsculas -- defensivo ante nombres que no se
    # hayan anticipado explícitamente.
    "patron_columnas_coordenadas": re.compile(r"(lat|lon|latitud|longitud)", re.IGNORECASE),

    # ── Todos los outputs intermedios de los tres pipelines (no solo los ─────
    # tres que se unen) -- se revisan al inicio, solo por nombre de columna
    # (nunca se leen datos ni coordenadas), para dejar constancia de en qué
    # etapa aparecen columnas de coordenadas antes de tocar nada.
    "todos_los_outputs": {
        "dmsp": [
            _HERE.parent.parent / "dmsp_ols_pipeline" / "data" / "processed" / "extraccion_buffer_dmsp.parquet",
            _HERE.parent.parent / "dmsp_ols_pipeline" / "data" / "processed" / "series_temporales_dmsp.parquet",
            _HERE.parent.parent / "dmsp_ols_pipeline" / "data" / "processed" / "indicadores_derivados_dmsp.parquet",
            _HERE.parent.parent / "dmsp_ols_pipeline" / "data" / "processed" / "variables_dmsp.parquet",
        ],
        "alos": [
            _HERE.parent.parent / "alos_palsar_pipeline" / "data" / "processed" / "extraccion_buffer_alos.parquet",
            _HERE.parent.parent / "alos_palsar_pipeline" / "data" / "processed" / "series_temporales_alos.parquet",
            _HERE.parent.parent / "alos_palsar_pipeline" / "data" / "processed" / "variables_alos.parquet",
        ],
        "landsat": [
            _HERE.parent.parent / "landsat5_pipeline" / "data" / "processed" / "extraccion_buffer_landsat.parquet",
            _HERE.parent.parent / "landsat5_pipeline" / "data" / "processed" / "series_temporales_landsat.parquet",
            _HERE.parent.parent / "landsat5_pipeline" / "data" / "processed" / "indicadores_derivados_landsat.parquet",
            _HERE.parent.parent / "landsat5_pipeline" / "data" / "processed" / "variables_landsat.parquet",
        ],
    },

    # ── Rondas que cada fuente cubre por diseño (para el reporte) ────────────
    "rondas_por_fuente": {
        "dmsp": [2010, 2013],
        "alos": [2010],
        "l5": [2010, 2013],
    },

    # ── Carpetas de salida ──────────────────────────────────────────────────────
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
        log.error("  CÓMO SOLUCIONARLO:")
        log.error(f"    Corre primero el pipeline correspondiente hasta su script de unión de variables.")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"  {nombre}: {len(df):,} filas x {df.shape[1]} columnas — {ruta.name}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1B: VERIFICACIÓN DE TODOS LOS OUTPUTS INTERMEDIOS (SOLO NOMBRES DE COLUMNA)
# ──────────────────────────────────────────────────────────────────────────────

def verificar_columnas_todos_los_outputs(cfg: dict) -> None:
    """
    Recorre los 11 outputs conocidos de los tres pipelines (no solo los 3
    que este script une) y reporta, por archivo, si alguna columna
    coincide con CONFIG['patron_columnas_coordenadas'] -- leyendo SOLO
    los nombres de columna, nunca los datos. Es puramente informativo: no
    bloquea la ejecución, porque los archivos intermedios (extraccion_
    buffer/series_temporales/indicadores_derivados) no se unen ni se
    exportan por este script.
    """
    patron = cfg["patron_columnas_coordenadas"]
    for fuente, rutas in cfg["todos_los_outputs"].items():
        log.info(f"  {fuente.upper()}:")
        for ruta in rutas:
            ruta = Path(ruta)
            if not ruta.exists():
                log.info(f"    {ruta.name}: no existe todavía (se omite)")
                continue
            columnas = pd.read_parquet(ruta).columns.tolist()
            sospechosas = [c for c in columnas if patron.search(c)]
            if sospechosas:
                log.warning(f"    {ruta.name}: {len(columnas)} columnas -- CONTIENE COORDENADAS: {sospechosas}")
            else:
                log.info(f"    {ruta.name}: {len(columnas)} columnas -- sin coordenadas")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: VALIDACIÓN ANTES DE UNIR
# ──────────────────────────────────────────────────────────────────────────────

def verificar_unicidad(df: pd.DataFrame, llave: list, nombre: str) -> None:
    """Verifica que `llave` identifique una única fila en `df`. Termina el proceso si encuentra duplicados."""
    n_dup = df.duplicated(subset=llave).sum()
    if n_dup > 0:
        log.error("-" * 62)
        log.error(f"ERROR: {nombre} tiene {n_dup:,} filas duplicadas en la llave {llave}.")
        log.error("  No se puede unir de forma segura sin resolver esto primero.")
        log.error("-" * 62)
        sys.exit(1)


def verificar_mismo_espinazo(dfs: dict, llave: list) -> None:
    """
    Verifica que los tres archivos compartan el mismo universo de hogares x
    ola (mismo conjunto de valores en la llave de unión). Una diferencia
    aquí indicaría que alguno de los pipelines se corrió con una versión
    distinta de coordenadas_hogares_s1.parquet -- advierte pero no bloquea,
    porque el merge subsiguiente es LEFT JOIN sobre la base DMSP-OLS.
    """
    claves = {nombre: set(map(tuple, df[llave].itertuples(index=False, name=None))) for nombre, df in dfs.items()}
    base_nombre = "dmsp"
    for nombre, claves_nombre in claves.items():
        if nombre == base_nombre:
            continue
        faltan_en_otro = claves[base_nombre] - claves_nombre
        sobran_en_otro = claves_nombre - claves[base_nombre]
        if faltan_en_otro or sobran_en_otro:
            log.warning(
                f"  ADVERTENCIA: {nombre} y {base_nombre} no comparten exactamente el mismo "
                f"espinazo (id_ola, ola) -- {len(faltan_en_otro):,} en {base_nombre} sin match en "
                f"{nombre}, {len(sobran_en_otro):,} en {nombre} sin match en {base_nombre}. "
                "Revisar si ambos pipelines usaron la misma versión de coordenadas_hogares_s1.parquet."
            )
        else:
            log.info(f"  {nombre}: mismo espinazo (id_ola, ola) que {base_nombre} -- OK")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: UNIÓN
# ──────────────────────────────────────────────────────────────────────────────

def unir_variables(df_dmsp: pd.DataFrame, df_alos: pd.DataFrame, df_landsat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Usa df_dmsp como base (trae el espinazo completo) y le pega SOLO las
    columnas nuevas de df_alos y df_landsat, para no duplicar columnas de
    espinazo (consecutivo, lat, lon, es_split, coord_valida_ola) con
    sufijos _x/_y.
    """
    llave = cfg["llave_union"]
    espinazo_cols = set(cfg["columnas_espinazo"])

    antes = len(df_dmsp)
    resultado = df_dmsp.copy()

    for nombre, df_otro in [("alos", df_alos), ("landsat", df_landsat)]:
        columnas_nuevas = [c for c in df_otro.columns if c not in espinazo_cols or c in llave]
        resultado = resultado.merge(df_otro[columnas_nuevas], on=llave, how="left", validate="one_to_one")
        assert len(resultado) == antes, f"El merge con {nombre} cambió el número de filas -- revisar unicidad de la llave."
        log.info(f"  Pegado {nombre}: {len(columnas_nuevas) - len(llave):,} columnas nuevas agregadas.")

    return resultado


def eliminar_columnas_coordenadas(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list]:
    """
    Elimina toda columna cuyo nombre coincida con CONFIG['patron_columnas_coordenadas'].
    Retorna el DataFrame sin esas columnas y la lista de nombres eliminados,
    para dejar constancia explícita en el reporte de qué se quitó.
    """
    patron = cfg["patron_columnas_coordenadas"]
    columnas_a_eliminar = [c for c in df.columns if patron.search(c)]
    df_limpio = df.drop(columns=columnas_a_eliminar)
    return df_limpio, columnas_a_eliminar


def verificar_sin_coordenadas(df: pd.DataFrame, cfg: dict) -> None:
    """
    Verificación final defensiva: termina el proceso si, después de
    eliminar_columnas_coordenadas(), sigue quedando alguna columna que
    coincida con el patrón de coordenadas. Nunca debe exportarse un
    archivo con rastro de lat/lon.
    """
    patron = cfg["patron_columnas_coordenadas"]
    quedan = [c for c in df.columns if patron.search(c)]
    if quedan:
        log.error("-" * 62)
        log.error(f"ERROR: quedaron columnas de coordenadas sin eliminar: {quedan}")
        log.error("  No se exporta el archivo. Revisa CONFIG['patron_columnas_coordenadas'].")
        log.error("-" * 62)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def calcular_cobertura(df: pd.DataFrame, cfg: dict) -> dict:
    """Cuenta, por ola, cuántos hogares tienen variables no-nulas de cada fuente."""
    prefijos = {"dmsp": "dmsp_", "alos": "alos_", "l5": "l5_"}
    cobertura = {}
    for ola, grupo in df.groupby("ola"):
        fila = {"n_total": len(grupo)}
        for fuente, prefijo in prefijos.items():
            cols_fuente = [c for c in df.columns if c.startswith(prefijo)]
            n_con_datos = int(grupo[cols_fuente].notna().any(axis=1).sum()) if cols_fuente else 0
            fila[fuente] = n_con_datos
        cobertura[int(ola)] = fila
    return cobertura


def generar_reporte(df: pd.DataFrame, cobertura: dict, columnas_eliminadas: list, cfg: dict) -> str:
    """Genera el texto del reporte de calidad de la unificación."""
    sep = "=" * 65
    lines = [
        sep,
        "REPORTE DE UNIFICACIÓN — VARIABLES GEOESPACIALES (DMSP-OLS + ALOS PALSAR + LANDSAT 5 TM)",
        sep,
        "",
        "── 1. COLUMNAS DE COORDENADAS ELIMINADAS ─────────────────",
        "",
    ]
    if columnas_eliminadas:
        lines.append(f"  Se eliminaron {len(columnas_eliminadas)} columna(s): {columnas_eliminadas}")
    else:
        lines.append("  No se encontró ninguna columna de coordenadas en los inputs (nada que eliminar).")
    lines.append(
        "  Verificación final: 0 columnas con 'lat'/'lon' en el archivo exportado (bloqueante si no)."
    )

    lines += [
        "",
        "── 2. COBERTURA POR OLA Y FUENTE ─────────────────────────",
        "",
        "  QUÉ MIDE ESTA SECCIÓN",
        "  Cuántos hogares (de los n_total en esa ola) tienen al menos una",
        "  variable no-nula de cada fuente. Rondas fuera del alcance por",
        "  diseño de cada fuente (ver CONFIG['rondas_por_fuente']) deberían",
        "  mostrar 0 -- no es un error.",
        "",
    ]
    for ola, fila in sorted(cobertura.items()):
        lines.append(f"  Ola {ola}: {fila['n_total']:,} hogares totales")
        for fuente in ["dmsp", "alos", "l5"]:
            cubierta = ola in cfg["rondas_por_fuente"].get(fuente, [])
            etiqueta = "" if cubierta else " [fuera de alcance por diseño]"
            lines.append(f"    {fuente}: {fila[fuente]:,} con datos{etiqueta}")

    lines += [
        "",
        "── 3. ESTRUCTURA FINAL DE LA TABLA ───────────────────────",
        "",
        f"  Filas (hogar x ola): {len(df):,}",
        f"  Columnas totales: {df.shape[1]}",
        f"  Columnas dmsp_*: {len([c for c in df.columns if c.startswith('dmsp_')])}",
        f"  Columnas alos_*: {len([c for c in df.columns if c.startswith('alos_')])}",
        f"  Columnas l5_*: {len([c for c in df.columns if c.startswith('l5_')])}",
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
    log.info("  01_unificar_variables.py")
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

    log.info("\n── Verificando columnas de los 11 outputs conocidos ──────")
    log.info("  (solo nombres de columna -- no se lee ningún dato)")
    verificar_columnas_todos_los_outputs(cfg)

    log.info("\n── Cargando los tres archivos de variables ───────────────")
    df_dmsp = cargar_parquet(cfg["variables_dmsp"], "variables_dmsp.parquet")
    df_alos = cargar_parquet(cfg["variables_alos"], "variables_alos.parquet")
    df_landsat = cargar_parquet(cfg["variables_landsat"], "variables_landsat.parquet")

    log.info("\n── Verificando unicidad de la llave en cada archivo ──────")
    verificar_unicidad(df_dmsp, cfg["llave_union"], "variables_dmsp")
    verificar_unicidad(df_alos, cfg["llave_union"], "variables_alos")
    verificar_unicidad(df_landsat, cfg["llave_union"], "variables_landsat")
    log.info("  OK -- sin duplicados en ninguno de los tres.")

    log.info("\n── Verificando que los tres compartan el mismo espinazo ──")
    verificar_mismo_espinazo({"dmsp": df_dmsp, "alos": df_alos, "landsat": df_landsat}, cfg["llave_union"])

    log.info("\n── Uniendo variables (DMSP-OLS como base) ────────────────")
    unificado = unir_variables(df_dmsp, df_alos, df_landsat, cfg)
    log.info(f"  Tabla unida: {len(unificado):,} filas x {unificado.shape[1]} columnas")

    log.info("\n── Eliminando columnas de coordenadas (lat/lon) ──────────")
    unificado, columnas_eliminadas = eliminar_columnas_coordenadas(unificado, cfg)
    if columnas_eliminadas:
        log.info(f"  Eliminadas: {columnas_eliminadas}")
    else:
        log.info("  No se encontró ninguna columna de coordenadas.")

    log.info("\n── Verificación final: sin rastro de coordenadas ─────────")
    verificar_sin_coordenadas(unificado, cfg)
    log.info("  OK -- confirmado, 0 columnas de coordenadas en el archivo final.")

    log.info("\n── Calculando cobertura por ola y fuente ─────────────────")
    cobertura = calcular_cobertura(unificado, cfg)

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "variables_geoespaciales_unificadas.parquet"
    unificado.to_parquet(out_path, index=False)
    log.info(f"  Tabla exportada: {out_path} ({len(unificado):,} filas x {unificado.shape[1]} columnas)")

    reporte = generar_reporte(unificado, cobertura, columnas_eliminadas, cfg)
    txt_path = Path(cfg["logs_dir"]) / "reporte_unificacion.txt"
    txt_path.write_text(reporte, encoding="utf-8")
    log.info(f"  Reporte exportado: {txt_path}")

    print("\n" + reporte)


if __name__ == "__main__":
    main()
