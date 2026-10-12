"""
03_series_temporales.py
=========================

Separa la tabla cruda hogar×año (extraccion_buffer_alos.parquet) en las
dos familias de variables del pipeline: ESTADO (año 2010) y VENTANA
ACUMULADA (estadísticos sobre 2008-2010) — ver la explicación completa en
dmsp_ols_pipeline/scripts/01_construir_coleccion_dmsp.py (misma lógica,
aplicada aquí a ALOS PALSAR).

QUÉ HACE

    1. Carga extraccion_buffer_alos.parquet.
    2. Separa en:
         a. Estado: fila con año == 2010, para cada banda (HH_db, HV_db,
            hh_hv_diff, hh_hv_ratio, hv_hh_ratio, indice_normalizado).
         b. Ventana acumulada: estadísticos (media, std, min, max, rango,
            cv, tendencia, crecimiento_acumulado) sobre los 3 años, para
            cada banda en CONFIG['columnas_acumulado'].
    3. Une ambos subconjuntos en un panel hogar×ronda (ronda=2010 siempre).
    4. Verifica consistencia aritmética.
    5. Exporta el panel y un reporte de calidad.

INPUTS

    data/processed/extraccion_buffer_alos.parquet

OUTPUTS

    data/processed/series_temporales_alos.parquet
    logs/reporte_series_temporales.txt

CÓMO CORRER

    python 03_series_temporales.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from utils.alos_utils import calcular_estadisticos_ventana_acumulada  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "extraccion_buffer": _HERE.parent / "data" / "processed" / "extraccion_buffer_alos.parquet",

    "ronda_objetivo": 2010,
    "anio_estado": 2010,
    "anios_acumulado": [2008, 2009, 2010],

    # Bandas sobre las que se calculan estadísticos de ventana acumulada.
    "columnas_acumulado": ["HH_db", "HV_db", "hh_hv_diff", "hh_hv_ratio", "indice_normalizado"],

    # Prefijo de columnas en el panel final (mismo criterio que S1: alos_*).
    "prefijo_variables": "alos",

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


def cargar_extraccion(cfg: dict) -> pd.DataFrame:
    """Lee extraccion_buffer_alos.parquet. Termina el proceso si no existe."""
    ruta = Path(cfg["extraccion_buffer"])
    if not ruta.exists():
        log.error("-" * 62)
        log.error("ERROR: no se encontró la tabla de extracción cruda.")
        log.error(f"  Se buscó en: {ruta.resolve()}")
        log.error("  CÓMO SOLUCIONARLO: corre primero 02_extraccion_buffer.py")
        log.error("-" * 62)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    log.info(f"Extracción cargada: {len(df):,} filas (hogar×año) — {ruta.name}")
    return df


def extraer_estado(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Filtra a la fila del año de estado (2010) y renombra columnas con
    prefijo alos_.

    La imagen anual trae más bandas de las que CONFIG['columnas_acumulado']
    lista explícitamente:
    - 'HH'/'HV': número digital (DN) crudo, sin calibrar -- artefacto de
      calibrar_a_gamma0_db() (alos_utils.py), que agrega HH_db/HV_db sin
      eliminar las bandas originales. No son interpretables sin
      calibración y son redundantes frente a alos_hh_db/alos_hv_db ya
      calibradas -- se descartan aquí.
    - 'hv_hh_ratio'/'log_hh_hv': variables derivadas válidas que
      agregar_bandas_derivadas() sí calcula, pero que CONFIG['columnas_acumulado']
      no incluye (no se les calcula ventana acumulada) -- se renombran
      con el mismo prefijo 'alos_' para que ninguna variable quede sin
      prefijo, aunque no tengan versión '_acum'.
    """
    prefijo = cfg["prefijo_variables"]
    estado = df[df["anio"] == cfg["anio_estado"]].copy()
    estado["ola"] = cfg["ronda_objetivo"]

    columnas_derivadas_extra = ["hv_hh_ratio", "log_hh_hv"]
    columnas_a_descartar = ["HH", "HV"]

    renombres = {c: f"{prefijo}_{c.lower()}" for c in cfg["columnas_acumulado"] if c in estado.columns}
    renombres.update({c: f"{prefijo}_{c.lower()}" for c in columnas_derivadas_extra if c in estado.columns})

    estado = estado.drop(columns=[c for c in columnas_a_descartar if c in estado.columns])
    estado = estado.rename(columns=renombres).drop(columns=["anio"])
    log.info(f"  Estado: {len(estado):,} filas (hogar, año {cfg['anio_estado']})")
    log.info(f"  Descartadas (DN crudo, redundante): {[c for c in columnas_a_descartar if c in df.columns]}")
    log.info(f"  Renombradas con prefijo '{prefijo}_' (antes sin prefijo): {[renombres[c] for c in columnas_derivadas_extra if c in renombres]}")
    return estado


def calcular_ventana_acumulada(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Agrupa por hogar y calcula, para cada banda, los estadísticos de la ventana de 3 años."""
    prefijo_base = cfg["prefijo_variables"]
    filas = []
    for (consecutivo, id_ola), grupo in df.groupby(["consecutivo", "id_ola"]):
        fila = {"consecutivo": consecutivo, "id_ola": id_ola, "ola": cfg["ronda_objetivo"]}
        for col in cfg["columnas_acumulado"]:
            if col not in grupo.columns:
                continue
            prefijo = f"{prefijo_base}_{col.lower()}_acum"
            fila.update(calcular_estadisticos_ventana_acumulada(grupo, col, prefijo))
        filas.append(fila)
    acumulado = pd.DataFrame(filas)
    log.info(f"  Ventana acumulada: {len(acumulado):,} filas (hogar) × {acumulado.shape[1]} columnas")
    return acumulado


def verificar_consistencia(panel: pd.DataFrame, cfg: dict) -> list:
    """Verifica rango = max-min y cv = std/media para cada variable acumulada."""
    problemas = []
    prefijo_base = cfg["prefijo_variables"]
    for col in cfg["columnas_acumulado"]:
        prefijo = f"{prefijo_base}_{col.lower()}_acum"
        cols_necesarias = [f"{prefijo}_maximo", f"{prefijo}_minimo", f"{prefijo}_rango"]
        if not all(c in panel.columns for c in cols_necesarias):
            continue
        validos = panel.dropna(subset=cols_necesarias)
        for _, fila in validos.iterrows():
            rango_calc = fila[f"{prefijo}_maximo"] - fila[f"{prefijo}_minimo"]
            if abs(fila[f"{prefijo}_rango"] - rango_calc) > 1e-6:
                problemas.append(f"consecutivo={fila['consecutivo']} {prefijo}: rango != max-min")
    return problemas


def generar_reporte(panel: pd.DataFrame, problemas: list, cfg: dict) -> str:
    """Genera el texto del reporte de calidad."""
    sep = "=" * 65
    prefijo_base = cfg["prefijo_variables"]
    lines = [
        sep, "REPORTE DE CALIDAD — SERIES TEMPORALES ALOS PALSAR (OLA 2010)", sep, "",
        "── 1. COBERTURA DEL PANEL ────────────────────────────────", "",
        f"  Hogares: {len(panel):,}",
    ]
    col_n = f"{prefijo_base}_hh_db_acum_n_anios"
    if col_n in panel.columns:
        lines.append(f"  Años en ventana acumulada (promedio): {panel[col_n].mean():.2f} de 3")

    lines += ["", "── 2. VERIFICACIÓN DE CONSISTENCIA ───────────────────────", ""]
    if not problemas:
        lines.append(f"  OK — sin problemas de consistencia en {len(panel):,} hogares.")
    else:
        lines.append(f"  {len(problemas)} problema(s) encontrado(s):")
        for p in problemas[:20]:
            lines.append(f"    - {p}")

    lines += ["", "── 3. ESTADÍSTICOS DESCRIPTIVOS ──────────────────────────", ""]
    cols_alos = [c for c in panel.columns if c.startswith(f"{prefijo_base}_")]
    resumen = panel[cols_alos].describe().T[["mean", "std", "min", "max"]]
    for col, fila in resumen.iterrows():
        lines.append(f"    {col}: media={fila['mean']:.3f}  std={fila['std']:.3f}  min={fila['min']:.3f}  max={fila['max']:.3f}")

    lines += ["", sep]
    return "\n".join(lines)


def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  03_series_temporales.py")
    log.info("=" * 62)

    cfg = CONFIG
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)

    log.info("\n── Cargando extracción cruda ──────────────────────────────")
    df = cargar_extraccion(cfg)

    log.info("\n── Separando en estado y ventana acumulada ───────────────")
    estado = extraer_estado(df, cfg)
    acumulado = calcular_ventana_acumulada(df, cfg)

    panel = estado.merge(acumulado, on=["consecutivo", "id_ola", "ola"], how="left")
    log.info(f"  Panel final: {len(panel):,} filas × {panel.shape[1]} columnas")

    log.info("\n── Verificando consistencia aritmética ───────────────────")
    problemas = verificar_consistencia(panel, cfg)
    if problemas:
        log.warning(f"  {len(problemas)} problema(s) encontrado(s).")
    else:
        log.info("  OK — sin problemas de consistencia.")

    log.info("\n── Exportando resultados ─────────────────────────────────")
    out_path = Path(cfg["output_dir"]) / "series_temporales_alos.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(f"  Panel exportado: {out_path} ({len(panel):,} filas)")

    reporte = generar_reporte(panel, problemas, cfg)
    (Path(cfg["logs_dir"]) / "reporte_series_temporales.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
