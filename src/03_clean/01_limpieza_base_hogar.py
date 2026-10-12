"""
Encuesta Longitudinal Colombiana (ELCA)
Limpieza de la base de hogares (olas 2010, 2013 y 2016).

Autor: Jonathan Melo Sarta
Fecha: Junio 2026

Contexto
--------
Este script parte del output de
src/01_download/01_descarga_ELCA/04_consolidacion_bases_hogar.py
(data/processed/hogar_elca_longitudinal.parquet), que únicamente UNIFICA
las seis bases de hogar (U y R × olas 2010, 2013, 2016) sin corregir
inconsistencias estructurales entre olas: ese script se limita a leer,
corregir codificación ('???') y concatenar.

El objetivo de ESTE script es la limpieza propiamente dicha de la base ya
consolidada, dejándola lista para el análisis: resolver columnas que
deberían representar lo mismo en las tres olas pero que, por diferencias
entre los instrumentos de cada ola, quedan fragmentadas o con NaN tras la
concatenación. El resultado se guarda como una base nueva e independiente
(hogar_elca_longitudinal_clean.parquet), sin sobreescribir el output crudo
unificado, para conservar trazabilidad entre "datos unificados" y "datos
limpios".

Este es un script vivo: por ahora corrige tres problemas identificados
(ver abajo). A medida que se detecten más inconsistencias en la base
consolidada, deben agregarse aquí como nuevas funciones de limpieza
(siguiendo el mismo patrón: una función por problema, documentada con el
mismo nivel de detalle del porqué), y listarse en el pipeline de main().

Aclaración sobre 'RegionLb'
-----------------------------
'RegionLb' es la columna que contiene la clasificación estándar ELCA de
9 regiones (Bogotá, Atlántica, Pacífica, Oriental, Central, Eje Cafetero,
Cundi-Boyacense, Centro-Oriente, Atlántica Media), PERO su disponibilidad
varía por ola:

  - Ola 1 (2010): 'RegionLb' no existe en el archivo fuente → NaN total.
                  La clasificación regional está en 'region'.
  - Ola 2 (2013): 'RegionLb' sí existe, con 9 valores, 0 NaN.
                  La columna 'region' en 2013 U contiene una clasificación
                  geográfica distinta (ver problema 1 abajo).
  - Ola 3 (2016): 'RegionLb' sí existe, con 9 valores, 0 NaN.
                  La columna 'region' queda en NaN (ver problema 2 abajo).

Este patrón irregular explica por qué la unificación de 'region' requiere
fuentes distintas según la ola: 'region' directa en 2010, 'RegionLb' en
2013, y 'region_2016' en 2016.

Problemas identificados y corregidos
--------------------------------------

1) 'region' en 2013 U no contiene la clasificación ELCA estándar de
   9 regiones
   -----------------------------------------------------------------------
   En la ola 2013 Urbano, la columna 'region' contiene una clasificación
   geográfica sub-regional DIFERENTE a las 9 categorías canónicas ELCA.
   La evidencia es que hogares clasificados como 'Bogotá' en 'RegionLb'
   aparecen con region = 'Cundi-Boyacense' u 'Oriental', y hogares de
   'Atlántica' o 'Pacífica' en 'RegionLb' aparecen con region = 'Central'
   o 'Eje Cafetero', categorías que geográficamente no corresponden.

   Los valores únicos de 'region' en 2013 U tras la consolidación son:
     Central, Oriental, Cundi-Boyacense, Centro-Oriente, Eje Cafetero,
     Atlántica Media, Orinoquía-Amazonas  (más 1 206 NaN)
   Estas son categorías de zona RURAL que no tienen sentido para el
   subconjunto urbano, lo que refuerza que 'region' en 2013 U es una
   variable con una asignación geográfica distinta (posiblemente heredada
   del diseño muestral rural) y no la región de residencia actual.

   En contraste, 'RegionLb' en 2013 U y R muestra las 9 regiones ELCA
   con conteos razonables y 0 NaN, ya con los caracteres U+FFFD corregidos
   por REGIONLB_2013U_MAPA en 04_consolidacion_bases_hogar.py:
     Atlántica=1054, Pacífica=1006, Oriental=937, Central=927, Bogotá=711,
     Eje Cafetero=141, Cundi-Boyacense=53, Centro-Oriente=45, Atl.Media=36
     (2013 U, 4 910 filas)

   Las 1 206 filas con region = NaN en 2013 U se deben también a este
   problema: 'region' estaba vacía en el archivo fuente para esas filas
   (831 Atlántica + 368 Pacífica + 7 otras según 'RegionLb'), y al
   concatenar heredaron NaN, no por ausencia de información sino porque
   la variable de origen ya era inconsistente para esas observaciones.

   Consecuencia práctica si no se corrige: cualquier tabla que cruce
   'region' × 'ola' pierde Bogotá completa en ola 2 (de 711 hogares a 0),
   y muestra apenas 13 Atlántica y 10 Pacífica (solo los rurales de 2013 R)
   en lugar de los ~1 000 hogares que tiene cada región en las otras olas.
   El problema pasa desapercibido porque la base no arroja error, sino que
   silenciosamente deja esas filas como None en el grupo 'region'.

   Corrección: para todas las filas de ola 2 (2013), se sobreescribe
   'region' con el valor de 'RegionLb':

       mask = df["ola"] == 2
       df.loc[mask, "region"] = df.loc[mask, "RegionLb"]

   La sustitución cubre tanto 2013 U (donde 'region' era incorrecto) como
   2013 R (donde 'region' y 'RegionLb' son equivalentes tras correcciones):
   esto es inofensivo para R y garantiza que 'region' sea siempre la
   misma fuente canónica para toda la ola 2.

   Se usa la máscara 'ola == 2' en lugar de 'ola == 2 & zona == Urbano'
   para no depender de la columna 'zona', que en 2013 está bien poblada
   pero cuya semántica podría cambiar en una actualización futura.

   Antes de aplicar la corrección, el script valida que 'RegionLb' no
   tenga NaN en las filas de ola 2.  Si los tuviera, la sustitución
   introduciría NaN donde antes había valores válidos, lo que sería
   peor que el error original; el script se detiene con AssertionError.

2) 'zona' y 'region' en NaN para toda la ola 2016
   -----------------------------------------------
   Los archivos fuente de la ola 2016 NO incluyen las columnas 'zona' ni
   'region' que sí existen en 2010 y 2013. En su lugar, 2016 trae:
     - 'zona_2016'   (equivalente a 'zona': 'Urbano'/'Rural')
     - 'region_2016' (equivalente a 'region')
   ('zona_2010' indica la zona de origen del hogar en la primera ola en
   la que aparece, no la zona vigente en 2016.)

   Como 04_consolidacion_bases_hogar.py se limita a concatenar las bases
   tal cual vienen (unión de columnas, NaN donde no existen), el resultado
   es que 'zona' y 'region' quedan en NaN para TODAS las filas de la ola
   2016, mientras que 'zona_2016'/'region_2016' solo están pobladas en esa
   ola. Si no se corrige, cualquier análisis que agrupe o filtre por
   'zona' o 'region' pierde silenciosamente el 31.6% de las filas (8 818
   de 27 932) correspondientes a 2016.

   Corrección: se rellenan los NaN de 'zona' y 'region' con los valores de
   'zona_2016' y 'region_2016' respectivamente:

       df['zona']   = df['zona'].fillna(df['zona_2016'])
       df['region'] = df['region'].fillna(df['region_2016'])

   Se usa fillna() en lugar de una asignación condicionada por 'ola'
   (p. ej. 'ola == 3') porque 'zona'/'region' y 'zona_2016'/'region_2016'
   son, por construcción, mutuamente excluyentes en cuanto a sus NaN: las
   primeras solo existen en 2010/2013 y las segundas solo en 2016.
   fillna() logra el mismo resultado sin depender de la codificación
   numérica interna de 'ola' (1 = 2010, 2 = 2013, 3 = 2016, heredada de
   IDS_POR_OLA en 04_consolidacion_bases_hogar.py), lo que hace el código
   más legible y robusto ante cambios en esa codificación.

   Nota: 'region' en 2016 (region_2016) tiene algunos NaN propios (37 de
   8 818 filas) que NO se imputan acá porque no existe una fuente
   alternativa para esas filas; quedan como NaN tras la limpieza.

   ORDEN DE APLICACIÓN IMPORTANTE: limpiar_region_2013() debe ejecutarse
   ANTES de limpiar_zona_region() porque esta última usa fillna() sobre
   'region', que solo rellena NaN.  Si se invirtiera el orden, los NaN de
   2013 U (que corresponden a filas con 'region_2016' también NaN, pues
   son de ola 2) quedarían sin corregir; la función de 2013 sí los cubre
   porque sobreescribe con 'RegionLb' independientemente de si hay NaN.
"""

import os

from pathlib import Path

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal.parquet")
OUTPUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet")


# ─── Funciones de limpieza ────────────────────────────────────────────────────
# Una función por problema identificado. Ver el docstring del módulo para el
# detalle completo de cada corrección.

def limpiar_region_2013(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza 'region' en la ola 2013 con 'RegionLb', que es la fuente
    correcta de la clasificación ELCA de 9 regiones para esa ola.

    En 2013 U la columna 'region' contiene una clasificación geográfica
    sub-regional diferente (no las 9 regiones ELCA estándar), y además
    tiene 1 206 NaN.  'RegionLb' sí contiene las 9 categorías canónicas
    con 0 NaN y ya viene con los caracteres U+FFFD corregidos por
    REGIONLB_2013U_MAPA en 04_consolidacion_bases_hogar.py.

    Ver el docstring del módulo (problema 1) para el análisis detallado.
    """
    df = df.copy()

    mask_ola2 = df["ola"] == 2

    nan_regionlb_ola2 = df.loc[mask_ola2, "RegionLb"].isna().sum()
    assert nan_regionlb_ola2 == 0, (
        f"'RegionLb' tiene {nan_regionlb_ola2} NaN en la ola 2; "
        f"la sustitución introduciría NaN en 'region'. "
        f"Revisar 04_consolidacion_bases_hogar.py (REGIONLB_2013U_MAPA)."
    )

    n_region_antes = df.loc[mask_ola2, "region"].notna().sum()
    n_total_ola2 = mask_ola2.sum()

    df.loc[mask_ola2, "region"] = df.loc[mask_ola2, "RegionLb"]

    n_region_despues = df.loc[mask_ola2, "region"].notna().sum()
    print(
        f"    'region' ola 2 poblada: {n_region_antes:>5} -> {n_region_despues:>5} "
        f"de {n_total_ola2} filas (desde 'RegionLb')"
    )

    return df


def limpiar_zona_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifica 'zona' y 'region' para que queden pobladas en las tres olas
    (2010, 2013, 2016).

    Rellena 'zona' y 'region' para la ola 2016, donde el archivo fuente
    usa 'zona_2016' y 'region_2016' en lugar de los nombres canónicos.
    Tras la consolidación esas columnas quedan en NaN para toda la ola 2016;
    esta función las rellena con los valores de 'zona_2016'/'region_2016'.

    Debe ejecutarse DESPUÉS de limpiar_region_2013(), que ya corrigió
    'region' para la ola 2 con 'RegionLb'.  Ver docstring del módulo
    (problemas 1 y 2) para el análisis completo y el orden requerido.
    """
    df = df.copy()

    # Validar el supuesto de mutua exclusión antes de mezclar columnas:
    # 'zona' y 'zona_2016' nunca deberían estar pobladas a la vez en la
    # misma fila (idem 'region'/'region_2016'). Si esto falla, algo cambió
    # en los datos fuente o en 04_consolidacion_bases_hogar.py y conviene
    # detenerse en vez de sobreescribir datos válidos en silencio.
    solapa_zona = df["zona"].notna() & df["zona_2016"].notna()
    assert not solapa_zona.any(), (
        f"'zona' y 'zona_2016' están pobladas simultáneamente en "
        f"{solapa_zona.sum()} filas; revisar antes de aplicar fillna()."
    )
    solapa_region = df["region"].notna() & df["region_2016"].notna()
    assert not solapa_region.any(), (
        f"'region' y 'region_2016' están pobladas simultáneamente en "
        f"{solapa_region.sum()} filas; revisar antes de aplicar fillna()."
    )

    n_zona_antes = df["zona"].notna().sum()
    n_region_antes = df["region"].notna().sum()

    df["zona"] = df["zona"].fillna(df["zona_2016"])
    df["region"] = df["region"].fillna(df["region_2016"])

    print(
        f"    'zona'   poblada: {n_zona_antes:>6} -> {df['zona'].notna().sum():>6} filas "
        f"(+{df['zona'].notna().sum() - n_zona_antes} desde 'zona_2016')"
    )
    print(
        f"    'region' poblada: {n_region_antes:>6} -> {df['region'].notna().sum():>6} filas "
        f"(+{df['region'].notna().sum() - n_region_antes} desde 'region_2016')"
    )

    return df


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Lee la base consolidada de hogares, aplica todas las funciones de
    limpieza definidas en este módulo, y guarda el resultado como una
    base nueva e independiente (hogar_elca_longitudinal_clean.parquet)
    sin sobreescribir el output crudo de 04_consolidacion_bases_hogar.py.

    Para agregar un nuevo paso de limpieza: escribir la función siguiendo
    el patrón de limpiar_zona_region() (recibe y devuelve el DataFrame
    completo) y encadenarla aquí.
    """
    print(f"Leyendo base consolidada: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Dimensiones de entrada : {df.shape[0]:>6} filas × {df.shape[1]:>4} columnas")

    # Orden importante: limpiar_region_2013 antes de limpiar_zona_region.
    # Ver docstring del módulo (nota de orden al final del problema 2).
    print("\nCorrigiendo 'region' en ola 2013 con 'RegionLb' (problema 1)...")
    df = limpiar_region_2013(df)

    print("\nLimpiando 'zona' y 'region' en ola 2016 (problema 2)...")
    df = limpiar_zona_region(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    print(f"\nBase limpia guardada en : {OUTPUT_PATH}")
    print(f"Dimensiones             : {df.shape[0]:>6} filas × {df.shape[1]:>4} columnas")

    return df


if __name__ == "__main__":
    df_clean = main()
