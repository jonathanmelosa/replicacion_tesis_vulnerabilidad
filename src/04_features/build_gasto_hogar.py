"""
Construccion del gasto total mensual del hogar (ELCA 2010, 2013, 2016).

Parte de gastos_elca_longitudinal.parquet, que trae el detalle de compra por
articulo (columnas gasto_{articulo}=compro si/no, vr_{articulo}=valor de esa
compra, per_{articulo}=periodicidad declarada). Ese archivo ya incluye una
columna `total_gasto`, pero NO se usa aqui: `total_gasto` es la suma directa
de vr_{articulo} sin normalizar por periodicidad, lo que mezcla en la misma
suma pesos gastados "por semana" con pesos gastados "por año" (ver
02_consolidacion_bases_gastos.py). Eso infla el gasto de los hogares que
compran mas articulos de periodicidad larga frente a los que compran mas de
periodicidad corta, y no es comparable con una linea de pobreza mensual.

Se normaliza cada articulo a un equivalente MENSUAL antes de sumar, usando la
periodicidad declarada (per_{articulo}) cuando existe. Factor de conversion =
numero esperado de veces que se repite esa compra en un mes:

    Diariamente      x 30      Trimestralmente  x 1/3
    Semanalmente     x 30/7    Semestralmente   x 1/6
    Quincenalmente   x 2       Anualmente       x 1/12
    Mensualmente     x 1

AUDITORÍA 2026-08-06: se reviso exhaustivamente la periodicidad de los 88
articulos de gasto, cruzando la cobertura de cada columna per_ por ola/zona
contra el manual metodologico original (data/interim/raw/diccionarios_elca/
ELCA-Manual-Hogar-Urbano2010.pdf) y el cod_articulo de los .tab crudos. Se
encontraron y corrigieron dos problemas:

1. BUG CRITICO (dato silenciosamente perdido): el codigo anterior decidia si
   usar la periodicidad declarada mirando si la COLUMNA per_{articulo}
   EXISTIA en el esquema, no si estaba poblada fila por fila. Para 9
   articulos, la columna per_ solo esta poblada en ola 1 urbana (el resto --
   ola 1 rural y TODA ola 2/3 -- la deja en blanco para esos articulos
   especificos, aunque el hogar si compro). El codigo anterior intentaba
   usar per_.map(...) para todas las filas de todas formas, y como el
   resultado es NaN cuando per_ esta vacio, ese NaN se propagaba SIN CAER a
   ningun supuesto de respaldo -- el articulo quedaba fuera de la suma para
   quienes si lo compraron. Impacto verificado: para "ropa para hombre,
   mujer, niño y niña", 10.296 de 10.301 hogares que reportaron haberla
   comprado quedaban con valor NaN (gasto de ropa desaparecido del total
   para el 81 %+ de la muestra). Los 9 articulos afectados y su tratamiento
   corregido:

     - 3 con periodo fijo confirmado por el manual (capitulo IX-B "Gastos
       trimestrales del hogar", articulos cod_articulo 21-25): ropa, calzado,
       reparacion de calzado/vestuario -> Trimestral (ver punto 2).
     - 6 sin periodo unico dominante (la distribucion real, solo observable
       en ola 1 urbana, se reparte entre varias categorias): articulos de
       aseo personal, articulos de aseo del hogar, corte de pelo, bombillos/
       pilas, algodon/gasas/botiquin, empleados del servicio domestico
       internos. Para estos, cuando no hay per_ propio, se usa un factor de
       respaldo = promedio ponderado por la distribucion real de
       periodicidad observada en ola 1 urbana (unica fuente con dato
       individual), calculado dinamicamente al correr el script -- no un
       valor fijo arbitrario. Ver _factores_respaldo_ponderados().

2. Periodicidad mal asumida para 2 de los 35 "articulos sin periodo": se
   cruzo el cod_articulo de los .tab crudos de 2010 urbano contra el manual
   metodologico y se confirmo que "reparacion_repuestos_y_mantenimiento_de_
   vehiculo_de_uso_del_hogar" (cod 24) y "libros_discos_y_cds" (cod 25)
   pertenecen al capitulo IX-B (Gastos TRIMESTRALES, periodo fijo de 3
   meses), no al capitulo IX-C (Gastos ANUALES) que se les asumia antes.
   Se movieron de /12 a /3. Los otros 28 articulos sin columna per_
   corresponden al capitulo IX-C (Gastos ANUALES, cod 26-35: muebles,
   nevera/electrodomesticos, colchones, ollas, vehiculo/moto, bienes
   raices, hoteles, pasajes de avion, primas de seguros) mas gastos
   excepcionales de salud/educacion/impuestos que no tienen equivalente en
   el manual -- para estos se mantiene el supuesto de recall anual (/12),
   documentado explicitamente porque no hay una fuente que lo confirme
   articulo por articulo.

Limitaciones documentadas que NO se corrigieron con codigo (requieren juicio
metodologico, no una formula):
  - Ola 1 urbana (2010) usa 35 categorias de gasto agregadas (ej. "alimentos
    procesados" junta varios rubros en un solo item); ola 1 rural y las olas
    2013/2016 (ambas zonas) usan 72 categorias detalladas para los mismos
    conceptos. Esto no afecta el TOTAL del hogar (cada hogar solo reporta en
    un sistema, sin doble conteo), pero si la granularidad/comparabilidad a
    nivel de articulo individual dentro de la ola 1.
  - "impuesto_a_la_renta_y_complementarios" (solo 2010) e "impuesto_a_la_
    renta_complementarios_y_predial" (solo 2013/2016) no son el mismo
    concepto: el segundo agrega impuesto predial. Se mantienen como
    articulos separados (ninguno se excluye), pero el monto de 2010 mide
    algo mas angosto que 2013/2016 para ese rubro puntual.
  - "Gastos en educación" y "Gastos en salud" en la ola 1 RURAL tienen una
    tasa de compra extremadamente baja (6/4578 y 254/4578 hogares
    respectivamente) frente al equivalente urbano ("educacion/salud de
    todos los miembros del hogar", ~33 %/9 %). La pregunta SI existe en el
    cuestionario rural (confirmado en RGastos-csv.tab), pero no hay una
    variable de filtro asociada (a diferencia del caso de ingreso, ver
    build_ingreso_hogar.py) que permita distinguir si es una tasa de
    compra real muy baja o un problema de ventana de recall. Queda como
    limitacion documentada, sin ajuste.

Si el hogar no compro el articulo (gasto_{a}=0), su aporte es 0. Si lo
compro pero el valor vr_{a} quedo en blanco (no-respuesta puntual, con
periodicidad ya resuelta), el articulo se trata como NaN y no se suma; el
hogar solo queda con gasto_mensual_hogar en NaN si TODOS sus articulos
comprados carecen de valor.

Output: data/processed/gasto_hogar_elca_longitudinal.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GASTOS_PATH = PROJECT_ROOT / "data" / "processed" / "gastos_elca_longitudinal.parquet"
HOGAR_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
DEFLACTOR_PATH = PROJECT_ROOT / "data" / "processed" / "deflactor_ipc_elca.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gasto_hogar_elca_longitudinal.parquet"
DOC_PATH = PROJECT_ROOT / "docs" / "variable_audit" / "gasto_hogar_construccion.csv"

# Factor de conversion a equivalente mensual por periodicidad declarada.
FACTOR_MENSUAL = {
    "Diariamente": 30.0,
    "Semanalmente": 30.0 / 7.0,
    "Quincenalmente": 2.0,
    "Mensualmente": 1.0,
    "Bimestralmente": 1.0 / 2.0,
    "Trimestralmente": 1.0 / 3.0,
    "Semestralmente": 1.0 / 6.0,
    "Anualmente": 1.0 / 12.0,
}

# Capitulo IX-B del manual metodologico (Gastos TRIMESTRALES, cod_articulo
# 21-25): periodo FIJO de 3 meses por diseno del cuestionario. Los 3
# primeros tienen columna per_ (poblada solo ~0.05% de las veces, un
# residuo de captura -- no se usa como fuente real, ver docstring); los 2
# ultimos no tienen columna per_ en absoluto. En ambos casos, cuando no hay
# per_ propio en la fila, el respaldo es Trimestral (/3).
ARTICULOS_TRIMESTRAL_FIJO = [
    "ropa_para_hombre_mujer_nino_y_nina",
    "calzado_para_hombre_mujer_nino_y_nina",
    "reparacion_de_calzado_y_o_vestuario",
    "reparacion_repuestos_y_mantenimiento_de_vehiculo_de_uso_del_hogar",
    "libros_discos_y_cds",
]

# Articulos con columna per_ poblada SOLO en ola 1 urbana (bug descrito en
# el docstring del modulo), sin un periodo dominante claro en esa muestra.
# El respaldo para las demas filas es el promedio ponderado por la
# distribucion real de periodicidad observada en ola 1 urbana -- se calcula
# en tiempo de ejecucion en _factores_respaldo_ponderados(), no es un valor
# fijo escrito a mano.
ARTICULOS_PROMEDIO_PONDERADO = [
    "articulos_de_aseo_personal_crema_dental_jabon_champu_papel_higienico_desodorantes_toallas_higienicas_panales_desechables_maquinas_y_cuchillas_de_afeitar_desechables",
    "articulos_para_el_aseo_del_hogar_detergentes_desinfectantes_escobas_ceras_servilletas_etc",
    "corte_de_pelo_arreglo_de_unas",
    "bombillos_pilas_y_otros_articulos_electricos_velas_y_velones",
    "algodon_gasas_alcohol_curitas_anticonceptivos_condones_aspirinas_y_otros_elementos_de_botiquin",
    "empleados_del_servicio_domestico_internos",
]

# Resto de articulos sin columna per_ (o sin cobertura util de ella):
# capitulo IX-C del manual (Gastos ANUALES, cod_articulo 26-35: muebles,
# electrodomesticos, colchones, ollas, vehiculo/moto, bienes raices,
# hoteles, pasajes de avion, primas de seguros) mas gastos excepcionales de
# salud/educacion/impuestos sin equivalente documentado en el manual. Se
# asume recall anual (/12); es un supuesto documentado, no verificado
# articulo por articulo para los que no estan en el capitulo IX-C.
ARTICULOS_SIN_PERIODO = [
    "anillos_relojes_y_otros_articulos_de_joyeria_artesanias_porcelanas_etc",
    "colchones_cobijas_manteles_y_ropa_de_cama",
    "compra_de_bienes_raices_diferentes_a_la_vivienda_que_ocupa_el_hogar",
    "compra_de_computador_personal",
    "compra_y_sostenimiento_de_mascotas",
    "conexion_o_pago_por_uso_de_internet",
    "cuadros_y_obras_originales_de_arte",
    "cuotas_extraordinarias_de_administracion_o_comunitarias",
    "diversiones_y_entretenimiento_espectaculos_discotecas_cines_deportes_etc",
    "educacion_pagos_de_matriculas_escolares_universitarias_uniformes_diferentes_a_las_reportadas_en_403_m",
    "gastos_en_educacion",
    "gastos_en_educacion_pensiones_libros_utiles_fotocopias_etc",
    "gastos_en_salud_medicamentos_examenes_consultas_etc",
    "gastos_en_vacaciones_hospedaje_transporte_pasajes_de_avion_otros",
    "impuesto_a_la_renta_complementarios_y_predial",
    "impuesto_a_la_renta_y_complementarios",
    "lavado_y_planchado_de_ropa_fuera_del_hogar",
    "medias_veladas_para_mujer",
    "muebles_para_el_hogar_sala_comedor_camas_etc",
    "nevera_estufa_tv_lavadora_brilladora_horno_y_otros_electrodomesticos_y_gasodomesticos",
    "ollas_vajillas_cubiertos_y_otros_utensilios_domesticos",
    "pago_de_hoteles",
    "pago_de_impuestos_de_vehiculos_de_uso_del_hogar_seguro_obligatorio_soat",
    "pago_del_servicio_de_celular_compra_de_minutos_o_recargas",
    "pasajes_de_avion",
    "polizas_o_seguro_de_salud",
    "primas_o_polizas_de_seguros",
    "reparaciones_y_mejoras_de_la_vivienda_plomeria_electricidad_pintura_resane_y_panetes",
    "salud_medicamentos_examenes_consultas_etc",
    "salud_terapias_hospitalizaciones_etc_diferentes_a_las_reportadas_en_403_l",
    "seguro_contra_incendios_o_contra_robo_de_la_vivienda_o_vehiculos_de_uso_del_hogar",
    "tela_para_vestuario_u_otros_usos",
    "vehiculo_moto_para_uso_del_hogar",
]


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    """consecutivo (ola 1) / llave (ola 2) / llave_n16 (ola 3): 1 fila = 1 sub-hogar."""
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def _factores_respaldo_ponderados(gastos: pd.DataFrame) -> dict:
    """
    Para ARTICULOS_PROMEDIO_PONDERADO: factor mensual = promedio ponderado
    por la distribucion real de periodicidad observada en ola 1 urbana
    (unica fuente con per_ poblado para estos articulos). No es un supuesto
    arbitrario -- se deriva de los datos cada vez que se corre el script.
    """
    base = gastos[(gastos["ola"] == 1) & (gastos["zona"] == "Urbano")]
    factores = {}
    for articulo in ARTICULOS_PROMEDIO_PONDERADO:
        proporciones = base[f"per_{articulo}"].value_counts(normalize=True)
        factores[articulo] = sum(
            proporciones[cat] * FACTOR_MENSUAL[cat] for cat in proporciones.index
        )
    return factores


def construir_gasto_mensual(gastos: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza cada vr_{articulo} a equivalente mensual y suma.

    Jerarquia por fila: (1) si per_{articulo} esta poblado PARA ESA FILA, se
    usa la periodicidad declarada; (2) si no, se usa el factor de respaldo
    que corresponda segun el articulo (trimestral fijo, promedio ponderado,
    o anual) -- ver las listas ARTICULOS_* arriba y el docstring del modulo
    para la justificacion de cada grupo.
    """
    vr_cols = sorted(
        c for c in gastos.columns if c.startswith("vr_") and not c.startswith("vr_obt_")
    )
    articulos = [c[len("vr_"):] for c in vr_cols]

    factores_ponderados = _factores_respaldo_ponderados(gastos)
    mensualizado = pd.DataFrame(index=gastos.index)

    for articulo in articulos:
        vr_col = f"vr_{articulo}"
        per_col = f"per_{articulo}"
        gasto_col = f"gasto_{articulo}"

        vr = gastos[vr_col]
        compro = gastos[gasto_col].fillna(0).astype(bool)

        if per_col in gastos.columns:
            factor_declarado = gastos[per_col].map(FACTOR_MENSUAL)
        else:
            factor_declarado = pd.Series(np.nan, index=gastos.index)

        if articulo in ARTICULOS_TRIMESTRAL_FIJO:
            factor_respaldo = 1.0 / 3.0
        elif articulo in ARTICULOS_PROMEDIO_PONDERADO:
            factor_respaldo = factores_ponderados[articulo]
        else:
            factor_respaldo = 1.0 / 12.0

        factor = factor_declarado.fillna(factor_respaldo)

        valor_mensual = (vr * factor).where(compro, 0.0)
        mensualizado[articulo] = valor_mensual

    todo_sin_comprar = (gastos[[f"gasto_{a}" for a in articulos]].fillna(0) == 0).all(axis=1)
    comprado_sin_valor = mensualizado.isna().all(axis=1) & ~todo_sin_comprar

    gasto_mensual = mensualizado.sum(axis=1, skipna=True)
    gasto_mensual[comprado_sin_valor] = np.nan

    resultado = gastos[["consecutivo", "ola", "hogar", "hogar_n16", "llave", "llave_n16", "zona"]].copy()
    resultado["gasto_mensual_hogar"] = gasto_mensual
    resultado["n_articulos_comprados"] = gastos[[f"gasto_{a}" for a in articulos]].fillna(0).sum(axis=1)
    return resultado


def agregar_series_reales(salida: pd.DataFrame) -> pd.DataFrame:
    """deflactor_ipc_ing_bajos es especifico de la zona del hogar; merge por ola y zona."""
    deflactor = pd.read_parquet(DEFLACTOR_PATH)[
        ["ola", "zona", "deflactor_ipc_total", "deflactor_ipc_ing_bajos"]
    ]
    salida = salida.merge(deflactor, on=["ola", "zona"], how="left")

    for sufijo, columna in [
        ("ipctotal", "deflactor_ipc_total"),
        ("ipcbajos", "deflactor_ipc_ing_bajos"),
    ]:
        salida[f"gasto_total_hogar_real_{sufijo}"] = salida["gasto_mensual_hogar"] / salida[columna]
        salida[f"gasto_percapita_hogar_real_{sufijo}"] = (
            salida["gasto_percapita_hogar"] / salida[columna]
        )

    return salida.drop(columns=["deflactor_ipc_total", "deflactor_ipc_ing_bajos"])


def documentar_supuesto_periodicidad(gastos: pd.DataFrame) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    factores_ponderados = _factores_respaldo_ponderados(gastos)

    filas = []
    for a in ARTICULOS_TRIMESTRAL_FIJO:
        filas.append({
            "articulo": a,
            "factor_respaldo": round(1 / 3, 4),
            "supuesto": "Capitulo IX-B del manual metodologico (Gastos TRIMESTRALES, "
            "cod_articulo 21-25): periodo fijo de 3 meses. Confirmado contra "
            "ELCA-Manual-Hogar-Urbano2010.pdf y cod_articulo de los .tab crudos.",
        })
    for a in ARTICULOS_PROMEDIO_PONDERADO:
        filas.append({
            "articulo": a,
            "factor_respaldo": round(factores_ponderados[a], 4),
            "supuesto": "Sin periodo dominante claro. Factor = promedio ponderado por la "
            "distribucion real de periodicidad observada en ola 1 urbana (unica fuente "
            "con dato individual para este articulo), calculado en tiempo de ejecucion.",
        })
    for a in ARTICULOS_SIN_PERIODO:
        filas.append({
            "articulo": a,
            "factor_respaldo": round(1 / 12, 4),
            "supuesto": "Sin columna per_ en ninguna ola. Se asume recall anual, por "
            "capitulo IX-C del manual metodologico (Gastos ANUALES, cod_articulo "
            "26-35) donde aplica, o por analogia con el mismo capitulo para gastos "
            "excepcionales de salud/educacion/impuestos sin equivalente documentado.",
        })

    pd.DataFrame(filas).to_csv(DOC_PATH, index=False)


def main() -> None:
    gastos = pd.read_parquet(GASTOS_PATH)
    hogar = pd.read_parquet(HOGAR_PATH)[["consecutivo", "ola", "llave", "llave_n16", "t_personas"]]

    salida = construir_gasto_mensual(gastos)

    llave_gastos = _llave_compuesta(salida)
    llave_hogar = _llave_compuesta(hogar)
    t_personas = hogar.set_index(llave_hogar)["t_personas"]
    salida["t_personas"] = llave_gastos.map(t_personas)

    salida["gasto_percapita_hogar"] = salida["gasto_mensual_hogar"] / salida["t_personas"]
    salida = agregar_series_reales(salida)

    salida.to_parquet(OUTPUT_PATH, index=False)
    documentar_supuesto_periodicidad(gastos)

    print(f"Guardado: {OUTPUT_PATH} ({len(salida):,} filas)")
    print(f"Supuestos de periodicidad documentados en: {DOC_PATH}")
    print()
    print(salida.groupby("ola")["gasto_mensual_hogar"].describe())


if __name__ == "__main__":
    main()
