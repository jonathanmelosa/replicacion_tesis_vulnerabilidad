"""
run_all.py
==========
Reproduce las tablas y figuras del artículo. Tres niveles:

    python run_all.py agregados   # tablas y figuras desde resultados/ agregados (cualquier persona)
    python run_all.py elca        # + descarga y procesamiento de la ELCA (datos públicos)
    python run_all.py completo    # + modelos y diagnósticos (requiere las variables
                                  #   geoespaciales por hogar; ver README)

Cada script se ejecuta desde su propia carpeta, como en el desarrollo.
"""
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# ── Nivel "elca": solo microdatos públicos de la ELCA ──────────────────────
ELCA = [
    "src/01_download/01_descarga_ELCA/00_descarga_API_bases.py",
    "src/01_download/01_descarga_ELCA/01_consolidacion_bases_choques.py",
    "src/01_download/01_descarga_ELCA/02_consolidacion_bases_gastos.py",
    "src/01_download/01_descarga_ELCA/04_consolidacion_bases_hogar.py",
    "src/01_download/01_descarga_ELCA/05_consolidacion_bases_comunidades.py",
    "src/01_download/01_descarga_ELCA/06_consolidacion_bases_personas.py",
    "src/01_download/01_descarga_ELCA/07_consolidacion_bases_ninos.py",
    "src/03_clean/01_limpieza_base_hogar.py",
    "src/03_clean/05_limpieza_corrupcion_hogar.py",
    "src/03_clean/02_limpieza_base_personas.py",
    "src/03_clean/03_limpieza_base_comunidades.py",
    "src/03_clean/04_limpieza_base_ninos.py",
    "src/04_features/build_deflactor_ipc.py",
    "src/04_features/build_ingreso_hogar.py",
    "src/04_features/build_gasto_hogar.py",
    "src/04_features/build_pobreza_monetaria.py",
    "src/04_features/build_hogar_features.py",
    "src/04_features/build_personas_hogar.py",
    "src/04_features/build_personas_hogar_ext.py",
    "src/04_features/build_educacion_ocupacion_hogar.py",
    "src/04_features/build_educacion_ocupacion_hogar_ext.py",
    "src/04_features/build_salud_discapacidad_hogar.py",
    "src/04_features/build_salud_discapacidad_hogar_ext.py",
    "src/04_features/build_ahorro_capital_social_hogar.py",
    "src/04_features/build_becas_subsidios_hogar.py",
    "src/04_features/build_participacion_civica_hogar.py",
    "src/04_features/build_comunidades_hogar.py",
    "src/04_features/build_ninos_hogar.py",
    "src/04_features/build_choques_hogar.py",
    "src/04_features/build_benchmark_consolidado.py",
    "src/04_features/build_ipm_multidimensional.py",
    "src/04_features/build_matriz_transicion_ipm.py",
    "src/04_features/build_pobreza_desagregaciones.py",
    "src/02_build/eda_hogares_panel.py",
    "src/02_build/eda_trayectorias_3olas.py",
    "src/tabla_cobertura_modulos.py",
]

# ── Nivel "completo": requiere data/processed/SALE_13082026/
#    variables_geoespaciales_unificadas.parquet (sala de datos) ────────────
COMPLETO = [
    "src/02_build/eda_dmsp_fondo_mapa.py",
    "src/02_build/eda_variables_modelo.py",
    "src/02_build/eda_transicion_covariables.py",
    "src/05_model/build_benchmark_train_test.py",
    "src/05_model/build_benchmark_train_test_ipm.py",
    "src/05_model/build_benchmark_train_test_4clases.py",
    "src/05_model/construir_pipeline_geo_dmsp.py",
    "src/05_model/construir_pipeline_geo_dmsp_ipm.py",
    "src/05_model/construir_pipeline_geo_dmsp_4clases.py",
    "src/05_model/construir_pipeline_geo3_cv.py",
    "src/05_model/modelo_fbeta2_cv10_comparacion.py",
    "src/05_model/modelo_fbeta2_cv10_comparacion_rf_lgbm.py",
    "src/05_model/modelo_ipm_comparacion.py",
    "src/05_model/modelo_ipm_comparacion_rf_lgbm.py",
    "src/05_model/modelo_geo3_baseline_cv.py",
    "src/05_model/modelo_geo3_robusto_cv.py",
    "src/05_model/modelo_multiclase_robusto_comparacion.py",
    "src/05_model/generar_predicciones_test_dmsp.py",
    "src/05_model/generar_predicciones_test_ipm.py",
    "src/05_model/diagnostico_curva_pr.py",
    "src/05_model/diagnostico_bootstrap_dmsp.py",
    "src/05_model/diagnostico_bootstrap_ipm.py",
    "src/05_model/diagnostico_bootstrap_cluster_dmsp.py",
    "src/05_model/diagnostico_bootstrap_cluster_ipm.py",
    "src/05_model/diagnostico_bootstrap_cluster_multiclase.py",
    "src/05_model/auc_pares_multiclase_semillas.py",
    "src/05_model/diagnostico_heterogeneidad_dmsp.py",
    "src/05_model/diagnostico_heterogeneidad_ipm.py",
    "src/05_model/diagnostico_shap_ab.py",
    "src/05_model/diagnostico_shap_2013_2016.py",
    "src/05_model/diagnostico_shap_ipm.py",
    "src/05_model/diagnostico_shap_ipm_2013_2016.py",
    "src/05_model/diagnostico_shap_signo.py",
    "src/05_model/diagnostico_shap_nucleo_perfil.py",
    "src/05_model/diagnostico_shap_variables_necesarias.py",
    "src/05_model/sensibilidad_shap_nucleo.py",
    "src/05_model/diagnostico_errores_modelo_principal.py",
    "src/05_model/diagnostico_variacion_dmsp_comunidad.py",
    "src/05_model/diagnostico_residual_dmsp_riqueza.py",
    "src/05_model/diagnostico_variabilidad_temporal_dmsp.py",
    "src/05_model/robustez_hogares_no_repetidos.py",
    "src/05_model/prevalencia_validacion_b.py",
    "src/05_model/robustez_historia_hogar.py",
    "src/05_model/robustez_techo_comunidad.py",
    "src/05_model/robustez_encuesta_desactualizada.py",
    "src/05_model/robustez_encuesta_desactualizada_diagnostico.py",
    "src/02_build/eda_split_trayectoria_nucleo.py",
    "src/graf_perfil_nucleo_grupos.py",
]

# ── Nivel "agregados": tablas y figuras desde los resultados publicados ────
AGREGADOS = [
    "src/descargar_gadm.py",
    "src/tabla_matriz_transicion_combinada.py",
    "src/tabla_trayectorias_3olas.py",
    "src/tabla_desempeno_pr.py",
    "src/tabla_aporte_dmsp_pr.py",
    "src/tabla_shap_variables_necesarias.py",
    "src/tabla_sensibilidad_shap.py",
    "src/tabla_shap_signo_nucleo.py",
    "src/tabla_marginal_dmsp_fbeta2_cv10.py",
    "src/tabla_marginal_dmsp_ipm_significancia.py",
    "src/tabla_variacion_dmsp_comunidad.py",
    "src/tabla_heterogeneidad_dmsp.py",
    "src/tabla_geo3_cv.py",
    "src/tabla_gsv_anexo.py",
    "src/tabla_temas_variables.py",
    "src/tabla_errores_modelo_principal.py",
    "src/tabla_multiclase_anexo.py",
    "src/tabla_split_trayectoria_nucleo.py",
    "src/tabla_hiperparametros.py",
    "src/tabla_robustez_validacion.py",
    "src/tabla_robustez_historia.py",
    "src/tabla_robustez_techo.py",
    "src/tabla_robustez_desactualizada.py",
    "src/graf_curva_pr.py",
    "src/graf_curva_pr_dmsp.py",
    "src/graf_dmsp_monotonico_categoria.py",
    "src/graf_mapa_territorial_tramas.py",
    "src/graf_perfil_nucleo_grupos.py --solo-figura",
    "src/graf_variacion_dmsp_comunidad.py",
    "src/timeline_cobertura_geoespacial.py",
    "src/05_model/graficar_residual_dmsp_riqueza.py",
    "src/04_features/build_pobreza_desagregaciones.py --solo-figuras-robustez",
    "src/02_build/eda_hogares_panel.py --solo-figuras",
    "src/02_build/eda_variables_modelo.py --solo-figura-inventario",
]

NIVELES = {
    "agregados": AGREGADOS,
    "elca": ELCA + AGREGADOS,
    "completo": ELCA + COMPLETO + AGREGADOS,
}


def correr(paso: str) -> None:
    script, *args = paso.split()
    ruta = RAIZ / script
    print(f"\n>>> {paso}", flush=True)
    subprocess.run([sys.executable, ruta.name, *args], cwd=ruta.parent, check=True)


def main() -> None:
    nivel = sys.argv[1] if len(sys.argv) > 1 else "agregados"
    if nivel not in NIVELES:
        sys.exit(f"Nivel desconocido: {nivel}. Opciones: {', '.join(NIVELES)}")
    for paso in NIVELES[nivel]:
        correr(paso)
    print(f"\nListo (nivel {nivel}). Tablas en paper/tables/, figuras en outputs/figures/ y paper/figures/.")


if __name__ == "__main__":
    main()
