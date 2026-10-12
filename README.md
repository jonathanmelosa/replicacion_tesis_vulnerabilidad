# Paquete de replicación

**¿Pueden los datos geoespaciales públicos mejorar la predicción de la vulnerabilidad a la pobreza? Evidencia para hogares colombianos con base en la Encuesta Longitudinal Colombiana (ELCA)**

Jonathan Melo Sarta, Maestría en Economía Aplicada, Universidad de los Andes.

Este paquete reproduce todas las tablas y figuras del artículo. Sigue la plantilla de README de los Social Science Data Editors.

## Resumen

El código construye el panel de hogares de la ELCA (2010, 2013 y 2016), predice qué hogares no pobres serán pobres en la encuesta siguiente y evalúa si las luces nocturnas de baja resolución (DMSP-OLS) y otras fuentes geoespaciales públicas mejoran esa predicción. El paquete permite reproducir los resultados en tres niveles, según los datos a los que se tenga acceso:

| Nivel | Qué reproduce | Datos que requiere |
|---|---|---|
| `agregados` | Las 46 tablas y figuras del artículo, salvo la Tabla 18 | Solo los resultados agregados incluidos en `data/` y `outputs/` |
| `elca` | Además, la construcción del panel, las transiciones de pobreza y la Tabla 18 | Microdatos públicos de la ELCA (se descargan automáticamente) |
| `completo` | Además, los modelos, SHAP, los diagnósticos y las pruebas de robustez | Lo anterior y las variables geoespaciales por hogar (de acceso restringido; ver abajo) |

## Disponibilidad y procedencia de los datos

### Declaración de derechos

El autor tiene acceso legítimo a todas las fuentes usadas y la autorización para publicar los resultados agregados incluidos en este paquete. Ninguno de los archivos publicados contiene información individual de los hogares.

### Fuentes

| Fuente | Acceso | Incluida en el paquete |
|---|---|---|
| **ELCA 2010, 2013 y 2016**, Centro de Estudios sobre Desarrollo Económico (CEDE), Universidad de los Andes | Pública. Repositorio de datos de la Universidad de los Andes, DOI [10.57924/DPF0M5](https://doi.org/10.57924/DPF0M5) (2010), [10.57924/DE9LP7](https://doi.org/10.57924/DE9LP7) (2013) y [10.57924/BLUILW](https://doi.org/10.57924/BLUILW) (2016). `src/01_download/01_descarga_ELCA/00_descarga_API_bases.py` las descarga sin credenciales. | No (se descargan) |
| **Coordenadas de los hogares de la ELCA** | Restringidas. Solo pueden consultarse en la sala de procesamiento de datos del CEDE, Universidad de los Andes, previa solicitud. | No |
| **Variables geoespaciales por hogar** (DMSP-OLS, ALOS PALSAR, Landsat 5 TM en un círculo de ~930 m alrededor de la vivienda) | Restringidas: se construyen dentro de la sala de datos con el código de `sala_datos/`. El nivel `completo` espera el resultado en `data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet`. | No |
| **Imágenes satelitales**: DMSP-OLS (serie temporal v4, NOAA NGDC), ALOS PALSAR (mosaico anual, JAXA), Landsat 5 TM (Collection 2, nivel 2, USGS) | Públicas, a través de Google Earth Engine (requiere una cuenta de servicio propia). | No |
| **Fondo de luces nocturnas del mapa** (serie armonizada CCNL 1992–2013, DOI [10.5281/zenodo.6644980](https://doi.org/10.5281/zenodo.6644980)) | Pública. `src/02_build/eda_dmsp_fondo_mapa.py` la procesa. | Sí, el recorte para Colombia (`data/interim/geo_referencia/fondo_dmsp_colombia_2010.npz`) |
| **Google Street View** | Requiere una clave de la API de Google y las coordenadas de los hogares. Las condiciones de uso no permiten redistribuir las imágenes. | No; solo resultados agregados |
| **Límites administrativos GADM 4.1** | Públicos, pero su licencia no permite redistribuirlos. `src/descargar_gadm.py` los descarga. | No (se descargan) |

### Resultados agregados incluidos

`data/processed/benchmark_resultados/` y `outputs/tables/` contienen los resultados agregados (métricas, intervalos de confianza, importancias SHAP por variable, tablas de transición, conteos) con los que se regeneran las tablas y figuras. `diagnostico_curva_pr_puntos.csv` tiene los puntos de las curvas precision-recall, sin identificadores.

## Requisitos de cómputo

- **Software:** Python 3.9.6 y los paquetes de `requirements.txt`, con las versiones usadas para los resultados del artículo. El código de la sala de datos y de Street View requiere además `requirements-sala.txt`.
- **Hardware usado:** Apple M4, 16 GB de RAM, macOS.
- **Tiempo:** el nivel `agregados` tarda unos minutos. Los niveles `elca` y `completo` no se han cronometrado de punta a punta; el `completo` puede tomar uno o más días, sobre todo por la búsqueda de hiperparámetros con validación cruzada de 10 particiones y 5 semillas.
- **Figuras:** se generaron con el *backend* gráfico por defecto de matplotlib en macOS. Con otro *backend* (por ejemplo, `Agg` en Linux) el recorte automático de márgenes puede diferir en un píxel; el contenido es el mismo.

## Descripción del código

La estructura de carpetas es la del proyecto original, para que las rutas relativas de los scripts funcionen sin cambios.

| Carpeta | Contenido |
|---|---|
| `src/01_download/01_descarga_ELCA/` | Descarga y consolidación de las tres olas de la ELCA |
| `src/01_download/02_scr_GoogleStreetView/` | Panel de coordenadas, cobertura y descarga de Street View, y extracción de representaciones de las imágenes (requiere coordenadas) |
| `src/03_clean/` | Limpieza de errores de codificación de caracteres |
| `src/04_features/` | Construcción de variables por módulo, ingreso, gasto, pobreza monetaria, IPM, transiciones y el panel consolidado |
| `src/02_build/` | Agregados descriptivos del panel y de las transiciones |
| `src/05_model/` | Bases de entrenamiento y prueba, modelos, SHAP, diagnósticos y pruebas de robustez |
| `src/` (raíz) | Scripts que producen las tablas (`tabla_*.py`) y figuras (`graf_*.py`) del artículo; estilo común (`estilo_*.py`, `etiquetas_variables.py`) |
| `sala_datos/` | Extracción de DMSP-OLS, ALOS PALSAR y Landsat 5 TM alrededor de cada hogar con Google Earth Engine, y unificación con el panel. Se ejecuta dentro de la sala de datos; cada carpeta necesita `credenciales/gee_service_account.json` (no incluido). |
| `run_all.py` | Ejecuta los scripts en orden según el nivel elegido |

Varias figuras se calculan con datos por hogar en el nivel `completo`. Para el nivel `agregados`, sus scripts tienen una opción que las redibuja solo desde el CSV agregado correspondiente (`--solo-figura`, `--solo-figuras`, `--solo-figuras-robustez`, `--solo-figura-inventario`).

## Instrucciones

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_all.py agregados    # o: elca, completo
```

Las tablas quedan en `paper/tables/` y las figuras en `outputs/figures/` y `paper/figures/`.

## Tablas y figuras

Numeración de la versión del artículo del 12 de octubre de 2026. Todas, salvo la Tabla 18, se regeneran en el nivel `agregados`; la Tabla 18 requiere el nivel `elca`.

| Elemento | Archivo | Script |
|---|---|---|
| Tabla 1 | `tab_matriz_transicion_combinada.tex` | `src/tabla_matriz_transicion_combinada.py` |
| Tablas 2 y 7 | `tab_desempeno_pr_monetaria.tex`, `tab_desempeno_pr_ipm.tex` | `src/tabla_desempeno_pr.py` |
| Tabla 3 | `tab_aporte_dmsp_pr.tex` | `src/tabla_aporte_dmsp_pr.py` |
| Tabla 4 | `tab_shap_variables_necesarias.tex` | `src/tabla_shap_variables_necesarias.py` |
| Tabla 5 | `tab_sensibilidad_shap.tex` | `src/tabla_sensibilidad_shap.py` |
| Tabla 6 | `tab_shap_signo_nucleo.tex` | `src/tabla_shap_signo_nucleo.py` |
| Tabla 8 | `tab_marginal_dmsp_fbeta2_cv10.tex` | `src/tabla_marginal_dmsp_fbeta2_cv10.py` |
| Tablas 9 y 10 | `tab_significancia_dmsp_ipm.tex`, `tab_marginal_dmsp_ipm_significancia.tex` | `src/tabla_marginal_dmsp_ipm_significancia.py` |
| Tabla 11 | `tab_variacion_dmsp_comunidad.tex` | `src/tabla_variacion_dmsp_comunidad.py` |
| Tablas 12 y 13 | `tab_heterogeneidad_dmsp.tex`, `tab_heterogeneidad_ipm.tex` | `src/tabla_heterogeneidad_dmsp.py` |
| Tabla 14 | `tab_geo3_cv.tex` | `src/tabla_geo3_cv.py` |
| Tablas 15 a 17 | `tab_gsv_prediccion_luz.tex`, `tab_gsv_pobreza.tex`, `tab_gsv_desfase_modelos.tex` | `src/tabla_gsv_anexo.py` |
| Tabla 18 | `tab_cobertura_modulos.tex` | `src/tabla_cobertura_modulos.py` (nivel `elca`) |
| Tabla 19 | `tab_temas_variables.tex` | `src/tabla_temas_variables.py` |
| Tablas 20 y 21 | `tab_errores_nucleo.tex`, `tab_errores_choques.tex` | `src/tabla_errores_modelo_principal.py` |
| Tablas 22 y 23 | `tab_multiclase_metricas.tex`, `tab_multiclase_auc_pares.tex` | `src/tabla_multiclase_anexo.py` |
| Tabla 24 | `tab_trayectorias_3olas.tex` | `src/tabla_trayectorias_3olas.py` |
| Tabla 25 | `tab_split_trayectoria_nucleo.tex` | `src/tabla_split_trayectoria_nucleo.py` |
| Tabla 26 | `tab_hiperparametros.tex` | `src/tabla_hiperparametros.py` |
| Tabla 27 | `tab_robustez_validacion.tex` | `src/tabla_robustez_validacion.py` |
| Tabla 28 | `tab_robustez_historia.tex` | `src/tabla_robustez_historia.py` |
| Tablas 29 y 30 | `tab_robustez_techo.tex`, `tab_robustez_icc.tex` | `src/tabla_robustez_techo.py` |
| Tablas 31 y 32 | `tab_robustez_desactualizada.tex`, `tab_robustez_desactualizada_grupos.tex` | `src/tabla_robustez_desactualizada.py` |
| Figura 1 | `curva_pr_monetaria.png` | `src/graf_curva_pr.py` |
| Figura 2 | `perfil_nucleo_grupos_monetaria.png` | `src/graf_perfil_nucleo_grupos.py` |
| Figura 3 | `dmsp_monotonico_categoria.png` | `src/graf_dmsp_monotonico_categoria.py` |
| Figura 4 | `prueba_estilo_tramas_regional.png` | `src/graf_mapa_territorial_tramas.py` (requiere `src/descargar_gadm.py`) |
| Figura 5 | `curva_pr_dmsp_monetaria.png` | `src/graf_curva_pr_dmsp.py` |
| Figura 6 | `05_residual_dmsp_riqueza.png` | `src/05_model/graficar_residual_dmsp_riqueza.py` |
| Figuras 7 a 9 | `08_sensibilidad_banda.png`, `07_robustez_transiciones.png`, `09_atricion_panel.png` | `src/04_features/build_pobreza_desagregaciones.py` |
| Figura 10 | `variacion_dmsp_comunidad.png` | `src/graf_variacion_dmsp_comunidad.py` |
| Figura 11 | `01_inventario_por_modulo.png` | `src/02_build/eda_variables_modelo.py` |
| Figura 12 | `01_hogares_por_ola.png`, `03_attrition_tasas.png` | `src/02_build/eda_hogares_panel.py` |
| Figura 13 | `fig_cobertura_temporal_fuentes.png` | `src/timeline_cobertura_geoespacial.py` (conteos de `src/conteo_gsv_por_anio.py`) |

Las cifras citadas en el texto que no aparecen en ninguna tabla salen de los mismos resultados agregados. Por ejemplo, la estabilidad temporal de DMSP-OLS sale de `diagnostico_variabilidad_temporal_dmsp.csv`, la prevalencia por grupo de la prueba de historia del hogar de `historia_resultados.csv`, y los conteos del panel y la atrición de `outputs/tables/eda_hogares/`.

## Licencia

Código y resultados agregados: licencia MIT (ver `LICENSE`).

## Cita

Ver `CITATION.cff`.
