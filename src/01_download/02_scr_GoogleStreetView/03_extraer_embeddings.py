"""
03_extraer_embeddings.py
========================
Extrae embeddings visuales a partir de las imágenes de Google Street View
descargadas por 02_descarga_fotos_GSV.py, usando modelos preentrenados
mediante transfer learning.

RESPONSABILIDAD EXCLUSIVA
    Transformar un directorio de imágenes JPEG en una matriz de representaciones
    vectoriales (embeddings) que capture contenido visual relevante para el
    análisis de entorno urbano.

    Este script NO realiza:
    · Descarga de imágenes (responsabilidad de script 02).
    · Análisis espacial ni consultas a la API (responsabilidad de scripts 01 y 02).
    · Entrenamiento, fine-tuning ni clasificación.
    · Reducción de dimensionalidad (PCA, UMAP, etc.).
    · Selección de variables ni análisis estadístico.

INPUT
    gsv/registro_descargas.csv  → log de script 02 (imágenes exitosas)
    gsv/inventario_panos.csv    → identificadores del hogar (llave, etc.)

OUTPUTS
    embeddings/embeddings_{modelo}.parquet      → embeddings (Output 1)
    embeddings/reporte_embeddings_{modelo}.txt  → informe (Output 2)

MODELOS DISPONIBLES
    "vgg19"     → VGG-19 preentrenada en ImageNet; embedding de 4 096 dimensiones.
    "resnet50"  → ResNet-50 preentrenada en ImageNet; embedding de 2 048 dimensiones.
    "places365" → ResNet-50 preentrenada en Places365 (escenas urbanas);
                  embedding de 2 048 dimensiones.
                  Los pesos se descargan automáticamente si no están en caché.

FORMATO DE SALIDA
    Una fila por imagen. Columnas principales:
        image_id, consecutivo, ola, pano_id, heading, nombre_archivo,
        llave (si disponible), llave_n16 (si disponible),
        modelo, embedding_dim, embedding (lista de float32).

CÓMO CORRER
    python 03_extraer_embeddings.py

    Para cambiar el modelo: modificar CONFIG["modelo"].
    Para procesar en GPU: modificar CONFIG["device"] = "cuda" o "mps".
"""

import sys
import time
import hashlib
import logging
import urllib.request
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import pyarrow as _pyarrow; del _pyarrow  # guard: falla aquí si falta, antes de extraer embeddings
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from tqdm import tqdm
except ImportError as _err:
    # Si ves este error, el entorno virtual no está activo o las dependencias
    # no están instaladas.
    # Solución paso a paso:
    #   1. Activa el entorno virtual:
    #        Mac/Linux : source .venv/bin/activate
    #        Windows   : .venv\Scripts\activate
    #   2. Instala las dependencias generales:
    #        pip install -r requirements.txt
    #   3. Si falta torch o torchvision específicamente:
    #        CPU / Mac  : pip install torch torchvision
    #        GPU CUDA   : pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    #        Apple MPS  : pip install torch torchvision   (ya incluye MPS en macOS ≥ 12)
    print("=" * 62)
    print(f"ERROR: librería no encontrada → {_err}")
    print("Revisa los comentarios junto a los imports en este script.")
    print("=" * 62)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent   # carpeta GoogleStreetView/

CONFIG = {
    # ── Modelo ─────────────────────────────────────────────────────────────────
    # "vgg19"     : VGG-19 ImageNet, embedding 4 096 dimensiones.
    # "resnet50"  : ResNet-50 ImageNet, embedding 2 048 dimensiones.
    # "places365" : ResNet-50 Places365 (escenas urbanas), 2 048 dimensiones.
    "modelo": "vgg19",

    # ── Inputs ─────────────────────────────────────────────────────────────────
    # Registro de descargas producido por 02_descarga_fotos_GSV.py.
    # Contiene las rutas de archivo y el estado de cada imagen (exito=True/False).
    "input_registro": _HERE / "gsv" / "registro_descargas.csv",

    # Inventario producido por 01_analisis_cobertura_gsv.py.
    # Se usa para incorporar identificadores adicionales del hogar (llave, llave_n16).
    # Si no existe, el output incluirá solo consecutivo y ola.
    "input_inventario": _HERE / "gsv" / "inventario_panos.csv",

    # ── Places365 (solo necesario si modelo="places365") ───────────────────────
    # Ruta donde se almacenarán (o ya están almacenados) los pesos del modelo.
    # Si el archivo no existe, se descarga automáticamente de places2.csail.mit.edu.
    "places365_cache": _HERE / "modelos" / "places365_resnet50.pth.tar",

    # ── Inferencia ─────────────────────────────────────────────────────────────
    "batch_size": 32,          # imágenes por lote; reducir a 8 si hay OOM en GPU
    "device":     "auto",      # "auto" | "cpu" | "cuda" | "mps"

    # ── Salida ─────────────────────────────────────────────────────────────────
    "output_dir": _HERE / "embeddings",
}


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

# URL oficial del checkpoint de Places365-ResNet50.
_PLACES365_URL = (
    "http://places2.csail.mit.edu/models_places365/"
    "resnet50_places365.pth.tar"
)

# Normalización estándar ImageNet.  Válida para VGG19, ResNet50 e incluso
# Places365 (que fue entrenado con el mismo pipeline de preprocesamiento).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Dimensión del embedding por modelo.
_EMBEDDING_DIM = {"vgg19": 4096, "resnet50": 2048, "places365": 2048}


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
# MÓDULO 1: CARGA DEL REGISTRO DE IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────

def cargar_imagenes(cfg: dict) -> pd.DataFrame:
    """
    Lee el registro de descargas y retorna las imágenes disponibles para
    extraer embeddings.

    Filtros aplicados:
    · exito == True  → solo imágenes descargadas con éxito por script 02.
    · ruta_archivo no nula.

    Si el inventario está disponible, incorpora identificadores adicionales del
    hogar (llave, llave_n16) mediante un join por (consecutivo, ola, pano_id).

    Retorna:
        DataFrame con una fila por imagen, con columnas de identificadores
        y la ruta absoluta del archivo en disco.
    """
    path_reg = Path(cfg["input_registro"])
    if not path_reg.exists():
        log.error("-" * 62)
        log.error("ERROR: Registro de descargas no encontrado.")
        log.error(f"Buscado en: {path_reg.resolve()}")
        log.error("SOLUCION: Ejecuta primero el script anterior:")
        log.error("  python 02_descarga_fotos_GSV.py")
        log.error("Ese script genera: gsv/registro_descargas.csv")
        log.error("Si el archivo existe pero en otra carpeta, actualiza")
        log.error("CONFIG['input_registro'] en este script con la ruta correcta.")
        log.error("-" * 62)
        sys.exit(1)

    # Carga el registro con tipos explícitos para los identificadores de texto
    reg = pd.read_csv(
        path_reg,
        dtype={"consecutivo": str, "pano_id": str},
        low_memory=False,
    )
    log.info(f"Registro cargado: {len(reg):,} filas ({path_reg.name})")

    # Filtra solo las imágenes marcadas como exitosas
    if "exito" not in reg.columns:
        log.error("-" * 62)
        log.error("ERROR: El registro no tiene la columna 'exito'.")
        log.error("Columnas presentes: " + str(list(reg.columns)))
        log.error("El archivo gsv/registro_descargas.csv esta incompleto o corrompido.")
        log.error("SOLUCION: Borra gsv/registro_descargas.csv y vuelve a ejecutar:")
        log.error("  python 02_descarga_fotos_GSV.py")
        log.error("Eso regenerara el registro completo.")
        log.error("-" * 62)
        sys.exit(1)

    # NOTA: pd.read_csv lee la columna 'exito' como strings "True"/"False", NO como
    # booleanos Python.  La comparación  reg["exito"] == True  siempre devuelve False
    # porque  "True" == True  es False en Python → df_ok quedaría vacío y el script
    # terminaría diciendo que no hay imágenes, aunque sí las haya.
    # Se corrige comparando como string con .astype(str).str.strip() == "True".
    df_ok = reg[reg["exito"].astype(str).str.strip() == "True"].copy()
    df_ok = df_ok[df_ok["ruta_archivo"].notna()].copy()
    log.info(f"  Imágenes con exito=True y ruta válida: {len(df_ok):,}")

    if df_ok.empty:
        log.warning("-" * 62)
        log.warning("AVISO: No hay imagenes con exito=True en el registro.")
        n_total_reg = len(reg)
        # Mismo motivo de comparación-string: "False" == False es False en Python.
        n_fallidas  = int((reg["exito"].astype(str).str.strip() == "False").sum()) if "exito" in reg.columns else 0
        log.warning(f"Total en registro: {n_total_reg:,}  |  Fallidas: {n_fallidas:,}")
        log.warning("Causas posibles:")
        log.warning("  · El script 02 no completo la descarga (interrupcion).")
        log.warning("    Solucion: vuelve a ejecutar python 02_descarga_fotos_GSV.py")
        log.warning("  · Todas las descargas fallaron (error de API o conexion).")
        log.warning("    Revisa gsv/reporte_descarga.txt para diagnostico detallado.")
        log.warning("  · La carpeta gsv/fotos/ esta vacia o no tiene archivos JPEG.")
        log.warning("    Verifica manualmente que existan imagenes en gsv/fotos/ola_XXXX/")
        log.warning("-" * 62)
        sys.exit(0)

    # ── Join opcional con el inventario para obtener llave y llave_n16 ────────
    path_inv = Path(cfg["input_inventario"])
    if path_inv.exists():
        inv = pd.read_csv(
            path_inv,
            dtype={"consecutivo": str, "pano_id": str, "llave": str, "llave_n16": str},
            low_memory=False,
        )
        # El inventario tiene múltiples filas por (consecutivo, ola, pano_id)
        # (una por radio); se desduplicó para conservar la fila más pequeña.
        # NOTA: llave y llave_n16 solo existen en olas 2013/2016 respectivamente.
        # Si el inventario no las incluye (versión antigua del pipeline o solo ola 2010),
        # inv[["llave", "llave_n16"]] lanzaría KeyError.  Se seleccionan dinámicamente.
        cols_ids = ["consecutivo", "ola", "pano_id"]
        for _c in ["llave", "llave_n16"]:
            if _c in inv.columns:
                cols_ids.append(_c)
        ids_extra = inv[cols_ids].drop_duplicates(subset=["consecutivo", "ola", "pano_id"])
        df_ok = df_ok.merge(ids_extra, on=["consecutivo", "ola", "pano_id"], how="left")
        log.info("  Identificadores llave y llave_n16 incorporados desde el inventario.")
    else:
        # Si no hay inventario, las columnas se añaden vacías para mantener el esquema
        df_ok["llave"]     = pd.NA
        df_ok["llave_n16"] = pd.NA
        log.warning(f"Inventario no encontrado ({path_inv.name}). "
                    "llave y llave_n16 quedarán vacíos.")

    return df_ok.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: CARGA DEL MODELO PREENTRENADO
# ──────────────────────────────────────────────────────────────────────────────

def _resolver_device(device_cfg: str) -> torch.device:
    """
    Resuelve el dispositivo de cómputo a usar.
    "auto" selecciona CUDA si está disponible, luego MPS (Apple Silicon), luego CPU.
    """
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():   # Apple Silicon
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def _cargar_vgg19(device: torch.device) -> tuple:
    """
    Carga VGG-19 preentrenada en ImageNet y elimina la última capa FC.

    Arquitectura del clasificador VGG-19:
        [0] Linear(25088 → 4096)
        [1] ReLU
        [2] Dropout
        [3] Linear(4096 → 4096)
        [4] ReLU
        [5] Dropout
        [6] Linear(4096 → 1000)   ← se elimina con classifier[:-1]
    El embedding resultante es la salida de 4 096 dimensiones después del Dropout [5],
    que en modo eval() equivale a las activaciones ReLU de [4].

    Retorna: (model, transform, embedding_dim)
    """
    try:
        # API nueva (torchvision >= 0.13): se especifica el set de pesos
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    except AttributeError:
        # Fallback para versiones anteriores de torchvision
        model = models.vgg19(pretrained=True)   # type: ignore[call-arg]

    # Elimina la última capa del clasificador (Linear 4096 → 1000)
    model.classifier = model.classifier[:-1]

    model = model.to(device).eval()             # modo evaluación: desactiva dropout y BN
    embedding_dim = _EMBEDDING_DIM["vgg19"]
    log.info(f"VGG-19 cargada (ImageNet). Embedding: {embedding_dim} dimensiones.")
    return model, _transform_imagenet(), embedding_dim


def _cargar_resnet50(device: torch.device) -> tuple:
    """
    Carga ResNet-50 preentrenada en ImageNet y reemplaza la FC final con identidad.

    La capa fc de ResNet-50 es Linear(2048 → 1000).
    Al reemplazarla con nn.Identity(), el modelo retorna el vector de 2048
    dimensiones proveniente del average pooling global.

    Retorna: (model, transform, embedding_dim)
    """
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except AttributeError:
        model = models.resnet50(pretrained=True)   # type: ignore[call-arg]

    model.fc = nn.Identity()                    # elimina la capa de clasificación
    model = model.to(device).eval()
    embedding_dim = _EMBEDDING_DIM["resnet50"]
    log.info(f"ResNet-50 cargada (ImageNet). Embedding: {embedding_dim} dimensiones.")
    return model, _transform_imagenet(), embedding_dim


def _descargar_places365(path_cache: Path) -> None:
    """
    Descarga el checkpoint de Places365-ResNet50 desde el sitio oficial del MIT.
    Solo se ejecuta si el archivo no existe en el path de caché.
    """
    path_cache.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Descargando pesos de Places365 desde {_PLACES365_URL} …")
    log.info("(Solo ocurre una vez; quedan en caché para futuras ejecuciones.)")

    def _progreso(bloques, tam_bloque, tam_total):
        if tam_total > 0:
            pct = min(100.0, bloques * tam_bloque / tam_total * 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(_PLACES365_URL, path_cache, _progreso)
    except Exception as e:
        print()
        log.error("-" * 62)
        log.error(f"ERROR: No se pudieron descargar los pesos de Places365: {e}")
        log.error(f"URL intentada: {_PLACES365_URL}")
        log.error("Causas posibles:")
        log.error("  · Sin conexion a internet o el servidor del MIT esta caido.")
        log.error("  · La red de la universidad bloquea descargas externas grandes.")
        log.error("Soluciones:")
        log.error("  1. Descarga el archivo manualmente desde otro equipo con internet:")
        log.error(f"     {_PLACES365_URL}")
        log.error(f"     Guardalo en: {path_cache.resolve()}")
        log.error("  2. Usa otro modelo que no requiere descarga:")
        log.error("     Cambia CONFIG['modelo'] a 'vgg19' o 'resnet50' en este script.")
        log.error("-" * 62)
        # Borra el archivo parcial si la descarga quedó incompleta
        if path_cache.exists():
            path_cache.unlink()
        sys.exit(1)
    print()
    log.info(f"Pesos guardados en: {path_cache}")


def _cargar_places365(device: torch.device, path_cache: Path) -> tuple:
    """
    Carga ResNet-50 preentrenada en el dataset Places365 (365 categorías de escenas).

    Places365 es especialmente relevante para análisis de entorno urbano porque
    fue entrenada sobre fotografías de escenas (calles, plazas, parques, etc.)
    en lugar de objetos (ImageNet).  El embedding captura características de la
    escena en lugar de características de objetos.

    El checkpoint del MIT contiene las claves con prefijo "module." (DataParallel);
    se eliminan antes de cargar para compatibilidad con modelos no-paralelos.

    Retorna: (model, transform, embedding_dim)
    """
    if not path_cache.exists():
        _descargar_places365(path_cache)

    # ResNet-50 con 365 clases de salida (en lugar de 1000 de ImageNet)
    model = models.resnet50(num_classes=365)

    checkpoint  = torch.load(path_cache, map_location="cpu", weights_only=False)
    state_dict  = checkpoint.get("state_dict", checkpoint)
    # Elimina el prefijo "module." que añade DataParallel al guardar el checkpoint
    state_dict  = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.fc = nn.Identity()                    # elimina la capa de clasificación (365)
    model = model.to(device).eval()
    embedding_dim = _EMBEDDING_DIM["places365"]
    log.info(f"ResNet-50 Places365 cargada. Embedding: {embedding_dim} dimensiones.")
    return model, _transform_imagenet(), embedding_dim


def _transform_imagenet() -> transforms.Compose:
    """
    Transformación estándar para modelos entrenados en ImageNet.
    Válida también para Places365 ya que se entrenó con el mismo pipeline.

    Pasos:
    1. Resize a 256px (lado más corto) manteniendo relación de aspecto.
    2. CenterCrop a 224×224 (tamaño de entrada estándar).
    3. ToTensor: convierte PIL Image a tensor [0, 1].
    4. Normalize: resta media y divide por desviación estándar por canal.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def cargar_modelo(cfg: dict) -> tuple:
    """
    Dispatcher: carga el modelo indicado en CONFIG["modelo"].

    Retorna: (model, transform, embedding_dim, device)
    """
    nombre = cfg["modelo"]
    if nombre not in _EMBEDDING_DIM:
        log.error("-" * 62)
        log.error(f"ERROR: Modelo desconocido: '{nombre}'.")
        log.error(f"Opciones validas: {list(_EMBEDDING_DIM)}")
        log.error("Edita CONFIG['modelo'] en este script con uno de los valores de arriba.")
        log.error("  'vgg19'     → 4096 dimensiones, facil de cargar, requiere mas memoria.")
        log.error("  'resnet50'  → 2048 dimensiones, mas rapido, pesos de ImageNet.")
        log.error("  'places365' → 2048 dimensiones, entrenado en escenas urbanas (recomendado).")
        log.error("-" * 62)
        sys.exit(1)

    device = _resolver_device(cfg["device"])
    log.info(f"Dispositivo de cómputo: {device}")

    if nombre == "vgg19":
        model, transform, dim = _cargar_vgg19(device)
    elif nombre == "resnet50":
        model, transform, dim = _cargar_resnet50(device)
    else:   # "places365"
        model, transform, dim = _cargar_places365(device, Path(cfg["places365_cache"]))

    return model, transform, dim, device


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: DATASET DE IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """
    Dataset de PyTorch que carga imágenes JPEG desde el disco y aplica
    la transformación del modelo.

    Si una imagen no puede cargarse (archivo corrompido, permisos, etc.),
    __getitem__ retorna None en lugar de lanzar una excepción, para que
    el proceso continúe con las imágenes restantes.
    """

    def __init__(self, rutas: list, transform: transforms.Compose):
        self.rutas     = rutas            # lista de rutas absolutas (strings)
        self.transform = transform        # preprocesamiento del modelo

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, idx: int):
        ruta = self.rutas[idx]
        try:
            img = Image.open(ruta).convert("RGB")   # fuerza 3 canales (descarta alpha)
            tensor = self.transform(img)             # aplica preprocesamiento
            return tensor, idx                       # retorna tensor + índice de origen
        except Exception as e:
            # Devuelve None para que el collate_fn lo filtre; el error se registra aparte
            log.warning(f"  No se pudo cargar imagen [{idx}]: {ruta} — {e}")
            return None, idx


def _collate_filtrar_nulos(batch: list) -> tuple:
    """
    Función de collation personalizada: separa los ejemplos válidos (tensor cargado)
    de los inválidos (tensor es None, imagen corrompida o ilegible).

    Retorna: (tensores_validos, indices_ok, indices_error)
    · tensores_validos : tensor (B_ok, C, H, W) con los ejemplos válidos del lote
    · indices_ok       : posiciones en df_ok de los ejemplos procesados
    · indices_error    : posiciones en df_ok de los ejemplos que fallaron al cargarse

    Esto permite que el DataLoader continúe sin interrumpirse y que el bucle
    principal pueda registrar qué imágenes tuvieron error.
    """
    validos   = [(tensor, idx) for tensor, idx in batch if tensor is not None]
    invalidos = [idx           for tensor, idx in batch if tensor is None]

    if not validos:
        # Lote completamente vacío: todos fallaron al cargarse
        return torch.tensor([]), [], invalidos

    tensores, indices = zip(*validos)
    return torch.stack(tensores), list(indices), invalidos


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: EXTRACCIÓN DE EMBEDDINGS
# ──────────────────────────────────────────────────────────────────────────────

def extraer_embeddings(
    model:     nn.Module,
    transform: transforms.Compose,
    df_ok:     pd.DataFrame,
    cfg:       dict,
) -> tuple:
    """
    Ejecuta el modelo sobre todas las imágenes exitosas en lotes (batches).

    Para cada imagen:
    1. El Dataset intenta cargarla y preprocesarla.
    2. El modelo extrae el embedding (vector de activaciones).
    3. El embedding se almacena asociado al índice original en df_ok.

    Los índices con error de carga se excluyen del output y se registran
    por separado para incluirlos en el reporte.

    Retorna:
        (embeddings, indices_ok, indices_error)
        · embeddings   : np.ndarray (N_ok, D) de float32
        · indices_ok   : lista de posiciones en df_ok con embedding exitoso
        · indices_error: lista de posiciones en df_ok con error de carga
    """
    rutas   = df_ok["ruta_archivo"].tolist()   # rutas de las imágenes a procesar
    dataset = ImageDataset(rutas, transform)

    loader  = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,                          # se preserva el orden de df_ok
        num_workers=0,                          # 0 para evitar conflictos con macOS/Windows
        collate_fn=_collate_filtrar_nulos,
    )

    # Preselecciona el device desde el modelo para no necesitarlo como parámetro
    dev           = next(model.parameters()).device
    todos_emb     = []   # lista de arrays (B, D) por lote
    indices_ok    = []   # posiciones en df_ok procesadas con éxito
    indices_error = []   # posiciones en df_ok que fallaron al cargarse

    log.info(f"Extrayendo embeddings de {len(df_ok):,} imágenes "
             f"(batch_size={cfg['batch_size']}, device={dev}) …")

    with torch.no_grad():
        for tensores, idxs_ok, idxs_err in tqdm(loader, desc="Lotes procesados", unit="lote"):
            # Si ves RuntimeError: CUDA out of memory o MPS out of memory,
            # reduce CONFIG['batch_size'] a 8 o incluso 4 y vuelve a ejecutar.
            # Acumula los índices con error de carga (imagen ilegible/corrompida)
            indices_error.extend(idxs_err)

            if tensores.numel() == 0:
                # Lote completamente vacío: todos los ejemplos del lote fallaron
                continue

            # Mueve el lote al dispositivo del modelo
            tensores = tensores.to(dev)

            # Forward pass: extrae el embedding como vector de activaciones
            salida = model(tensores)             # tensor (B, D)

            # Convierte a CPU y numpy para el post-procesamiento
            emb_np = salida.cpu().float().numpy()   # garantiza float32

            todos_emb.append(emb_np)
            indices_ok.extend(idxs_ok)

    if not todos_emb:
        log.error("-" * 62)
        log.error("ERROR: Ningun embedding pudo extraerse.")
        log.error(f"Total de imagenes intentadas: {len(df_ok):,}")
        log.error(f"Errores de carga: {len(indices_error):,}")
        log.error("Causas posibles:")
        log.error("  · Las rutas en registro_descargas.csv no existen en este equipo.")
        log.error("    Las imagenes se descargaron en otro equipo y no se copiaron aqui.")
        log.error("    Verifica que la carpeta gsv/fotos/ exista y tenga archivos .jpg")
        log.error("  · Todas las imagenes estan corrompidas (JPEG invalido).")
        log.error("    Solucion: borra gsv/registro_descargas.csv y re-ejecuta script 02.")
        log.error("  · Error de memoria (OOM): reduce CONFIG['batch_size'] a 4 o 8.")
        log.error("-" * 62)
        sys.exit(1)

    embeddings = np.vstack(todos_emb)       # (N_ok, D) — matriz final de embeddings
    log.info(f"Embeddings extraídos: {len(indices_ok):,} exitosos, "
             f"{len(indices_error):,} errores.")
    return embeddings, indices_ok, indices_error


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5: CONSTRUCCIÓN DEL OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def _generar_image_id(nombre_archivo: str) -> str:
    """
    Genera un identificador único para la imagen basado en el nombre del archivo.

    Se usa SHA-256 del nombre del archivo (no del pano_id de Google).
    Los primeros 16 caracteres del hash hexadecimal ofrecen ~64 bits de entropía,
    suficiente para evitar colisiones en colecciones de millones de imágenes.

    El ID es determinístico: el mismo archivo siempre produce el mismo image_id.
    """
    return hashlib.sha256(nombre_archivo.encode()).hexdigest()[:16]


def construir_output(
    df_ok:         pd.DataFrame,
    embeddings:    np.ndarray,
    indices_ok:    list,
    nombre_modelo: str,
    embedding_dim: int,
) -> pd.DataFrame:
    """
    Construye el DataFrame de output con una fila por imagen exitosa.

    Columnas del output:
        image_id       : identificador único derivado del nombre del archivo
        consecutivo    : ID del hogar (de script 00)
        ola            : año de la ola (2010, 2013, 2016)
        pano_id        : ID del panorama (de la Metadata API — sirve para trazabilidad)
        heading        : ángulo de la imagen (grados, de script 01/02)
        nombre_archivo : nombre del archivo JPEG en disco
        llave          : ID de panel 2013/2016 (si disponible)
        llave_n16      : ID de panel 2016 (si disponible)
        modelo         : nombre del modelo usado para extraer el embedding
        embedding_dim  : dimensión del vector
        embedding      : lista de float32 con el vector completo
    """
    # Subset de df_ok con las filas que se procesaron exitosamente
    df_proc = df_ok.iloc[indices_ok].copy().reset_index(drop=True)

    # Genera image_id determinístico para cada imagen
    df_proc["image_id"] = df_proc["nombre_archivo"].apply(_generar_image_id)

    # Añade metadatos del modelo
    df_proc["modelo"]        = nombre_modelo
    df_proc["embedding_dim"] = embedding_dim

    # Agrega el embedding como lista de float32 (compatible con Parquet)
    df_proc["embedding"] = [row.astype(np.float32).tolist() for row in embeddings]

    # Columnas de salida en orden canónico; las opcionales se incluyen si existen
    cols_base = [
        "image_id", "consecutivo", "ola", "pano_id", "heading",
        "nombre_archivo", "modelo", "embedding_dim", "embedding",
    ]
    cols_extra = [c for c in ["llave", "llave_n16"] if c in df_proc.columns]

    # Inserta los identificadores adicionales después de 'ola'
    idx_ola  = cols_base.index("ola") + 1
    cols_out = cols_base[:idx_ola] + cols_extra + cols_base[idx_ola:]

    return df_proc[[c for c in cols_out if c in df_proc.columns]]


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: EXPORTACIÓN A PARQUET
# ──────────────────────────────────────────────────────────────────────────────

def exportar_parquet(df_emb: pd.DataFrame, cfg: dict) -> Path:
    """
    Exporta el DataFrame de embeddings a formato Parquet.

    Se usa Parquet porque:
    · Soporta columnas de tipo lista (listas de float32) de forma nativa.
    · Es eficiente para matrices de alta dimensión (compresión por columna).
    · Permite lectura parcial por columnas (útil para trabajar solo con IDs sin
      cargar los embeddings).

    El archivo se nombra según el modelo: embeddings_{modelo}.parquet.

    Retorna:
        Ruta absoluta del archivo Parquet generado.
    """
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    nombre_modelo = cfg["modelo"]
    ruta_parquet  = out_dir / f"embeddings_{nombre_modelo}.parquet"

    # Usa el motor pyarrow que maneja columnas de lista de forma nativa
    df_emb.to_parquet(ruta_parquet, engine="pyarrow", index=False)
    log.info(f"Embeddings exportados: {ruta_parquet}")
    return ruta_parquet


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 7: REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(
    df_emb:       pd.DataFrame,
    n_total:      int,
    n_errores:    int,
    t_inicio:     float,
    cfg:          dict,
) -> str:
    """
    Genera un reporte de texto plano con estadísticas del proceso de extracción.

    Secciones:
    1. Parámetros de ejecución
    2. Resumen de imágenes procesadas
    3. Tiempos y rendimiento
    4. Cobertura por ola (si disponible)
    5. Notas metodológicas
    """
    t_total  = time.time() - t_inicio                      # segundos totales
    n_proc   = len(df_emb)                                 # imágenes con embedding exitoso
    t_prom   = t_total / n_proc if n_proc > 0 else 0       # segundos por imagen

    lineas = [
        "=" * 72,
        "REPORTE DE EXTRACCIÓN DE EMBEDDINGS GSV",
        "=" * 72,
        "",
        "── 1. Parámetros de ejecución ─────────────────────────────────────────",
        f"  Modelo             : {cfg['modelo']}",
        f"  Dimensión embedding: {df_emb['embedding_dim'].iloc[0] if n_proc > 0 else 'N/A'}",
        f"  Batch size         : {cfg['batch_size']}",
        f"  Device             : {cfg['device']}",
        "",
        "── 2. Resumen de imágenes procesadas ──────────────────────────────────",
        f"  Total programadas  : {n_total:>8,}",
        f"  Exitosas           : {n_proc:>8,}",
        f"  Con error de carga : {n_errores:>8,}",
        f"  Tasa de éxito      : {n_proc / n_total * 100:.1f}%" if n_total > 0 else "",
        "",
        "── 3. Tiempos y rendimiento ────────────────────────────────────────────",
        f"  Tiempo total       : {t_total / 60:.1f} minutos",
        f"  Tiempo por imagen  : {t_prom:.3f} segundos",
        "",
    ]

    # Cobertura por ola (si la columna ola está disponible)
    if "ola" in df_emb.columns and n_proc > 0:
        lineas += [
            "── 4. Embeddings por ola ───────────────────────────────────────────────",
        ]
        for ola, grp in df_emb.groupby("ola"):
            n_hogares = grp["consecutivo"].nunique()
            lineas.append(
                f"  Ola {ola}: {len(grp):>6,} imágenes  |  {n_hogares:>5,} hogares únicos"
            )
        lineas.append("")

    lineas += [
        "── 5. Notas metodológicas ──────────────────────────────────────────────",
        "  · Los embeddings fueron extraídos de la penúltima capa del modelo,",
        "    sin entrenamiento ni fine-tuning adicional.",
        "  · Cada imagen genera un vector de activaciones que representa el",
        "    contenido visual del entorno fotografiado.",
        "  · Para VGG19: embedding de 4 096 dimensiones (FC penúltima capa).",
        "  · Para ResNet50 e ImageNet: 2 048 dimensiones (salida del avg-pool).",
        "  · Para Places365: 2 048 dimensiones — modelo entrenado en escenas",
        "    urbanas y naturales, más apropiado para análisis de entorno.",
        "  · La columna 'embedding' en el Parquet contiene listas de float32.",
        "    Para convertir a numpy: np.array(df['embedding'].iloc[0]).",
        "",
        "=" * 72,
    ]

    return "\n".join(lineas)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_inicio = time.time()
    log.info("=" * 62)
    log.info("  03_extraer_embeddings.py")
    log.info(f"  Modelo configurado: {CONFIG['modelo']}")
    log.info("  Si ves ModuleNotFoundError, activa el entorno virtual:")
    log.info("    Mac/Linux : source .venv/bin/activate")
    log.info("    Windows   : .venv\\Scripts\\activate")
    log.info("    Luego     : pip install -r requirements.txt")
    log.info("  Si ves 'No module named torch', instala PyTorch segun tu hardware:")
    log.info("    CPU/Mac   : pip install torch torchvision")
    log.info("    GPU CUDA  : pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    log.info("  Para reducir consumo de memoria si hay error de OOM (memoria agotada):")
    log.info(f"    reduce CONFIG['batch_size'] de {CONFIG['batch_size']} a 8 o menos.")
    log.info("=" * 62)

    # ── 1. Carga las imágenes disponibles ─────────────────────────────────────
    df_ok = cargar_imagenes(CONFIG)
    n_total = len(df_ok)

    # ── 2. Carga el modelo preentrenado ───────────────────────────────────────
    model, transform, embedding_dim, device = cargar_modelo(CONFIG)

    # ── 3. Extrae embeddings en lotes ─────────────────────────────────────────
    embeddings, indices_ok, indices_error = extraer_embeddings(
        model, transform, df_ok, CONFIG
    )

    # ── 4. Construye el DataFrame de output ───────────────────────────────────
    df_emb = construir_output(
        df_ok, embeddings, indices_ok,
        nombre_modelo=CONFIG["modelo"],
        embedding_dim=embedding_dim,
    )

    # ── 5. Exporta a Parquet ──────────────────────────────────────────────────
    ruta_parquet = exportar_parquet(df_emb, CONFIG)

    # ── 6. Genera y guarda el reporte ─────────────────────────────────────────
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    reporte = generar_reporte(
        df_emb,
        n_total=n_total,
        n_errores=len(indices_error),
        t_inicio=t_inicio,
        cfg=CONFIG,
    )

    nombre_modelo = CONFIG["modelo"]
    ruta_reporte  = out_dir / f"reporte_embeddings_{nombre_modelo}.txt"
    ruta_reporte.write_text(reporte, encoding="utf-8")

    log.info(f"Reporte guardado: {ruta_reporte}")
    log.info(f"Proceso completado en {(time.time() - t_inicio) / 60:.1f} minutos.")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
