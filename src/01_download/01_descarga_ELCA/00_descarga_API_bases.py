# %%
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlencode
import zipfile

import requests


BASE_URL = "https://datahub.uniandes.edu.co"
DANE_BASE_URL = "https://microdatos.dane.gov.co"
TIMEOUT = 60
MAX_WORKERS = 6

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "interim" / "raw"


DATASETS = {
    "elca_2010": "doi:10.57924/DPF0M5",
    "elca_2013": "doi:10.57924/DE9LP7",
    "elca_2016": "doi:10.57924/BLUILW",
}

# Olas adicionales publicadas en el catálogo de microdatos del DANE (plataforma
# NADA), en vez del Dataverse de Uniandes. Cada una corresponde a un único
# archivo zip identificado por su catalog_id y el file_id de descarga.
DANE_DATASETS = {
    "elca_2019": {
        "catalog_id": 814,
        "file_id": 23283,
        "filename": "ELCO_2019.zip",
    },
    "elca_2022": {
        "catalog_id": 902,
        "file_id": 24583,
        "filename": "BDATOS-ELCO-2022.zip",
    },
}

@dataclass
class DownloadJob:
    label: str
    url: str
    destination: Path


def safe_filename(name: str) -> str:
    """
    Limpia nombres de archivos para evitar problemas con caracteres especiales.
    """
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^\w\-.() ]", "_", name, flags=re.UNICODE)
    return name


def get_dataset_metadata(base_url: str, dataset_doi: str) -> dict:
    """
    Consulta los metadatos de un dataset en Dataverse usando su DOI.
    """
    url = f"{base_url}/api/datasets/:persistentId/?{urlencode({'persistentId': dataset_doi})}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "OK":
        raise RuntimeError(f"No se pudo obtener el dataset {dataset_doi}: {payload}")

    return payload["data"]


def extract_files_from_metadata(metadata: dict) -> list[dict]:
    """
    Extrae la lista de archivos disponibles dentro de los metadatos del dataset.
    """
    latest_version = metadata.get("latestVersion", {})
    files = latest_version.get("files", [])

    extracted = []
    for f in files:
        data_file = f.get("dataFile", {})
        extracted.append(
            {
                "id": data_file.get("id"),
                "filename": data_file.get("filename"),
                "description": f.get("description"),
                "contentType": data_file.get("contentType"),
                "filesize": data_file.get("filesize"),
                "md5": data_file.get("md5"),
                "persistentId": data_file.get("persistentId"),
            }
        )
    return extracted


def get_dane_catalog_metadata(base_url: str, catalog_id: int) -> dict:
    """
    Consulta los metadatos de un catálogo del DANE (plataforma NADA) en formato JSON.
    """
    url = f"{base_url}/index.php/metadata/export/{catalog_id}/json"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def download_url_to_file(url: str, destination: Path) -> None:
    """
    Descarga el contenido de una URL a un archivo local, en streaming.
    """
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with open(destination, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)


def run_download_jobs(jobs: list[DownloadJob], max_workers: int = MAX_WORKERS) -> None:
    """
    Ejecuta una lista de descargas en paralelo usando un pool de hilos.
    """
    if not jobs:
        return

    print(f"\nDescargando {len(jobs)} archivo(s) en paralelo (max_workers={max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(download_url_to_file, job.url, job.destination): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                future.result()
                print(f"Completado: {job.label}")
            except Exception as e:
                print(f"Error descargando {job.label}: {e}")


def extract_zip(zip_path: Path) -> Path | None:
    """
    Descomprime un .zip en una carpeta hermana con su mismo nombre (sin extensión).
    Si esa carpeta ya existe y tiene contenido, no hace nada.
    """
    extract_dir = zip_path.with_suffix("")
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return None

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # Si el zip envuelve todo en una única carpeta contenedora (p. ej.
    # "ELCO_2019/ELCO_2019/..."), la aplanamos para evitar la doble anidación.
    entries = list(extract_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        for item in wrapper.iterdir():
            item.rename(extract_dir / item.name)
        wrapper.rmdir()

    return extract_dir


def extract_all_zips(root_dir: Path) -> None:
    """
    Busca todos los .zip descargados dentro de root_dir y los descomprime.
    """
    zip_files = sorted(root_dir.rglob("*.zip"))
    if not zip_files:
        return

    print(f"\nDescomprimiendo {len(zip_files)} archivo(s) .zip...")
    for zip_path in zip_files:
        label = zip_path.relative_to(root_dir)
        try:
            extracted = extract_zip(zip_path)
            if extracted is None:
                print(f"Ya estaba descomprimido, se omite: {label}")
            else:
                print(f"Descomprimido: {label} -> {extracted.name}/")
        except Exception as e:
            print(f"Error descomprimiendo {label}: {e}")


def save_json(data: dict | list, path: Path) -> None:
    """
    Guarda un objeto JSON con codificación UTF-8 e indentación legible.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_dataset_jobs(dataset_name: str, dataset_doi: str) -> list[DownloadJob]:
    """
    Consulta los metadatos de un dataset de Dataverse y arma la lista de
    descargas pendientes (omitiendo los archivos que ya existen).
    """
    output_dir = RAW_DATA_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Procesando {dataset_name} ===")
    print(f"DOI: {dataset_doi}")
    print(f"Carpeta de salida: {output_dir}")

    metadata = get_dataset_metadata(BASE_URL, dataset_doi)
    save_json(metadata, output_dir / "dataset_metadata.json")

    files = extract_files_from_metadata(metadata)
    save_json(files, output_dir / "files_manifest.json")

    if not files:
        print(f"No se encontraron archivos en {dataset_name}.")
        return []

    print(f"Se encontraron {len(files)} archivos en {dataset_name}.")

    jobs = []
    skipped = 0
    for file_info in files:
        file_id = file_info["id"]
        filename = safe_filename(file_info["filename"] or f"file_{file_id}")
        destination = output_dir / filename

        if destination.exists():
            skipped += 1
            continue

        jobs.append(
            DownloadJob(
                label=f"{dataset_name}/{filename}",
                url=f"{BASE_URL}/api/access/datafile/{file_id}",
                destination=destination,
            )
        )

    if skipped:
        print(f"{skipped} archivo(s) ya existen en {dataset_name}, se omiten.")

    download_info = {
        "dataset_name": dataset_name,
        "dataset_doi": dataset_doi,
        "download_timestamp": datetime.now().isoformat(),
    }
    save_json(download_info, output_dir / "download_info.json")

    return jobs


def collect_dane_dataset_jobs(dataset_name: str, dataset_info: dict) -> list[DownloadJob]:
    """
    Consulta los metadatos de una ola del DANE (plataforma NADA) y arma la
    descarga pendiente (si el archivo ya existe, no se agrega ninguna).
    """
    catalog_id = dataset_info["catalog_id"]
    file_id = dataset_info["file_id"]
    filename = safe_filename(dataset_info["filename"])

    output_dir = RAW_DATA_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Procesando {dataset_name} ===")
    print(f"Catálogo DANE: {catalog_id}")
    print(f"Carpeta de salida: {output_dir}")

    metadata = get_dane_catalog_metadata(DANE_BASE_URL, catalog_id)
    save_json(metadata, output_dir / "dataset_metadata.json")

    download_info = {
        "dataset_name": dataset_name,
        "catalog_id": catalog_id,
        "file_id": file_id,
        "download_timestamp": datetime.now().isoformat(),
    }
    save_json(download_info, output_dir / "download_info.json")

    destination = output_dir / filename
    if destination.exists():
        print(f"Ya existe, se omite: {filename}")
        return []

    return [
        DownloadJob(
            label=f"{dataset_name}/{filename}",
            url=f"{DANE_BASE_URL}/index.php/catalog/{catalog_id}/download/{file_id}",
            destination=destination,
        )
    ]


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    jobs: list[DownloadJob] = []

    for dataset_name, dataset_doi in DATASETS.items():
        try:
            jobs.extend(collect_dataset_jobs(dataset_name, dataset_doi))
        except Exception as e:
            print(f"Error procesando {dataset_name} ({dataset_doi}): {e}")

    for dataset_name, dataset_info in DANE_DATASETS.items():
        try:
            jobs.extend(collect_dane_dataset_jobs(dataset_name, dataset_info))
        except Exception as e:
            print(f"Error procesando {dataset_name} ({dataset_info}): {e}")

    run_download_jobs(jobs)
    extract_all_zips(RAW_DATA_DIR)

    print("\nProceso terminado.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error general: {exc}")
        sys.exit(1)


