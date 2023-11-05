"""Interact with Google Cloud Platform (GCP) services."""
import functools
from io import BytesIO

import dotenv
import polars as pl
from google.cloud import storage
from loguru import logger

import src


@functools.cache
def get_client():
    """Return a cached instance of the GCS client."""
    dotenv.load_dotenv()
    return storage.Client()


def download_data(bucket: str, blob: str) -> pl.DataFrame:
    """Download csv from GCS and return as a DataFrame."""
    client = get_client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(blob)
    temp_bytes = BytesIO()
    blob.download_to_file(temp_bytes)
    return pl.read_csv(temp_bytes)


@functools.cache
def get_data() -> pl.DataFrame:
    """Get the training data, download and cache if necessary."""
    config = src.config.get_config()
    if not config.TRAIN_PATH.exists():
        logger.debug("Downloading data from GCP")
        data = download_data(config.BUCKET_NAME, config.BLOB_NAME)
        config.TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.write_csv(config.TRAIN_PATH)

    return pl.read_csv(config.TRAIN_PATH)
