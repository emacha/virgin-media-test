"""
Sane project configuration.

https://leontrolski.github.io/sane-config.html
"""
import dataclasses
import functools
from pathlib import Path


@dataclasses.dataclass
class Config:
    """Project configuration."""

    BUCKET_NAME: str
    BLOB_NAME: str
    SEED: int
    MODEL_PATH: Path
    OUTPUT_PATH: Path
    TRAIN_PATH: Path


@functools.cache
def get_config() -> Config:
    """Return constants."""
    return Config(
        BUCKET_NAME="cloud-samples-data",
        BLOB_NAME="ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
        SEED=1337,
        MODEL_PATH=Path("artifacts/model"),
        OUTPUT_PATH=Path("output/results.csv"),
        TRAIN_PATH=Path("output/train.csv"),
    )
