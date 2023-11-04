"""ML model code."""
from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb


def frequency_encoding_mapping(frame: pl.DataFrame, column: str) -> dict[str, int]:
    """Return a mapping of values to their frequency rank."""
    frequencies = (
        frame[column]
        .value_counts()
        .sort(by="counts", descending=True)
        .with_columns(pl.col("counts").rank(method="ordinal").alias("rank"))
    )
    mapping = dict(zip(frequencies[column], frequencies["rank"]))
    return mapping


def any_nan(data: pl.DataFrame) -> bool:
    """Return True if any column contains null values."""
    return data.null_count().transpose()["column_0"].sum() > 0


@dataclasses.dataclass
class Model:
    """Custom wrapper around an XGBoost model."""

    target: str
    features: list[str]
    categorical_mapping: dict[str, dict[str, int]]
    seed: int = 1

    booster: xgb.Booster = dataclasses.field(init=False, default=None)
    is_fit: bool = dataclasses.field(init=False, default=False)

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        """Predict and return class labels."""
        probabilities = self.predict_proba(frame)
        class_int = probabilities.round(0).astype(int)
        return self.class_labels(class_int)

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        """Predict and return class probabilities."""
        if not self.is_fit:
            raise RuntimeError("Model must be fit before predictions can be made")
        frame = self.preprocess(frame)
        return self.booster.predict(xgb.DMatrix(frame[self.features]))

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        """Fit the model."""
        train, validation = self.preprocess(train), self.preprocess(validation)
        dtrain = xgb.DMatrix(train[self.features], label=train[self.target])
        dvalid = xgb.DMatrix(validation[self.features], label=validation[self.target])
        training = xgb.train(
            {
                "objective": "binary:logistic",
                "seed": self.seed,
            },
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            verbose_eval=False,
            early_stopping_rounds=6,
        )
        # Get the model at best iteration
        # Horribly inefficient, but I can't be bothered to write a callback
        self.booster = xgb.train(
            {
                "objective": "binary:logistic",
                "seed": self.seed,
            },
            dtrain,
            num_boost_round=training.best_iteration,
        )
        self.is_fit = True

    def preprocess(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Transform categorical features to numeric."""
        data = frame.with_columns(
            [
                pl.col(column).map_dict(mapping)
                for column, mapping in self.categorical_mapping.items()
            ]
        )
        if any_nan(data):
            raise RuntimeWarning("Data contains null values")

        return data

    def class_labels(self, predictions: np.ndarray) -> np.ndarray:
        """Return the class labels for the given predictions."""
        inverse_mapping = {
            v: k for k, v in self.categorical_mapping[self.target].items()
        }
        func = np.vectorize(inverse_mapping.get)
        return func(predictions)

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> Model:
        """Load the model from disk."""
        path = Path(path)
        with path.open("rb") as file:
            model = pickle.load(file)
        return model

    def __eq__(self, other: Model) -> bool:
        """Check if two Model instances are equivalent."""
        if not isinstance(other, Model) or not self.is_fit or not other.is_fit:
            return False
        return self.booster.get_dump() == other.booster.get_dump()
