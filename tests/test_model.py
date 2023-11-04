"""Tests for the model module."""
import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError

import src
from src.model import Model

config = src.config.get_config()


@pytest.fixture(scope="module")
def model():
    """Get the trained model."""
    return Model.load(config.MODEL_PATH)


@pytest.fixture(scope="module")
def train():
    """Get the training data."""
    return src.gcp.get_data()


def test_predicts(model, train):
    """Test that the model predicts correctly."""
    prediction = model.predict(train)
    assert len(prediction) == len(train)


def test_extra_columns(model, train):
    """Test that the model correctly ignores extra columns."""
    predictions = model.predict(train)
    train = train.with_columns(extra=pl.lit(1))
    extra_preds = model.predict(train)
    assert (predictions == extra_preds).all()


def test_missing_columns(model, train):
    """Test that the model doesn't silently ignore missing columns."""
    train = train.drop("Breed1")
    with pytest.raises(ColumnNotFoundError):
        model.predict(train)


def test_unknown_values(model, train):
    """Test that the model doesn't silently ignore unknown values."""
    train = train.with_columns(Breed1=pl.lit("I'm not a breed"))
    with pytest.raises(RuntimeWarning):
        model.predict(train)


def test_saving_loading(model, tmp_path):
    """Test that we don't lose information when saving/loading."""
    model.save(tmp_path / "model.pkl")
    loaded = Model.load(tmp_path / "model.pkl")
    assert model == loaded
