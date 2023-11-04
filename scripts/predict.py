"""Task 2: Predict."""
import polars as pl
from loguru import logger

import src
from src.model import Model

logger.info("Running task 2: Predict")
config = src.config.get_config()

logger.info("Downloading data from GCP")
data = src.gcp.get_data()

logger.info("Calculating results")
model = Model.load(config.MODEL_PATH)
prediction = model.predict(data)
# Why not AdoptedPrediction? :(
data = data.with_columns(pl.Series("Adopted_prediction", prediction))

logger.info("Saving results")
config.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
data.write_csv(config.OUTPUT_PATH)

logger.info("Done!")
