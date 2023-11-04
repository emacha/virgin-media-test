"""Task 1: Training."""
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

import src
from src.model import Model

logger.info("Running task 1: Training")
config = src.config.get_config()

logger.info("Downloading data from GCP")
# We have 4 examples where breed is "0", clearly a data entry error
# If this was a one-off I'd just drop it. But it's a "common" error.
# If we drop this here, we can't keep the pipeline erroring on unknown values
# Or it will fail in prod. It's unclean, but we'll keep it for now.
data = src.gcp.get_data()

categorical_mapping = {
    "Type": {"Cat": 0, "Dog": 1},
    # Frequency encoding doesn't leak information about the target
    # So use the full dataset
    "Breed1": src.model.frequency_encoding_mapping(data, "Breed1"),
    "Gender": src.model.frequency_encoding_mapping(data, "Gender"),
    "Color1": src.model.frequency_encoding_mapping(data, "Color1"),
    "Color2": src.model.frequency_encoding_mapping(data, "Color2"),
    "MaturitySize": {"Small": 0, "Medium": 1, "Large": 2},
    "FurLength": {"Short": 0, "Medium": 1, "Long": 2},
    "Vaccinated": {"No": 0, "Not Sure": 1, "Yes": 2},
    "Sterilized": {"No": 0, "Not Sure": 1, "Yes": 2},
    "Health": {"Healthy": 0, "Minor Injury": 1, "Serious Injury": 2},
    "Adopted": {"No": 0, "Yes": 1},
}
features = [column for column in data.columns if column != "Adopted"]

logger.info("Splitting data")
train, all_test = train_test_split(
    data, test_size=0.4, random_state=config.SEED, stratify=data["Adopted"]
)
validation, test = train_test_split(
    all_test, test_size=0.5, random_state=config.SEED, stratify=all_test["Adopted"]
)

logger.info("Fitting model")
model = Model(
    target="Adopted",
    features=features,
    categorical_mapping=categorical_mapping,
    seed=config.SEED,
)
model.fit(train, validation)

test_predictions = model.predict(test)
logger.info(f"Test Accuracy: {accuracy_score(test['Adopted'], test_predictions)}")
logger.info(
    f"Test Recall: {recall_score(test['Adopted'], test_predictions, pos_label='Yes')}"
)
logger.info(f"Test F1: {f1_score(test['Adopted'], test_predictions, pos_label='Yes')}")

logger.info("Saving model")
model.save(config.MODEL_PATH)

logger.success("Done!")
