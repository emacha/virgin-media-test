# Virgin Media Tech Test

Tech test for the Virgin Media MLOps role

## Pre requisites

Make sure you have the following installed:

- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installation)
- [Make](https://en.wikipedia.org/wiki/Make_%28software%29)
- A Google Cloud account with access to the [Cloud Storage](https://cloud.google.com/storage) service

## Alternatives

### Poetry

If you can't or don't want to install poetry, you can use the `requirements.txt` file to install the dependencies.

### Make

If you don't have `make`, just open the `Makefile` and run the commands manually.

## Setup

### Virtual environment

```shell
poetry install
poetry shell
```

### GCloud

You need a way to authenticate with GCP. I've created a service account and downloaded the JSON key. If using the same approach, just create a `.env` file with the following content:

```ini
GOOGLE_APPLICATION_CREDENTIALS=<path to the JSON key>
```

## Running the code

- `make train` to train the model (Task 1)
- `make predict` to predict the results (Task 2)
- `make test` to run the tests
