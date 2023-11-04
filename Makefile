lint:
	ruff format src scripts tests -q
	ruff src scripts tests --fix

train:
	python scripts/train.py

predict:
	python scripts/predict.py

test:
	JUPYTER_PLATFORM_DIRS=1 pytest

get-requirements:
	poetry export -f requirements.txt --output requirements.txt
