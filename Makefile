.PHONY: install lint format test check

install:
	uv sync --no-install-project

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

check: lint test

.PHONY: data

data:
	$(PYTHON) -m src.data.make_dataset


.PHONY: load_dataset

load_dataset:
	$(PYTHON) -m src.data.load_dataset

.PHONY: prepare_data

prepare_data:
	$(PYTHON) -m src.data.prepare_data

PYTHON ?= uv run --no-sync python
MODEL ?= all
EPOCHS ?= 5
N_TRIALS ?= 30
DEVICE ?=
K ?= 10
SPLIT ?= val
PROCESSED_DATA_DIR ?= data/processed
RAW_DATA_DIR ?= data/raw
MODELS_DIR ?= artifacts
REPORTS_DIR ?= reports
MODEL_NAMES := pop content item svd hybrid adaptive two_tower balanced
TRAIN_TARGETS := $(addprefix train-,$(MODEL_NAMES))
REPORT_TARGETS := $(addprefix report-,$(MODEL_NAMES))

.PHONY: train report train-report $(TRAIN_TARGETS) $(REPORT_TARGETS)

train:
	$(PYTHON) -m src.models.train --model "$(MODEL)" \
		--processed-data-dir "$(PROCESSED_DATA_DIR)" --raw-data-dir "$(RAW_DATA_DIR)" \
		--models-dir "$(MODELS_DIR)" --epochs "$(EPOCHS)" --n-trials "$(N_TRIALS)" \
		$(if $(DEVICE),--device "$(DEVICE)")

$(TRAIN_TARGETS): train-%:
	$(MAKE) train MODEL=$*

report:
	$(PYTHON) -m src.metrics.report --model "$(MODEL)" --split "$(SPLIT)" --k "$(K)" \
		--processed-data-dir "$(PROCESSED_DATA_DIR)" --models-dir "$(MODELS_DIR)" \
		--reports-dir "$(REPORTS_DIR)"

$(REPORT_TARGETS): report-%:
	$(MAKE) report MODEL=$*

train-report: train
	$(MAKE) report


# Сквозной пайплайн из модулей src: данные → обучение → выдача → отчёт.
PIPELINE_OUTPUT_DIR ?= artifacts/notebook_pipeline
MAX_USERS ?= none
TUNING_USERS ?= 200
EVAL_USERS ?= none
SVD_EPOCHS ?= 50
BATCH_SIZE ?= 1024
CANDIDATES_K ?= 100
RANDOM_STATE ?= 42
THREADS ?= 2
USE_NEURAL ?= 1
DOWNLOAD ?= 0
RECOMMEND_USERS ?=

.PHONY: pipeline pipeline-smoke pipeline-classical

pipeline:
	$(PYTHON) -m src.pipeline \
		--data-dir "$(RAW_DATA_DIR)" --output-dir "$(PIPELINE_OUTPUT_DIR)" \
		--max-users "$(MAX_USERS)" --tuning-users "$(TUNING_USERS)" --eval-users "$(EVAL_USERS)" \
		--epochs "$(EPOCHS)" --batch-size "$(BATCH_SIZE)" --svd-epochs "$(SVD_EPOCHS)" \
		--n-trials "$(N_TRIALS)" --k "$(K)" --candidates-k "$(CANDIDATES_K)" \
		--random-state "$(RANDOM_STATE)" --threads "$(THREADS)" \
		$(if $(DEVICE),--device "$(DEVICE)") \
		$(if $(filter 1 true yes,$(DOWNLOAD)),--download) \
		$(if $(filter 0 false no,$(USE_NEURAL)),--no-neural) \
		$(if $(RECOMMEND_USERS),--recommend-user-ids "$(RECOMMEND_USERS)")

pipeline-smoke:
	$(MAKE) pipeline MAX_USERS=200 EPOCHS=2 SVD_EPOCHS=5 N_TRIALS=3 \
		TUNING_USERS=100 EVAL_USERS=none PIPELINE_OUTPUT_DIR="$(PIPELINE_OUTPUT_DIR)_smoke"

pipeline-classical:
	$(MAKE) pipeline USE_NEURAL=0 PIPELINE_OUTPUT_DIR="$(PIPELINE_OUTPUT_DIR)_classical"
