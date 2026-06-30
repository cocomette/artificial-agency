# FACE-OF-AGI Kaggle submission workflow.

PYTHON ?= python3.12
ROOT := ..
-include .env
export FACE_OF_AGI_KAGGLE_OWNER
export FACE_OF_AGI_KAGGLE_TOKEN_FILE

KAGGLE_TOKEN := $(if $(filter /%,$(FACE_OF_AGI_KAGGLE_TOKEN_FILE)),$(FACE_OF_AGI_KAGGLE_TOKEN_FILE),$(ROOT)/$(FACE_OF_AGI_KAGGLE_TOKEN_FILE))
KAGGLE_CLI ?= uv run --with kaggle kaggle
KAGGLE := KAGGLE_API_TOKEN=$$(cat "$(KAGGLE_TOKEN)") $(KAGGLE_CLI)
KAGGLE_ACCELERATOR ?= NvidiaRtxPro6000
KAGGLE_ACCELERATOR_ARGS := $(if $(strip $(KAGGLE_ACCELERATOR)),--accelerator $(KAGGLE_ACCELERATOR),)
PUBLIC_GAMES_PYTHON ?= uv run --no-dev python
KAGGLE_SUBMISSION_CONFIG ?= src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_qwen36_35b_fp8_parallel.yaml
KAGGLE_SUBMISSION_MODEL_DATASET_SLUG ?= face-of-agi-qwen36-35b-fp8-weights
KAGGLE_SUBMISSION_KERNEL_ID ?= $(FACE_OF_AGI_KAGGLE_OWNER)/face-of-agi-arc-agi-3-rtx6000
KAGGLE_SUBMISSION_KERNEL_TITLE ?= FACE-OF-AGI ARC-AGI-3 RTX6000
KAGGLE_DEBUG_CONFIG ?= src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_qwen36_35b_fp8_debug.yaml
KAGGLE_DEBUG_MODEL_DATASET_SLUG ?= face-of-agi-qwen36-35b-fp8-weights
DEBUG_KERNEL_TITLE_SUFFIX ?=
DEBUG_KERNEL_TITLE_SUFFIX_ARG := $(if $(strip $(DEBUG_KERNEL_TITLE_SUFFIX)),--kernel-title-suffix "$(DEBUG_KERNEL_TITLE_SUFFIX)",)

WHEELHOUSE_DIR := build/wheelhouse
WHEEL_PLATFORMS ?= manylinux_2_31_x86_64 manylinux_2_28_x86_64 manylinux_2_24_x86_64 manylinux2014_x86_64
WHEEL_PLATFORM_ARGS := $(foreach platform,$(WHEEL_PLATFORMS),--platform $(platform))
WHEEL_PYTHON_VERSION ?= 312
WHEEL_ABI ?= cp312
PUBLIC_GAMES_DIR := build/public-games
DEBUG_OUTPUT_DIR := $(ROOT)/runs/kaggle-debug
UPLOAD_MODE ?= create
VERSION_NOTES ?= FACE-OF-AGI Kaggle submission artifact update
DEBUG_KERNEL ?=

.PHONY: sync-metadata wheelhouse wheelhouse-upload public-games public-games-upload notebook submit status debug-metadata debug-notebook debug-submit debug-status debug-pull clean _check-kaggle

sync-metadata:
	$(PYTHON) scripts/sync_kaggle_metadata.py

_check-kaggle:
	@if [ -z "$(strip $(FACE_OF_AGI_KAGGLE_OWNER))" ]; then \
		echo "ERROR: FACE_OF_AGI_KAGGLE_OWNER is missing from the environment or kaggle/.env."; \
		exit 1; \
	fi
	@if [ -z "$(strip $(FACE_OF_AGI_KAGGLE_TOKEN_FILE))" ]; then \
		echo "ERROR: FACE_OF_AGI_KAGGLE_TOKEN_FILE is missing from the environment or kaggle/.env."; \
		exit 1; \
	fi
	@if [ ! -s "$(KAGGLE_TOKEN)" ]; then \
		echo "ERROR: $(KAGGLE_TOKEN) is missing or empty."; \
		echo "Generate a Kaggle API token and save the token value there."; \
		exit 1; \
	fi

wheelhouse: sync-metadata
	$(PYTHON) scripts/build_wheelhouse.py \
		$(WHEEL_PLATFORM_ARGS) \
		--python-version $(WHEEL_PYTHON_VERSION) \
		--abi $(WHEEL_ABI) \
		--requirements requirements-kaggle.txt \
		--metadata upload/wheelhouse/dataset-metadata.json \
		--output $(WHEELHOUSE_DIR)

wheelhouse-upload: _check-kaggle wheelhouse
	@if [ "$(UPLOAD_MODE)" = "create" ]; then \
		$(KAGGLE) datasets create -p $(WHEELHOUSE_DIR); \
	elif [ "$(UPLOAD_MODE)" = "version" ]; then \
		$(KAGGLE) datasets version -p $(WHEELHOUSE_DIR) -m "$(VERSION_NOTES)" --delete-old-versions; \
	else \
		echo "ERROR: UPLOAD_MODE must be create or version"; \
		exit 1; \
	fi

public-games: sync-metadata
	$(PUBLIC_GAMES_PYTHON) scripts/prepare_public_games.py

public-games-upload: _check-kaggle public-games
	@if [ "$(UPLOAD_MODE)" = "create" ]; then \
		$(KAGGLE) datasets create -p $(PUBLIC_GAMES_DIR) --dir-mode zip; \
	elif [ "$(UPLOAD_MODE)" = "version" ]; then \
		$(KAGGLE) datasets version -p $(PUBLIC_GAMES_DIR) -m "$(VERSION_NOTES)" --delete-old-versions --dir-mode zip; \
	else \
		echo "ERROR: UPLOAD_MODE must be create or version"; \
		exit 1; \
	fi

notebook: sync-metadata
	FACE_OF_AGI_KAGGLE_SUBMISSION_CONFIG="$(KAGGLE_SUBMISSION_CONFIG)" \
	FACE_OF_AGI_KAGGLE_MODEL_DATASET_SLUG="$(KAGGLE_SUBMISSION_MODEL_DATASET_SLUG)" \
	FACE_OF_AGI_KAGGLE_SUBMISSION_KERNEL_ID="$(KAGGLE_SUBMISSION_KERNEL_ID)" \
	FACE_OF_AGI_KAGGLE_SUBMISSION_KERNEL_TITLE="$(KAGGLE_SUBMISSION_KERNEL_TITLE)" \
	$(PYTHON) scripts/build_notebook.py

submit: notebook _check-kaggle
	$(KAGGLE) kernels push -p notebooks/ $(KAGGLE_ACCELERATOR_ARGS)

status: sync-metadata _check-kaggle
	@KERNEL_ID=$$($(PYTHON) -c "import json; print(json.load(open('notebooks/kernel-metadata.json'))['id'])"); \
	$(KAGGLE) kernels status $$KERNEL_ID

debug-metadata: sync-metadata
	$(PYTHON) scripts/build_debug_notebook.py \
		--config-relative-path "$(KAGGLE_DEBUG_CONFIG)" \
		--model-dataset-slug "$(KAGGLE_DEBUG_MODEL_DATASET_SLUG)" \
		$(DEBUG_KERNEL_TITLE_SUFFIX_ARG) \
		--metadata-only

debug-notebook: sync-metadata
	$(PYTHON) scripts/build_debug_notebook.py \
		--config-relative-path "$(KAGGLE_DEBUG_CONFIG)" \
		--model-dataset-slug "$(KAGGLE_DEBUG_MODEL_DATASET_SLUG)" \
		$(DEBUG_KERNEL_TITLE_SUFFIX_ARG)

debug-submit: debug-notebook _check-kaggle
	$(KAGGLE) kernels push -p debug-notebooks/ $(KAGGLE_ACCELERATOR_ARGS)

debug-status: debug-metadata _check-kaggle
	@KERNEL_ID=$$($(PYTHON) scripts/resolve_kaggle_kernel_ref.py "$(DEBUG_KERNEL)" debug-notebooks/kernel-metadata.json); \
	$(KAGGLE) kernels status $$KERNEL_ID

debug-pull: debug-metadata _check-kaggle
	mkdir -p $(DEBUG_OUTPUT_DIR)
	@KERNEL_ID=$$($(PYTHON) scripts/resolve_kaggle_kernel_ref.py "$(DEBUG_KERNEL)" debug-notebooks/kernel-metadata.json); \
	$(KAGGLE) kernels output $$KERNEL_ID -p $(DEBUG_OUTPUT_DIR) -o --file-pattern '.*\.sqlite$$'

clean:
	rm -rf build notebooks/submission.ipynb debug-notebooks/debug.ipynb debug-notebooks/kernel-metadata.json __pycache__ scripts/__pycache__
