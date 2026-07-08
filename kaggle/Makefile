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
MODAL_CLI ?= uv run --with modal modal
MODAL_KAGGLE_SECRET ?= kaggle-api-token
KAGGLE_SUBMISSION_CONFIG ?= src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_qwen36_35b_fp8_parallel.yaml
KAGGLE_SUBMISSION_MODEL_DATASET_SLUG ?= face-of-agi-qwen36-35b-fp8-weights
KAGGLE_SUBMISSION_KERNEL_ID ?= $(FACE_OF_AGI_KAGGLE_OWNER)/face-of-agi-arc-agi-3-rtx6000
KAGGLE_SUBMISSION_KERNEL_TITLE ?= FACE-OF-AGI ARC-AGI-3 RTX6000
GEMMA_SUBMISSION_CONFIG ?= src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_gemma4_parallel.yaml
GEMMA_SUBMISSION_MODEL_DATASET_SLUG ?= face-of-agi-gemma4-31b-it-qat-w4a16-ct-weights
GEMMA_SUBMISSION_KERNEL_ID ?= $(FACE_OF_AGI_KAGGLE_OWNER)/face-of-agi-arc-agi-3-rtx6000-gemma4
GEMMA_SUBMISSION_KERNEL_TITLE ?= FACE-OF-AGI ARC-AGI-3 RTX6000 Gemma4
KAGGLE_DEBUG_CONFIG ?= src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_qwen36_35b_fp8_debug.yaml
KAGGLE_DEBUG_MODEL_DATASET_SLUG ?= face-of-agi-qwen36-35b-fp8-weights
DEBUG_KERNEL_TITLE_SUFFIX ?=
DEBUG_KERNEL_TITLE_SUFFIX_ARG := $(if $(strip $(DEBUG_KERNEL_TITLE_SUFFIX)),--kernel-title-suffix "$(DEBUG_KERNEL_TITLE_SUFFIX)",)
DEBUG_WHEELHOUSE_DATASET_SLUG ?= face-of-agi-wheelhouse
GEMMA4_FP8_DEBUG_CONFIG ?= src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_gemma4_26b_a4b_fp8_dynamic_debug.yaml
GEMMA4_FP8_DEBUG_MODEL_DATASET_SLUG ?= face-of-agi-gemma4-26b-a4b-it-fp8-dynamic-weights
GEMMA4_FP8_DEBUG_KERNEL_TITLE_SUFFIX ?= gemma4-fp8
MINICPM_V46_THINKING_DEBUG_CONFIG ?= src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_minicpm_v46_thinking_debug.yaml
MINICPM_V46_THINKING_DEBUG_MODEL_DATASET_SLUG ?= face-of-agi-minicpm-v46-thinking-weights
MINICPM_V46_THINKING_DEBUG_KERNEL_TITLE_SUFFIX ?= minicpm-v46-thinking
MINICPM_V46_THINKING_DEBUG_VLLM_VERSION ?= 0.22.0
MINICPM_V46_THINKING_WHEELHOUSE_DIR ?= build/wheelhouse-minicpm-v46-thinking
MINICPM_V46_THINKING_WHEELHOUSE_METADATA ?= upload/wheelhouse-minicpm-v46-thinking/dataset-metadata.json
MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG ?= face-of-agi-minicpm-v46-thinking-wheelhouse
MINICPM_V46_THINKING_WHEEL_ABIS ?= cp312 abi3
MODAL_MODEL_UPLOAD_REPO_ID ?= Qwen/Qwen3.6-35B-A3B-FP8
MODAL_MODEL_UPLOAD_REVISION ?= main
MODAL_MODEL_UPLOAD_SNAPSHOT_PATH ?=
MODAL_MODEL_UPLOAD_SNAPSHOT_ARG := $(if $(strip $(MODAL_MODEL_UPLOAD_SNAPSHOT_PATH)),--snapshot-path $(MODAL_MODEL_UPLOAD_SNAPSHOT_PATH),)
MODEL_DATASET_METADATA := upload/model-dataset/dataset-metadata.json

WHEELHOUSE_DIR := build/wheelhouse
WHEELHOUSE_METADATA ?= upload/wheelhouse/dataset-metadata.json
WHEELHOUSE_DATASET_SLUG ?= face-of-agi-wheelhouse
WHEEL_PLATFORMS ?= manylinux_2_31_x86_64 manylinux_2_28_x86_64 manylinux_2_24_x86_64 manylinux2014_x86_64
WHEEL_PLATFORM_ARGS := $(foreach platform,$(WHEEL_PLATFORMS),--platform $(platform))
WHEEL_PYTHON_VERSION ?= 312
WHEEL_ABI ?= cp312
WHEEL_ABIS ?= $(WHEEL_ABI)
WHEEL_ABI_ARGS := $(foreach abi,$(WHEEL_ABIS),--abi $(abi))
MODEL_BUILD_DIR := build/model/default
MODEL_SOURCE_DIR ?=
MODEL_REF ?= $(FACE_OF_AGI_KAGGLE_OWNER)/face-of-agi-qwen36-35b-fp8/pytorch/default
PUBLIC_GAMES_DIR := build/public-games
DEBUG_OUTPUT_DIR := $(ROOT)/runs/kaggle-debug
UPLOAD_MODE ?= create
VERSION_NOTES ?= FACE-OF-AGI Kaggle submission artifact update
DEBUG_KERNEL ?=

.PHONY: sync-metadata wheelhouse wheelhouse-upload model-upload modal-kaggle-secret modal-model-upload modal-model-upload-dry-run modal-model-finalize modal-model-dataset-upload modal-model-dataset-upload-dry-run model-bootstrap-notebook model-bootstrap-submit model-bootstrap-submit-with-token model-bootstrap-status public-games public-games-upload notebook submit status gemma-notebook gemma-submit gemma-status debug-metadata debug-notebook debug-submit debug-status debug-pull gemma4-fp8-debug-metadata gemma4-fp8-debug-notebook gemma4-fp8-debug-submit gemma4-fp8-debug-status gemma4-fp8-debug-pull minicpm-v46-thinking-wheelhouse minicpm-v46-thinking-wheelhouse-upload minicpm-v46-thinking-debug-metadata minicpm-v46-thinking-debug-notebook minicpm-v46-thinking-debug-submit minicpm-v46-thinking-debug-status minicpm-v46-thinking-debug-pull clean _check-kaggle

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
		$(WHEEL_ABI_ARGS) \
		--requirements requirements-kaggle.txt \
		--metadata $(WHEELHOUSE_METADATA) \
		--dataset-slug $(WHEELHOUSE_DATASET_SLUG) \
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

model-upload: sync-metadata _check-kaggle
	@if [ "$(UPLOAD_MODE)" = "create-model" ]; then \
		$(KAGGLE) models create -p upload/model; \
		exit 0; \
	fi
	@if [ -z "$(MODEL_SOURCE_DIR)" ]; then \
		echo "ERROR: set MODEL_SOURCE_DIR to the local Hugging Face model directory."; \
		exit 1; \
	fi
	rm -rf $(MODEL_BUILD_DIR)
	mkdir -p $(MODEL_BUILD_DIR)
	cp -R $(MODEL_SOURCE_DIR)/. $(MODEL_BUILD_DIR)/
	cp upload/model/model-instance-metadata.json $(MODEL_BUILD_DIR)/model-instance-metadata.json
	@if [ "$(UPLOAD_MODE)" = "create" ]; then \
		$(KAGGLE) models variations create -p $(MODEL_BUILD_DIR) -q; \
	elif [ "$(UPLOAD_MODE)" = "version" ]; then \
		$(KAGGLE) models variations versions create $(MODEL_REF) -p $(MODEL_BUILD_DIR) -n "$(VERSION_NOTES)"; \
	else \
		echo "ERROR: UPLOAD_MODE must be create-model, create, or version"; \
		exit 1; \
	fi

modal-kaggle-secret: _check-kaggle
	@tmp=$$(mktemp); \
	trap 'rm -f "$$tmp"' EXIT; \
	$(PYTHON) -c 'import json, pathlib, sys; token = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip(); pathlib.Path(sys.argv[2]).write_text(json.dumps({"KAGGLE_API_TOKEN": token}), encoding="utf-8")' "$(KAGGLE_TOKEN)" "$$tmp"; \
	$(MODAL_CLI) secret create --force "$(MODAL_KAGGLE_SECRET)" --from-json "$$tmp"

modal-model-upload: sync-metadata
	FACE_OF_AGI_MODAL_KAGGLE_SECRET="$(MODAL_KAGGLE_SECRET)" $(MODAL_CLI) run scripts/upload_modal_model_to_kaggle.py \
		--repo-id "$(MODAL_MODEL_UPLOAD_REPO_ID)" \
		--revision "$(MODAL_MODEL_UPLOAD_REVISION)" \
		--upload-mode "$(UPLOAD_MODE)" \
		--version-notes "$(VERSION_NOTES)" \
		$(MODAL_MODEL_UPLOAD_SNAPSHOT_ARG)

modal-model-upload-dry-run: sync-metadata
	FACE_OF_AGI_MODAL_KAGGLE_SECRET="$(MODAL_KAGGLE_SECRET)" $(MODAL_CLI) run scripts/upload_modal_model_to_kaggle.py \
		--repo-id "$(MODAL_MODEL_UPLOAD_REPO_ID)" \
		--revision "$(MODAL_MODEL_UPLOAD_REVISION)" \
		--upload-mode "$(UPLOAD_MODE)" \
		--version-notes "$(VERSION_NOTES)" \
		$(MODAL_MODEL_UPLOAD_SNAPSHOT_ARG) \
		--dry-run

modal-model-finalize: sync-metadata
	FACE_OF_AGI_MODAL_KAGGLE_SECRET="$(MODAL_KAGGLE_SECRET)" $(MODAL_CLI) run scripts/upload_modal_model_to_kaggle.py \
		--repo-id "$(MODAL_MODEL_UPLOAD_REPO_ID)" \
		--revision "$(MODAL_MODEL_UPLOAD_REVISION)" \
		--upload-mode finalize \
		$(MODAL_MODEL_UPLOAD_SNAPSHOT_ARG)

modal-model-dataset-upload: sync-metadata
	FACE_OF_AGI_MODAL_KAGGLE_SECRET="$(MODAL_KAGGLE_SECRET)" $(MODAL_CLI) run scripts/upload_modal_model_to_kaggle.py \
		--artifact-kind dataset \
		--metadata-path "$(MODEL_DATASET_METADATA)" \
		--repo-id "$(MODAL_MODEL_UPLOAD_REPO_ID)" \
		--revision "$(MODAL_MODEL_UPLOAD_REVISION)" \
		--upload-mode "$(UPLOAD_MODE)" \
		--version-notes "$(VERSION_NOTES)" \
		$(MODAL_MODEL_UPLOAD_SNAPSHOT_ARG)

modal-model-dataset-upload-dry-run: sync-metadata
	FACE_OF_AGI_MODAL_KAGGLE_SECRET="$(MODAL_KAGGLE_SECRET)" $(MODAL_CLI) run scripts/upload_modal_model_to_kaggle.py \
		--artifact-kind dataset \
		--metadata-path "$(MODEL_DATASET_METADATA)" \
		--repo-id "$(MODAL_MODEL_UPLOAD_REPO_ID)" \
		--revision "$(MODAL_MODEL_UPLOAD_REVISION)" \
		--upload-mode "$(UPLOAD_MODE)" \
		--version-notes "$(VERSION_NOTES)" \
		$(MODAL_MODEL_UPLOAD_SNAPSHOT_ARG) \
		--dry-run

model-bootstrap-notebook: sync-metadata
	$(PYTHON) scripts/build_model_bootstrap_notebook.py

model-bootstrap-submit: model-bootstrap-notebook _check-kaggle
	$(KAGGLE) kernels push -p model-bootstrap/

model-bootstrap-submit-with-token: sync-metadata _check-kaggle
	MODEL_BOOTSTRAP_EMBED_KAGGLE_TOKEN=true $(PYTHON) scripts/build_model_bootstrap_notebook.py
	$(KAGGLE) kernels push -p model-bootstrap/

model-bootstrap-status: sync-metadata _check-kaggle
	@KERNEL_ID=$$($(PYTHON) -c "import json; print(json.load(open('model-bootstrap/kernel-metadata.json'))['id'])"); \
	$(KAGGLE) kernels status $$KERNEL_ID

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

gemma-notebook:
	$(MAKE) notebook \
		KAGGLE_SUBMISSION_CONFIG="$(GEMMA_SUBMISSION_CONFIG)" \
		KAGGLE_SUBMISSION_MODEL_DATASET_SLUG="$(GEMMA_SUBMISSION_MODEL_DATASET_SLUG)" \
		KAGGLE_SUBMISSION_KERNEL_ID="$(GEMMA_SUBMISSION_KERNEL_ID)" \
		KAGGLE_SUBMISSION_KERNEL_TITLE="$(GEMMA_SUBMISSION_KERNEL_TITLE)"

gemma-submit: gemma-notebook _check-kaggle
	$(KAGGLE) kernels push -p notebooks/ $(KAGGLE_ACCELERATOR_ARGS)

gemma-status: _check-kaggle
	$(KAGGLE) kernels status "$(GEMMA_SUBMISSION_KERNEL_ID)"

debug-metadata: sync-metadata
	$(PYTHON) scripts/build_debug_notebook.py \
		--config-relative-path "$(KAGGLE_DEBUG_CONFIG)" \
		--model-dataset-slug "$(KAGGLE_DEBUG_MODEL_DATASET_SLUG)" \
		--wheelhouse-dataset-slug "$(DEBUG_WHEELHOUSE_DATASET_SLUG)" \
		$(DEBUG_KERNEL_TITLE_SUFFIX_ARG) \
		--metadata-only

debug-notebook: sync-metadata
	$(PYTHON) scripts/build_debug_notebook.py \
		--config-relative-path "$(KAGGLE_DEBUG_CONFIG)" \
		--model-dataset-slug "$(KAGGLE_DEBUG_MODEL_DATASET_SLUG)" \
		--wheelhouse-dataset-slug "$(DEBUG_WHEELHOUSE_DATASET_SLUG)" \
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

gemma4-fp8-debug-metadata:
	$(MAKE) debug-metadata \
		KAGGLE_DEBUG_CONFIG="$(GEMMA4_FP8_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(GEMMA4_FP8_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(GEMMA4_FP8_DEBUG_KERNEL_TITLE_SUFFIX)"

gemma4-fp8-debug-notebook:
	$(MAKE) debug-notebook \
		KAGGLE_DEBUG_CONFIG="$(GEMMA4_FP8_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(GEMMA4_FP8_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(GEMMA4_FP8_DEBUG_KERNEL_TITLE_SUFFIX)"

gemma4-fp8-debug-submit:
	$(MAKE) debug-submit \
		KAGGLE_DEBUG_CONFIG="$(GEMMA4_FP8_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(GEMMA4_FP8_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(GEMMA4_FP8_DEBUG_KERNEL_TITLE_SUFFIX)"

gemma4-fp8-debug-status:
	$(MAKE) debug-status \
		KAGGLE_DEBUG_CONFIG="$(GEMMA4_FP8_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(GEMMA4_FP8_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(GEMMA4_FP8_DEBUG_KERNEL_TITLE_SUFFIX)"

gemma4-fp8-debug-pull:
	$(MAKE) debug-pull \
		KAGGLE_DEBUG_CONFIG="$(GEMMA4_FP8_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(GEMMA4_FP8_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(GEMMA4_FP8_DEBUG_KERNEL_TITLE_SUFFIX)"

minicpm-v46-thinking-wheelhouse:
	FACE_OF_AGI_KAGGLE_VLLM_VERSION="$(MINICPM_V46_THINKING_DEBUG_VLLM_VERSION)" \
	FACE_OF_AGI_KAGGLE_FULL_VLLM_DEPS=true \
	$(MAKE) wheelhouse \
		WHEELHOUSE_DIR="$(MINICPM_V46_THINKING_WHEELHOUSE_DIR)" \
		WHEELHOUSE_METADATA="$(MINICPM_V46_THINKING_WHEELHOUSE_METADATA)" \
		WHEELHOUSE_DATASET_SLUG="$(MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG)" \
		WHEEL_ABIS="$(MINICPM_V46_THINKING_WHEEL_ABIS)"

minicpm-v46-thinking-wheelhouse-upload:
	FACE_OF_AGI_KAGGLE_VLLM_VERSION="$(MINICPM_V46_THINKING_DEBUG_VLLM_VERSION)" \
	FACE_OF_AGI_KAGGLE_FULL_VLLM_DEPS=true \
	$(MAKE) wheelhouse-upload \
		WHEELHOUSE_DIR="$(MINICPM_V46_THINKING_WHEELHOUSE_DIR)" \
		WHEELHOUSE_METADATA="$(MINICPM_V46_THINKING_WHEELHOUSE_METADATA)" \
		WHEELHOUSE_DATASET_SLUG="$(MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG)" \
		WHEEL_ABIS="$(MINICPM_V46_THINKING_WHEEL_ABIS)"

minicpm-v46-thinking-debug-metadata:
	FACE_OF_AGI_KAGGLE_DEBUG_VLLM_VERSION="$(MINICPM_V46_THINKING_DEBUG_VLLM_VERSION)" \
	FACE_OF_AGI_KAGGLE_DEBUG_FULL_VLLM_DEPS=true \
	$(MAKE) debug-metadata \
		KAGGLE_DEBUG_CONFIG="$(MINICPM_V46_THINKING_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(MINICPM_V46_THINKING_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_WHEELHOUSE_DATASET_SLUG="$(MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(MINICPM_V46_THINKING_DEBUG_KERNEL_TITLE_SUFFIX)"

minicpm-v46-thinking-debug-notebook:
	FACE_OF_AGI_KAGGLE_DEBUG_VLLM_VERSION="$(MINICPM_V46_THINKING_DEBUG_VLLM_VERSION)" \
	FACE_OF_AGI_KAGGLE_DEBUG_FULL_VLLM_DEPS=true \
	$(MAKE) debug-notebook \
		KAGGLE_DEBUG_CONFIG="$(MINICPM_V46_THINKING_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(MINICPM_V46_THINKING_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_WHEELHOUSE_DATASET_SLUG="$(MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(MINICPM_V46_THINKING_DEBUG_KERNEL_TITLE_SUFFIX)"

minicpm-v46-thinking-debug-submit:
	FACE_OF_AGI_KAGGLE_DEBUG_VLLM_VERSION="$(MINICPM_V46_THINKING_DEBUG_VLLM_VERSION)" \
	FACE_OF_AGI_KAGGLE_DEBUG_FULL_VLLM_DEPS=true \
	$(MAKE) debug-submit \
		KAGGLE_DEBUG_CONFIG="$(MINICPM_V46_THINKING_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(MINICPM_V46_THINKING_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_WHEELHOUSE_DATASET_SLUG="$(MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(MINICPM_V46_THINKING_DEBUG_KERNEL_TITLE_SUFFIX)"

minicpm-v46-thinking-debug-status:
	FACE_OF_AGI_KAGGLE_DEBUG_VLLM_VERSION="$(MINICPM_V46_THINKING_DEBUG_VLLM_VERSION)" \
	FACE_OF_AGI_KAGGLE_DEBUG_FULL_VLLM_DEPS=true \
	$(MAKE) debug-status \
		KAGGLE_DEBUG_CONFIG="$(MINICPM_V46_THINKING_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(MINICPM_V46_THINKING_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_WHEELHOUSE_DATASET_SLUG="$(MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(MINICPM_V46_THINKING_DEBUG_KERNEL_TITLE_SUFFIX)"

minicpm-v46-thinking-debug-pull:
	FACE_OF_AGI_KAGGLE_DEBUG_VLLM_VERSION="$(MINICPM_V46_THINKING_DEBUG_VLLM_VERSION)" \
	FACE_OF_AGI_KAGGLE_DEBUG_FULL_VLLM_DEPS=true \
	$(MAKE) debug-pull \
		KAGGLE_DEBUG_CONFIG="$(MINICPM_V46_THINKING_DEBUG_CONFIG)" \
		KAGGLE_DEBUG_MODEL_DATASET_SLUG="$(MINICPM_V46_THINKING_DEBUG_MODEL_DATASET_SLUG)" \
		DEBUG_WHEELHOUSE_DATASET_SLUG="$(MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG)" \
		DEBUG_KERNEL_TITLE_SUFFIX="$(MINICPM_V46_THINKING_DEBUG_KERNEL_TITLE_SUFFIX)"

clean:
	rm -rf build notebooks/submission.ipynb debug-notebooks/debug.ipynb debug-notebooks/kernel-metadata.json model-bootstrap/model_bootstrap.ipynb __pycache__ scripts/__pycache__
