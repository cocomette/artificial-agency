# Models Overview

The models module owns provider-neutral model role boundaries. It should
contain model-specific folders, contracts, configs, and adapters, while
keeping backend details swappable.

The target model roles are:

- orchestrator agent `X`
- world model tool `S`
- goal model tool `G`
- updater `P`

World and goal are tools used by the orchestrator agent. They are not called
directly by runtime and they do not own persistence.

## Target Folder Intent

The source tree should keep model roles separate:

- `models/orchestrator_agent`: decision model `X`
- `models/tools/world`: world-model tool `S`
- `models/tools/goal`: goal-model tool `G`
- `models/updater`: updater model `P`

Each role exposes a provider-neutral contract to orchestration. Provider
adapters live inside role-local `providers/` folders and translate between the
framework shape and a concrete backend. Provider-common utilities live in
`models/providers/`.

Current provider layout:

- `models/orchestrator_agent/providers`: `random`, `openai`, `ollama`, plus
  unsupported placeholders for `huggingface` and `configurable`
- `models/tools/world/providers`: `huggingface`, `openai`, plus an unsupported
  `configurable` placeholder
- `models/tools/goal/providers`: `huggingface`, `openai`, plus an unsupported
  `configurable` placeholder

Adapters can point to local VLMs, custom neural networks, LoRA-updated models,
or hybrid backends without changing orchestration.

## Current Backend Notes

The current concrete world and goal tool providers live under the role-local
`providers/` folders. Local Diffusers providers share
`models/providers/huggingface.py` and support Qwen Image Edit,
InstructPix2Pix-style editors, and FLUX Kontext qint8. OpenAI-backed providers
use the Responses API plus hosted image generation through shared OpenAI
utilities.

World predictions are action-conditioned. Goal predictions are observation and
goal-context conditioned; they do not require an action.

Other model roles may use different providers or local runtimes. Sharing a
provider is an implementation choice, not a module boundary.
