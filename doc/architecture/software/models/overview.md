# Models Overview

The models module owns provider-neutral model role boundaries. It should
contain model-specific folders, contracts, configs, and adapters, while
keeping backend details swappable.

The target model roles are:

- orchestrator agent `X`
- world prediction model `S`
- goal prediction model `G`
- updater `P`

World `S` is the active prediction role in normal runtime. Goal `G` remains in
the source tree as a dormant role contract, but runtime assembly does not build
or call it. Maintained world context is fed to updater `P`, which revises
Agent `X` context for later decisions. Model roles do not own persistence.

## Target Folder Intent

The source tree should keep model roles separate:

- `models/orchestrator_agent`: decision model `X`
- `models/world`: world prediction model `S`
- `models/goal`: goal prediction model `G`
- `models/updater`: updater model `P`
- `models/description`: shared structured description capability used by
  world and goal

Each role exposes a provider-neutral contract to orchestration. Provider
adapters may live at a shared capability boundary when roles use the same
provider/output contract. `models/providers/` is the final provider-call layer:
it owns provider SDK/runtime request assembly, the actual provider call, and
raw response normalization. It should not own role concepts such as
`PredictionResult`, `AgentTrace`, updater tasks, or instruction-folder
selection.

Current provider layout:

- `models/orchestrator_agent/providers`: `openai`, `ollama`, plus an
  unsupported `configurable` placeholder
- `models/description/providers`: description-producing OpenAI/Ollama/vLLM
  providers shared by world and dormant goal
- `models/updater/providers`: real text-backed `openai`, `ollama`, and `vllm` providers
  for world/agent game-context updater tasks and the shared general updater
  task, plus dormant goal updater support and an unsupported `configurable`
  placeholder; runtime wires the active updater task slots from these providers

Adapters can point to local VLMs, custom neural networks, LoRA-updated models,
or hybrid backends without changing orchestration.

## Backend Notes

World and dormant goal adapters share the `models/description` capability.
Role adapters provide the role spec, instruction folder, and
action-conditioning rule; shared description providers translate prompt/image
requests into OpenAI, Ollama, or vLLM
calls and normalize responses into provider-neutral `PredictionResult` values
that updater `P` uses to improve future role contexts.

Each prompt-backed role call combines immutable role instructions with mutable
context. Agent `X` uses `models/orchestrator_agent/instructions/system_prompt.md`
as its immutable instruction and carries its mutable `K + L` context in the
Markdown user prompt. Agent X provider calls use one shared step loop that
sends the final action schema through the provider structured-output field
with each step and handles any future generic tool calls before accepting final
structured output. World `S` and goal `G`
use their role-local `instructions/instruction_prompt*.md` files as provider
instruction/system content. The mutable role context remains `K + L` and is
supplied through `RoleContext.composed()` in the user payload. World user
payloads also include the committed action because `S` is action-conditioned.
The implementation-facing model input/output reference lives in
[`inputs.md`](inputs.md) and [`outputs.md`](outputs.md).

World predictions are action-conditioned. Goal predictions are observation and
goal-context conditioned; they do not require an action.

Other model roles may use different providers or local runtimes. Sharing a
provider is an implementation choice, not a module boundary.
