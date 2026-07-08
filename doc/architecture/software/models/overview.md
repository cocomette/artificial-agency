# Models Overview

The models module owns provider-neutral model role boundaries. It should
contain model-specific folders, contracts, configs, and adapters, while
keeping backend details swappable.

The target model roles are:

- orchestrator agent `X`
- world prediction model `S`
- goal prediction model `G`
- updater `P`

World `S` and goal `G` are self-improving model roles. Their maintained
contexts are fed to Agent `X` for decisions and to updater `P` for revision
after observed transitions. They do not own persistence.

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
- `models/description/providers`: description-producing OpenAI/Ollama
  providers shared by world and goal
- `models/updater/providers`: real text-backed `openai` and `ollama` providers
  for world/goal/agent game-context updater tasks and the shared general
  updater task, plus an unsupported `configurable` placeholder; runtime wires
  four explicit updater task slots from these providers

Adapters can point to local VLMs, custom neural networks, LoRA-updated models,
or hybrid backends without changing orchestration.

## Backend Notes

World and goal share the `models/description` capability. Role adapters provide
the role spec, instruction folder, and action-conditioning rule; shared
description providers translate prompt/image requests into OpenAI or Ollama
calls and normalize responses into provider-neutral `PredictionResult` values
that updater `P` uses to improve future role contexts.

Each prompt-backed role call combines immutable role instructions with mutable
context. Agent `X` uses `models/orchestrator_agent/instructions/system_prompt.md`
as its immutable instruction and appends its mutable `K + L` context to the
same provider instruction/system content as `AGENT_CONTEXT`; the Agent X user
payload does not carry `role_context`. Agent X provider calls use one shared
step loop that sends the final action schema with each step and handles any
future generic tool calls before accepting final structured output. World `S`
and goal `G` use their role-local `instructions/instruction_prompt*.md` files
as provider instruction/system content. The mutable role context remains
`K + L` and is supplied through `RoleContext.composed()` in the user payload.
World user payloads also include the committed action because `S` is
action-conditioned.
The implementation-facing model input/output reference lives in
[`inputs.md`](inputs.md) and [`outputs.md`](outputs.md).

World predictions are action-conditioned. Goal predictions are observation and
goal-context conditioned; they do not require an action.

Other model roles may use different providers or local runtimes. Sharing a
provider is an implementation choice, not a module boundary.
