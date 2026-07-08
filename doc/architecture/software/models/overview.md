# Models Overview

The models module owns provider-neutral role boundaries and provider adapters.
Model adapters receive typed inputs from orchestration and return typed outputs;
they do not own persistence or environment side effects.

## Active Roles

- `models/orchestrator_agent`: dormant Agent X adapter path.
- `models/change`: visual transition summary for action history.
- `models/compacter`: current world model plus compact current-level action
  and strategy summaries.
- `models/updater`: updater P for agent game context and action selection.

Shared provider utilities live under `models/providers/`. Role-specific
provider implementations live below each active role package. Runtime configs
wire OpenAI, Ollama, or vLLM backends for the active roles only.

## Role Boundaries

Updater P is the active model role that chooses environment actions through its
`next_actions` output. The change summary role describes observed transitions
after the environment response.
The compacter role runs over the previous compacter context, action history,
strategy history, allowed actions, and the current frame for the latest
action-history row. Animation evidence stays in action history; compacter
provider calls attach only the final current frame. It returns world
fields plus `previous_actions_summary` and `previous_strategy_summary`.
Updater P then receives previous strategy summary, previous actions summary,
world model context, the previous turn's `current_strategy`, and the current
raw action/strategy windows.
It revises `current_strategy`,
then selects
a bounded ordered action array that
orchestration stores for upcoming controllable environment steps.
When the latest action-history row shows a completed-level increase,
orchestration stores the completion compacter summaries as the solved-level
summary carried to the next level.

Generic Agent X tool plumbing may carry arbitrary `ToolResult.output` values in
the dormant adapter path, but there are no role-specific visual forecast models
in the active runtime.
