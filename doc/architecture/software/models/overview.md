# Models Overview

The models module owns provider-neutral role boundaries and provider adapters.
Model adapters receive typed inputs from orchestration and return typed outputs;
they do not own persistence or environment side effects.

## Active Roles

- `models/orchestrator_agent`: dormant Agent X adapter path.
- `models/change`: visual transition summary for action history.
- `models/world`: current world description, special-event memory, and
  per-action effect summaries.
- `models/historizer`: compact probing/policy evolution and update-mode
  selection.
- `models/level_summary`: completed-level solution method summary.
- `models/updater`: updater P for agent probing, agent policy, and
  agent general context.

Shared provider utilities live under `models/providers/`. Role-specific
provider implementations live below each active role package. Runtime configs
wire OpenAI, Ollama, or vLLM backends for the active roles only.

## Role Boundaries

Updater P is the active model role that chooses environment actions through its
`next_actions` output. The change summary role describes observed transitions
after the environment response.
The world role runs over the previous world-model context, action history,
allowed actions, and the current frame for the latest action-history row.
Animation evidence stays in action history; world-model provider calls attach
only the final current frame.
The historizer role then summarizes recent updater strategy snapshots plus the
fresh world-model output into compact probing and policy evolution summaries
and proposes the next update mode. Orchestration may deterministically cap a
probing proposal to policy before dispatch. Updater P revises
probing or policy strategies and selects a bounded ordered action array that
orchestration stores for upcoming controllable environment steps.
When a level is completed, the level-summary role summarizes the same-run
strategy snapshots that led to that completion. The next updater call receives
the latest same-run `solution_method` as optional guidance for the next level.

Generic Agent X tool plumbing may carry arbitrary `ToolResult.output` values in
the dormant adapter path, but there are no role-specific visual forecast models
in the active runtime.
