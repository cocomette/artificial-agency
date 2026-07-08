# Models Overview

The models module owns provider-neutral role boundaries and provider adapters.
Model adapters receive typed inputs from orchestration and return typed outputs;
they do not own persistence or environment side effects.

## Active Roles

- `models/orchestrator_agent`: Agent X action selection.
- `models/change`: visual transition summary for action history.
- `models/memory`: fresh free-form memory document from first/current frames
  and sanitized action/change/reward ledger rows.
- `models/world`: change-summary-style prediction for candidate actions.
- `models/goal`: structured goal and remaining-step prediction.
- `models/interest`: candidate value prediction for expected proxy LP and task
  progress.
- `models/reward_judge`: text/VLM judge for World prediction quality.

Shared provider utilities live under `models/providers/`. Role-specific
provider implementations live below each active role package. The committed
no-LoRA runtime path is static `vllm` inference using the configured FP8 model.

## Role Boundaries

Agent X is the only model role that chooses environment actions. It first
proposes coordinate candidates, then selects one final action after World has
predicted candidate outcomes. The change summary role describes observed
transitions after the environment response. Memory regenerates a complete text
document from first/current frames and sanitized ledger rows every turn. Goal
reads Memory and estimates the goal, subgoals, remaining steps, and confidence.
Interest scores candidate actions after World predictions are available. Reward
Judge compares World prediction text to Change Summary ground truth.

World, Interest, and Agent are not trainable in this branch. Their feedback
comes through prompt-visible reward and proxy learning-progress history.
Generic Agent X tool plumbing may carry arbitrary `ToolResult.output` values,
but the v1 action loop does not expose runtime tools to the agent.
