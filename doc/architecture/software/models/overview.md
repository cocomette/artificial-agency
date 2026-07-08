# Models Overview

The models module owns provider-neutral role boundaries and provider adapters.
Model adapters receive typed inputs from orchestration and return typed outputs;
they do not own persistence or environment side effects.

## Active Roles

- `models/orchestrator_agent`: Agent X action selection.
- `models/change`: visual transition summary for action history.
- `models/memory`: fresh free-form memory document from first/current frames
  and sanitized action/change ledger rows.
- `models/world`: change-summary-style prediction for candidate actions.
- `models/goal`: structured goal and remaining-step prediction.
- `models/interest`: candidate value prediction for expected LP and task
  progress.
- `models/reward_judge`: text/VLM judge for World prediction quality.

Shared provider utilities live under `models/providers/`. Role-specific
provider implementations live below each active role package. Active v1 roles
support the existing `vllm` backend and the opt-in `hf_transformers` backend.
The HF backend uses one shared multimodal Transformers engine for all roles and
online LoRA training.

## Role Boundaries

Agent X is the only model role that chooses environment actions. It first
proposes coordinate candidates, then selects one final action after World has
predicted candidate outcomes. The change summary role describes observed
transitions after the environment response. Memory regenerates a complete text
document from first/current frames and sanitized action/change ledger rows
every turn. Goal reads Memory and estimates the goal, subgoals, remaining
steps, and confidence. Interest scores candidate actions after World
predictions are available. Reward Judge compares World prediction text to
Change Summary ground truth.

World is the trainable image-aware supervised transition role. Interest and
Agent are trainable GRPO roles. Memory, Goal, Change Summary, and Reward Judge
are inference-only in v1.

Generic Agent X tool plumbing may carry arbitrary `ToolResult.output` values,
but the v1 action loop does not expose runtime tools to the agent.
