# System Architecture

FACE-OF-AGI is a Python runtime for ARC-AGI-3 experiments. The active system is
centered on an orchestration-owned game loop, provider-neutral model roles, and
SQLite memory.

## Active Roles

- Agent X chooses actions on controllable frames.
- Change summary produces compact transition evidence.
- Memory regenerates a free-form run document from first/current frames and a
  sanitized action/change/reward ledger, preserving reset history.
- World predicts candidate transition summaries.
- Goal estimates the current goal, subgoals, and remaining steps.
- Interest scores candidate actions for expected proxy LP and task progress.
- Reward Judge scores World predictions against observed change summaries.

## Runtime Loop

The environment produces frame observations and action spaces. Orchestration
prewrites state, bootstraps Memory/Goal, asks Agent X for coordinate
candidates, runs World for each candidate, asks Agent X for the final action,
submits real actions, summarizes observed changes, judges World, computes
reward with a reward-only Goal call, appends the finalized ledger entry,
regenerates Memory/Goal, and completes state memory rows. Models never call the
environment or write persistence directly.

## Memory

`M` stores committed frame turns, Agent X trace, metrics, metadata, v1 role
artifacts, rewards, and model-input debug records. `E` stores generic Agent X
tool outputs. Older SQLite run databases are incompatible with the current
schema and should be reset.

## Providers

The v1 active runtime path is static vLLM FP8 inference. Runtime configs under
`src/face_of_agi/runtime/configs/` define the active role set and vLLM startup
options. There is no online training or adapter activation path in this branch.
