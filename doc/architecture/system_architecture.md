# System Architecture

FACE-OF-AGI is a Python runtime for ARC-AGI-3 experiments. The active system is
centered on an orchestration-owned game loop, provider-neutral model roles, and
SQLite memory.

## Active Roles

- Change summary produces compact transition evidence.
- Compacter maintains current mechanics, per-action effect summaries, and
  compact current-level action/strategy summaries.
- Updater P revises agent game context during runs and returns the next
  controllable action chain.

Agent X adapters remain in the codebase, but Agent X is dormant in the current
runtime game loop.

## Runtime Loop

The environment produces frame observations and action spaces. Orchestration
prewrites state, submits updater-produced actions or synthesizes `NONE`,
computes visible changed-pixel percentage, summarizes changed controllable
transitions, updates compacter context, updates agent context, and completes
state memory rows. Models never call the environment or write
persistence directly.

## Memory

`M` stores committed frame turns, decision traces, metrics, metadata, agent
context, and solved-level compacter summaries. `E` stores generic Agent X tool
outputs for the dormant adapter path.
Older SQLite run databases
with wider state rows are incompatible with the current schema and should be
reset.

## Providers

OpenAI, Ollama, and vLLM provider adapters are kept for active roles. Runtime
configs under `src/face_of_agi/runtime/configs/` define the active role set for
each backend family.
