# Step 02: Contracts

## Objective

Define shared architecture-level contracts that future modules can implement
independently.

## Implementation

- Add `src/face_of_agi/contracts.py`.
- Use dataclasses for observations, actions, refs, tool calls, tool results,
  traces, context documents, reward/update quantities, runtime config, and
  decision results.
- Use `Protocol` for environment and model role boundaries.
- Keep frame payloads as `Any`.
- Keep metadata and record payloads generic.

## Parallelism

Can run in parallel with Step 01. Steps 03 and 04 depend on this step.

## Acceptance Check

- Core contracts can be imported without any model or ARC runtime installed at import time.
