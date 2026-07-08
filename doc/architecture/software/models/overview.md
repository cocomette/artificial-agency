# Models Overview

The models module owns provider-neutral model role boundaries. Role adapters
translate between orchestration contracts and concrete providers while keeping
backend details swappable.

## Active Roles

- `models/orchestrator_agent`: Agent X decision model.
- `models/change`: transition change-summary model.
- `models/historizer`: prior agent game-context history summarizer.
- `models/memory`: same-run game memory summarizer.
- `models/updater`: updater P for agent game context and agent general context.

World and goal tool modules are not present in the active runtime for this
branch.

## Provider Boundary

Each role exposes a provider-neutral contract to orchestration. Provider
adapters live inside role-local `providers/` folders and translate prompts,
schemas, images, and repair requests to a concrete backend.

The shared provider utilities under `models/providers/` are final transport
helpers. They should not own role concepts such as agent traces, memory
documents, or updater tasks.

## Structured Output

Prompt-backed roles validate structured JSON output and use bounded repair
attempts when providers support repair. Current output caps are:

- change-summary element fields: 2000 characters
- change-summary elements array: 20 items
- historizer field evolution values: 2000 characters
- game memory: configurable, default 10000 characters
- updater general context: 20000 characters
- updater agent game context: 12000 characters total, 6000 per field
- vLLM invalid-output repair previews: configurable, default 8000 characters
