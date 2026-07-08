# Orchestration Overview

Orchestration owns the game loop and coordinates environment, models, memory,
debug tracing, and runtime lifecycle.

Per controllable frame turn it:

1. builds a frame snapshot and optional source `M` row
2. models the previous-to-current transition when one exists
3. runs the agent-context historizer and updater P when a fresh queued action
   chain is needed
4. submits the updater-selected action to the environment
5. completes the `M` row

Agent X is dormant in the current game loop. The orchestrator-agent adapters
remain in the codebase, but runtime bootstrap registers no Agent X model and
controllable frame decisions are wrappers around updater-selected actions.

Updater P returns an ordered queued action chain. Valid queued actions are
submitted on later controllable frames without rerunning the world model,
historizer, or updater. If the latest real action has zero net visible
first-to-last frame change, orchestration clears the remaining queued actions
before deciding whether fresh context is needed. Animation bundles still run
through change summary even when their final frame matches the first frame.

Post-action animation bundles produce one synthetic `NONE` raw history entry
that renders on the same prompt-facing line as the action that caused it. When
fresh context is needed, the bundled transition is sent through one world-model
update. Historizer mode selection and updater P run on the final controllable
animation frame before the next environment action is submitted, so updater
decisions are based on the actual actionable state.
