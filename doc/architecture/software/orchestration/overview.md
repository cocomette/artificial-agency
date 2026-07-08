# Orchestration Overview

Orchestration owns the game loop and coordinates environment, models, memory,
debug tracing, and runtime lifecycle.

Per controllable frame turn it:

1. builds a frame snapshot and optional source `M` row
2. models the previous-to-current transition when one exists
3. runs the compacter, then updater P when a fresh queued action chain is
   needed
4. submits the updater-selected action to the environment
5. completes the `M` row

Agent X is dormant in the current game loop. The orchestrator-agent adapters
remain in the codebase, but runtime bootstrap registers no Agent X model and
controllable frame decisions are wrappers around updater-selected actions.

Updater P returns an ordered queued action chain. Valid queued actions are
submitted on later controllable frames without rerunning the compacter or
updater. If the latest real action has zero net visible
first-to-last frame change, orchestration clears the remaining queued actions
before deciding whether fresh context is needed. Locally summarized identical
direct transitions carry forward the latest element names and descriptions in
action history. Animation bundles still run through change summary even when
their final frame matches the first frame.

Updater P receives the current raw action-history and strategy-history windows
plus previous action and strategy summaries from the compacter. The raw window
size is configured by `updater_context_history_window`; `1` means only the
latest raw entry/snapshot. On a level-completion boundary, the compacter sees
the latest retained solved-level frame instead of the new-level frame,
orchestration stores the compact strategy summary as the next previous-level
strategy summary, blanks previous `current_strategy`, and sends updater P the reset notice
`You just solved a level, you start solving a new level now`.

After updater P selects a single action, orchestration may invoke the
known-state simulation sidecar. The sidecar uses only real, non-simulated prior
M rows with submitted environment actions as transition edges: if the selected
action was previously submitted from the same exact current-frame hash, it
replays that historical transition without calling change summary or the
environment. Synthetic `NONE` lifecycle rows and simulated rows are
persisted for traceability and agent-facing history, but they are excluded from
future simulation transition edges. While simulating,
orchestration keeps one continuous action history: it first folds in any pending
real transition evidence from the entry turn, then appends known successor-row
transition evidence read from memory instead of rerunning change summary. The
updater receives fresh compacter output captured on each simulated row before
each simulated updater decision. When simulation exits on an unknown
action, orchestration
builds a bounded historical graph path from the pre-simulation frame hash to
the simulated endpoint hash. For a single simulated action it submits that
action directly unless the simulated endpoint is already the entry frame; for
longer simulations it breadth-first searches real, non-simulated prior
transition edges up to the simulated path length and uses the first shortest
path that reaches the simulated endpoint. If no path is found, it falls back to
the simulated action list. It submits the selected catch-up actions immediately
without change summary, compacter, updater, or intermediate memory rows, then
queues only the unknown action for normal processing. The catch-up endpoint
hash check is recorded as debug metadata only. Runtime telemetry treats
simulated rows as ordinary completed turns for live `turns`, `avg_turn_sec`,
`turns_per_min`, and controllable-action totals, and counts submitted
environment actions from environment step events.

For `ACTION6` simulation matching, historical rows persist the submitted full
ARC-grid coordinate plus the targeted ARC cell value. The current updater bbox
is kept only as transient runtime metadata. Simulation compares in full ARC-grid
space: the current and historical target values must match, and the current
bbox must wrap the historical submitted coordinate after applying the same
ARC-grid crop transform.

Post-action animation bundles produce one synthetic `NONE` raw history entry
that renders on the same prompt-facing line as the action that caused it. When
fresh context is needed, the bundled transition is sent through one compacter
update. Updater P runs on the final controllable animation frame before the next
environment action is submitted, so updater decisions are based on the actual
actionable state.
