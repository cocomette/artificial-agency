# Updates Overview

The updates module owns the post-step update boundary for the updater role
`P`. Updates happen after a real environment step, not after every simulated
tool call.

In the frame-unrolled orchestration loop, the updater boundary also runs after
each real observed frame turn. During frame unrolling it compares the current
frame with the next buffered frame; after a controllable final frame it
compares against the first frame returned by the next real environment step.

The updater improves game-specific context documents based on what the agent
predicted, what it did, and what actually happened.

Updater inputs are orchestration-managed live transition objects, not direct
database reads by the updater model. `M` remains the durable source of truth,
but orchestration may pass the current in-turn observations, trace, tool
results, contexts, and update quantities as Python objects while the turn is
being processed.

## Target Behavior

After orchestration applies a real action and receives the next observation,
it calls updater `P` with the trace, transition, predictions, and update
quantities. In the frame-unrolled loop, the same boundary also runs between
observed animation frames. The updater returns revised game-specific context
documents for later model calls; orchestration applies those documents to its
working `ContextDocuments` and persists the resulting authoritative state into
`M`.
