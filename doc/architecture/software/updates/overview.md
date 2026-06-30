# Updates Overview

The updates module owns the post-step update boundary for the updater role
`P`. Updates happen after a real environment step, not after every simulated
tool call.

In the frame-unrolled orchestration loop, the updater boundary also runs after
each real observed frame turn. During frame unrolling it compares the current
frame with the next buffered frame; after a controllable final frame it
compares against the first frame returned by the next real environment step.

The updater improves context documents based on what the agent predicted, what
it did, and what actually happened. Orchestration decides the task timing:
role-specific game `L` updaters run during frame/game-loop transitions, while
the shared general `K` updater runs only at the end of a game loop/run.

Updater inputs are orchestration-managed live transition objects, not direct
database reads by the updater model. `M` remains the durable source of truth,
but orchestration may pass the current in-turn observations, trace, tool
results, contexts, and update quantities as Python objects while the turn is
being processed.

## Target Behavior

After orchestration applies a real action and receives the next observation,
it calls the appropriate role-specific game updater with the trace,
transition, predictions, and update quantities. In the frame-unrolled loop,
the same game-update boundary also runs between observed animation frames. At
terminal run completion, orchestration calls the shared general updater with
run summary data once per role. The updater returns revised context documents
for later model calls; orchestration applies those documents to its working
`ContextDocuments` and persists the resulting authoritative state into `M`.

At the start of a later run, orchestration hydrates contexts by combining the
latest persisted game-agnostic `K` across all games with the latest
game-specific `L` for the selected game. This carries general knowledge forward
without leaking one game's game context into another.
