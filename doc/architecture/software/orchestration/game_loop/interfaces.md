# Game Loop Interfaces

Important shared contracts:

- `FrameTurnSnapshot`: immutable in-loop snapshot for one frame turn.
- `FrameTurnContext`: provider-neutral model input context derived from the
  snapshot.
- `UpdaterFrameTransitionInput`: observed transition, decision trace, metrics,
  selected action, and action-history entry for updater/persistence steps.
- `ContextDocuments`: agent context documents only.
- `MStateRecord`: durable state row with agent context, trace, metrics, and
  metadata.

The game loop passes typed objects between action modules. Persistence converts
them to JSON only at the memory boundary.
