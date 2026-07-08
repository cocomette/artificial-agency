# Orchestration Responsibilities

## Owns

- Main running loop for one game run.
- Turn-by-turn communication with the ARC-AGI environment adapter.
- Focused sub-orchestration components for concrete workflows, such as the
  game-loop state machine.
- Reading memory-backed observations, ledgers, role outputs, and reward history
  for model roles.
- Calling the orchestrator agent model `X`.
- Persisting model outputs, tool results, traces, actions, observations,
  transition timing, and score/progress metadata.
- Building candidate actions, running World predictions, computing rewards, and
  feeding proxy learning-progress back through action history and Memory.
- Keeping callable source-state ids internal to orchestration and memory.
- Keeping rolling `E` artifacts as non-callable experiment/debug records.
- Owning the target frame-unrolled game-loop state machine described in
  [`game_loop/overview.md`](game_loop/overview.md).

## Does Not Own

- ARC-AGI toolkit internals.
- Model backend implementation details.
- SQLite connection primitives.
- The content of provider prompts beyond composing stored context documents.
- Rendering or visualization policy except as environment metadata passed
  through the loop.

## Central Rule

Models may return descriptions, predictions, explanations, traces, actions,
judgments, memory documents, or goal estimates. Orchestration decides where
those outputs go, persists them, and resolves later references to them.

No model adapter writes directly to or reads directly from SQLite in the target
architecture.

Final action choice lives behind the configured `X` adapter after
orchestration has assembled the candidate set. Orchestration validates and
applies that decision.
