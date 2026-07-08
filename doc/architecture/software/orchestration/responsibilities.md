# Orchestration Responsibilities

## Owns

- Main running loop for one game run.
- Turn-by-turn communication with the ARC-AGI environment adapter.
- Focused sub-orchestration components for concrete workflows, such as the
  game-loop state machine.
- Reading memory-backed observations, description predictions, and context for
  model roles.
- Calling the orchestrator agent model `X`.
- Providing a controlled per-turn `AgentToolRuntime` extension point to `X`.
- Persisting model outputs, tool results, traces, actions, observations,
  transition timing, and score/progress metadata.
- Applying updater-returned contexts to the live working context documents
  before those contexts are committed to memory.
- Keeping callable source-state ids internal to orchestration and memory.
- Keeping rolling `E` artifacts as non-callable experiment/debug records.
- Invoking updater `P` after a real environment step.
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

Models may return description predictions, explanations, traces, actions, or
context updates. Orchestration decides where those outputs go, persists them,
and resolves later references to them.

No model adapter writes directly to or reads directly from SQLite in the target
architecture.

Action choice lives behind the configured `X` adapter. Orchestration validates
and applies that decision; it does not choose the action directly.
