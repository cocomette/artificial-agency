# Orchestration Responsibilities

## Owns

- Main running loop for one game run.
- Turn-by-turn communication with the ARC-AGI environment adapter.
- Focused sub-orchestration components for concrete workflows, such as the
  game-loop state machine.
- Reading memory-backed observations and agent context.
- Calling Agent `X`, the change summary model, the agent context historizer,
  and updater `P`.
- Providing a controlled per-turn `AgentToolRuntime` to `X`.
- Persisting model outputs, traces, actions, observations, action history, and
  update quantities.
- Applying updater-returned contexts to live working context documents before
  those contexts are committed to memory.
- Owning the frame-unrolled game-loop state machine described in
  [`game_loop/overview.md`](game_loop/overview.md).

## Does Not Own

- ARC-AGI toolkit internals.
- Model backend implementation details.
- SQLite connection primitives.
- The content of provider prompts beyond composing stored context documents and
  deterministic runtime evidence.
- Rendering or visualization policy except as environment metadata passed
  through the loop.

## Central Rule

Models may return actions, traces, transition summaries, or context updates.
Orchestration decides where those outputs go, persists them, and resolves later
references to them.

No model adapter writes directly to or reads directly from SQLite.
