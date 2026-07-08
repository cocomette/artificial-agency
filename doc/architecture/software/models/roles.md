# Model Roles

## Orchestrator Agent `X`

`X` is the decision-making agent. It receives the current context,
observations, action space, and callable tools. It may call world and goal
tools any number of times before returning one final real action.

`X` can build experimental tree paths by reusing memory references. It may ask
orchestration to call `S` or `G` from a current observation, a past real state
in `M`, or a prior prediction in `E`.

### Tool Runtime Framework

`X` does not call model tools, read memory, or write SQLite directly. During
each frame turn, orchestration passes `X` an `AgentToolRuntime`. This is the
only tool interface visible to the agent.

The intended loop is:

1. Orchestration builds the frame-turn input for `X`: agent context, first
   observation, current observation, action space, and `AgentToolRuntime`.
2. `X` reasons over the visible observations and references.
3. If `X` wants an experiment, it emits a `ToolCall` and invokes the provided
   runtime.
4. The runtime delegates back to orchestration.
5. Orchestration resolves the source `ObservationRef` from `M`, `E`, or the
   current in-flight frame, calls the world or goal tool, persists the output in
   `E`, and returns a `ToolResult` plus `ObservationRef(memory="experimental",
   id=...)`.
6. `X` may add that result and reference to its active reasoning context and
   request more tool calls.
7. `X` finally returns one `DecisionResult` containing a final action and
   `AgentTrace`.

This mirrors standard tool-calling agent patterns: the model requests a tool,
the deterministic runtime validates and executes it, then the result is fed
back to the model. The difference in this project is that orchestration also
owns memory reference resolution and `E` persistence before a tool result can
be reused.

OpenAI and Ollama use the same provider-neutral Agent X loop. That loop owns
tool-call budgets, repair attempts, final-action parsing, tool invocation
through `AgentToolRuntime`, and `AgentTrace` construction. Provider adapters
translate only the normalized turn request, provider function calls, and tool
feedback. The current default `random` provider remains an empty shell and
does not invoke tools.

Output:

- final `ActionSpec`
- full `AgentTrace`

## World Tool `S`

`S` predicts how the environment may change after a candidate action from a
referenced observation.

The current concrete world-tool providers use image-editing backends. The role
adapter receives the framework input, provider code translates it into the
backend prompt/request shape, and the role returns a provider-neutral
`ToolResult`.

Output:

- `ToolResult` containing a predicted observation and optional explanation

## Goal Tool `G`

`G` reasons about objective hypotheses, progress, and goal-relevant outcomes
from a referenced observation and the current goal context. Unlike `S`, it is
not action-conditioned in the current tool contract.

Output:

- `ToolResult` containing a goal-relevant prediction or explanation

## Updater `P`

`P` runs after an observed transition. In the frame-unrolled game loop, that
means after each frame decision: animation-frame transitions compare the
current frame to the next buffered frame, while controllable final-frame
transitions compare the selected action against the first frame returned by
the next real environment step.

`P` updates game-specific context documents for the world, goal, and agent
roles. World and goal updates use the same input shape with a role
discriminator. Agent updates use a separate input shape because `X` is updated
from the full decision trace and reward/update quantities rather than a single
tool prediction stream.

Output:

- updated `L^S`, `L^G`, and `L^X` context documents

The updater does not own persistence. Its outputs return to orchestration,
which applies them to the live working `ContextDocuments` and persists the
resulting state into `M`. Future updater backends may revise text prompts,
trigger loss-based updates, or coordinate LoRA-style model updates while
preserving this boundary.

## Model Adapter Rule

Adapters translate between provider-specific calls and shared model contracts.
They do not own the runtime loop, environment stepping, or SQLite persistence.
They also do not read memory directly; memory access is mediated by
orchestration.

Provider-specific adapters live in `providers/` folders under each model role.
Shared provider utilities that are reused across roles live in
`models/providers/`.
