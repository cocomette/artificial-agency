# Model Roles

## Orchestrator Agent `X`

`X` is the decision-making agent. It receives its own mutable agent context in
the provider instruction/system content, and receives world/goal game
contexts, observations, action history, and action space in the turn payload.
It returns one final real action as structured output from the shared Agent X
step loop.

### Tool Runtime Framework

The `AgentToolRuntime` and provider tool-loop code remain available for future
agent tools. Provider requests receive tool specs only when orchestration
explicitly exposes them.

The current default loop is:

1. Orchestration builds the frame-turn input for `X`: agent context in the
   instruction/system message, plus world game context, goal game context,
   `history_anchor` frame, current frame, bounded recent action history, and
   action space in the turn payload.
2. The shared Agent X loop calls the provider with the same turn input, any
   available tool specs, and the final structured action schema.
3. If the provider returns tool calls, orchestration executes them, appends
   tool feedback, and the same loop continues until the tool-call budget is
   exhausted or a final output is returned.
4. If the provider returns final structured output without tool calls, the
   shared adapter validates the final action and builds `DecisionResult` plus
   `AgentTrace`.

OpenAI and Ollama use the same provider-neutral Agent X loop. That loop owns
repair attempts, final-action parsing, dormant tool-call budget config, and
`AgentTrace` construction. Provider adapters translate only the normalized turn
request, tool specs, and final structured action schema.

Output:

- final `ActionSpec`
- full `AgentTrace`

## World Prediction Model `S`

`S` predicts how the environment may change from the current frame and action.
Its game-specific context is updated by `P`, then fed back into later `S`
predictions and Agent `X` decisions.

The role adapter receives the framework input, provider code translates it into
the shared description provider request shape, and the role returns a
provider-neutral `PredictionResult` carrying `predicted_description`.

Output:

- `PredictionResult` containing a predicted description and optional explanation

## Goal Prediction Model `G`

`G` reasons about objective hypotheses, progress, and goal-relevant outcomes
from the current frame and goal context. Its game-specific context is updated
by `P`, then fed back into later `G` predictions and Agent `X` decisions.
Unlike `S`, it is not action-conditioned in the current contract.

Output:

- `PredictionResult` containing a goal-relevant predicted description or explanation

## Updater `P`

`P` runs after an observed transition. In the frame-unrolled game loop, that
means after each frame decision: animation-frame transitions compare the
current frame to the next buffered frame, while controllable final-frame
transitions compare the selected action against the first frame returned by
the next real environment step. World and goal game-context updates run for
both cases; animation-frame updates use the synthetic `NONE` action and the
S/G predictions for that frame.

`P` is wired as four updater tasks. Three role-specific game tasks update
`L^S`, `L^G`, and `L^X` during frame/game-loop transitions. One shared
general task updates role-specific `K` contexts at end-of-run and is invoked
separately for world, goal, and agent. Orchestration owns this timing and
chooses which task is called. Runtime config must declare each updater slot
explicitly; each slot chooses OpenAI or Ollama with an explicit model. The
shared general updater uses one backend/model configuration and role-specific
instruction prompts for world, goal, and agent `K` updates.

Output:

- updated `L^S`, `L^G`, and `L^X` during frame/game-loop updates
- updated `K^S`, `K^G`, and `K^X` at end-of-run through the shared general
  updater task

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

Provider-specific adapters live either under a role or under a shared
capability when multiple roles use the same concrete output contract. World and
goal share the `models/description` capability. Shared provider utilities that
are reused across roles live in `models/providers/`. The shared provider layer
is only the final provider-call boundary: role adapters build prompts,
conversations, and model-role results; shared provider helpers build/send
provider requests and normalize raw provider responses.
