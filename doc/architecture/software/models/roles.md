# Model Roles

## Orchestrator Agent `X`

`X` is the decision-making agent. It receives immutable role rules in provider
instruction/system content, and receives its mutable agent context,
observations, action history, and action space in the turn payload. It returns
one final real action as structured output from the shared Agent X step loop.

### Tool Runtime Framework

The `AgentToolRuntime` and provider tool-loop code remain available for future
agent tools. Provider requests receive tool specs only when orchestration
explicitly exposes them.

The current default loop is:

1. Orchestration builds the frame-turn input for `X`: agent context,
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

OpenAI, Ollama, and vLLM use the same provider-neutral Agent X loop. That loop
owns repair attempts, final-action parsing, dormant tool-call budget config,
and `AgentTrace` construction. Provider adapters translate only the normalized
turn request, tool specs, and final structured action schema.

Output:

- final `ActionSpec`
- full `AgentTrace`

## World Prediction Model `S`

`S` predicts how the environment may change from the current frame and action.
Its game-specific context is updated by `P`, then fed back into later `S`
predictions. The agent game-context updater summarizes relevant world context
into `L^X` for later Agent `X` decisions.

The role adapter receives the framework input, provider code translates it into
the shared description provider request shape, and the role returns a
provider-neutral `PredictionResult` carrying `predicted_description`.

Output:

- `PredictionResult` containing a predicted description and optional explanation

## Goal Prediction Model `G`

`G` reasons about objective hypotheses, progress, and goal-relevant outcomes
from the current frame and goal context. The goal model and its updater source
code remain available as dormant role contracts, but the normal runtime loop
does not build or call them. Goal context may remain in persisted memory rows
for schema continuity, but it is not fed into Agent `X` or the agent updater.
Unlike `S`, the dormant goal contract is not action-conditioned.

Output:

- `PredictionResult` containing a goal-relevant predicted description or explanation

## Updater `P`

`P` runs after an observed transition. In the frame-unrolled game loop, that
means after each frame decision: animation-frame transitions compare the
current frame to the next buffered frame, while controllable final-frame
transitions compare the selected action against the first frame returned by
the next real environment step. The active world and agent game-context
updates run for both cases; animation-frame world updates use the synthetic
`NONE` action and the world prediction for that frame.

`P` is actively wired as world game, agent game, and shared general updater
tasks. The dormant goal game updater remains in the model package but is not
called by orchestration. The shared general task updates role-specific `K`
contexts at end-of-run and is invoked separately for world and agent only.
Runtime config must declare each active updater slot explicitly; goal updater
config is optional and ignored by normal runtime assembly.

Output:

- updated `L^S` and `L^X` during frame/game-loop updates
- updated `K^S` and `K^X` at end-of-run through the shared general updater
  task

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
