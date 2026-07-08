# Model I/O Reference

This page is the living implementation reference for the current
provider-neutral inputs and outputs of the model-role components:

- world model tool `S`
- goal model tool `G`
- orchestrator agent `X`
- updater `P`, split into world, goal, agent, and shared general update tasks

It intentionally does not enumerate backend-specific compact prompt variants,
provider request options, or provider metadata fields. Backend-specific
world/goal prompt variants such as `instruction_prompt_flux_kontext.md` and
`instruction_prompt_instruct_pix2pix.md` are out of scope here.

## Source Files

Primary contracts:

- `src/face_of_agi/contracts.py`
- `src/face_of_agi/models/tools/world/contracts.py`
- `src/face_of_agi/models/tools/goal/contracts.py`
- `src/face_of_agi/models/orchestrator_agent/contracts.py`
- `src/face_of_agi/models/updater/contracts.py`

Prompt and instruction assembly:

- `src/face_of_agi/models/tools/world/providers/openai.py`
- `src/face_of_agi/models/tools/world/providers/huggingface.py`
- `src/face_of_agi/models/tools/goal/providers/openai.py`
- `src/face_of_agi/models/tools/goal/providers/huggingface.py`
- `src/face_of_agi/models/orchestrator_agent/tooling.py`
- `src/face_of_agi/models/updater/adapter.py`
- `src/face_of_agi/models/updater/config.py`
- `src/face_of_agi/models/updater/providers/openai.py`
- `src/face_of_agi/models/updater/providers/ollama.py`

Human-editable instruction files:

- `src/face_of_agi/models/tools/world/instructions/instruction_prompt.md`
- `src/face_of_agi/models/tools/goal/instructions/instruction_prompt.md`
- `src/face_of_agi/models/orchestrator_agent/instructions/system_prompt.md`
- `src/face_of_agi/models/updater/instructions/world_game_context_updater_prompt.md`
- `src/face_of_agi/models/updater/instructions/goal_game_context_updater_prompt.md`
- `src/face_of_agi/models/updater/instructions/agent_game_context_updater_prompt.md`
- `src/face_of_agi/models/updater/instructions/world_general_context_updater_prompt.md`
- `src/face_of_agi/models/updater/instructions/goal_general_context_updater_prompt.md`
- `src/face_of_agi/models/updater/instructions/agent_general_context_updater_prompt.md`

## Shared Data Shapes

`RoleContext` is the mutable context document passed to model roles:

- `general`: game-agnostic context `K^m`
- `game`: game-specific context `L^m`
- `composed()`: `K^m + L^m`, joined with a blank line when both parts exist

`ActionSpec` is one candidate, proposed, or submitted action:

- `action_id`: ARC `GameAction` or string action id
- `data`: optional structured payload for complex actions
- `name`: stable display string derived from `action_id`
- `NONE`: internal synthetic action for non-controllable animation frames

`Observation` is a resolved observed frame or frame bundle:

- `id`
- `step`
- `frame`
- `frames`
- `raw_frame_data`
- `metadata`
- `frame_count()`

`ObservationRef` is the reusable reference form:

- `memory`: `state` for persistent `M`, or `experimental` for rolling `E`
- `id`: record or observation id inside that memory domain

`ToolResult` is the provider-neutral output of world or goal tools:

- `id`
- `tool`: `world` or `goal`
- `predicted_observation`
- `source_observation_ref`
- `action`: set for world results, unset for goal results
- `explanation`
- `metadata`

## World Model Tool `S`

Contract:

```python
WorldToolModel.predict(
    context: RoleContext,
    action: ActionSpec,
    observation: Observation,
) -> ToolResult
```

### Inputs

Contract input fields:

- `context`: world role context. Prompt-backed implementations use
`context.composed()` as `WORLD MODEL DOC (K^S + L^S)`.
- `action`: proposed candidate action. The world model is action-conditioned.
- `observation`: resolved source observation. Orchestration resolves any
agent-facing `ObservationRef` before calling `S`.

Invocation input before this model contract:

- `ToolCall.tool`: `world`
- `ToolCall.observation_ref`: source ref requested by `X`, or selected by
orchestration for committed post-decision predictions
- `ToolCall.action`: required action for the world prediction

The current `ToolRouter` resolves `observation_ref` outside the model role and
passes only `context`, `action`, and the resolved `observation` into `S`.

Prompt request sent to the model provider:

- fixed instruction text loaded from
`src/face_of_agi/models/tools/world/instructions/instruction_prompt.md`
- composed world context text, or `(no game-specific world context supplied)`
when empty
- source observation metadata:
  - `id`
  - `step`
  - `frame_count`
- proposed action metadata:
  - `action_id`
  - deterministic JSON `data`, or `{}`
- source observation image converted from `Observation`

Standard long-context prompt body:

```text
<world instruction prompt>

WORLD MODEL DOC (K^S + L^S):
<context.composed() or fallback>

SOURCE OBSERVATION:
id: <observation.id>
step: <observation.step>
frame_count: <observation.frame_count()>

PROPOSED ACTION:
action_id: <action name>
data: <action data as JSON or {}>
```

### Outputs

The world model returns `ToolResult`:

- `id`: generated world result id, currently prefixed with `world-`
- `tool`: `world`
- `predicted_observation`: predicted next visual observation
- `source_observation_ref`: source observation ref at the model boundary
- `action`: the proposed `ActionSpec`
- `explanation`: optional text explanation
- `metadata`: opaque provider metadata

When this result comes from an Agent X tool call, orchestration also writes the
predicted observation to rolling `E` and returns an
`ExperimentToolInvocationResult` containing the `ToolResult`, the new
experimental `ObservationRef`, and the `EExperimentRecord`.

### Instruction Text

Path:
`src/face_of_agi/models/tools/world/instructions/instruction_prompt.md`

```text
# World Model Instruction

You are the world model tool for an ARC-AGI-3 game agent.

Your task is to predict the next visual observation after the proposed action is
applied to the supplied current observation image. Treat the input image as the
authoritative current game state. Use the world context as working hypotheses
about the game's transition rules, objects, controls, and visual conventions.

Preserve all visual details that should not change. Change only what should
change because of the proposed action and the current transition hypotheses.
When the action has coordinates, interpret them with a top-left origin where
`(0, 0)` is the top-left pixel or cell. Do not invent goals or rewards here.
Return the best predicted next observation image.
```

## Goal Model Tool `G`

Contract:

```python
GoalToolModel.predict(
    context: RoleContext,
    observation: Observation,
) -> ToolResult
```

### Inputs

Contract input fields:

- `context`: goal role context. Prompt-backed implementations use
`context.composed()` as `GOAL MODEL DOC (K^G + L^G)`.
- `observation`: resolved source observation. Orchestration resolves any
agent-facing `ObservationRef` before calling `G`.

Invocation input before this model contract:

- `ToolCall.tool`: `goal`
- `ToolCall.observation_ref`: source ref requested by `X`, or selected by
orchestration for committed post-decision predictions

The current `ToolRouter` resolves `observation_ref` outside the model role and
passes only `context` and the resolved `observation` into `G`.

Prompt request sent to the model provider:

- fixed instruction text loaded from
`src/face_of_agi/models/tools/goal/instructions/instruction_prompt.md`
- composed goal context text, or `(no game-specific goal context supplied)`
when empty
- source observation metadata:
  - `id`
  - `step`
  - `frame_count`
- source observation image converted from `Observation`

Standard long-context prompt body:

```text
<goal instruction prompt>

GOAL MODEL DOC (K^G + L^G):
<context.composed() or fallback>

SOURCE OBSERVATION:
id: <observation.id>
step: <observation.step>
frame_count: <observation.frame_count()>
```

### Outputs

The goal model returns `ToolResult`:

- `id`: generated goal result id, currently prefixed with `goal-`
- `tool`: `goal`
- `predicted_observation`: predicted goal-relevant visual observation
- `source_observation_ref`: source observation ref at the model boundary
- `action`: unset
- `explanation`: optional text explanation
- `metadata`: opaque provider metadata

When this result comes from an Agent X tool call, orchestration also writes the
predicted observation to rolling `E` and returns an
`ExperimentToolInvocationResult` containing the `ToolResult`, the new
experimental `ObservationRef`, and the `EExperimentRecord`.

### Instruction Text

Path:
`src/face_of_agi/models/tools/goal/instructions/instruction_prompt.md`

```text
# Goal Model Instruction

You are the goal model tool for an ARC-AGI-3 game agent.

Your task is to predict the next visual observation after the agent takes the
best goal-directed action available from the supplied current observation
image. Treat the input image as the authoritative current game state. Use the
goal context as working hypotheses about the game's objective, progress
evidence, reward evidence, and success or failure conditions.

Infer what action an effective agent would choose if it were trying to make
immediate progress toward the current objective hypothesis. Then return the
single next observation frame that would most likely result from that optimal
goal-directed action. Do not merely reproduce the source frame unless no
controllable object, objective, or progress direction can be inferred.

Prefer the smallest clear progress edit: usually one legal-looking move toward
the inferred goal, exit, target, collectible, success region, or other
goal-relevant state. Preserve grid alignment, object identity, colors, walls,
obstacles, and all visual details that should not change after one optimal
action. If multiple objectives are plausible, choose the action that best
improves progress under the goal context.

Return the best next observation image after that optimal goal-directed action.
```

## Orchestrator Agent `X`

Contract:

```python
OrchestratorAgentModel.decide(
    context: RoleContext,
    first_observation: Observation,
    current_observation: Observation,
    action_space: Sequence[ActionSpec],
    tool_runtime: AgentToolRuntime | None = None,
    previous_observation: Observation | None = None,
) -> DecisionResult
```

### Inputs

Contract input fields:

- `context`: agent role context. Prompt-backed implementations use
`context.composed()` as `role_context`.
- `first_observation`: first real observation for the current game.
- `previous_observation`: immediately preceding X frame-turn observation, or
`None` on the first turn.
- `current_observation`: current frame-turn observation.
- `action_space`: authoritative set of actions that may be submitted this turn.
- `tool_runtime`: orchestration-owned tool interface, if tools are available.

Additional values exposed through `AgentToolRuntime`:

- `turn_id`
- `first_observation_ref`
- `previous_observation_ref`
- `current_observation_ref`
- `available_observation_refs()`
- `recent_action_history()`
- `available_tools()`
- `tool_metadata()`
- `invoke(call, metadata=None)`

Each `ActionHistoryEntry` visible to `X` contains:

- `turn_id`
- `step`
- `frame_index`
- `frame_count`
- `observation_ref`
- `action`
- `controllable`
- `control_reason`
- optional compact `reasoning_summary`

### Prompt Payload

Prompt-backed implementations load fixed system instructions from
`src/face_of_agi/models/orchestrator_agent/instructions/system_prompt.md`.

They then build a JSON user prompt with these fields:

- `role_context`: `context.composed()`
- `first_observation`: observation metadata for the first observation
- `previous_observation`: observation metadata for the previous observation, or
`null`
- `current_observation`: observation metadata for the current observation
- `image_order`: role and observation id for each attached image
- `allowed_actions`: serialized `ActionSpec` values from `action_space`
- `recent_action_history`: compact prior decision records from
`tool_runtime.recent_action_history()`, or `[]`
- `visible_observation_refs`: refs from
`tool_runtime.available_observation_refs()`, or `[]`
- `tool_policy`: `tool_runtime.tool_metadata()` plus normalized
`available_tools` and `tools_enabled`

Behavioral rules and prompt-field interpretation live in the loaded system
instructions, not in a duplicate JSON prompt field.

Observation metadata fields in the JSON prompt:

- `id`
- `step`
- `frame_count`
- JSON-safe `metadata`

Action payload fields in the JSON prompt:

- `action_id`
- `data`
- `requires_data`

Attached images:

- first observation image, unless duplicated with current
- previous observation image, unless missing or duplicated with current or first
- current observation image, always attached and always last

### Native Tool Calls And Feedback

World call arguments:

```json
{
  "observation_ref": {"memory": "state|experimental", "id": "<id>"},
  "action": {"action_id": "<allowed action>", "data": null}
}
```

Goal call arguments:

```json
{
  "observation_ref": {"memory": "state|experimental", "id": "<id>"}
}
```

Tool feedback returned into the active `X` conversation after orchestration
executes a world or goal call:

- JSON text with:
  - `tool`
  - `observation_ref`: the new experimental ref for the tool output
- the predicted observation image when the provider path can attach it

The reusable identity exposed back to `X` is the orchestration-created
`ObservationRef(memory="experimental", id=<experiment record id>)`, not the raw
tool result id.

Repair input appended after an invalid provider response:

```text
Invalid response: <error>. Repair by using native tools only. Allowed final actions: <allowed action names>.
```

Terminal action call arguments:

```json
{
  "action": {"action_id": "<allowed action>", "data": null},
  "reasoning_summary": "<short summary>"
}
```

For complex actions, `data` must include integer `x` and `y` in the inclusive
range `0..63`.

### Outputs

`X` returns `DecisionResult`:

- `final_action`: one validated `ActionSpec` from the current `action_space`
- `trace`: `AgentTrace`

`AgentTrace` contains:

- `step`
- `first_observation_ref`
- `current_observation_ref`
- `final_action`
- `tool_calls`
- `tool_results`
- `reasoning_summary`
- `metadata`

Tool calls made during deliberation are not outputs to the environment. They
are routed through orchestration, resolved against `M` or `E`, stored when
needed, and then included in the final trace.

### Instruction Text

Path:
`src/face_of_agi/models/orchestrator_agent/instructions/system_prompt.md`

```text
# Orchestrator Agent Instruction

You are model role X for an ARC-AGI-3 agent.

Choose one valid action for the current frame. You may call the world and goal
tools only when they are listed as available for this frame. Tool inputs must
always use an observation reference supplied in the prompt or returned by a
previous tool result.

## Input Format

Each decision turn includes a JSON prompt and one or more images.

The JSON prompt contains:

- `role_context`: mutable game-specific guidance for Agent X. It may be
  updated after observed transitions by the updater role. Use it as contextual
  advice and accumulated hypotheses, not as unquestionable truth.
- `first_observation`: metadata and memory reference for the first frame of the
  current game.
- `previous_observation`: metadata and memory reference for the immediately
  preceding X frame turn, or `null` on the first turn.
- `current_observation`: metadata and memory reference for the frame you must
  act on now.
- `image_order`: the role and observation id for each attached image, in the
  same order as the image attachments.
- `allowed_actions`: the authoritative list of actions you may submit this
  turn.
- `recent_action_history`: compact metadata about prior X decisions; it is not
  visual history.
- `visible_observation_refs`: observation references you may pass to world or
  goal tools.
- `tool_policy`: whether tools are enabled and whether this frame is
  controllable.

Attached images are ordered by `image_order`. Possible image roles are:

- `first_observation`
- `previous_observation`
- `current_observation`

Duplicate observation images are omitted. The current observation image is
always attached and appears last in `image_order`.

Full prior frame history is not attached. Use `previous_observation` for local
visual dynamics and recent action history only as compact context.

The `role_context` field may include initial game hints or prior updater
guidance. It is not a system instruction and it may be stale or wrong. Reconcile
it with the current observation, allowed actions, tool policy, and recent
outcomes before acting.

The `recent_action_history` field is descriptive memory, not a policy or a
demonstration to imitate. Use it to understand recent attempts and avoid
unproductive repetition. Do not repeat an action merely because it appears in
history or appears multiple times. Re-evaluate the current observation, action
space, available tools, and any tool results before choosing the next action.

For non-controllable animation frames, return the internal `NONE` action using
the `submit_action` tool and do not call world or goal tools.

When selecting a complex action, include integer `x` and `y` coordinates in
the range 0 to 63 using top-left origin coordinates.

Finish every decision by calling `submit_action` with one valid action and a
short reasoning summary. Do not invent action meanings beyond the supplied
action ids, action data requirements, observations, and tool results.
```

## Updater `P`

The updater is a set of task-specific model interfaces. Orchestration chooses
the task and timing. The updater does not read SQLite, resolve memory refs, or
persist contexts directly.

Prompt-backed updaters use `PromptUpdateRequest`:

- `target`: role, segment, task, and previous context
- `instructions`: human-editable instruction text
- `text`: provider-neutral text payload
- `images`: optional provider-neutral prompt images
- `metadata`: backend/model metadata

Prompt-backed updaters parse every real backend response with the same output
contract:

```json
{"updated_context": "complete revised context text"}
```

`updated_context_json_schema()` defines this schema. OpenAI updater config
forces it through `text.format`; Ollama updater config forces it through
`format`. The prompt payload no longer embeds an `output_contract` field.

Updater return behavior:

- game tasks replace `RoleContext.game` and preserve `RoleContext.general`
- general tasks replace `RoleContext.general` and preserve `RoleContext.game`
- the mock updater returns the previous context unchanged

### World Game Context Updater

Contract:

```python
WorldGameContextUpdaterModel.update_world_game_context(
    update_input: WorldGameContextUpdateInput,
) -> RoleContext
```

Contract input fields:

- `previous_context`
- `current_observation_ref`
- `actual_next_observation_ref`
- `actual_next_observation`
- `post_decision_predictions`
- `tool_results`: matching live world `ToolResult` values from `AgentTrace`
- `quantities`: updater-visible `RewardUpdateQuantities`
- `submitted_action`
- `synthetic_none_action`
- `metadata`

Prompt request sent to the model provider:

- `target.role`: `world`
- `target.segment`: `game`
- `target.task`: `world_game`
- `instructions`: loaded from
`src/face_of_agi/models/updater/instructions/world_game_context_updater_prompt.md`
- `text`:

```text
Previous world context:
<previous_context.game>

Action:
<submitted_action or synthetic_none_action>
```

- image `predicted_future_frame`: from
`post_decision_predictions.world_prediction.predicted_observation`
- image `actual_next_frame`: from `actual_next_observation.frame`, or the last
item in `actual_next_observation.frames`

Returned output:

- `RoleContext(general=previous_context.general, game=updated_context)`

Instruction path:
`src/face_of_agi/models/updater/instructions/world_game_context_updater_prompt.md`

```text
# World Game Context Updater Prompt

You recieve two frames from a game, and a corresponding action.
- the first frame was the prediction of the future frame, according given the action.
- the second frame is the actual real frame that resulted from the action.

Based on this state you are responsible to update the world description to help an agent understand how the game behaves according to the possible set of actions.
```

### Goal Game Context Updater

Contract:

```python
GoalGameContextUpdaterModel.update_goal_game_context(
    update_input: GoalGameContextUpdateInput,
) -> RoleContext
```

Contract input fields:

- `previous_context`
- `current_observation_ref`
- `actual_next_observation_ref`
- `actual_next_observation`
- `post_decision_predictions`
- `tool_results`: matching live goal `ToolResult` values from `AgentTrace`
- `quantities`: updater-visible `RewardUpdateQuantities`
- `submitted_action`
- `synthetic_none_action`
- `metadata`

Prompt request sent to the model provider:

- `target.role`: `goal`
- `target.segment`: `game`
- `target.task`: `goal_game`
- `instructions`: loaded from
`src/face_of_agi/models/updater/instructions/goal_game_context_updater_prompt.md`
- `text`:

```text
Previous goal context:
<previous_context.game>
```

- image `predicted_goal_frame`: from
`post_decision_predictions.goal_prediction.predicted_observation`
- image `actual_next_frame`: from `actual_next_observation.frame`, or the last
item in `actual_next_observation.frames`

Returned output:

- `RoleContext(general=previous_context.general, game=updated_context)`

Instruction path:
`src/face_of_agi/models/updater/instructions/goal_game_context_updater_prompt.md`

```text
# Goal Game Context Updater Prompt

You recieve two frames from a game, and a goal description.
- the first frame was the prediction of the future frame, according to the given goal description.
- the second frame is the frame that resulted from an agent action.
- this agent is responsible to either follow the goal or explore actions not necessary following the goal.

Based on this state you are responsible to update the goal description to help the agent solve the game.
```

### Agent Game Context Updater

Contract:

```python
AgentGameContextUpdaterModel.update_agent_game_context(
    update_input: AgentGameContextUpdateInput,
) -> RoleContext
```

Contract input fields:

- `previous_context`: previous agent `RoleContext`
- `previous_observation`: observed frame before Agent X's last action, `o_t`
- `current_observation`: observed frame after that action, `o_t+1`
- `trace`: full live `AgentTrace`
- `quantities`: updater-visible `RewardUpdateQuantities`

Prompt request sent to the model provider:

- `target.role`: `agent`
- `target.segment`: `game`
- `target.task`: `agent_game`
- `instructions`: loaded from
`src/face_of_agi/models/updater/instructions/agent_game_context_updater_prompt.md`
- `text`: JSON generated from:
  - `task`: `agent_game`
  - `role`: `agent`
  - `segment`: `game`
  - `current_context`: `previous_context.game`
  - `previous_context`: full previous `RoleContext`
  - `transition`: JSON-safe `AgentGameContextUpdateInput`
  - `attached_images`: metadata mapping image labels back to transition fields
- images:
  - `previous_observation_frame`, from `previous_observation.frame` or last
  `previous_observation.frames`
  - `current_observation_frame`, from `current_observation.frame` or last
  `current_observation.frames`

Frame-like payloads embedded in JSON are summarized for text prompts. Raw
base64 frame data is omitted in `transition`; the actual frames are sent as
attached images.

Returned output:

- `RoleContext(general=previous_context.general, game=updated_context)`

Instruction path:
`src/face_of_agi/models/updater/instructions/agent_game_context_updater_prompt.md`

```text
# Agent Game Context Updater Prompt

You update the game-specific context document `L^X` for the orchestrator agent
`X`.

You receive the previous agent context, the live decision trace, deterministic
reward/update quantities computed by orchestration, and two attached observed
frames from the last transition:

- `previous_observation_frame`: the frame before Agent X's last action, `o_t`;
- `current_observation_frame`: the frame observed after that action, `o_t+1`.

The live `AgentTrace` contains the final action and any world/goal tool calls
and results Agent X used while deciding. Treat the attached observation images
as the authoritative visual before/after evidence. Do not produce generic
image-chat instructions; your job is only to revise Agent X's game-specific
context.

You own the full mutable `L^X` text. The current context may have been seeded
from cheat action context at startup, but it is now ordinary updater-owned
context. You may preserve, rewrite, shorten, or remove any part of it when the
transition evidence supports doing so. Do not quote or embed the whole previous
context as JSON; rewrite it as concise guidance for Agent X.

Return only a JSON object:

```json
{"updated_context": "revised agent game-context text"}
```

Reward/update quantities are evidence, not a scalar reward:

- `prediction_error_delta`: previous real-step prediction error minus current
  prediction error. Positive values mean the world model predicted the latest
  outcome better than the prior real step; negative values mean prediction got
  worse. Maximize this value to push the rate of world-model improvement
  higher. This is an improvement delta, not raw prediction error.
- `goal_distance`: normalized visual distance between the committed goal
  prediction and the current observed frame. Higher values mean the agent is less
  aligned with the inferred goal and is exploring more; lower values mean it is
  following the goal more closely.
- `time_cost`: cumulative real environment steps spent in this game. It rises
  as the agent spends more actions, increasing pressure to improve performance
  rather than continue unproductive exploration.
- `trace_cost`: measured wall-clock seconds spent inside the full Agent X
  decision call, including any requested tool calls.
- `score_delta`: change in completed levels when lifecycle counters expose it.
  Positive values are direct progress evidence.
- `notes`: optional implementation notes from reward computation.

Use the evidence to improve how Agent X should act in this specific game. Good
updates usually clarify:

- tool-use policy: when world or goal calls helped, when they were wasteful,
  and which comparisons should be requested next;
- action choice: which action patterns appeared useful, harmful, or uncertain;
- exploration versus exploitation: whether to test unknown mechanics or repeat
  a successful strategy;
- hypotheses: current beliefs about what the agent is trying to achieve and
  what observations support or contradict them;
- internal reward interpretation: how reward evidence should change the next
  behavioral guidance you write for Agent X;
- failure handling: what to avoid on the next frame turn.

Optimization direction for your interpretation: minimize `goal_distance`,
`time_cost`, and `trace_cost`; maximize `prediction_error_delta` so the agent
increasingly chooses actions that improve world-model prediction quality.
Balance learning against the need to make goal progress as cumulative time cost
grows.

Do not copy this reward glossary, metric names, metric values, model-quality
phrases, or generic optimization objective into the revised context. Agent X
does not need reward-component definitions. Avoid abstract reward phrasing such as
"reward", "metric", "goal distance", "visual distance", "time cost",
"trace cost", "prediction improvement", "prediction error", "world model
improvement", "model ability", "minimize", or "maximize". It is acceptable to
give tool-use advice about when to call world or goal tools, but do not discuss
reward quantities or prediction-quality objectives. Convert reward evidence
into concrete game guidance such as which action to try, which pattern to
avoid, whether to explore or exploit, and when tool use is worth the latency.
For example, write "try a different direction if repeated right moves stop
changing the useful state" instead of "maximize prediction improvement", or
"avoid spending many turns on repeated moves that do not change the scene"
instead of "minimize time cost". If
`prediction_error_delta` is negative, translate that internally as evidence
that the last action made outcomes harder to predict; do not describe it as an
improvement.

Keep useful prior context, remove stale or contradicted advice, and keep the
revised context concise enough to fit in future Agent X prompts. Do not invent
observations or outcomes that are not supported by the observed frames, trace,
or reward quantities. If the evidence is weak, preserve the
previous context and add at most one concrete, evidence-grounded note. Never
return placeholders such as "No update provided."
```

### World General Context Updater

Contract:

```python
GeneralKnowledgeUpdaterModel.update_general_knowledge(
    update_input: GeneralKnowledgeUpdateInput(role="world", ...),
) -> RoleContext
```

Contract input fields:

- `role`: `world`
- `previous_context`: previous world `RoleContext`, including final `L^S` and
current `K^S`
- `run_id`
- `game_id`
- `stop_reason`
- `step_count`
- `completed_levels`
- `final_state`
- `state_record_ids`
- `metadata`

Prompt request sent to the model provider:

- `target.role`: `world`
- `target.segment`: `general`
- `target.task`: `general`
- `instructions`: loaded from
`src/face_of_agi/models/updater/instructions/world_general_context_updater_prompt.md`
- `text`:

```text
Game world model text:
<previous_context.game>

General world model text:
<previous_context.general>
```

Returned output:

- `RoleContext(general=updated_context, game=previous_context.game)`

Instruction path:
`src/face_of_agi/models/updater/instructions/world_general_context_updater_prompt.md`

```text
# World model
You are given 2 texts:
- 1 describing the world model of a game
- 1 discribing how to figure out a world model in general for unknown games.

You should update the general worl model text in order to include mechanics from this game, things that you can identify and generalize.
```

### Goal General Context Updater

Contract:

```python
GeneralKnowledgeUpdaterModel.update_general_knowledge(
    update_input: GeneralKnowledgeUpdateInput(role="goal", ...),
) -> RoleContext
```

Contract input fields:

- `role`: `goal`
- `previous_context`: previous goal `RoleContext`, including final `L^G` and
current `K^G`
- `run_id`
- `game_id`
- `stop_reason`
- `step_count`
- `completed_levels`
- `final_state`
- `state_record_ids`
- `metadata`

Prompt request sent to the model provider:

- `target.role`: `goal`
- `target.segment`: `general`
- `target.task`: `general`
- `instructions`: loaded from
`src/face_of_agi/models/updater/instructions/goal_general_context_updater_prompt.md`
- `text`:

```text
Game goal model text:
<previous_context.game>

General goal model text:
<previous_context.general>
```

Returned output:

- `RoleContext(general=updated_context, game=previous_context.game)`

Instruction path:
`src/face_of_agi/models/updater/instructions/goal_general_context_updater_prompt.md`

```text
# Goal model
You are given a 2 texts:
- 1 describing the goal model of a game.
- 1 discribing how to figure out a goal in general for unknown games.

You should update the general goal text in order to include mechanics from this game, things that you can identify and generalize.

```

### Agent General Context Updater

Contract:

```python
GeneralKnowledgeUpdaterModel.update_general_knowledge(
    update_input: GeneralKnowledgeUpdateInput(role="agent", ...),
) -> RoleContext
```

Contract input fields:

- `role`: `agent`
- `previous_context`: previous agent `RoleContext`, including final `L^X` and
current `K^X`
- `run_id`
- `game_id`
- `stop_reason`
- `step_count`
- `completed_levels`
- `final_state`
- `state_record_ids`
- `metadata`

Prompt request sent to the model provider:

- `target.role`: `agent`
- `target.segment`: `general`
- `target.task`: `general`
- `instructions`: loaded from
`src/face_of_agi/models/updater/instructions/agent_general_context_updater_prompt.md`
- `text`: JSON generated from:
  - `task`: `general`
  - `role`: `agent`
  - `segment`: `general`
  - `current_context`: `previous_context.general`
  - `previous_context`: full previous `RoleContext`
  - `transition`: JSON-safe `GeneralKnowledgeUpdateInput`
- images: none

Returned output:

- `RoleContext(general=updated_context, game=previous_context.game)`

Instruction path:
`src/face_of_agi/models/updater/instructions/agent_general_context_updater_prompt.md`

```text
# Agent model
You are given 2 texts:
- 1 describing the agent strategy for a game
- 1 discribing how to act in general for unknown games.

You should update the general agent text in order to include strategies from this game, things that you can identify and generalize.
```

## Updater Transition Inputs

`UpdaterFrameTransitionInput` is orchestration's transition-level input before
it fans out into the role-specific updater contracts. It contains:

- `current_observation_ref`
- `actual_next_observation_ref`
- `decision_trace`
- `previous_observation`
- `actual_next_observation`
- `post_decision_predictions`
- `reward_update_quantities`
- `submitted_action`
- `synthetic_none_action`
- `metadata`

World and goal game updaters receive refs, post-decision predictions, selected
actions, matching live tool results, metadata, and reward/update quantities.
Agent game updater receives only the previous/current observed frames, the
trace, previous context, and reward/update quantities.

`RewardUpdateQuantities` may contain:

- `prediction_error`
- `prediction_error_delta`
- `goal_distance`
- `time_cost`
- `trace_cost`
- `score_delta`
- `notes`

Before updater prompts see quantities, orchestration uses
`updater_visible_quantities(...)` so raw `prediction_error` is hidden from
updater models while `prediction_error_delta` remains visible.

`PostDecisionPredictions` may contain:

- `world_prediction`: committed world `ToolResult`
- `goal_prediction`: committed goal `ToolResult`

The updater receives these objects from orchestration. It does not choose when
to run, and it does not decide whether a returned context is durable. After the
updater returns, orchestration applies the returned `RoleContext` to the live
`ContextDocuments` and persists the authoritative state into `M`.
