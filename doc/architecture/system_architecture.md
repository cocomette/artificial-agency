# ARC-AGI-3 Agent Architecture Plan

## Purpose

This document describes a high-level, swappable architecture for an ARC-AGI-3 agent based on the whiteboard design and provided description.

The core idea is to run an online-learning agent over a turn-based game environment. The environment returns visual frames as observations. The agent chooses actions. During each real environment step, the agent can call a world model and a goal model as tools, store temporary simulations in experimental memory, then emit one final action to apply to the game. After the environment advances, an updater model updates the game-specific context documents used by the world model, goal model, and agent.

The initial implementation should use VLMs and text context updates. The architecture must allow replacing any VLM with a custom neural network later, including loss-based LoRA updates.

Detailed target software modules and execution diagrams live in
[`software/overview.md`](software/overview.md).

## ARC-AGI-3 Environment Interface

ARC-AGI-3 games are turn-based interactive environments. Each game advances through discrete action-response cycles.

The software should assume:

- Observations are one or more frame objects per environment step.
- The usable visual observation is a 64x64 frame or grid-like image.
- Frame metadata can include game state and available actions.
- Actions are game-specific and should be read from the environment action space each step.
- Simple actions are represented by action IDs such as `ACTION1`, `ACTION2`, etc.
- A coordinate action, when available, is represented as a complex action with `x, y` coordinates.
- Coordinates use the top-left origin convention, with `(0, 0)` at the top-left.
- The environment wrapper exposes:
  - `reset()` for the initial observation.
  - `step(action, data=None, reasoning=None)` for applying a selected action.
  - `observation_space` for the last returned observation.
  - `action_space` for currently available actions.
  - `info` for environment metadata.
- If the game is over, the implementation should use reset rather than sending another ordinary action.

The agent architecture should not hard-code meanings such as up, down, left, right, click, jump, or fire. The available action list and metadata from the current frame are the authoritative interface.

## Core Notation

Indices:

- `i`: game index. It is used only for game-specific quantities.
- `t`: real environment step within game `i`.
- `k`: temporary tool-call or simulation index within one agent decision step.
- `m`: model role, where `m ∈ {S, G, X}`.

Model-role superscripts:

- `S`: world model.
- `G`: goal model.
- `X`: agent.

Core objects:

- `O_{i,t}`: real observation for game `i` at step `t`.
- `O_{i,0}`: first observation of game `i`.
- `A_{i,t}`: real action applied in game `i` at step `t`.
- `T^X_{i,t}`: full agent trace for game `i` at step `t`.
- `M_i`: persistent state memory for game `i`.
- `E_{i,t}`: temporary experimental memory used only during the agent decision at step `t`.
- `Q^X_{i,t}`: reward/update quantity packet passed to the agent updater. This is not collapsed into a final scalar reward.

Context notation:

```text
C^m_{i,t} = K^m + L^m_{i,t},     m ∈ {S, G, X}
```

Where:

- `K^m` is the game-agnostic context document for model role `m`. It has no game index `i` and is fixed while playing a single game.
- `L^m_{i,t}` is the game-specific context document for model role `m`, game `i`, and step `t`.
- `C^m_{i,t}` is the composed context passed to model role `m`.

Expanded forms:

```text
C^S_{i,t} = K^S + L^S_{i,t}
C^G_{i,t} = K^G + L^G_{i,t}
C^X_{i,t} = K^X + L^X_{i,t}
```

## Top-Level Components

### 1. Game Environment

External ARC-AGI-3 environment.

**Input:** one action from the agent.

**Output:** next observation frame or frames, plus metadata such as game state and available actions.

The environment is treated as the only source of ground-truth transitions.

### 2. Orchestration Layer

The central controller that connects the environment, memory, tools, agent, and updater.

Responsibilities:

- Reset and step the environment.
- Maintain state memory `M_i`.
- Maintain per-step experimental memory `E_{i,t}`.
- Compose model contexts `C^m_{i,t}` from fixed game-agnostic documents `K^m` and mutable game-specific documents `L^m_{i,t}`.
- Route tool calls from the agent to the world model `S` or goal model `G`.
- Return tool outputs immediately to the agent context.
- Submit the final agent action to the environment.
- Run the updater `P` after each real environment step.
- Reset experimental memory at the end of each real step.
- Update game-agnostic context `K^m` only after finishing a game.

### 3. Model Registry and Adapters

All models should be accessed through small adapter interfaces rather than direct calls.

Required model roles:

- World model `S`
- Goal model `G`
- Agent `X`
- Updater `P`

Initial backend:

- VLMs for all roles.
- VLM plus image generator for models that need to return image predictions.

Future backends:

- Custom neural networks.
- LoRA-updated versions of the same models.
- Hybrid models that combine VLM reasoning with learned predictors.

The orchestration layer should depend only on typed inputs and outputs, not on a specific model provider or architecture.

## Main Model Roles

### World Model `S`

The world model predicts how the environment changes.

Conceptual form:

```text
S(C^S_{i,t}, A, O_ref) -> O_hat^S
```

Where:

- `C^S_{i,t} = K^S + L^S_{i,t}`.
- `A` is a proposed action.
- `O_ref` is an observation reference resolved from state memory `M_i` or experimental memory `E_{i,t}`.
- `O_hat^S` is the predicted next observation.

The output should be stored as a prediction object, including the predicted frame or image and any text explanation needed by the updater.

### Goal Model `G`

The goal model represents hypotheses about the game objective and progress.

Conceptual form:

```text
G(C^G_{i,t}, A, O_ref) -> O_hat^G
```

Where:

- `C^G_{i,t} = K^G + L^G_{i,t}`.
- `A` is a proposed action.
- `O_ref` is an observation reference resolved from state memory `M_i` or experimental memory `E_{i,t}`.
- `O_hat^G` is a goal-relevant predicted or desired observation.

The goal model should support reasoning about what the agent is trying to achieve, why that hypothesis changed, and what visual or reward evidence supports it.

### Agent `X`

The agent is the decision model. It reasons, calls tools, chains tool calls, and eventually returns one real action.

Conceptual form:

```text
X(C^X_{i,t}, O_{i,0}, O_{i,t}, action_space_{i,t}, tools={S,G}) -> A_{i,t}, T^X_{i,t}
```

Where:

- `C^X_{i,t} = K^X + L^X_{i,t}`.
- `O_{i,0}` is the first observation frame of the current game.
- `O_{i,t}` is the current observation frame.
- `action_space_{i,t}` is the current set of valid environment actions.
- X also receives a bounded recent action history from prior frame turns:
  compact metadata with turn, step, observation reference, action,
  controllability, control reason, and optional reasoning summary.
- `S` and `G` are callable tools, not embedded logic.
- `A_{i,t}` is the final real action selected for the environment.
- `T^X_{i,t}` is the full agent trace for the current step.

The agent may call `S` and `G` any number of times before producing the final action. Each tool call must include:

```text
{
  tool: "world" | "goal",
  action: ActionSpec,
  observation_ref: ObservationRef
}
```

The `observation_ref` may point to:

- the first observation `O_{i,0}` in state memory,
- the current observation `O_{i,t}` in state memory,
- a past real observation in state memory,
- or an intermediate output in experimental memory `E_{i,t}`.

The agent immediately receives each tool result in its active context before deciding whether to call another tool or output the final action.
The result is also referenceable. This lets the agent build experimental tree
paths by asking orchestration to call `S` or `G` against prior prediction ids
instead of carrying the whole imagined path in context.
Every experiment-loop tool input must be a memory reference. `X` may see the
predicted frame in its active context, but the next tool call must pass the
reference id so orchestration can resolve the exact persisted frame from `M_i`
or `E_{i,t}`.

### Updater `P`

The updater performs online context updates after the real environment step.

It updates text context documents initially. Later, the same role can trigger loss-based updates such as LoRA.

World/goal update form:

```text
P(O_hat^m, O_{i,t+1}, L^m_{i,t}) -> L^m_{i,t+1},     m ∈ {S, G}
```

Agent update form:

```text
P(Q^X_{i,t}, T^X_{i,t}, L^X_{i,t}, O_{i,0}, O_{i,t}, O_{i,t+1}) -> L^X_{i,t+1}
```

Where:

- `O_hat^m` is a committed world-model or goal-model prediction produced by
  orchestration after the final action is selected.
- `O_{i,t+1}` is the real next observation.
- `Q^X_{i,t}` is the packet of reward/update quantities, not a final scalar reward.
- `T^X_{i,t}` is the full agent trace, including tool calls and final action.

At the software boundary, orchestration packages these quantities into
role-specific updater inputs: a shared world/goal context update input and a
separate agent context update input. These inputs are built from
orchestration-managed live transition objects and references. The updater does
not read or write memory directly; it returns updated context documents to
orchestration, which applies them to the working contexts and persists the
resulting authoritative state into `M_i`.

The updater should preserve the distinction between:

- Game-agnostic context `K^m`: updated only after finishing a game.
- Game-specific context `L^m_{i,t}`: updated during the game after each real environment step.

## Memory Design

### State Memory `M_i`

Persistent for the full game.

Stores:

```text
M_i = {
  observations: O_{i,0}...O_{i,t},
  actions: A_{i,0}...A_{i,t},
  world_predictions: O_hat^S,
  goal_predictions: O_hat^G,
  agent_traces: T^X_{i,0}...T^X_{i,t},
  reward_update_quantities: Q^X_{i,0}...Q^X_{i,t},
  contexts: {
    K^S, K^G, K^X,
    L^S_{i,0}...L^S_{i,t},
    L^G_{i,0}...L^G_{i,t},
    L^X_{i,0}...L^X_{i,t}
  }
}
```

State memory is the durable record of the game.

The implementation should allow observations and predictions to be referenced by ID rather than copied into every model context.

### Experimental Memory `E_{i,t}`

Temporary memory for simulations during one agent decision step.

Stores:

```text
E_{i,t} = {
  world_tool_outputs: O_hat^{S,E}_{i,t,0}...O_hat^{S,E}_{i,t,k},
  goal_tool_outputs: O_hat^{G,E}_{i,t,0}...O_hat^{G,E}_{i,t,k}
}
```

Experimental memory is reset after the real environment step and after the updater has used the relevant tool outputs.

This memory exists so the agent can chain simulations and refer to intermediate model outputs before committing to a real action.
The orchestrator agent can reuse `E_{i,t}` records by reference id to branch
from earlier world or goal predictions. It can also request experiments from
past real states stored in `M_i`; orchestration resolves those references and
keeps the distinction between real and imagined records.
Every world or goal tool output is persisted in `E_{i,t}` before it is exposed
as a reusable reference to the agent.

## Context Structure

Each model context is the concatenation of two documents:

```text
C^m_{i,t} = K^m + L^m_{i,t},     m ∈ {S, G, X}
```

### `K^m`: Game-Agnostic Context

Fixed during a single game. No game index `i` is used.

Examples of content:

- General instructions for the model role.
- General knowledge about how to interpret observations and actions.
- General policies for using predictions, goals, and tool calls.

Updated only after finishing a game.

### `L^m_{i,t}`: Game-Specific Context

Updated during a single game.

For `L^S_{i,t}`, store current hypotheses about how the game dynamics work, tested transitions, prediction failures, and revised mechanics.

For `L^G_{i,t}`, store current hypotheses about the objective, visible evidence for progress, reward evidence, and why the goal hypothesis changed.

For `L^X_{i,t}`, store current action and tool-use strategy, mistakes from prior steps, useful tool-call patterns, and current policy guidance.

## Data Contracts

### Observation

```text
Observation = {
  id: string,
  step: int,
  frame: image_or_grid_64x64,
  metadata: {
    game_state: optional,
    levels_completed: optional,
    available_actions: list[ActionSpec],
    raw: optional
  }
}
```

### ActionSpec

```text
ActionSpec = {
  action_id: string,
  data: optional dict
}
```

For complex coordinate actions:

```text
ActionSpec = {
  action_id: "ACTION6",
  data: {
    x: int,
    y: int
  }
}
```

### ObservationRef

```text
ObservationRef = {
  memory: "state" | "experimental",
  id: string
}
```

For decision-time agent calls, state-memory refs should be limited to the first
observation, immediately previous frame-turn observation, current observation,
and any past real states explicitly exposed by the orchestrator through the
current context. Experimental refs can point to temporary predictions created
during the current decision step.

### ToolCall

```text
ToolCall = {
  tool: "world" | "goal",
  action: ActionSpec,
  observation_ref: ObservationRef
}
```

### ToolResult

```text
ToolResult = {
  id: string,
  tool: "world" | "goal",
  predicted_observation: image_or_grid_64x64,
  explanation: optional text,
  source_observation_ref: ObservationRef,
  action: ActionSpec
}
```

### AgentTrace

```text
AgentTrace = {
  step: int,
  first_observation_ref: ObservationRef,
  current_observation_ref: ObservationRef,
  tool_calls: list[ToolCall],
  tool_results: list[ToolResult],
  final_action: ActionSpec,
  reasoning_summary: optional text
}
```

The full trace `T^X_{i,t}` should be passed to the updater, because the updater for the agent needs to improve how the agent uses tools and chooses actions.

### RewardUpdateQuantities

The agent updater receives selected individual quantities that would go into a
reward calculation, rather than a final scalar reward. Orchestration may retain
additional internal quantities for computing future fields.

```text
RewardUpdateQuantities = {
  prediction_error: optional float,          # internal D^S baseline; not sent to updater prompts
  prediction_error_delta: optional float,    # prior prediction error minus current prediction error
  goal_distance: optional float,             # D^G: distance from current/next state to inferred goal
  time_cost: optional float,                 # cumulative real environment steps spent
  trace_cost: optional float,                # wall-clock time spent deciding
  score_delta: optional float,               # raw environment score/progress change, if available
  notes: optional text
}
```

The updater system prompt should explain how to use the fields it receives:

- `prediction_error_delta`: detect whether the agent is finding outcomes the
  world model predicts better than before; higher positive values mean the
  prediction error is falling faster and are the learning-improvement signal.
- `goal_distance`: update whether the action appeared to move toward or away
  from the inferred goal; higher values mean less goal-following and more
  exploration.
- `time_cost`: prefer strategies that make progress in fewer real environment
  steps; this is cumulative pressure as real actions are spent.
- `trace_cost`: discourage slow decisions when they do not improve action quality.
- `score_delta`: use direct game feedback, when available, as evidence for goal and policy updates.

Updater prompts receive `prediction_error_delta` rather than raw
`prediction_error`. The raw prediction error may still be retained by
orchestration/memory to compute the next delta.

## Main Game Loop

```text
initialize ARC-AGI-3 environment
O_{i,0} = env.reset()
store O_{i,0} in M_i
initialize K^S, K^G, K^X
initialize L^S_{i,0}, L^G_{i,0}, L^X_{i,0}

for each environment step t:

  E_{i,t} = empty experimental memory

  read first observation O_{i,0} from M_i
  read current observation O_{i,t}
  read any exposed past-state refs from M_i
  read any active experimental refs from E_{i,t}
  read current action_space_{i,t} from environment metadata or env.action_space

  compose contexts:
    C^S_{i,t} = K^S + L^S_{i,t}
    C^G_{i,t} = K^G + L^G_{i,t}
    C^X_{i,t} = K^X + L^X_{i,t}

  call agent X with:
    C^X_{i,t}
    first observation O_{i,0}
    current observation O_{i,t}
    current action_space_{i,t}
    callable tool interfaces for S and G

  while X requests a tool call:
    parse ToolCall(tool, action, observation_ref)
    resolve observation_ref from allowed state refs in M_i or E_{i,t}
    reject inline frame inputs that are not memory references

    if tool == world:
      run S(C^S_{i,t}, action, referenced_observation)
      store result in E_{i,t}
      return result and reference id immediately to X

    if tool == goal:
      run G(C^G_{i,t}, action, referenced_observation)
      store result in E_{i,t}
      return result and reference id immediately to X

  receive final action A_{i,t} and trace T^X_{i,t} from X

  run committed post-decision predictions:
    O_hat^S = S(C^S_{i,t}, A_{i,t}, O_{i,t})
    O_hat^G = G(C^G_{i,t}, O_{i,t})

  execute real environment step:
    O_{i,t+1} = env.step(A_{i,t}.action_id, data=A_{i,t}.data, reasoning=trace_summary)

  compute or extract reward/update quantities:
    Q^X_{i,t} = RewardUpdateQuantities(...)

  store transition in M_i:
    O_{i,t}, A_{i,t}, O_{i,t+1}, T^X_{i,t}, O_hat^S, O_hat^G, E_{i,t}, Q^X_{i,t}

  run updater P:
    update L^S_{i,t} -> L^S_{i,t+1}
    update L^G_{i,t} -> L^G_{i,t+1}
    update L^X_{i,t} -> L^X_{i,t+1} using Q^X_{i,t}, not a scalar reward

  clear E_{i,t}

  if game is finished:
    update K^S, K^G, K^X if desired
    stop or reset according to benchmark flow
```

## Update Flow

The updater should run after each real step, not after every simulated tool call.

### World-Model Update

Inputs:

- Current world context `L^S_{i,t}`.
- Source observation `O_{i,t}`.
- Real action `A_{i,t}`.
- World predictions from the agent trace.
- Real next observation `O_{i,t+1}`.
- Prediction discrepancy `D^S` if computed.

Output:

- Updated game-specific world context `L^S_{i,t+1}`.

Purpose:

- Record which dynamics hypotheses were supported.
- Record which predictions failed.
- Revise the current understanding of state transitions.

### Goal-Model Update

Inputs:

- Current goal context `L^G_{i,t}`.
- First, current, and next observations.
- Goal-model outputs from the agent trace.
- Reward/update quantities `Q^X_{i,t}`.
- Goal discrepancy `D^G` if computed.

Output:

- Updated game-specific goal context `L^G_{i,t+1}`.

Purpose:

- Record current goal hypothesis.
- Record evidence for or against that hypothesis.
- Track what appears to count as progress.

### Agent Update

Inputs:

- Current agent context `L^X_{i,t}`.
- Full trace `T^X_{i,t}`.
- First observation `O_{i,0}`.
- Current observation `O_{i,t}`.
- Real next observation `O_{i,t+1}`.
- Real action `A_{i,t}`.
- Tool calls and results.
- Reward/update quantities `Q^X_{i,t}`.

Output:

- Updated game-specific agent context `L^X_{i,t+1}`.

Purpose:

- Improve how the agent chooses actions.
- Improve how it uses the world and goal tools.
- Record mistakes and useful patterns from the current step.
- Adjust strategy under the current game hypothesis.

## Loss and Update Signals

The architecture should store discrepancy and reward-update terms, but keep their exact definitions pluggable.

Core quantities:

```text
D^S = distance(O_hat^S, O_{i,t+1})
D^G = goal-model discrepancy or goal distance
Q^X_{i,t} = {D^S, ΔD^S, D^G, time_cost, trace_cost, score_delta, ...}
```

`Q^X_{i,t}` is passed to the agent updater as structured evidence. The updater should decide how those quantities change `L^X_{i,t+1}` based on its system prompt. The implementation should not require a single scalar reward.

## Software Architecture View

```text
ARC-AGI-3 Game Engine
  |
  | reset / step(action, data, reasoning)
  v
Environment Adapter
  |
  | observations, action_space, metadata
  v
Orchestration Layer
  |
  |-- read/write --> State Memory M_i
  |-- read/write --> Experimental Memory E_{i,t}
  |
  |-- compose C^X_{i,t} --> Agent X
  |                     |
  |                     | sees O_{i,0}, O_{i,t}, action_space
  |                     | tool call: S or G with action + observation_ref
  |                     v
  |                  Tool Router
  |                     |
  |                     |-- C^S_{i,t} --> World Model S --> predicted observation
  |                     |
  |                     |-- C^G_{i,t} --> Goal Model G  --> goal-relevant prediction
  |
  |-- final action --> Environment Adapter
  |
  |-- transition + trace + Q^X_{i,t} --> Updater P
                              |
                              |-- update L^S_{i,t} -> L^S_{i,t+1}
                              |-- update L^G_{i,t} -> L^G_{i,t+1}
                              |-- update L^X_{i,t} -> L^X_{i,t+1}
```

## Implementation Boundaries

### Environment Adapter Boundary

The environment adapter owns all direct ARC-AGI-3 calls.

It should expose a minimal internal interface:

```text
reset() -> Observation
step(ActionSpec, reasoning=None) -> Observation
get_action_space() -> list[ActionSpec]
get_info() -> EnvironmentInfo
```

No model should call the ARC-AGI-3 environment directly.

### Model Adapter Boundary

Each model adapter should expose one role-specific call:

```text
WorldModel.predict(context, action, observation_ref) -> ToolResult
GoalModel.predict(context, action, observation_ref) -> ToolResult
Agent.act(
  context,
  current_observation,
  tools,
  action_space
) -> AgentTrace + ActionSpec
Updater.update(update_packet) -> UpdatedContexts
```

The rest of the system should not depend on whether the implementation is a VLM, VLM plus image generator, LoRA-updated model, or custom network.

### Memory Boundary

Memory should store objects and return references.

The decision-time agent should receive only the first, immediately previous,
and current frame-turn observations as images. Duplicate observations may be
omitted from the image list when those roles point to the same observation.
Full frame history remains in state memory for logging, updater use, and
future training, but it should not be copied into the agent context. X may
receive a bounded recent action history as compact metadata; this does not
include prior frame payloads beyond the single previous frame-turn
observation.

Tool calls should pass observation references. This avoids copying full histories into model contexts and matches the intended state-memory and experimental-memory split.
Orchestration reads both `M_i` and `E_{i,t}` to resolve those references, then
writes new tool outputs back to `E_{i,t}` or committed run records back to
`M_i`.
The resolved frame used as model input must be the exact persisted object
behind the reference, even when the agent context contains a visual copy of the
same prediction.

## Minimal Runtime Sequence

1. Start game and store `O_{i,0}`.
2. Build `C^X_{i,t}`, `C^S_{i,t}`, and `C^G_{i,t}`.
3. Give the agent only `O_{i,0}`, `O_{i,t}`, the current action space, and tool interfaces.
4. Agent reasons and calls `S` and `G` as tools.
5. Store tool outputs in `E_{i,t}` and immediately return them with reference ids to the agent.
6. Agent emits one final valid action.
7. Environment advances one step.
8. Store the real transition and trace in `M_i`.
9. Compute or extract `Q^X_{i,t}` and pass it to the updater.
10. Updater revises `L^S_{i,t}`, `L^G_{i,t}`, and `L^X_{i,t}`.
11. Clear `E_{i,t}`.
12. Repeat until the game is finished.
13. After the game, optionally update game-agnostic `K^m` documents.

## References Used

- ARC-AGI-3 Quickstart: [https://docs.arcprize.org/](https://docs.arcprize.org/)
- ARC-AGI-3 Games: [https://docs.arcprize.org/games](https://docs.arcprize.org/games)
- ARC-AGI-3 Actions: [https://docs.arcprize.org/actions](https://docs.arcprize.org/actions)
- ARC-AGI Toolkit EnvironmentWrapper: [https://docs.arcprize.org/toolkit/environment_wrapper](https://docs.arcprize.org/toolkit/environment_wrapper)
- ARC-AGI Toolkit GitHub repository: [https://github.com/arcprize/arc-agi](https://github.com/arcprize/arc-agi)
