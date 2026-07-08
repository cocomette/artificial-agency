# ARC-AGI-3 Agent Architecture Plan

## Purpose

This document describes a high-level, swappable architecture for an ARC-AGI-3 agent based on the whiteboard design and provided description.

The core idea is to run an online-learning agent over a turn-based game
environment. The environment returns visual frames as observations. Agent `X`
chooses actions using its own context plus the maintained world `S` and goal
`G` contexts. `S` and `G` produce predictions for each frame flow, and updater
`P` revises the game-specific context documents used by `S`, `G`, and `X`.

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

- `i`: game index. It is used only for game-specific values.
- `t`: real environment step within game `i`.
- `k`: temporary future-tool or simulation index within one agent decision step.
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
- `E_{i,t}`: temporary experimental memory reserved for Agent X tool outputs at
  step `t` when tools are configured.
- `H_{i,t}`: transition timing metadata for game `i` at step `t`.
- `cumulative_score_{i,t}`: cumulative environment progress when exposed by the
  environment; in the current ARC runtime this is completed levels so far.

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

The central controller that connects the environment, memory, model roles,
agent, and updater.

Responsibilities:

- Reset and step the environment.
- Maintain state memory `M_i`.
- Maintain per-step experimental memory `E_{i,t}` for tool outputs.
- Compose model contexts `C^m_{i,t}` from fixed game-agnostic documents `K^m` and mutable game-specific documents `L^m_{i,t}`.
- Run world predictions for updater evidence; goal prediction code is dormant
  in normal runtime.
- Submit the final agent action to the environment.
- Run the updater `P` after each real environment step.
- Reset experimental memory at the end of each real step.
- Update game-agnostic context `K^m` only after finishing a game.

### 3. Model Registry and Adapters

All models should be accessed through small adapter interfaces rather than direct calls.

Model roles:

- World model `S`
- Goal model `G` (dormant in normal runtime)
- Agent `X`
- Updater `P`

Initial backend:

- VLMs for active roles.
- Description-producing model calls for world predictions.

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
S(C^S_{i,t}, A, O_ref) -> D_hat^S
```

Where:

- `C^S_{i,t} = K^S + L^S_{i,t}`.
- `A` is a proposed action.
- `O_ref` is the current observation reference resolved from state memory `M_i`.
- `D_hat^S` is a structured description prediction for the next visual state.

The output is stored as a prediction object containing a `predicted_description`
that follows `DescriptionPrediction`, plus any text explanation or metadata
needed by the updater.

### Goal Model `G`

The goal model represents hypotheses about the game objective and progress.

Conceptual form:

```text
G(C^G_{i,t}, O_ref) -> D_hat^G
```

Where:

- `C^G_{i,t} = K^G + L^G_{i,t}`.
- `O_ref` is the current observation reference resolved from state memory `M_i`.
- `D_hat^G` is a structured description prediction for the goal-relevant
  visual state.

The goal model should support reasoning about what the agent is trying to
achieve, why that hypothesis changed, and what visual or progress evidence
supports it.

### Agent `X`

The agent is the decision model. It reasons over the visible frames and action
space, then returns one real action.

Conceptual form:

```text
X(C^X_{i,t}, O^anchor_{i,t}, O_{i,t}, action_space_{i,t}) -> A_{i,t}, T^X_{i,t}
```

Where:

- `C^X_{i,t} = K^X + L^X_{i,t}`.
- `L^S_{i,t}` is maintained by the world updater and summarized into `C^X`
  by the agent updater.
- `O^anchor_{i,t}` is the observation frame before the oldest visible recent
  action, or `O_{i,0}` when no recent history is visible.
- `O_{i,t}` is the current observation frame.
- `action_space_{i,t}` is the current set of valid environment actions.
- X also receives a bounded recent action history from prior frame turns:
  compact metadata with turn, step, observation reference, action,
  controllability, control reason, and optional reasoning summary.
- `A_{i,t}` is the final real action selected for the environment.
- `T^X_{i,t}` is the full agent trace for the current step.

World calls include a candidate action and goal calls use the current
observation. Their outputs become updater evidence and durable replay data.

```text
PredictionCall = {
  tool: "world",
  action: ActionSpec,
  source_state_id: int
}

PredictionCall = {
  tool: "goal",
  source_state_id: int
}
```

The `source_state_id` points to the current frame source row in committed
state memory. Agent X tools should use memory references rather than inline
frames.

Tool outputs may become referenceable through `E_{i,t}` when a concrete tool
is introduced. Every experiment-loop tool input should be a memory reference so
orchestration can resolve the exact persisted artifact from `M_i` or `E_{i,t}`.

### Updater `P`

The updater performs online context updates after the real environment step.

It updates text context documents initially. Later, the same role can trigger loss-based updates such as LoRA.

World update form:

```text
P(D_hat^S, O_{i,t+1}, A_{i,t}, L^S_{i,t}) -> L^S_{i,t+1}
```

Goal update form:

```text
P(D_hat^G, O_{i,t+1}, L^G_{i,t}) -> L^G_{i,t+1}
```

Agent update form:

```text
P(action_history, L^S_{i,t}, L^G_{i,t}, O_{i,t}, O_{i,t+1},
  H_{i,t}, cumulative_score_{i,t}, word_count(L^X_{i,t}), L^X_{i,t})
  -> L^X_{i,t+1}
```

Where:

- `D_hat^m` is a world-model or goal-model description prediction produced by
  orchestration.
- `O_{i,t+1}` is the real next observation.
- `A_{i,t}` is the submitted real action or synthetic `NONE` action used by
  the world updater.
- `action_history` is bounded prior actions plus the submitted real action or
  synthetic `NONE` action that produced the current frame.
- `H_{i,t}` is transition timing, such as elapsed real steps and Agent X
  decision duration.
- `cumulative_score_{i,t}` is cumulative environment progress metadata when
  available.
- `word_count(L^X_{i,t})` is compactness feedback for Agent X's prompt updater;
  lower is better when useful strategy is preserved.

At the software boundary, orchestration packages role-specific updater inputs.
World and goal prompt updaters receive only their previous game context, the
role-specific prediction, and current observation frame; the world updater also
receives the selected real action or synthetic `NONE` action. Agent X's prompt
updater receives compact action history, observed frames, transition timing,
score/progress metadata, and agent context word count. World and goal prompt
updaters do not receive timing metadata. The
updater does not read or write memory
directly and does not receive a precomputed scoring formula. It returns
updated context documents to orchestration, which applies them to the working
contexts and persists the resulting authoritative state into `M_i`.

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
  world_predictions: D_hat^S,
  goal_predictions: D_hat^G,
  agent_traces: T^X_{i,0}...T^X_{i,t},
  transition_timing: H_{i,0}...H_{i,t},
  cumulative_scores: cumulative_score_{i,0}...cumulative_score_{i,t},
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

Temporary memory reserved for Agent X tool outputs when tools are configured.

Stores:

```text
E_{i,t} = {
  tool_outputs: D_hat^{E}_{i,t,0}...D_hat^{E}_{i,t,k}
}
```

Experimental memory is pruned as a rolling buffer after frame turns. It is not
used for current world predictions.

This memory exists so Agent X tools can store temporary outputs before a real
action is committed. World predictions are stored with the frame turn in `M_i`;
goal prediction storage remains dormant and unset in normal runtime.

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
- General policies for using predictions and goals.

Updated only after finishing a game.

### `L^m_{i,t}`: Game-Specific Context

Updated during a single game.

For `L^S_{i,t}`, store current hypotheses about how the game dynamics work, tested transitions, prediction failures, and revised mechanics.

For `L^G_{i,t}`, store current hypotheses about the objective, visible evidence
for progress, score/progress evidence, and why the goal hypothesis changed.

For `L^X_{i,t}`, store current action strategy, mistakes from prior steps, and
current policy guidance.

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

For decision-time agent calls, state-memory refs should be limited to the
history-anchor observation, current observation, and any past real states
explicitly exposed by the orchestrator through the current context.
Experimental refs can point to temporary predictions created during the current
decision step.

### DescriptionPrediction

World predictions use the same description schema as `DESCRIPTION_SCHEMA` in
`tests/e2e/core.py`. The dormant goal prediction adapter uses the same schema
when called directly:

```text
DescriptionPrediction = list[DescriptionArea]

DescriptionArea = {
  bbox_2d: [x0: number, y0: number, x1: number, y1: number],
  description: string
}
```

`DescriptionPrediction` is a top-level array. Each item has exactly `bbox_2d`
and `description`; each `bbox_2d` is exactly four numbers in `[x0, y0, x1, y1]`
order. The schema allows overlapping areas when the areas describe meaningfully
different visual concepts.

### PredictionCall

```text
PredictionCall = {
  tool: "world",
  action: optional ActionSpec,
  source_state_id: int
}
```

`action` is required for world predictions. The goal prediction contract is
dormant in normal runtime.

### PredictionResult

```text
PredictionResult = {
  id: string,
  tool: "world" | "goal",
  predicted_description: DescriptionPrediction,
  explanation: optional text,
  source_observation_ref: ObservationRef,
  action: optional ActionSpec
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

The full trace `T^X_{i,t}` is persisted for replay and debug inspection. The
agent prompt updater receives compact action history instead of the full trace.
`tool_calls` and `tool_results` are dormant future-extension fields in the
current runtime.

### TransitionTiming

Agent updater input includes timing as turn metrics, not as an
optimization value. It also includes the current agent context word count as a
compactness reward that the updater should minimize when possible.

```text
TransitionTiming = {
  elapsed_real_steps: optional number,
  frame_turn_index: optional number,
  decision_duration_seconds: optional number
}
```

`cumulative_score` is persisted separately as raw environment progress metadata
when the environment exposes it. Orchestration does not compute prediction
discrepancy, goal-distance metrics, or a separate scoring value for updater input; the
updater models compare
their role-specific prediction evidence, actual observations, compact action
history, timing, score/progress metadata, and context-size feedback directly.

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

  receive final action A_{i,t} and trace T^X_{i,t} from X

  run world prediction:
    D_hat^S = S(C^S_{i,t}, A_{i,t}, O_{i,t})

  execute real environment step:
    O_{i,t+1} = env.step(A_{i,t}.action_id, data=A_{i,t}.data, reasoning=trace_summary)

  assemble turn metrics:
    H_{i,t} = TransitionTiming(...)
    cumulative_score_{i,t} = cumulative environment progress when available
    agent_context_word_count = word count of L^X_{i,t}

  store transition in M_i:
    O_{i,t}, A_{i,t}, O_{i,t+1}, T^X_{i,t}, D_hat^S, D_hat^G, E_{i,t}, H_{i,t}, cumulative_score_{i,t}

  run updater P:
    update L^S_{i,t} -> L^S_{i,t+1} using D_hat^S, O_{i,t+1}, and A_{i,t}
    update L^G_{i,t} -> L^G_{i,t+1} using D_hat^G and O_{i,t+1}
    update L^X_{i,t} -> L^X_{i,t+1} using compact action history,
      observations, current world/goal contexts, and turn metrics

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
- Real action `A_{i,t}`.
- Real next observation `O_{i,t+1}`.
- Committed world description prediction for the chosen action.

Output:

- Updated game-specific world context `L^S_{i,t+1}`.

Purpose:

- Record which dynamics hypotheses were supported.
- Record which predictions failed.
- Revise the current understanding of state transitions.

### Goal-Model Update

Inputs:

- Current goal context `L^G_{i,t}`.
- Real next observation `O_{i,t+1}`.
- Committed goal description prediction for the transition.

Output:

- Updated game-specific goal context `L^G_{i,t+1}`.

Purpose:

- Record current goal hypothesis.
- Record evidence for or against that hypothesis.
- Track what appears to count as progress.

### Agent Update

Inputs:

- Current agent context `L^X_{i,t}`.
- Current observation `O_{i,t}`.
- Real next observation `O_{i,t+1}`.
- Compact action history including real action `A_{i,t}` or synthetic `NONE`.
- Current-turn world game context.
- Previous-turn world game context when available.
- Transition timing `H_{i,t}`.
- Score/progress evidence when available.

Output:

- Updated game-specific agent context `L^X_{i,t+1}`.

Purpose:

- Improve how the agent chooses actions.
- Improve how it uses world prediction feedback.
- Record mistakes and useful patterns from the current step.
- Adjust strategy under the current game hypothesis.

## Transition Evidence

The architecture stores the evidence needed by updater `P`, but it does not
define a deterministic scoring module or executed scoring formula. Agent X's
updater compares action history, actual observations, transition timing, and
raw score/progress metadata directly.

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
  |
  |-- world prediction runner
  |                     |
  |                     |-- C^S_{i,t} --> World Model S --> predicted_description
  |
  |-- final action --> Environment Adapter
  |
  |-- turn metrics + action history --> Updater P
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
WorldModel.predict(context, action, observation_ref) -> PredictionResult
GoalModel.predict(context, observation_ref) -> PredictionResult
Agent.act(
  context,
  history_anchor_observation,
  current_observation,
  action_space
) -> AgentTrace + ActionSpec
Updater.update(update_packet) -> UpdatedContexts
```

The rest of the system should not depend on whether the implementation is a
VLM, LoRA-updated model, or custom network.

### Memory Boundary

Memory should store objects and return references.

The decision-time agent should receive only the history-anchor and current
frame-turn observations as images. The history anchor is the observation frame
that precedes the oldest action in the bounded recent action history, or the
initial run observation when no recent history is visible.
Full frame history remains in state memory for logging, updater use, and
future training, but it should not be copied into the agent context. X may
receive a bounded recent action history as compact metadata.

Agent X tool calls should pass memory references. This avoids copying
full histories into model contexts and matches the intended state-memory and
experimental-memory split. World predictions use the current committed frame
source selected by orchestration.

## Minimal Runtime Sequence

1. Start game and store `O_{i,0}`.
2. Build `C^X_{i,t}` and `C^S_{i,t}`.
3. Give the agent `C^X`, `O_{i,0}`, `O_{i,t}`, and
   the current action space.
4. Agent emits one final valid action.
5. Run world prediction.
6. Environment advances one step.
7. Store the real transition, trace, and predictions in `M_i`.
9. Assemble turn metrics and agent context word count for Agent X's prompt
   updater.
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
