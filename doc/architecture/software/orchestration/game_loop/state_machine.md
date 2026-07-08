# Game Loop State Machine

The source implementation lives in
`src/face_of_agi/orchestration/game_loop/state_machine.py` as
`GameLoopStateMachine`. `Orchestrator` wires dependencies and invokes this
component; it does not own the state-machine internals.

## Implementation Shape

The concrete implementation keeps `state_machine.py` as the coordinator and
splits behavior into state-machine concepts:

- `lifecycle.py`: run start, lifecycle checks, reset/stop, and finish behavior
- `actions/steps.py`: ordered frame-turn step actions
- `actions/context_updates.py`: updater actions
- `actions/post_decision_predictions.py`: world prediction action
- `actions/metrics.py`: action timing and progress metrics
- `persistence.py`: M-state and model-input debug persistence actions
- `session.py`: mutable `GameLoopSession` and immutable `FrameTurnSnapshot`
- `helpers.py`: small frame, action, validation, and history helpers

`run()` has one entry point after `start_run(...)` initializes the session, and
one exit path through `finish_run(...)` after a terminal lifecycle state sets
`session.running` to false.

## Invariants

- `X` is called only on controllable final frames.
- Non-final buffer frames synthesize the internal `NONE` action in
  orchestration without calling `X`.
- Final buffer frames expose the real action list from the environment.
- Orchestration rejects any non-final frame decision that is not `NONE`.
- Orchestration rejects `NONE` on a final controllable frame unless ARC exposes
  a separate real no-op action in the real action list.
- The ARC environment is called only after a final controllable frame.
- Every ARC frame is persisted as real state in `M`, including animation
  frames.
- Agent X tool calls, when configured, must be routed through orchestration and
  persisted in `E` before reuse.
- The updater boundary runs after every frame decision.

## States

### `START_RUN`

Select and initialize the ARC game. Reset the environment and receive the
first `EnvironmentObservationBundle`.

Terminal lifecycle states are checked here and after each real environment
step.

### `LOAD_FRAME_BUFFER`

Normalize the latest environment response into a `FrameUnrollBuffer`.

If the environment returns one frame, the buffer has one controllable
`FrameTurn`. If it returns multiple frames, all frames except the last are
non-controllable `FrameTurn`s with synthetic `NONE` as their only action.
Consecutive duplicate raw frames are filtered while loading the buffer: a frame
is kept when it differs from the next incoming frame, and the last incoming
frame is always kept.

### `ENTER_FRAME_TURN`

Load or persist the current frame context from state memory `M`.

The current frame must be represented by a stable memory reference before it is
used in predictions or a trace. The first observation for the run remains
available as a stable reference for `X` on controllable frames. The previous
frame reference, when present, is the immediately prior frame turn processed by
orchestration; it may be an animation `NONE` turn or a real environment-action
turn.

### `BUILD_DECISION_INPUT`

For controllable final frames, compose the input for `X`:

- agent role context
- first observation reference, previous frame reference, and current frame
  reference
- current frame payload for active perception
- bounded recent action history from prior frame turns
- action space for this frame turn
- frame control metadata such as `controllable` and buffer position

The recent action history is bounded by runtime config and includes both
synthetic `NONE` animation decisions and real environment actions. The
model-facing prompt receives only prior action payloads plus controllability;
full frame history and richer transition metadata stay outside the prompt.

For non-final frames, orchestration does not build `X` input.

### `CALL_X`

Call the orchestrator agent model `X` only for controllable final frames.

Decision contract:

- return one action from the provided real action list

### `SYNTHESIZE_NONE`

For non-controllable animation frames, orchestration creates the frame
decision directly:

- final action is synthetic `NONE`
- no `AgentToolRuntime` is created
- no `X` provider call is made
- the trace is marked as orchestration-generated

### `RUN_POST_DECISION_PREDICTIONS`

Run world predictions after a frame decision is available.

This state runs on both orchestration-synthesized animation decisions and
Agent-X-selected controllable decisions:

- world prediction receives the current frame reference, current frame, and
  chosen action, including synthetic `NONE` for animation frames

The output is a description prediction artifact carried separately for updater
input and persistence. The goal prediction field remains optional and unset in
normal runtime.

### `RUN_UPDATER`

Run the updater boundary after each frame decision.

For non-final frames, the actual next frame is `buffer[index + 1]`.

For final frames, the actual next frame is the first frame of the newly
received environment response.

The transition-level update object contains the current frame, decision trace,
world predictions when present, transition timing, score/progress metadata, and
the actual next frame. Timing and score/progress metadata are used by Agent X's
prompt updater and persistence, not by the world prompt updater.

Agent X's prompt updater receives compact action history plus progress
feedback, not the live `AgentTrace`. The world prompt updater consumes the
world prediction. Predictions are carried separately as transition artifacts
and become durable replay data when orchestration later persists the turn into
`M`.

The updater returns revised context documents. Orchestration applies them to
the live working `ContextDocuments` before persistence and before later model
calls are composed.

### `PERSIST_TURN`

Persist the frame turn into `M`:

- current observed frame
- frame control mode
- decision trace from `X` or the orchestration-synthesized animation decision
- world prediction when present; dormant goal prediction remains unset
- real action if one was submitted
- synthetic `NONE` decision if this was an animation frame
- transition timing and score/progress metadata
- transition metadata
- current world, goal, and agent context documents after updater output has
  been applied
- references needed for replay and inspection

Temporary tool outputs remain in `E` unless orchestration explicitly commits a
selected artifact into `M`.

### `ADVANCE`

If the current `FrameUnrollBuffer` has more frames, advance to the next
`FrameTurn` and return to `ENTER_FRAME_TURN`.

If the buffer is exhausted, load the latest environment response as the next
buffer and return to `LOAD_FRAME_BUFFER`.

## Terminal States

`GAME_WIN` stops the run after the environment reports a win.

`GAME_OVER_RESET` resets the environment when ARC reports game over and the
runtime policy allows reset.

`ACTION_LIMIT_REACHED` stops the run when the configured per-level action
budget is exhausted.

`ERROR` stops the run when a contract invariant fails, an environment call
fails, or a required dependency is missing.
