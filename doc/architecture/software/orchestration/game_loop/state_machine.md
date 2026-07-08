# Game Loop State Machine

The source implementation lives in
`src/face_of_agi/orchestration/game_loop/state_machine.py` as
`GameLoopStateMachine`. `Orchestrator` wires dependencies and invokes this
component; it does not own the state-machine internals.

## Invariants

- `X` is called once per frame turn, including non-controllable animation
  frames.
- Non-final buffer frames expose exactly one valid action to `X`: synthetic
  `NONE`.
- Final buffer frames expose the real action list from the environment.
- Orchestration rejects any non-final frame decision that is not `NONE`.
- Orchestration rejects `NONE` on a final controllable frame unless ARC exposes
  a separate real no-op action in the real action list.
- The ARC environment is called only after a final controllable frame.
- Every ARC frame is persisted as real state in `M`, including animation
  frames.
- Tool calls from `X` are always routed through orchestration. Tool outputs are
  persisted in `E` before `X` can reuse them by reference.
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

### `ENTER_FRAME_TURN`

Load or persist the current frame context from state memory `M`.

The current frame must be represented by a stable memory reference before it is
passed to tools or used in a trace. The first observation for the run remains
available as a stable reference for `X`.

### `BUILD_X_INPUT`

Compose the input for `X`:

- agent role context
- first observation reference and current frame reference
- current frame payload for active perception
- tool handles for world `S` and goal `G`, routed through orchestration
- action space for this frame turn
- frame control metadata such as `controllable` and buffer position

For non-final frames, the action space is `[NONE]`. For the final frame, the
action space is the real action list from the latest environment info.

### `CALL_X`

Call the orchestrator agent model `X` for every frame turn.

Decision contract:

- if action space is `[NONE]`, return `NONE`
- otherwise return one action from the provided real action list

### `HANDLE_TOOL_CALLS`

If `X` asks to call world `S` or goal `G`, orchestration resolves the requested
`ObservationRef` from `M` or `E`, calls the tool model, persists the tool
result in `E`, and returns the resulting reference to `X`.

This path exists on both controllable and non-controllable frames. During
frame unrolling, `X` may still run experiments even though it cannot affect the
environment until the final frame.

### `APPLY_DECISION`

For non-final frames:

- require `X` to return `NONE`
- do not call the environment
- continue to updater and persistence

For final frames:

- require `X` to return one action from the real environment action space
- run post-decision world and goal predictions for the chosen action
- submit that action to the ARC environment adapter
- receive the next `EnvironmentObservationBundle`
- keep that response available for updater comparison and the next
  `LOAD_FRAME_BUFFER`

### `RUN_POST_DECISION_PREDICTIONS`

Run committed S/G predictions after `X` returns a valid real action and before
the environment is stepped.

This state runs only on controllable final frames:

- world prediction receives the current frame reference, current frame, and
  chosen final action
- goal prediction receives the current frame reference and current frame

These calls are internal orchestration prediction calls. They are not
agent-requested tool calls, are not appended to `AgentTrace`, and are not
stored in `E`.

### `RUN_UPDATER`

Run the updater boundary after each frame decision.

For non-final frames, the actual next frame is `buffer[index + 1]`.

For final frames, the actual next frame is the first frame of the newly
received environment response.

The updater input is the current frame, the decision trace, committed
post-decision predictions when present, and the actual next frame.

Agent-requested tool results remain in the live `AgentTrace`. Committed
post-decision predictions are carried separately as transition artifacts. Both
become durable replay data only when orchestration later persists the turn
into `M`.

The updater returns revised context documents. Orchestration applies them to
the live working `ContextDocuments` before persistence and before later model
calls are composed.

### `PERSIST_TURN`

Persist the frame turn into `M`:

- current observed frame
- frame control mode
- decision trace from `X`
- committed world and goal post-decision predictions when present
- real action if one was submitted
- synthetic `NONE` decision if this was an animation frame
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
