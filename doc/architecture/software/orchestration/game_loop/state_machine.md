# Game Loop State Machine

The source implementation lives in
`src/face_of_agi/orchestration/game_loop/state_machine.py` as
`GameLoopStateMachine`. `Orchestrator` wires dependencies and invokes this
component; it does not own the state-machine internals.

## Invariants

- Agent `X` is called only on controllable final frames.
- Non-final buffer frames synthesize the internal `NONE` action in
  orchestration without calling `X`.
- Final buffer frames expose the real action list from the environment.
- Orchestration rejects any non-final frame decision that is not `NONE`.
- Orchestration rejects `NONE` on a final controllable frame unless ARC exposes
  a separate real no-op action in the real action list.
- The ARC environment is called only after a final controllable frame.
- Every ARC frame is persisted as real state in `M`, including animation
  frames.
- The updater boundary runs after every frame decision.
- The change summary boundary runs before updater context updates.

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

Load or prewrite the current frame context in state memory `M`.

The current frame must be represented by a stable memory reference before it is
used in a trace. The first observation for the run remains available as a
stable reference for `X` on controllable frames. The previous frame reference,
when present, is the immediately prior frame turn processed by orchestration.

### `BUILD_DECISION_INPUT`

For controllable final frames, compose the input for `X`:

- agent role context
- first observation reference, previous frame reference, and current frame
  reference
- first, previous, and current observations as `ObservationText`
- bounded recent action history from prior frame turns
- empty tool policy for the current vLLM-only runtime
- action space for this frame turn
- frame control metadata such as `controllable` and buffer position

The recent action history is bounded by runtime config and includes both
synthetic `NONE` animation decisions and real environment actions.

For non-final frames, orchestration does not build `X` input.

### `CALL_X`

Call Agent `X` only for controllable final frames.

Decision contract:

- return one action from the provided real action list

### `SYNTHESIZE_NONE`

For non-controllable animation frames, orchestration creates the frame decision
directly:

- final action is synthetic `NONE`
- no `X` provider call is made
- the trace is marked as orchestration-generated

### `RESOLVE_NEXT_SNAPSHOT`

For non-final frames:

- use the orchestration-synthesized `NONE`
- do not call the environment
- compare the current frame to the next buffered frame

For final frames:

- require `X` to return one action from the real environment action space
- submit that action to the ARC environment adapter
- receive the next `EnvironmentObservationBundle`
- compare the current frame to the first relevant new frame

### `SUMMARIZE_CHANGE`

Run the change summary model on the observed transition. The prompt is
text-only and uses `ObservationText` plus component deltas.

The resulting summary and cropped changed-cell count become action history and
updater evidence.

### `SUMMARIZE_AGENT_CONTEXT_HISTORY`

Run the agent context historizer when configured. It summarizes recent context
revisions for updater input.

### `RUN_UPDATER`

Run updater `P` after each frame decision.

For non-final frames, the actual next frame is `buffer[index + 1]`.

For final frames, the actual next frame is the first frame of the newly
received environment response.

The updater input is the current frame, observed next frame, decision trace,
change summary, recent action history, context history, current context, and
update quantities.

The updater returns revised context documents. Orchestration applies them to
the live working `ContextDocuments` before persistence and before later model
calls are composed.

### `PERSIST_TURN`

Persist the frame turn into `M`:

- current observed frame
- frame control mode
- decision trace from `X` or the orchestration-synthesized animation decision
- real action if one was submitted
- synthetic `NONE` decision if this was an animation frame
- transition summary and turn metrics
- current agent context after updater output has been applied
- references needed for replay and inspection

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
