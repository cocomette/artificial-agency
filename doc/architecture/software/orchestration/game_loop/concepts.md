# Game Loop Concepts

## Ownership

Orchestration owns deterministic side effects: environment calls, memory
writes, frame unrolling, action validation, deadlines, and debug events.

`OnlineLearnerAgent` owns action selection and online updates for controllable
frames. During non-controllable animation frames, orchestration synthesizes the
internal `NONE` action.

## Environment Observation Bundle

One ARC reset or step can return one or more observed frames. Only the final
retained frame in the current buffer is controllable.

## Frame Unroll Buffer

The frame buffer is the ordered list of kept frames after consecutive duplicate
frames and below-threshold animation frames are filtered. Threshold `0` keeps
every non-duplicate frame. The final incoming frame is always retained.

## Frame Turn

A frame turn builds a `FrameTurnContext`, chooses or synthesizes an action,
resolves the next frame, records a `TransitionRecord`, updates the learner, and
persists a `LearnerTurnTrace`.

## Synthetic `NONE`

`NONE` is an orchestration-only action for retained animation frames. It is
never submitted to the ARC environment.
