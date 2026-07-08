# Game Loop Concepts

## Ownership

Orchestration is the deterministic software owner of the loop, side effects,
memory writes, tool routing, and environment calls.

The orchestrator agent model `X` owns real action selection on controllable
frames. During non-controllable animation frames, orchestration synthesizes
the internal `NONE` decision and handles all downstream side effects itself.

## Environment Observation Bundle

`EnvironmentObservationBundle` is one response from the ARC environment. It
contains one or more real observed frames plus environment metadata and the
current real action space.

The environment can return multiple consecutive frames for one reset or step.
Those frames are real observations, but only the last frame in the bundle is a
point where the agent may affect the environment.

## Frame Unroll Buffer

`FrameUnrollBuffer` is the ordered list of frames extracted from the current
environment response after consecutive duplicate raw frames are filtered. The
buffer is processed one frame at a time.

If the response contains one frame, the buffer has one controllable frame. If
the response contains multiple frames, all frames except the last are
non-controllable animation frames.

Duplicate filtering is local to one environment response. For each consecutive
identical run, orchestration keeps the rightmost frame and drops earlier
identical frames. The final incoming frame is always retained, so the buffer
always has a controllable final frame.

## Frame Turn

`FrameTurn` is one pass through the orchestration loop for a single frame. A
frame turn:

- builds input for `X`
- lets `X` deliberate and call tools only when the frame is controllable
- applies the decision according to the frame control mode
- runs the updater boundary
- persists the turn

The downstream transition shape is intentionally the same for controllable and
non-controllable frames, so updater and persistence logic can stay shared.

## Frame Control Mode

`controllable=false` applies to every frame except the last frame in the
current buffer. The frame is real observed state, but it is part of an
uncontrollable animation or transition. Agent `X` is not called for these
frames.

`controllable=true` applies only to the last frame in the current buffer. On
this frame, `X` receives the real action space exposed by the environment and
orchestration may submit the chosen action.

## Synthetic `NONE`

`NONE` is a synthetic internal action available only when
`controllable=false`. Orchestration creates it directly for non-final
unrolled frames.

`NONE` is never sent to the ARC environment adapter. It means orchestration is
processing a real observed frame where no agent action can affect the
environment.

If ARC later exposes a real no-op action, that action should remain distinct
from this internal `NONE` contract.
