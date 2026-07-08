# Game Loop Concepts

## Ownership

Orchestration is the deterministic software owner of the loop, side effects,
memory writes, tool routing, and environment calls.

The orchestrator agent model `X` owns decision selection. Orchestration calls
`X`, validates its result against the current frame control mode, and handles
all side effects around that result.

## Environment Observation Bundle

`EnvironmentObservationBundle` is one response from the ARC environment. It
contains one or more real observed frames plus environment metadata and the
current real action space.

The environment can return multiple consecutive frames for one reset or step.
Those frames are real observations, but only the last frame in the bundle is a
point where the agent may affect the environment.

## Frame Unroll Buffer

`FrameUnrollBuffer` is the ordered list of frames extracted from the current
environment response. The buffer is processed one frame at a time.

If the response contains one frame, the buffer has one controllable frame. If
the response contains multiple frames, all frames except the last are
non-controllable animation frames.

## Frame Turn

`FrameTurn` is one pass through the orchestration loop for a single frame. A
frame turn:

- builds input for `X`
- lets `X` deliberate and call tools
- applies the decision according to the frame control mode
- runs the updater boundary
- persists the turn

The shape is intentionally the same for controllable and non-controllable
frames, so agent logic does not need a separate animation-frame pathway.

## Frame Control Mode

`controllable=false` applies to every frame except the last frame in the
current buffer. The frame is real observed state, but it is part of an
uncontrollable animation or transition. The agent may inspect it and use tools,
but it has no control over the environment at that instant.

`controllable=true` applies only to the last frame in the current buffer. On
this frame, `X` receives the real action space exposed by the environment and
orchestration may submit the chosen action.

## Synthetic `NONE`

`NONE` is a synthetic internal action available only when
`controllable=false`. It is exposed to `X` as the only returnable action for
non-final unrolled frames.

`NONE` is never sent to the ARC environment adapter. It means the agent is
observing a frame it cannot control.

If ARC later exposes a real no-op action, that action should remain distinct
from this internal `NONE` contract.
