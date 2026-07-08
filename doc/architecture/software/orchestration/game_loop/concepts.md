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

`FrameUnrollBuffer` is the ordered list of kept frames extracted from the
current environment response after consecutive duplicate raw frames and
below-threshold animation frames are filtered. The buffer is processed one kept
frame at a time.

If the response contains one frame, the buffer has one controllable frame. If
the response contains multiple frames, all frames except the last are
non-controllable animation frames.

Duplicate filtering is local to one environment response. For each consecutive
identical run, orchestration keeps the rightmost frame and drops earlier
identical frames.

After duplicate filtering, orchestration keeps animation keyframes by comparing
each candidate frame to the last retained frame. Post-action bundles use the
pre-action controllable frame as the initial comparison anchor; reset/start
bundles use the first kept bundle frame as the anchor. A candidate becomes a
retained keyframe when its changed-pixel count reaches
`animation_keyframe_pixel_threshold`. Threshold `0` keeps every non-duplicate
frame. The final incoming frame is always retained, so the buffer always has a
controllable final frame.

Skipped intermediate animation frames do not become frame turns. Their count is
recorded on the action-history row for the retained transition that skipped
them, including controllable action-to-first-keyframe transitions.

## Frame Turn

`FrameTurn` is one pass through the orchestration loop for one kept frame. A
frame turn:

- builds current Memory and Goal context
- lets `X` propose coordinate candidates only when the frame is controllable
- runs World on each candidate and lets `X` select one final action
- applies the decision according to the frame control mode
- summarizes and judges the executed World prediction
- calls Goal once for reward-only Goal delta, computes immediate reward,
  appends the finalized transition ledger entry, then regenerates Memory and
  next-turn Goal from sanitized action/change/reward ledger rows
- persists the turn

The downstream transition shape stays explicit for both controllable and
non-controllable frames, so persistence and dashboard inspection can use the
same ledger shape.

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
