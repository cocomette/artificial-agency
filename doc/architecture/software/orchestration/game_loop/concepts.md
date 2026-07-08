# Game Loop Concepts

## Ownership

Orchestration is the deterministic software owner of the loop, side effects,
memory writes, tool routing, and environment calls.

Agent X is dormant in the current loop. Runtime bootstrap registers no Agent X
model, and controllable actions come from updater P. Updater P updates the
selected agent game context and returns `next_actions`; orchestration submits
queued actions in order on controllable frames.

Post-action animation bundles are stored as one synthetic `NONE` raw history
entry tied to the real action that produced them. Prompt-facing action history
renders that animation evidence on the same line as the real action.

## Environment Observation Bundle

`EnvironmentObservationBundle` is one response from the ARC environment. It
contains one or more real observed frames plus environment metadata and the
current real action space.

The environment can return multiple consecutive frames for one reset or step.
Those frames are real observations, but only the last frame in a post-action
bundle is the next point where the agent may affect the environment.

## Frame Buffer And Animation Bundles

`FrameUnrollBuffer` remains the ordered list used to normalize multi-frame
observations that are not post-action animation bundles.

Post-action animation bundles are not processed one frame at a time. The loop
keeps the full ordered image array for model inputs, records one bundled
animation history row, and advances directly to the last bundle frame as the
next controllable state.

The model-input array starts with the pre-action controllable frame, continues
with every frame returned by the environment response, and ends with the final
controllable frame. Any frames between the first and last image are animation
frames.

Bundled animation evidence renders on the action row as
`[animation: X frames] [animation_avg_changed_pixels=N%]`. `X` is the number of
environment-returned frames in the bundle. `animation_avg_changed_pixels` is the
average consecutive changed visible area percentage across the model-input
array. When a controllable action has bundled animation evidence, that action
row carries the animation metrics and `Elements and associated changes:`
summary. If the action
triggered animation but the first and last visible frames are identical, the
action row introduces the summary with `This action triggered an animation
feedback, but previous and current frame remain identical. So there is no
progress but the animation teach you something. Elements and associated
animation feedback changes:` followed by the summary on the next line.
When local pixel metrics show nonzero direct or animation change but the
change-summary role returns `change_detected=false`, orchestration discards the
model summary text for that row and stores an uncertainty summary instead.

## Frame Turn

`FrameTurn` is one pass through the orchestration loop for a single frame. A
frame turn:

- applies the current frame control mode
- runs change-summary transition updates for direct transitions or bundled
  animation arrays when available
- runs world-model, historizer, and updater selection when the current frame is
  controllable and no valid queued action remains
- submits an updater-produced action when the frame is controllable
- persists the turn

Post-action animation bundles land on their final controllable frame before
historizer/updater selection runs.

## Updater Action Chains

Agent probing and policy updaters must return exactly their configured
action-window count. During the middle of a queued action chain, the loop
summarizes the observed transition, appends action-history entries, skips
world-model/historizer/updater calls, and submits the next queued action.

When the queue is exhausted, the next observed transition runs the full
world-model, historizer, and selected updater sequence before choosing another
action. If a queued action is no longer present in the current action space,
the queue is abandoned and the full context refresh runs immediately for the
current frame.

Level completion summarizes the strategy snapshots for the solved level when a
level-summary model is configured, clears any remaining queued actions, and
forces a full context refresh for the level-completing transition. The next
updater call receives the latest same-run `solution_method`. `GAME_OVER` clears
queued actions, models the failed transition, refreshes context, persists the
transition, then resets the environment.

## Frame Control Mode

`controllable=false` applies only to non-final frames that still enter a local
frame buffer, such as non-action startup/reset observations. Post-action
animation bundles do not create non-controllable frame turns.

`controllable=true` applies only to the last frame in the current buffer. On
this frame, orchestration first models the previous-to-current transition, then
submits the updater-produced action to the environment. If no updater action is
available, the loop fails fast instead of falling back to Agent X.

## Synthetic `NONE`

`NONE` is a synthetic internal action used for model-facing animation evidence.
For post-action bundles, orchestration records one nested `NONE` animation row
under the preceding real action.

`NONE` is never sent to the ARC environment adapter. It means orchestration is
processing a real observed frame where no agent action can affect the
environment.

If ARC later exposes a real no-op action, that action should remain distinct
from this internal `NONE` contract.
