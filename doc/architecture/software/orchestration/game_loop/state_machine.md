# Game Loop State Machine

`GameLoopStateMachine` owns the frame-turn loop. It is constructed by the
top-level orchestrator with state memory, live contexts, Agent X, change
summary, updater tasks, tool runtime factory, and debug bus.

Agent X is a dormant dependency path in the current runtime; the active loop
submits updater-produced actions for controllable decisions.

## Turn States

1. Load the current frame buffer.
2. Enter a frame turn and prewrite source state when memory is enabled.
3. For non-initial turns, build the observed previous-to-current transition.
4. Compute visible changed-pixel percentage and summarize changed direct
   transitions or bundled animation arrays.
5. Update compacter context for the transition.
6. Run agent context history and updater P when the current frame is
   controllable.
7. Submit an updater-produced action.
8. Resolve the next frame by stepping the environment; post-action animation
   bundles land on their final controllable frame and are retained as ordered
   model-input arrays.
9. Persist the completed state row.
10. Advance session cursors.

The session stores only transient turn data needed by these states.
