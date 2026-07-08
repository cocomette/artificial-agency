# Game Loop State Machine

`GameLoopStateMachine` owns the frame-turn loop. It is constructed by the
top-level orchestrator with state memory, live contexts, Agent X, change
summary, Memory, World, Goal, Reward Judge, and debug bus.

## Turn States

1. Load or unroll the current frame buffer.
2. Enter a frame turn and prewrite source state when memory is enabled.
3. Bootstrap Memory and Goal at reset, then reuse the latest outputs.
4. For controllable frames, assemble candidates, run World, and ask Agent X for
   the final selection.
5. Resolve the next frame by stepping the environment or advancing animation.
6. Summarize visible change.
7. Judge the executed World prediction.
8. Call Goal once with previous Memory plus the next frame for reward-only Goal
   delta, compute immediate reward, and append the finalized ledger entry.
9. Regenerate Memory from sanitized action/change/reward ledger rows, then call Goal
   again for next-turn state.
10. Persist the completed state row and v1 artifact tables.
11. Advance session cursors.

The internal ledger row is finalized before Memory regeneration, but the Memory
role receives sanitized `turn_id`, prompt-facing `action`, `change_summary`,
and concise reward feedback. `TurnReward.learning_progress` is the immediate
Reward Judge prediction-accuracy proxy in this branch.
