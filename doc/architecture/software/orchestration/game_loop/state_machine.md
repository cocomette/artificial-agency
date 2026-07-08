# Game Loop State Machine

`GameLoopStateMachine` owns one frame-unrolled ARC run.

Current turn order:

1. Load or unroll the current frame buffer.
2. Prewrite the current frame source state when memory is enabled.
3. Build the frame-turn context.
4. Call Agent X on controllable frames or synthesize `NONE` on animation
   frames.
5. For controllable Agent X decisions, try known-state simulation before
   stepping ARC.
6. Resolve the next observed or replayed frame.
7. Run the change-summary role for observed transitions, or replay historical
   change evidence for known-state simulation rows.
8. Run game memory on controllable real-action and known-state simulation
   turns.
9. Run the historizer when enough prior agent context exists.
10. Run updater P when context updates are enabled.
11. Persist the completed frame turn.
12. Advance session cursors.

Known-state simulation rows skip ARC `environment.step()` and the
change-summary model. They still use the existing Agent X, game-memory,
historizer, updater, persistence, and debug contracts. On simulation exit,
orchestration submits catch-up actions to ARC before resuming normal
environment stepping for the exit decision.

Known-state frame hashes use the same visible ARC-grid crop as the
change-summary model input crop. Orchestration converts that normalized crop
box to 64x64 grid crop edges and passes the edges through frame prewrite and
simulation catch-up hash validation.

For ACTION6 known-state matching, orchestration compares the current
model-visible bounding box against the historical submitted ARC-grid click
coordinate using those same crop edges. The final ARC-grid target value derived
from the selected click coordinate must match.

Model/provider/output failures use narrowed fallbacks. Non-model errors raise.
