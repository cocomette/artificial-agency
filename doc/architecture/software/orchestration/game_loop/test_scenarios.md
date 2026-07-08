# Game Loop Test Scenarios

These scenarios define the observable behavior of the game-loop state machine.

- Initial reset returns one frame: `X` receives real action space and the
  chosen action is sent to the environment.
- Initial reset returns three frames: first two `X` calls receive only `NONE`;
  the third receives real actions.
- Non-final frame returns anything other than `NONE`: orchestration rejects it
  as an invalid decision.
- Final frame returns `NONE`: orchestration rejects it unless ARC explicitly
  exposes a separate real no-op action.
- Controllable final frames produce committed post-decision world and goal
  predictions before the environment is stepped.
- Non-controllable animation frames do not produce post-decision predictions.
- Updater receives `current_frame -> next_buffer_frame` during unrolling.
- Updater receives `last_buffer_frame -> first_new_environment_frame` after a
  real step.
- Updater receives post-decision predictions separately from `AgentTrace`.
- Updater-returned contexts are injected into later `X`, `S`, and `G` model
  calls.
- Updater-returned contexts are persisted into `M` with the frame-turn state.
- Memory `M` stores all real observed frames, including animation frames.
- Environment is never called during non-final unrolled frames.
- Tool calls remain routed through orchestration and may run during both
  controllable and non-controllable frame turns.
