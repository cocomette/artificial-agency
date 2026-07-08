# Game Loop Test Scenarios

These scenarios define the observable behavior of the game-loop state machine.

- Initial reset returns one frame: `X` receives real action space and the
  chosen action is sent to the environment.
- Initial reset returns three frames: first two frames synthesize `NONE`
  without calling `X`; the third `X` call receives real actions.
- Non-final frame turns persist orchestration-generated `NONE` decisions with
  no agent-requested tool calls or results.
- Final frame returns `NONE`: orchestration rejects it unless ARC explicitly
  exposes a separate real no-op action.
- Controllable final frames produce world and goal description predictions
  before the environment is stepped.
- Non-controllable animation frames produce world and goal description
  predictions for updater input.
- Updater receives `current_frame -> next_buffer_frame` during unrolling.
- Updater receives `last_buffer_frame -> first_new_environment_frame` after a
  real step.
- Updater receives S/G predictions separately from `AgentTrace`.
- Updater-returned contexts are injected into later `X`, `S`, and `G` model
  calls.
- Updater-returned contexts are persisted into `M` with the frame-turn state.
- Memory `M` stores all real observed frames, including animation frames.
- Environment is never called during non-final unrolled frames.
- Agent-requested tool calls remain routed through orchestration and exist only
  on controllable frame turns where `X` is called.
