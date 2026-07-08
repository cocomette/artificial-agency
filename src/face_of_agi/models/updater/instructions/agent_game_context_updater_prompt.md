Output format: return exactly one top-level `updated_context` field whose value
is the complete revised context string; do not return arrays,
or nested objects.

Input:
- `previous_context`: current game-specific agent strategy context.
- `previous_observation_frame`: observed frame before the last agent decision.
- `current_observation_frame`: observed frame after the last decision.
- `current_turn_world_game_context`: world model game context used for this
  transition before the current world updater mutation.
- `current_turn_goal_game_context`: goal model game context used for this
  transition before the current goal updater mutation.
- `previous_turn_world_game_context`: world model game context used for the
  previous frame turn, or null when no previous turn context exists yet.
- `trace`: final action, reasoning summary, and any `world` or `goal` tool
  calls/results used by the agent.
- `turn_metrics`: 
  - `time_cost` gives the number of actions taken during this game. A level shall be solved in less than 100 actions,
  - `score_delta` is 1 or 0, 1 means a level was completed, so it is a really good feedback.

Action glossary:
- `RESET`: initialize or restart the game or level state.
- `ACTION1`: simple action, semantically mapped to up.
- `ACTION2`: simple action, semantically mapped to down.
- `ACTION3`: simple action, semantically mapped to left.
- `ACTION4`: simple action, semantically mapped to right.
- `ACTION5`: simple game-specific action, such as interact, select, rotate,
  attach/detach, or execute.
- `ACTION6`: coordinate action targeting `x,y` on the 64x64 game grid.
- `ACTION7`: undo-style simple action.

Task: revise the agent game context so future decisions use better strategy
and exploration for this game.

Keep useful strategy, fix contradicted assumptions, and encode what the agent has
learned about objectives, action choice, visible state, progress, failure, and
what to avoid repeating.

Use the committed `world` and `goal` post-decision areas to improve strategy.
Do not recommend calling world or goal as Agent X tools; they are not exposed
to the agent in the current runtime.

Balance exploration and exploitation strategy: ask to explore unknown actions, states if game seems stuck, follow the goal if things seem to progress.
