Input:
- `world_game_context`: the current action to consequences understanding for this game.
- `goal_game_context`: the current goal understanding for this game from an external agent.
- `frames`: attached game frame images labeled `history_anchor` and `current`.
- `allowed_actions`: the authoritative list of actions you must choose from.
- `recent_action_history`: ordered action memory from prior frame decisions.

Action glossary:
- `RESET`: initialize or restart the game or level state.
- `ACTION1`: simple action, semantically mapped to up.
- `ACTION2`: simple action, semantically mapped to down.
- `ACTION3`: simple action, semantically mapped to left.
- `ACTION4`: simple action, semantically mapped to right.
- `ACTION5`: simple game-specific action, such as interact, select, rotate,
  attach/detach, or execute.
- `ACTION6`: coordinate action targeting `x,y` on the 64x64 game grid.
  Follow the active `allowed_actions` schema for whether and how to include
  coordinate data in the final output.
- `ACTION7`: undo-style simple action.

Rules:
- Choose only from `allowed_actions`.
- If the chosen action has `requires_data: true`, include numeric
  `action.data.x` and `action.data.y`.
- If the chosen action has `requires_data: false`, omit `action.data`.
- Treat `recent_action_history` as descriptive context, not as examples to
  imitate.

Task:
- Figure out the next action to move the game state forward to solve the game.
- Return the structured action JSON when you are ready to choose the next
  action.
