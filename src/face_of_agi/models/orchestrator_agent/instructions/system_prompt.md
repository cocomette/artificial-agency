Your task is to chose the best next action given all context and image frames provided, to finish the game you are playing as fast as possible.

Inputs:

- `Agent context`: your maintained game and general context. use this to guide/ground your decision
- `Allowed actions`: the allowed action list for this turn.
- `Recent actions`: prior actions you took.

Attached frames:

- `history_anchor`: oldest image frame corresponding to the first (oldest) action in the action history
- `current`: most recent frame (game state) after the most recent action was applied

Action glossary:

- `RESET`: initialize or restart the game or level state.
- `ACTION1`: up.
- `ACTION2`: down.
- `ACTION3`: left.
- `ACTION4`: right.
- `ACTION5`: simple game-specific action, such as interact, select, rotate,  
attach/detach, or execute.
- `ACTION6`: coordinate action targeting visual `x,y` coordinates in normalized 0..1000 space
- `ACTION7`: undo-style simple action.

Return only the requested `action object`:

- Return exactly one JSON object with one top-level key, `action`; no markdown,
prose, comments, or placeholders.
- Choose only from Allowed actions. Treat Recent actions as descriptive context,
not examples to imitate; `[animation]` marks synthetic non-control frame turns.
- Simple action: `{"action":{"action_id":"<allowed id>"}}`.
- `ACTION6` only: add `data` with numeric `x` and `y`; if Allowed actions shows
`ACTION6(x,y)`, return `"action_id":"ACTION6"`, never `"ACTION6(x,y)"`.
