Output format: return exactly one JSON object with a top-level
`updated_context` object containing five required string fields.

Example shape:
{"updated_context":{"goals":"","game_mechanics":"","policy":"","history":"","extras":""}}

Inputs:

- `Previous agent game context`: current game-specific agent strategy context.
- `Current-turn world game context`: world model game context used for this
transition before the current world updater mutation.
- `Previous-turn world game context`: world model game context used for the
previous frame turn, or `none` when no previous turn context exists yet.
- Attached images: image 1 is `previous_observation_frame` , image 2 is  
`current_observation_frame` after the listed transition action.
- `Action history`: bounded prior actions plus the action that produced the
current frame. Entries marked `[animation]` are synthetic non-control frame
turns, usually `NONE`.
- `Progress feedback`:
  - `time_cost` gives the number of actions taken during this game. A level
  should be solved in less than 100 actions.
  - `cumulative_score` is the current total completed levels so far. Higher
  values mean progress; `none` means unavailable.
  - `agent_context_word_count` is the word count of the current agent game  
  context before this update. Lower values are preferred when useful guidance
  is preserved.

Reward guidance:

Before revising context, estimate these proxy rewards qualitatively. Use them
to decide what strategy to preserve, remove, or correct, then express the
result in `updated_context`.

- `cumulative_score`: positive reward to maximize. Higher completed-level total
means stronger progress.
- `time_cost`: negative reward to minimize. More actions spent means less
efficient progress.
- `agent_context_word_count`: negative reward to minimize. Prefer shorter
context when it preserves the same useful strategy.
- `learning_progress`: positive reward to maximize. Compare Current-turn world
game context with Previous-turn world game context; useful (large) semantic change
means the world model is learning.

Field guidance:

- `goals`: current objective, progress target, and goal hypothesis.
- `game_mechanics`: useful world/action dynamics and uncertainty.
- `policy`: general guidance for how to approach playing this game.
- `history`: compact learnings from past outcomes and progress evidence.
- `extras`: any other useful agent guidance.

Action glossary:

- `RESET`: initialize or restart the game or level state.
- `ACTION1`: up.
- `ACTION2`: down.
- `ACTION3`: left.
- `ACTION4`: right.
- `ACTION5`: simple game-specific action, such as interact, select, rotate,  
attach/detach, or execute.
- `ACTION6`: coordinate action mapped to the game grid.
- `ACTION7`: undo-style simple action.

Your only task is to update the agent game context so that the agent using it
will play the game better and progress faster.
