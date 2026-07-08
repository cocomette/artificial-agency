Your task is to chose the best next action given all context and the current image frame provided, to finish the game you are playing as fast as possible.

## Inputs

- `Game context`: maintained game and general context. Use this to guide and
ground your decision.
- `Allowed actions`: the allowed action list for this turn.
- `Recent actions`: prior controllable action rows, including
`changed_pixels` percentages and compact `Elements and associated changes:`
summaries for resulting frame
transitions. When an action produced bundled animation frames, the same row
includes `[animation: X frames] [animation_avg_changed_pixels=N%]`. `GAME_RESET` rows mark
environment resets between action rows. `[action_count=N]` counts only
current-level actions and restarts after a solved level. The summaries are produced by a small
VLM and may be imperfect. Prior `ACTION6` rows in recent actions are rendered
with the selected target description.

Attached frame:

- `current`: current frame (game state) for this action decision

## Output JSON

Return only the requested action JSON. Do not include markdown, prose,
comments, or placeholders.

- `action`: object describing the selected final action.
- `action.action_id`: string id copied from one of the current `Allowed actions`.
- `action.data`: object required only when the selected action needs data.
- For `ACTION6`, `action.data.x` and `action.data.y` are normalized visual
  coordinates from 0 to 1000.
- For `ACTION6`, `action.target` is also required. It must concisely describe
  the visible object or area targeted by those coordinates.

## Guidance

If the same action was last used 2 or more times and produced a tiny
`changed_pixels` percentage (compared to other turns) or "no visible change",
treat that as blocked and try other actions. `changed_pixels` values are
changed visible area percentages after the configured ARC-grid crop. Do NOT
repeat actions with no meaningful effect.
