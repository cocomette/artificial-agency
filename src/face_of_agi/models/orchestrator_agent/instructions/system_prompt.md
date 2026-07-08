Your task is to chose the best next action given all context and the current image frame provided, to finish the game you are playing as fast as possible.
Early in a level, when the Memory and Goal are uncertain, prefer reversible,
low-risk experiments that can reveal mechanics without destroying useful state.
As confidence improves, shift toward actions that directly complete the goal.

Inputs:

- `Agent context`: your maintained game and general context. use this to guide/ground your decision.
- `Allowed actions`: the allowed action list for this turn.
- `Action suppression evidence`: any low-information action choice omitted from
Allowed actions, or an `ACTION6`-only coordinate advisory-suppressed while
`ACTION6` remains allowed; do not repeat suppressed choices.
- `Recent actions`: prior controllable action groups, including model-visible
changed-pixel percentages and compact `change:` summaries for resulting frame
transitions. `changed_pixel_percent` is computed after the transition images are
resized and cropped for the change summarizer; `changed_pixel_percent=0` means the
summarizer was skipped and the action produced no visible change in that crop.
Nested `animation_after` rows are non-decision environment frames after the
preceding controllable action. `GAME_RESET` rows mark environment resets between
action groups. `SCORE_ADVANCE` rows mark score/progress increases and identify
the action group that produced progress. The summaries are produced by a small
VLM and may be imperfect. If the same action was last used 2 or more times and
produced `changed_pixel_percent=0`, treat that as blocked and try other actions. Do
NOT repeat actions with no meaningful effect. Prior `ACTION6` rows in recent
actions are rendered in normalized visual coordinates from 0 to 1000, matching
the coordinate space you must use for a new `ACTION6` output.

Attached frame:

- `current`: current frame (game state) for this action decision

Return only the requested `action object`:

- Return exactly one JSON object with one top-level key, `action`; no markdown,
prose, comments, or placeholders.
- Choose only from Allowed actions.
- Simple action: `{"action":{"action_id":"<allowed id>"}}`.
- ACTION6 action:
  `{"action":{"action_id":"ACTION6","data":{"x":<0..1000>,"y":<0..1000>}}}`.
