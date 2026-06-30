Your task is to choose the best next action from the current ARC observation and all context, to finish the game you are playing as fast as possible.

Inputs:

- `Agent context`: your maintained game and general context. use this to guide/ground your decision.
- `Current observation`: the current serialized ARC grid plus an attached
cropped image of the same cells. Use the image for visual pattern recognition
and the serialized text as authoritative evidence for exact symbols,
coordinates, component facts, and ACTION6 targets.
- `Allowed actions`: the allowed action list for this turn.
- `Action suppression evidence`: any low-information action choice omitted from
Allowed actions, or an `ACTION6` coordinate advisory-suppressed while `ACTION6`
remains allowed; do not repeat suppressed choices.
- `Recent actions`: prior controllable action groups, including model-visible
changed-cell counts and compact `change:` summaries for resulting frame
transitions. `changed_cells` is the cropped model-visible ARC cell count,
comparing the first and final serialized evidence observations. For bundled
animation transitions, `changed_cells=0` can
still include transient intermediate-frame changes; use the `change:` summary
to distinguish transient animation from no visible effect.
`changed_cells_pct` is the same first-to-final count as a percentage of the
visible crop. `completed_levels` and `action_count` give progress and current
level action count when available.
Nested `animation_after` rows are non-decision environment frames after the
preceding controllable action. `GAME_RESET` rows mark environment resets between
action groups. `SCORE_ADVANCE` rows mark score/progress increases and identify
the action group that produced progress. The summaries are produced by a model
and may be imperfect. If the same action was last used 2 or more times and
produced `changed_cells=0` with no useful `change:` effect, treat that as
blocked and try other actions. Do NOT repeat actions with no meaningful effect.
Prior `ACTION6` rows in recent actions are rendered in original ARC grid
coordinates and may include target text. For a new `ACTION6` output, choose
visible cropped coordinates inside the range stated in the action glossary and
allowed-action list, matching the serialized observation rows, and include a
non-empty `target` string naming the visible object, cell, or region.

Observation evidence:

- `current`: current serialized ARC grid observation for this action decision.
It is accompanied by one cropped image covering the same coordinate range.
When the image and serialized rows appear to disagree, trust the serialized
symbols and coordinates.

Return only the requested `action object`:

- Return exactly one JSON object with one top-level key, `action`; no markdown,
prose, comments, or placeholders.
- Choose only from Allowed actions.
- Simple action: `{"action":{"action_id":"<allowed id>"}}`.
- ACTION6 action:
  `{"action":{"action_id":"ACTION6","data":{"x":<visible-crop-x>,"y":<visible-crop-y>},"target":"<visible target>"}}`.
