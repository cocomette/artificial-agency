Your task is to choose the best next action given all context and the current image frame provided, to finish the game you are playing as fast as possible.

Inputs:

- `Agent context`: your maintained game and general context. use this to guide/ground your decision.
- `Allowed actions`: the allowed action list for this turn. The action glossary
is only a helper; observed facts from the frame and recent transitions win.
- `Action suppression evidence`: any low-information action choice omitted from
Allowed actions, or an `ACTION6`-only coordinate advisory-suppressed while
`ACTION6` remains allowed; do not repeat suppressed choices.
- `Game memory`: compact same-run memory of actions taken, visible changes,
score/progress changes, likely mechanics, failed patterns, and current
objective hypotheses. On the first turn this may be `not available`. Use it as
orientation for the whole game so far, but treat the current frame, Allowed
actions, Action suppression evidence, and Recent actions as more immediate
evidence when they conflict. Do not output or mention run ids or game ids.
- `Recent actions`: prior controllable action groups, including model-visible
changed-pixel counts and either compact `change:` summaries or structured
`Elements and associated changes` bullets for resulting frame transitions.
Elements may be targets, triggers, objects, characters, buttons, collectibles,
obstacles, paths, layout, or any other game-relevant artifacts. `changed_pixels`
is computed after the transition images are resized and cropped for the change
summarizer, comparing the first and final evidence frames. `changed_area` is the
same evidence as a display-only percentage. For bundled animation transitions,
`changed_pixels=0` can still include transient intermediate-frame changes; use
the change text or element bullets to distinguish transient animation from no
visible effect.
Nested `animation_after` rows are non-decision environment frames after the
preceding controllable action. `GAME_RESET` rows mark environment resets between
action groups. `SCORE_ADVANCE` rows mark score/progress increases and identify
the action group that produced progress. The summaries are produced by a small
VLM and may be imperfect. If the same action was last used 2 or more times and
produced `changed_pixels=0` with no useful change or element effect, treat that
as blocked and try other actions. Do NOT repeat actions with no meaningful
effect.
Prior `ACTION6` rows in recent actions are rendered as target strings naming
the visible object or area that was selected. New `ACTION6` output shape is
defined by the Allowed actions line: `ACTION6(x,y normalized_0_1000,target)`
requires coordinate `data`; `ACTION6(target,bbox)` requires a tight target
bounding box and clicks its center; `ACTION6(target,bbox,target_rgb_color)`
requires a tight target bounding box and RGB target color. Avoid blindly
repeating the same coordinate, object, or region when recent history says that
target was stale or uninformative.

Use Game memory and Recent actions together: Game memory gives compact long-run
state and hypotheses, while Recent actions gives the exact latest transition
evidence. Do not let a stale memory hypothesis override a more recent failed or
successful action row.

Attached frame:

- `current`: current frame (game state) for this action decision

Return only the requested `action object`:

- Return exactly one JSON object with one top-level key, `action`; no markdown,
prose, comments, or placeholders.
- Choose only from Allowed actions.
- Simple action: `{"action":{"action_id":"<allowed id>"}}`.
- ACTION6 coordinate action:
  `{"action":{"action_id":"ACTION6","data":{"x":<0..1000>,"y":<0..1000>},"target":"<visible object or area>"}}`.
- ACTION6 bbox/color action:
  `{"action":{"action_id":"ACTION6","target":"<visible object or area>","bbox":[<x0>,<y0>,<x1>,<y1>],"target_rgb_color":[<r>,<g>,<b>]}}`.
  The bbox is normalized visual 0..1000 around the target area, and
  `target_rgb_color` is the exact RGB color of target pixels inside it.
- ACTION6 bbox-center action:
  `{"action":{"action_id":"ACTION6","target":"<visible object or area>","bbox":[<x0>,<y0>,<x1>,<y1>]}}`.
  The bbox is normalized visual 0..1000 around the target area; the runtime
  selects the bbox center.
