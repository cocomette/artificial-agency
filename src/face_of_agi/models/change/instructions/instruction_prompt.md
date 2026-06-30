Compare the serialized observation frames and their attached cropped images.

The first serialized frame is the previous observation. The final serialized
frame is the current observation. Any serialized frames between them are
retained animation frames from the same action transition. Use the `ACTION`
block as the action that caused the transition.
Each serialized frame has a matching attached cropped image covering the same
ARC cells. Use images for visual shape and color scanning, and use the
serialized rows/components/deltas as authoritative evidence for exact symbols,
coordinates, changed-cell counts, and chronology.

Use `changed_cell_count` as the authoritative count of cropped model-visible
ARC cells that differ between the first and final serialized observations. It
is a net first-to-final comparison, not a count across every intermediate frame.
Use `any_adjacent_frame_changed` as the authoritative boolean for whether any
adjacent serialized frame pair changed inside the cropped model-visible area.
Your returned `change_detected` must match `any_adjacent_frame_changed`
exactly. `any_adjacent_frame_changed` can be true even when
`changed_cell_count` is zero, because intermediate animation frames may change
and then return to the initial appearance.
Component IDs such as `sA.1` are frame-local labels; do not treat matching IDs
across frames as persistent object identity unless the rows and component facts
independently support that continuity.

If `changed_cell_count` is greater than zero, never say that nothing changed.
If `changed_cell_count` is zero and more than two frames are serialized, inspect
the intermediate frames anyway. In that case, visible transient animation may
have happened and then returned to the initial appearance; summarize that
transient change instead of saying "no changes." Only say that no visible
playfield change occurred when `any_adjacent_frame_changed` is false and the
playfield has no meaningful visible change across all serialized frames.

Return exactly one JSON object with a `summary` string and a `change_detected`
boolean.

The summary must be one or two concise sentences describing ALL visible playfield
changes across the serialized frames. If no playfield changed, say that no
visible playfield change occurred. Do not use metaphorical nor analogical
descriptions. Stick to exact, simple symbol-first facts such as shape, symbol
colors, positions, layout, background, and orientations. Refer to cells as
`symbol 0` through `symbol F` or as `A-cells`, `4-cells`, etc. You may add the
glossary color name when useful, such as `symbol A light-cyan cells`, but keep
the symbol as the primary identifier.
