You reconcile multiple ordered partial summaries of one ARC transition.

Use the ordered partial summaries as the primary evidence. Use the selected
keyframe rows and attached cropped keyframe images only to resolve
contradictions and preserve chronology. The first selected keyframe is from the
beginning of the transition, the final selected keyframe is from the end, and
any selected keyframes between them are ordered boundary frames. Use serialized
rows as authoritative for exact symbols and coordinates; use attached images
for visual shape and color scanning.

Use `changed_cell_count` as the authoritative count of cropped model-visible
ARC cells that differ between the first and final observations. It is a net
first-to-final comparison, not a count across every intermediate frame. Use
`any_adjacent_frame_changed` as the authoritative boolean for whether any
adjacent retained frame pair changed inside the cropped model-visible area.
Your returned `change_detected` must match `any_adjacent_frame_changed`
exactly. `any_adjacent_frame_changed` can be true even when
`changed_cell_count` is zero, because intermediate animation frames may change
and then return to the initial appearance.

Return exactly one JSON object with a `summary` string and a `change_detected`
boolean.

The summary must be one or two concise sentences describing the visible
playfield changes across the whole transition. If
`any_adjacent_frame_changed` is false and no playfield changed, say that no
visible playfield change occurred. Do not use metaphorical nor analogical
descriptions. Stick to exact, simple symbol-first facts such as shape, symbol
colors, positions, layout, background, and orientations. Refer to cells as
`symbol 0` through `symbol F` or as `A-cells`, `4-cells`, etc. You may add the
glossary color name when useful, but keep the symbol as the primary identifier.
