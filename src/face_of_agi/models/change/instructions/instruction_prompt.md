Compare the attached observation frame array.

The first image is the previous observation. The final image is the current
observation. Any images between them are retained animation frames from the same
action transition. Use the `ACTION` block as the action that caused the
transition.

Return only the requested JSON object. Do not include markdown, prose, comments,
or placeholders.

Use the changed_pixel_percent value as authoritative. It is computed
from the exact images you see. If it is greater than 0,
never say that nothing changed.

Return exactly one JSON object with a `summary` string and a `change_detected`
boolean.

Set `change_detected` to true if any visible change is detected anywhere across
or between the attached frames. Set it to false only when the attached frames
show no visible changes at all and are fully identical.

The summary must be one or two concise sentences describing ALL visible playfield
changes across the attached frames. If no playfield changed, say that no visible
playfield change occurred. Do not use metaphorical nor analogical descriptions. Stick to exact, simple visual facts such as shape, colors, patterns, positions, layout, background, and orientations.

For `ACTION6`, the `ACTION.data.x` and `ACTION.data.y` coordinates are
normalized visual coordinates from 0 to 1000. Describe the area or element
targeted by those coordinates on the first image before describing the
transition.
