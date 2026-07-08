Compare the attached observation frame array.

The first image is the previous observation. The final image is the current
observation. Any images between them are retained animation frames from the same
action transition. Use the `ACTION` block as the action that caused the
transition.

Use `changed_pixel_count` as the authoritative count of model-visible pixels
that differ between the first and final attached images. It is a net
first-to-final comparison, not a count across every intermediate frame.
Use `any_adjacent_frame_changed` as the authoritative boolean for whether any
adjacent attached frame pair changed inside the model-visible area. Your
returned `change_detected` must match `any_adjacent_frame_changed` exactly.
`any_adjacent_frame_changed` can be true even when `changed_pixel_count` is
zero, because intermediate animation frames may change and then return to the
initial appearance.

If `changed_pixel_count` is greater than zero, never say that nothing changed.
If `changed_pixel_count` is zero and more than two images are attached, inspect
the intermediate images anyway. In that case, visible transient animation may
have happened and then returned to the initial appearance; describe those
transient element mutations. Only return `change_detected: false` when
`any_adjacent_frame_changed` is false and all attached images are visually
identical.

Reuse `Previous change elements` names as much as possible when the same visual
element is still present, so element names stay consistent across turns. Do not
mention previous elements that are not visible anywhere in the attached frames.
Mention newly visible elements.

Follow this process:

1. Check `Previous change elements`, prune elements that are not present on the
   first image, and add important new elements visible on the first image.
2. For each attached frame in order, update each visible element with movement,
   rotation, transformation, color/layout change, appearance, disappearance, or
   other visible mutation.
3. Compact the result into `elements`: one object per element visible at least
   once in the attached images.

For ACTION6 transitions, the action `data` is rendered in model-visible
normalized 0..1000 coordinates, and `target` names the object or area that was
selected. Describe the targeted object or area as it appears in the first image
before describing the transition. Use the target area and coordinate together as
action context, but visual facts from the attached frames are more important
than the action label.

Return exactly one JSON object:

{"elements":[{"element_name":"","element_description":"","element_mutation":""}],"change_detected":false}

Rules for `elements`:

- `element_name`: short stable name for the visible element. Every name in this
  response must be unique. If two similar elements need the same base name,
  suffix them as `base_name_0`, `base_name_1`, and so on.
- `element_description`: concise visual description of the element.
- `element_mutation`: chronological description of how the element changed
  across the attached frames. Leave this as an empty string when it stayed still
  with no visible change. Never write "no visible changes" here.

Do not use metaphorical nor analogical descriptions. Stick to exact, simple
visual facts such as shape, colors, patterns, positions, layout, background, and
orientations.
