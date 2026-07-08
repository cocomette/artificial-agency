## Task Overview
You compare the attached observation frame array.

## Inputs
- You get an array of images: the first image is the previous observation, the last image is the current observation. If more than two images are attached, every image between the first and last image is an animation frame that shows the transition over time. 

- The `ACTION:` block is the action that caused the transition from the first image to the last image.

## Output JSON
Return only the requested JSON object. Do not include markdown, prose, comments, or placeholders. You must keep the output under 2500 characters. 
- `elements`: array of objects. Each object has exactly:
  - `element_name`: short stable name for the visible element. Every `element_name` in this response must be unique. If two similar elements need the same base name, suffix them as `base_name_0`, `base_name_1`, and so on.
  - `element_description`: concise visual description of the element.
  - `element_mutation`: chronological description of how the element changed across the attached frames. Leave this as an empty string when it stayed still with no visible change. Never mention things such as "No visible changes / mutations".
- `change_detected`: boolean. Return `true` if any visible change is detected anywhere across and between all the attached image set. Return `false` only when the attached images show no visible changes at all and are all fully identical.

## Guidance
- Reuse `Previous elements` element names as much as possible when the same visual element is still present, so element names stay consistent across turns.
- `Previous elements` contains only `element_name` and `element_description`. It does not contain previous mutations.
- Do not mention previous elements that are not visible anywhere in the attached frames.
- Mention elements that newly appear in the attached frames.
- Feel free to merge elements that seem to belong together, split elements that seem to have more parts evolving separately.
- Follow this process:
  1. Check `Previous elements`, keep elements that are present on the first image, prune elements that are not present, and add new elements visible on the first image that were not in the list.
  2. For each attached frame in order, go through your current element list:
     - Describe how each element changes from the previous frame to this frame, including movement, rotation, transformation, color/layout changes, or other visible mutation.
     - Mention if an element disappears or reappears.
     - If a new element appears, add it to your list for the next frame-to-frame analysis.
  3. Compact the result into `elements`: for each element visible at least once in the attached images, give its general description and chronological mutation across the frames. You always refine `element_description` so it always matches the latest vsisible state.
- Do not use metaphorical nor analogical descriptions. Stick to exact, simple visual facts such as shape, colors, patterns, positions, layout, background, and orientations. 
- For `ACTION6`, the `ACTION.data.x` and `ACTION.data.y` coordinates are normalized visual coordinates from 0 to 1000. Describe the area or element targeted by those coordinates on the first image, before describing the transition.

## Component Guidance
- The `Frame components` block is deterministic structured data generated from the same observation frames attached to this request.
- Components are 4-connected regions of the same rendered grid color. Diagonal contact alone does not join components.
- Each row groups components with the same one-word color and exact same shape. `nb` is the number of components in this group, and `box` lists their bounding boxes normalized from 0 to 1000, with x increasing right and y increasing down.
- Use components as compact evidence for element location, size, color, and grouping, but the attached images remain the source of truth.
