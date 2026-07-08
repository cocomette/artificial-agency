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
- Reuse `Previous change elements` element names as much as possible when the same visual element is still present, so element names stay consistent across turns.
- Do not mention previous elements that are not visible anywhere in the attached frames.
- Mention elements that newly appear in the attached frames.
- Follow this process:
  1. Check `Previous change elements`, prune elements that are not present on the first image, and add new elements visible on the first image that were not in the list.
  2. For each attached frame in order, go through your current element list:
     - Describe how each element changes from the previous frame to this frame, including movement, rotation, transformation, color/layout changes, or other visible mutation.
     - Mention if an element disappears or reappears.
     - If a new element appears, add it to your list for the next frame-to-frame analysis.
  3. Compact the result into `elements`: for each element visible at least once in the attached images, give its general description and chronological mutation across the frames.
- Do not use metaphorical nor analogical descriptions. Stick to exact, simple visual facts such as shape, colors, patterns, positions, layout, background, and orientations.
- For `ACTION6`, the `ACTION.data.x` and `ACTION.data.y` coordinates are normalized visual coordinates from 0 to 1000. Describe the area or element targeted by those coordinates on the first image, before describing the transition.
- You never mention coordinates or pixel numbers in your output, don't say things such as "x=...", "y=...", simply don't mention numbers as coordinates or pixels to describe positions and movement. You describe positions or movements in natural language ONLY without outputing coordinates or pixel numbers.
