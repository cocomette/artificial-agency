## Task
Describe the current state and predict what will happen in the very next frame of this game (if this is the optimal frame after following any current goal hypotheses).

## Input
- `role_context`: current game goal and hypotheses.
- `observation`: the current game frame.

## Output
Return an exhaustive array of all identifiable areas you see, nothing left empty.
- Each and every area must be an object with exactly :
    -`bbox_2d`: [x0, y0, x1, y1] box coordinates in the image. The image starts at the top-left corner. x goes right. y goes down. (x0, y0) is the box top-left corner. (x1, y1) is the box bottom-right corner. You must keep x0 < x1 and y0 < y1.
    -`description`: visually describe everything you see such as precise colors, shape patterns, orientation, layout, background and other identifiable concept. Include how the area will change in the next frame given `role_context`.

## Rules
- In `description` to predict the next frame you follow the goals described in `role_context` and always analyze the goals with regard to the current `observation`.