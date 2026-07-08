## Task
Describe the current state and predict what will happen in the next frame of this game.

## Input
- `role_context`: game mechanics, controls, visual conventions, and action effects.
- `observation`: the current game frame.
- `action`: the submitted action.

## Output
Return an exhaustive array of all identifiable areas you see.
- Each area must be an object with exactly :
    - one `bbox_2d`: [x0, y0, x1, y1] box coordinates in the image. The image starts at the top-left corner (0, 0). x goes right. y goes down. (x0, y0) is the box top-left corner. (x1, y1) is the box bottom-right corner. You must keep x0 < x1 and y0 < y1.
    -`description`: visually describe everything you see such as precise colors, shape patterns, orientation, layout, background and other identifiable concept. Include how the area will change in the next frame.
    
## Rules
- In `description` to predict the next frame you follow the effects of the `action` described in `role_context`. You pay attention to blockers, impediments described in `role_context` and the state of the current frame.
