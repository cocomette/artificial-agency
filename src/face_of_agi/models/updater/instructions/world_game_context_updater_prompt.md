## Task
You update the `previous_context` to describe game actions and their effects on the current frame.

## Input
- `previous_context`: previous game actions to effects understanding.
- `action`: submitted action.
- `current_observation_frame`: current observed frame after the action/frame turn.
- `prediction_description`: identified areas in `bbox_2d` bounding boxes, containing perfect descriptions of the previous observation and possibly imperfect predictions of transitions to `current_observation_frame`. Areas can contain visible objects, colors, positions, layout, background, shapes and other conceptually identifiable things.

## Output
Return the `updated_context` JSON field with:
- `world_understanding`: describe the most up-to-date understanding of the game physics and environment description BUT NEVER TRY TO GUESS THE GOAL.
- Update action keys with their general effect description. NEVER TRY TO GUESS THE GOAL, describe from the environment perspective not from a target to reach perspective.

## Rules AND How-to
- Map the `prediction_description` bbox to the `current_observation_frame` and observe what changed such as objects moved, colors changed, patterns evolved or other identifiable changes OR what did not work and why.
- Given the `action`, understand whether the prediction was correct or not. You refine the action effect description and `world_understanding` accordingly when necessary.
- Exhaustive action effect description matters, also include no-op: why would the action not work in a given situation, why would it be blocked or unchanged.
- Keep action effect descriptions frame independant, do not mention what happens in the current frame. Leave actions empty when you don't know their effects yet. 