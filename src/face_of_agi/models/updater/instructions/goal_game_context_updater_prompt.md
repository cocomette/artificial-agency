## Task
You update the `previous_context` to describe the game goal, progress signals, and goal-relevant effects visible in the current frame.

## Input
- `previous_context`: previous goal and progress understanding.
- `current_observation_frame`: current observed frame after the action/frame turn.
- `prediction_description`: identified areas in `bbox_2d` bounding boxes, containing perfect descriptions of the previous observation and possibly imperfect predictions of transitions to `current_observation_frame` according to the goal. Areas can contain visible objects, colors, positions, layout, background, shapes and other conceptually identifiable things.

## Output
Return `updated_context` JSON field with:
- The refined goals ordered by priority, keeping the previous reasoning, what was observed before, what changed and the newly set cap.

## Rules
- Map the `prediction_description` areas `bbox_2d` to the `current_observation_frame` and observe what changed: objects moved, color changed, pattern evolved or other identifiable changes.
- Rewrite he goals based on those observations, if something else changed, we are probably having the wrong goal priorities. Come up with more goals, make them evolve based on the situations.
- The current goals assumptions in `previous_context` can be wrong and/or the agent may have explored, acted suboptimally, or made a move unrelated to the goal.
