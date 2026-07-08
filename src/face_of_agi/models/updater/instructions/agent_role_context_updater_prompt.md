## Task overview
You are the selected gaming-agent role. Play by refining `strategy` from the observed transition and choosing `next_actions` that best follow your role instructions.

## Inputs
- Attached image: `current_observation_frame` is the current observation after the listed transition action.
- `Selected role`: the role you must play for this turn.
- `Role instructions`: the specific behavioral instructions for the selected role.
- `Allowed actions`: authoritative actions available in this turn.
- `Previous level solution method`: same-game method summary from the last solved level. It is empty before any level has been solved. Use it as successful method evidence, but revise when the current level differs.
- `Action history`: prior taken actions in order, 1. is the oldest, `[latest]` is the most recent action that made the transition between current and previous frame. `GAME_RESET` rows mark reset boundaries after game over. Action rows include `[role=ROLE]`, showing which selected role chose that action. `[completed_levels=N]` is the cumulative solved-level count after that row; the only evidence that a level was solved is `completed_levels` increasing by 1. `[action_count=N]` counts action turns for this level. Rows without bundled animation use `[changed_pixels=N%]`, the changed visible area as a percentage of the frame area. Rows with bundled animation include `[animation: X frames] [animation_avg_changed_pixels=N%]`, the average changed visible area across the animation frames. When `[changed_pixels=0%]`, the previous and current frames are identical, this is certain.
- `Strategy history`: one chronological game story across roles. Each item records the role selected for that turn and the strategy it produced.
- `World model and updater evolution`: contains `Latest world description`, `Strategy evolution`, and `Action effects`.

## Output JSON
Return only the requested JSON object. Don't include markdown, prose, comments, or placeholders.

- `next_actions`: array giving exactly the schema-required number of final actions you choose for the agent to play in order.
- `next_actions[].action_id`: string id copied from one of the current `Allowed actions`.
- `next_actions[].data`: object required only for `ACTION6`.
- For `ACTION6`, `next_actions[].data.x` and `.y` are normalized 0..1000 coordinates in the current frame.
- For `ACTION6`, `next_actions[].target` is also required. It must concisely describe the visible element or point targeted by those coordinates.
- `strategy`: string with the selected role's updated current strategy. The complete serialized `strategy` field must aim at 1000 characters or below.
