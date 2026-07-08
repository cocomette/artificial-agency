## Task Overview
Update compact game context for a game-playing agent.

## Inputs
- `Previous compacter context`: previous compact context with `world_description`, `special_events`, per-action `action_effects`, `previous_actions_summary`, and `previous_strategy_summary`, or `not available`.
- `Allowed actions`: playable actions reported by the current game environment.
- Attached image `current_observation_frame`: current observed game frame.
- `Current frame components`: structured data generated from `current_observation_frame`. Components are 4-connected regions of the same rendered grid color. Diagonal contact alone does not join components. Each row groups components with the same one-word color and exact same shape. `nb` is the number of components in this group, and `box` lists their bounding boxes normalized from 0 to 1000, with x increasing right and y increasing down.
- `Action history`: prior taken actions in order, 1. is the oldest, `[latest]` is the most recent action that made the transition between current and previous frame. `GAME_RESET` rows mark reset boundaries after game over. `[completed_levels=N]` is the cumulative solved-level count after that row; a level was solved when this value increases. After a solved-level transition, raw action history starts over for the new current level. `[action_count=N]` counts only actions in this current-level history and restarts at 1 after the reset. Rows without bundled animation use `[changed_pixels=N%]`, the changed visible area as a percentage of the frame area. Rows with bundled animation include `[animation: X frames] [animation_avg_changed_pixels=N%]`, the average changed visible area across the animation frames. When `[changed_pixels=0%]`, the previous and current frames are identical, this is certain.
- `Strategy history`: previous agent `current_strategy` entries in order, 1. is the oldest and `[latest]` is the newest.

## Output JSON
Return only the requested JSON object. Don't include markdown, prose, comments, or placeholders.
- `world_description`: the most up-to-date understanding and description of the game environment, game physics, visual mechanics, static and dynamic elements and areas. You must keep this field under 1000 characters.
- `special_events`: isolated, sporadic, or one-off events that seem separated from the systematic action effects. This is the memory center for things that happen sporadically rather than systematically. Keep this field under 750 characters.
- `action_effects`: object wrapping one string field per action in `Allowed actions`, using the exact action id as the field name. Each action field describes the most up-to-date general effect behavior for that action. Keep it short and precise. Keep each of these fields under 500 characters.
- `previous_actions_summary`: compact analysis of what worked, what failed, where actions were no-ops, where repeated states/cycles occurred, and which path currently seems useful to solve the level. You must aim at 2000 characters or below for this field.
- `previous_strategy_summary`: compact analysis of how the agent strategy evolved, what hypotheses worked or failed, what stale goal should be avoided, and what strategic understanding should guide the next agent update. You must aim at 2000 characters or below for this field.

## Guidance
### General Guidance
- NEVER guess the goal of the game.
- When you describe elements or areas and how they change, don't use metaphorical nor analogical descriptions. Stick to exact, simple visual facts such as shape, colors, patterns, positions, layout, background, and orientations.
- Action fields from `action_effects` in `Previous compacter context` are empty when unknown. Leave them empty until the action effects are observed.
- Don't prune useful information in `special_events` even if it does not appear in `Action history` anymore.
- Use `Action history`, `Strategy history`, `Previous compacter context`, `current_observation_frame`, and `Current frame components` to update the compact context.
- Use `current_observation_frame` and `Current frame components` to visually relate to what happened when an action was taken given the descriptions of changes in `Action history`.
- Use `Current frame components` as compact evidence for exact visible element location, size, color, and grouping, while keeping `current_observation_frame` as the source of truth.
- All available actions have an impact. Don't say an action does nothing or is useless. If it appeared to do nothing, explain why it may have been a situational no-op by mentioning the elements at stake, their interactions, blockers, positions, and connections when evidence supports it.
- In `world_description` and `action_effects`, write general rules rather than the current situation. Avoid "is currently at ..." or "in the last turn this happened..."; prefer "if the ... is at ..., this happens ...".
- Refine `Previous compacter context` based on new facts. Add new facts and delete wrong facts when there is strong evidence.
- When `[completed_levels=N]` increases, the attached frame and latest action history still describe the level that was just completed. Keep any relevant completion event or final mechanic in `world_description`. After a solved-level boundary, raw action history and `action_count` restart for the new level; then previous facts may be stale, so rewrite drastically when new evidence shows a different layout or mechanics.

### Field Guidance
- For `world_description`, describe the game environment and how things work in general. Explain visible elements or areas with features such as pattern, color, position, orientation, layout, connections, interactions, alignment, and dynamics.
- For `special_events`, analyze feedback that looks like the game is pointing out something, such as flashes, progress feedback, level-completion feedback, unusual state jumps, or rare reactions.
- Keep per-action effects in `action_effects`; don't duplicate action-specific behavior into `world_description` unless it is needed to explain a general world rule.
- For each `action_effects.<allowed action id>` field, include known effects, blockers, and situational no-op causes. Sometimes changes may be caused by a situation and not a specific action, such as objects, elements, or areas interacting, aligning, or evolving. Encode that in `special_events` rather than the specific action key when appropriate.
- For `ACTION6`, don't paste coordinates. Explain which areas or elements can be targeted and for which purpose, such as "click with ACTION6 on ... does ...".
- In `previous_actions_summary`, describe at a high level what was tried, what worked, what failed, repeated states/cycles/no-ops, and the conceptual path that seems useful. Don't copy a chain of actions.
- In `previous_strategy_summary`, describe the strategy that should guide the agent. Keep failed or cyclic attempts only when they explain why the useful strategy is better.
- If a level was just solved, `previous_actions_summary` should describe the high-level conceptual path used to solve it, and `previous_strategy_summary` should describe the successful strategy that solved it.
