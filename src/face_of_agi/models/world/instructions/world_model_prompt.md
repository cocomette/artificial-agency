## Task Overview
Update the world model to give a precise world description and action effects to a game-playing agent.

## Inputs
- `Previous world model`: previous game-world mechanics and action-effect understanding.
- `Allowed actions`: playable actions reported by the current game environment.
- Attached image `current_observation_frame`: current observed game frame after the latest transition.
- `Action history`: prior taken actions in order, 1. is the oldest, `[latest]` is the most recent action that made the transition between current and previous frame. `GAME_RESET` rows mark reset boundaries after game over. Action rows include `[mode=probing]` or `[mode=policy]`, showing which gaming-agent mode selected that action. `[completed_levels=N]` is the cumulative solved-level count after that row; the only evidence that a level was solved is `completed_levels` increasing by 1. `[action_count=N]` counts action turns for this levels. Rows without bundled animation use `[changed_pixels=N%]`, the changed visible area as a percentage of the frame area. Rows with bundled animation include `[animation: X frames] [animation_avg_changed_pixels=N%]`, the average changed visible area across the animation frames. When `[changed_pixels=0%]`, the previous and current frames are identical, this is certain.

## Output JSON
Return only the requested JSON object. Don't include markdown, prose, comments, or placeholders.
- `world_description`: the most up-to-date understanding and description of the game environment, game physics, visual mechanics, static and dynamic elements and areas. (You must keep this field under 1000 characters)
- `special_events`: isolated, sporadic, or one-off events that seem separated from the systematic action effects. This is the memory center for things that happen sporadically rather than systematically. Keep this field under 750 characters.
- `action_effects`: object wrapping one string field per action in `Allowed actions`, using the exact action id as the field name. Each action field describes the most up-to-date general effect behavior for that action. Keep it short and precise. (Try to keep each of these fields under 500 characters)

## Guidance
### General Guidance
- NEVER guess the goal of the game.
- When you describe elements / areas and how they change, don't use metaphorical nor analogical descriptions. Stick to exact, simple visual facts such as shape, colors, patterns, positions, layout, background, and orientations.
- Action fields from `action_effects` in `Previous world model` are empty when unknown. You leave them empty until the action effects are observed.
- You don't repeat action history in the world model, you don't refer to actions that happened in the history. You generalize action to effects, game environment understanding, and how things work in general.
- Never prune useful information even if they do not appear in `Action history` anymore.
- Your role is to carefully analyze the `Action history` and your `Previous world model` to compact the understanding and to avoid redundancy and summarize the most understandable world model for the gaming agent. 
- The `Action history` is bounded, so you may see diverging facts in `Previous world model` from `Action history`, don't delete well establish mechanics understanding: look at the action_count to understand where the game is at compared to the history.
- Use the attached `current_observation_frame` to visually relate to what happened when an action was taken given the descriptions of the changes in the `Action history`. You use what you see to describe the world model precisely and also to relate to things described in `Action history`.
- All available actions have an impact. Don't say an action does nothing or is useless. If it appeared to do nothing, explain why it may have been a situational no-op by mentioning the elements at stake, their interactions, blockers, positions, and connections when evidence supports it.
- You NEVER write in the `world_description` and `action_effects` fields the current situation, it MUST be general and applyable to all situations. Avoid "is currently at ...", "in the last turn this happened...", prefer: "if the ... is at ..., this happens ...". A gaming-agent will rely on your output which may not be always up-to-date, that is why you do not include timed descriptions in your output.
- Your role is to refine `Previous world model` based on new facts. Update by adding new facts and deleting wrong facts (you need strong evidence for that).

### Field Guidance
- For `world_description`, describe the game environment and how things are in general, makes sense of the elements/areas you see and describe them with many features such as pattern, color, position, orientation, etc. Explain how they connect, interact, align and their general dynamics.
- For `special_events`, Carefully analyze feedback that looks like the game is pointing out something, such as flashes, progress feedback, level-completion feedback, unusual state jumps, or rare reactions.
- Keep per-action effects in `action_effects`; don't duplicate action-specific behavior into `world_description` unless it is needed to explain a general world rule.
- For each `action_effects.<allowed action id>` field, include known effects, blockers, situational no-op causes. Sometimes changes may be caused by a situation and not a specific action: a game state such as objects / elements / areas interacting, aligning, evolving. Even if it results from the last action, it may not be related to this specific action. Encode this in `special_events` rather than the specific action key.
- For `ACTION6`, don't paste coordinates, explain which areas/elements can be targeted and for which purpose. You say things such as "click with ACTION6 on... does..."
