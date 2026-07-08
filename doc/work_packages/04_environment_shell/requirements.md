# Work Package 04: Minimal Environment Shell

## Goal

Implement the first real environment shell between this repo runtime and the
ARC-AGI framework loop.

The result should stay intentionally simple:
- one selected game from environment config
- random action selection from the valid current ARC actions
- restart on `GameState.GAME_OVER`
- continue while `GameState.NOT_FINISHED`
- stop on final `GameState.WIN`
- stop when the per-level action counter is exhausted
- emit a condensed stdout trace

## Source References

- `doc/architecture/arch.md`
- `doc/architecture/techstack.md`
- `doc/project/arc-agi-3_technicals.md`

## Requirements

- Keep the environment config local to `face_of_agi.environment`.
- Use a YAML config file with:
  - `game_id`
  - `max_actions_per_level`
- Allow the same config to set basic ARC runtime settings such as
  `operation_mode`, `environments_dir`, and `seed`.
- Keep the environment adapter as the only direct ARC-AGI integration point.
- Do not redefine ARC-AGI framework enums locally.
- Use `arc_agi.Arcade.make(...)` as the real game creation path.
- Use `arcengine.GameAction`, `arcengine.GameState`, and `arcengine.FrameDataRaw`
  directly in the adapter/runtime boundary.
- Keep orchestration nearly empty and use random action selection only.
- Use the current `FrameDataRaw.available_actions` list as the only action source.
- Reset the per-level action counter when `levels_completed` increases.
- Restart the same game on game over.
- Emit a condensed stdout trace with frame count and selected action for each turn.
- Provide one simple runnable entrypoint for trying the shell locally.
- Do not add tests in this work package.
- Do not expand model, updater, or persistence behavior in this work package.

## Acceptance Criteria

- The environment module contains a YAML-backed starter config loader.
- The environment adapter can select one game through the ARC toolkit and
  surface `GameAction`, `GameState`, and `FrameDataRaw` data.
- The orchestration layer can return one random valid action for a turn.
- The runtime loop follows the stop and restart rules described above.
- Runtime writes the condensed trace to stdout only.
