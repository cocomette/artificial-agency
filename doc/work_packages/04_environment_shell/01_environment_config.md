# Step 01: Environment Config

## Objective

Add the smallest environment-local config boundary needed for the starter loop.

## Implementation

- Add a small dataclass in `face_of_agi.environment`.
- Load config from YAML.
- Keep the required shell fields to:
  - selected `game_id`
  - `max_actions_per_level`
- Allow a few ARC runtime settings with defaults:
  - `operation_mode`
  - `environments_dir`
  - `recordings_dir`
  - `seed`
  - `render_mode`
- Keep this config out of the shared runtime config.

## Parallelism

This step can run in parallel with Step 02.

## Acceptance Check

- Environment config is loaded from YAML.
- Only the minimal shell fields are required.
