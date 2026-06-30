# Environment Inputs

## Startup Inputs

- Environment config from runtime assembly.
- Selected ARC game id or game index.
- ARC toolkit paths such as environment and recording directories.
- Visualization and rendering settings when enabled.

## Loop Inputs

The environment module receives loop commands only from orchestration:

- select a game
- reset the selected game
- apply one real `ActionSpec`
- return action space and environment info

## Action Inputs

Actions are passed as shared `ActionSpec` values. Simple actions carry only the
ARC `GameAction`. Complex actions may include data, such as `x` and `y`
coordinates for coordinate actions.
