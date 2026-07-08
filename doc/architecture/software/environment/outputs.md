# Environment Outputs

## Observations

The environment module returns normalized `Observation` values to
orchestration. An observation may contain:

- stable observation id
- step number
- one frame or a frame bundle
- raw ARC frame data
- metadata copied from the ARC toolkit

## Environment Info

The environment module exposes `EnvironmentInfo` values containing:

- current game id
- lifecycle state
- available actions reported by ARC-AGI
- completed levels
- target win levels when known
- pass-through metadata

## Boundary Rule

The environment module reports facts from ARC-AGI. It does not decide what the
agent should do with those facts.
