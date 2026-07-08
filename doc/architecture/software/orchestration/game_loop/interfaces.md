# Game Loop Interface Sketches

These names describe architecture-level contracts. They are not required exact
class names.

## `FrameControlMode`

Represents whether the current frame can affect the real environment.

- `controllable`: boolean
- `allowed_actions`: `[NONE]` for non-final frames or real environment actions
  for final frames
- `reason`: for example `animation_unroll` or `real_environment_turn`

## `FrameUnrollBuffer`

Represents one ARC environment response after normalization.

- `bundle_id`
- ordered frame entries
- current index
- original environment info
- real action space from the environment

## `FrameTurnContext`

Represents one frame pass through orchestration.

- run id and game id
- first observation reference
- current frame reference
- current frame payload
- frame index and buffer length
- frame control mode
- current context documents
- memory references visible to `X`

## `XDecisionInput`

Represents the provider-neutral input to the orchestrator agent role.

- role context for `X`
- first observation reference and current frame reference
- current frame payload
- allowed action space for this frame
- tool interface exposed through orchestration
- frame control metadata

## `UpdaterFrameTransitionInput`

Represents the post-decision update boundary for one frame turn.

- current frame reference
- actual next frame reference or payload
- decision trace
- committed post-decision prediction packet
- submitted real action, if any
- synthetic `NONE` decision, if any
- tool outputs and predictions available for comparison
- previous and current context document references

## `ToolContextUpdateInput`

Represents the updater input for world or goal context documents.

- role discriminator: `world` or `goal`
- previous role context
- current frame reference
- actual next frame reference when available
- committed post-decision predictions
- matching tool results from the live decision trace
- submitted real action or synthetic `NONE`
- reward/update quantities and transition metadata

## `AgentContextUpdateInput`

Represents the updater input for the orchestrator agent context document.

- previous agent context
- current frame reference
- actual next frame reference when available
- full live decision trace
- committed post-decision prediction packet
- submitted real action or synthetic `NONE`
- reward/update quantities and transition metadata

## `PostDecisionPredictions`

Represents committed S/G predictions produced by orchestration after `X`
returns a valid real action.

- world prediction from current frame plus final action
- goal prediction from current frame
- absent on non-controllable animation frames
