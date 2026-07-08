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
- previous frame-turn reference, when available
- current frame reference
- current frame payload
- frame index and buffer length
- frame control mode
- bounded recent action history
- current context documents
- internal current source-state reference for committed predictions

## `XDecisionInput`

Represents the provider-neutral input to the orchestrator agent role. This is
created only for controllable final frames.

- role context for `X`
- first observation reference, previous frame-turn reference, and current frame
  reference
- current frame payload
- bounded recent action history from prior frame turns, limited to action and
  controllability
- allowed action space for this frame
- current world and goal game contexts
- optional tool-runtime extension point
- frame control metadata

## `UpdaterFrameTransitionInput`

Represents the update boundary for one frame turn.

- current frame reference
- actual next frame reference or payload
- decision trace
- S/G description prediction packet
- submitted real action, if any
- synthetic `NONE` decision, if any
- transition timing and score/progress metadata for Agent X prompt updates
- previous and current context document references

## Game Context Update Inputs

`WorldGameContextUpdateInput` and `GoalGameContextUpdateInput` represent the
updater input for world and goal game-context documents.

- previous role context
- committed role-specific post-decision description prediction
- previous/current observation frame attachments
- submitted real action or synthetic `NONE` for world updates

They do not receive live Agent X tool results, transition timing, or
score/progress metadata.

## `AgentGameContextUpdateInput`

Represents the updater input for the orchestrator agent context document.

- previous agent context
- previous observed frame, `o_t`
- current observed frame after the last action, `o_t+1`
- current-turn world and goal game contexts from the same pre-update context
  generation used by the S/G updater inputs
- previous-turn world game context when available
- full live decision trace
- action-progress `time_cost` and `score_delta` when available

The decision trace remains the source of the submitted action or synthetic
`NONE` and reasoning summary. Synthetic animation-frame traces are generated
by orchestration.

## `GeneralKnowledgeUpdateInput`

Represents the end-of-run input for the shared general updater task.

- target role: world, goal, or agent
- previous role context with final game context and current general context
- run summary metadata and persisted state record ids

## `PostDecisionPredictions`

Represents S/G description predictions produced by orchestration for updater
evidence.

- world prediction from current frame plus final action, including synthetic
  `NONE` on animation frames
- goal prediction from current frame
