# Audit: Goal Model Module

## Sources

- `doc/architecture/system_architecture.md`
- `doc/architecture/software/models/inputs.md`
- `doc/architecture/software/models/outputs.md`
- `doc/architecture/software/models/roles.md`
- `doc/architecture/software/orchestration/inputs.md`
- `doc/architecture/software/shared_contracts/contracts.md`
- `src/face_of_agi/contracts.py`
- `src/face_of_agi/tools/router.py`
- `src/face_of_agi/models/tools/goal/`
- `src/face_of_agi/models/tools/_image_editor.py`
- `tests/test_goal_tool_adapter.py`

## Findings

- The goal tool call signature now matches the architecture direction:
  `GoalToolModel.predict(context, observation)` and
  `GoalToolAdapter.predict(context, observation)` do not accept a candidate
  action.
- `ToolCall.action` and `ToolResult.action` are optional, and the router calls
  the goal model without an action while still requiring an action for world
  calls.
- The goal adapter uses the same Diffusers image-editor backends as the world
  adapter through the shared `_image_editor.py` backend.
- Goal prompt construction is role-specific and action-free. Long-context
  prompts include `GOAL MODEL DOC (K^G + L^G)` and source observation metadata;
  compact Pix2Pix/FLUX prompts use only goal instructions plus a short goal
  context hint.
- Tests cover goal prompt construction, all supported pipeline call shapes,
  goal `ToolResult` identity, no-action router dispatch, and absence of action
  text in goal prompts.

## Gaps

- `GoalToolAdapter.predict()` reconstructs
  `source_observation_ref=ObservationRef(memory="state", id=observation.id)`.
  The docs allow `O_ref` to come from either state memory `M_i` or experimental
  memory `E_i,t`; the current adapter cannot preserve that domain because it
  receives only the resolved `Observation`, not the original `ObservationRef`.
- The architecture says the goal model should support reasoning about the
  objective, hypothesis changes, and supporting visual or reward evidence. The
  current Diffusers-backed implementation returns an image plus a generic static
  explanation, but no model-derived textual reasoning or evidence.

## Suggested Follow-Up

- Pass the original `ObservationRef` through the tool boundary, or add it to
  the resolved observation payload, so goal and world results preserve state vs
  experimental provenance.
- Decide whether goal-model reasoning belongs in `ToolResult.explanation`,
  metadata, or a new structured contract. If it is required for updater input,
  implement a VLM/text step or backend response that actually produces it.
