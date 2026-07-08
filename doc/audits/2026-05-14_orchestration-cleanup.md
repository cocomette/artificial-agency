# Audit: Orchestration Cleanup

## Sources

- `doc/architecture/software/overview.md`
- `doc/architecture/software/orchestration/overview.md`
- `doc/architecture/software/orchestration/responsibilities.md`
- `doc/architecture/software/orchestration/game_loop/state_machine.md`
- `doc/architecture/software/orchestration/game_loop/interfaces.md`
- `src/face_of_agi/orchestration/game_loop/state_machine.py`
- `src/face_of_agi/orchestration/game_loop/lifecycle.py`
- `src/face_of_agi/orchestration/game_loop/session.py`
- `src/face_of_agi/orchestration/game_loop/persistence.py`
- `src/face_of_agi/orchestration/game_loop/helpers.py`
- `src/face_of_agi/orchestration/game_loop/actions/*.py`
- `src/face_of_agi/orchestration/orchestrator.py`
- `src/face_of_agi/debug/sinks/terminal.py`
- `src/face_of_agi/debug/capture/model_io.py`
- `src/face_of_agi/debug/capture/model_inputs.py`
- `src/face_of_agi/debug/sanitize.py`
- `src/face_of_agi/orchestration/tool_runtime.py`
- `src/face_of_agi/models/orchestrator_agent/tooling.py`
- `src/face_of_agi/models/updater/contracts.py`
- `src/face_of_agi/models/updater/adapter.py`
- `src/face_of_agi/contracts.py`
- `src/face_of_agi/runtime/configs/*.yaml`

## Findings

- [x] **Resolved: main game-loop code is now shaped like the documented state machine.**
  The docs describe explicit states such as `START_RUN`, `LOAD_FRAME_BUFFER`,
  `ENTER_FRAME_TURN`, `BUILD_DECISION_INPUT`, `CALL_X`, `SYNTHESIZE_NONE`,
  `RUN_POST_DECISION_PREDICTIONS`, `RUN_UPDATER`, `PERSIST_TURN`, and
  `ADVANCE`. The implementation now keeps
  `src/face_of_agi/orchestration/game_loop/state_machine.py` as the compact
  coordinator and moves the concrete lifecycle, action, persistence, helper,
  and session concepts into focused modules under `game_loop/`.

- [x] **Resolved: `run()` is split into small transition handlers.**
  `GameLoopStateMachine.run()` now owns the explicit loop entry/exit and calls
  named handlers for lifecycle checks, frame-buffer loading, frame-turn entry,
  decision, S/G predictions, updater calls, persistence, and advance.
  This was done as a behavior-preserving cleanup before deeper state semantics
  changes.

- [x] **Resolved: `_persist_observation_once()` was removed.**
  `src/face_of_agi/orchestration/game_loop/state_machine.py` has a helper named
  `_persist_observation_once`, but it only returns an `ObservationRef` and adds
  the observation id to a local set. Actual durable persistence happens through
  `StateMemory.prewrite_state()` and `StateMemory.complete_state()`. The helper
  and local observation-id set were removed; the state machine now constructs
  `ObservationRef(memory="state", id=...)` directly where a reference is needed.

- [x] **Kept: `previous_source_state_id` is a tactical updater breadcrumb.**
  `FrameTurnContext.previous_source_state_id` is currently needed to resolve
  the previous-turn world game context for Agent X's prompt updater. This
  should be revisited during the planned M/turn-ledger redesign, but it is not
  removable in the current implementation.

- [x] **Resolved: `previous_observation` was removed from `FrameTurnContext`.**
  `FrameTurnContext` carries `previous_observation`, but Agent X input does not
  currently use it. Updater input receives the current frame as the previous
  side of the transition through `UpdaterFrameTransitionInput`, not through
  `FrameTurnContext.previous_observation`. The frame-context field has been
  removed; `UpdaterFrameTransitionInput.previous_observation` remains.

- [x] **Resolved: `first_source_state_id` was removed from `FrameTurnContext`.**
  The state machine computes `first_source_state_id` with recent-history
  fallback logic, but current Agent X prompts intentionally omit
  `source_state_ids`, and all checked runtime configs set
  `agent.max_tool_calls: 0`. If live Agent X tools remain disabled, this source
  id machinery is unnecessary in the active path and has been removed.

- [x] **Resolved: Agent X tool runtime no longer carries S/G role behavior.**
  Current runtime configs under `src/face_of_agi/runtime/configs/` all set
  `agent.max_tool_calls: 0`. The orchestration path still builds
  `OrchestrationAgentToolRuntime` on controllable frames and debug trace can
  report available tools. The runtime keeps tool-passing separate from the
  S/G model-role flow.

- [x] **Resolved: Agent X tool schemas were separated from S/G model roles.**
  The active orchestration path now keeps S/G as model roles with prediction
  outputs for updater evidence. Provider-side world/goal tool schemas and
  descriptions were removed from the Agent X path.

- [x] **Resolved: `ActionHistoryEntry` now carries only current X prompt data.**
  The model prompt helper currently sends only `action` and `controllable` from
  recent action history. The core history object was trimmed to those two
  fields, so source state id, control reason, observation ref, turn id, step,
  frame index, frame count, and reasoning summary no longer circulate through
  the model-facing hot path.

- [x] **Resolved: model-facing history is no longer debug/replay history.**
  `ActionHistoryEntry` is now the prompt-facing recent-action DTO only. Richer
  replay/debug facts remain in the committed M-state trace, control metadata,
  and turn metrics rather than the prompt history object.

- [x] **Resolved: M-state metadata no longer stores `update_input`.**
  `_persist_turn_shell()` used to store `metadata["update_input"]` after
  stripping raw observations. Much of that data was already stored in
  first-class M columns: action, trace, S/G predictions, transition
  evidence, current observation, and contexts. The only persisted reader was
  the dashboard overview panel, and it only displayed a few duplicated
  transition refs. The snapshot was removed from M metadata and the dashboard
  now reads only first-class persisted state fields.

- [x] **Resolved: runtime transition data is separated from persistence/debug boundaries.**
  The loop now keeps mutable run state in `GameLoopSession` and stable
  beginning-of-turn inputs in frozen `FrameTurnSnapshot`. Persistence is
  centralized in `game_loop/persistence.py`, with reusable memory helpers
  handling M-state prewrite/complete operations. Debug rendering remains a
  separate open concern below.

- [x] **Resolved: debug terminal rendering moved out of orchestration.**
  `DebugTrace` now lives under `src/face_of_agi/debug/sinks/terminal.py`, with
  provider capture/drain helpers in `debug/capture/model_inputs.py`, trace I/O
  collection in `debug/capture/model_io.py`, and sanitization in
  `debug/sanitize.py`. Orchestration emits typed debug events, but it no longer
  owns the Rich terminal renderer or debug helper framework.

- [x] **Resolved: direct debug calls replaced with neutral instrumentation.**
  Orchestration now emits typed debug events through `DebugBus`, while
  `DebugTrace` acts as the terminal sink under `debug/`. Existing model-input
  debug persistence remains behind the debug boundary and continues to write
  the dedicated `model_input_debug_records` table.

- [ ] **Clarify: debug trace modes are volume levels, not task-oriented views.**
  Current modes are `off`, `minimal`, `agent_decision`, `verbose`, and
  `model_inputs`. They gate methods inside `DebugTrace`, but do not answer
  specific debugging questions. This creates both noise and blind spots.
  Consider replacing them with event views such as run summary, turn summary,
  decisions, model I/O, and full JSONL.

- **Verified keep: `AgentTrace` is core runtime data, not merely debug output.**
  The pretty rendering of traces belongs in `debug/`, but `AgentTrace` is still
  built as part of `DecisionResult`, carried through `UpdaterFrameTransitionInput`,
  passed into the agent game-context updater, and persisted on `m_states`.
  It should remain part of core contracts unless the architecture changes.

- **Verified keep: S/G predictions are required by current configs with real S/G models.**
  The refactored loop still calls S/G predictions every frame
  turn before updater execution. The runner fails if world/goal models are not
  configured, the runtime configs wire OpenAI or Ollama world/goal roles, and
  the resulting predictions are persisted to M.

- **Verified keep: updater calls are required by current configs.**
  The orchestrator now requires an updater task registry before constructing
  the game loop. Each frame turn calls world, goal, and agent game updaters,
  while normal game-end completion calls the general updater. Current runtime
  configs wire OpenAI or Ollama updater slots, so updater execution remains
  active runtime behavior.

- **Verified keep: M persistence is required for learned contexts, source rows, dashboard, and cleanup policy.**
  The runtime shell wires `StateMemory(SQLiteDatabase(...))` for normal runs.
  The game loop prewrites and completes one M row per frame turn, learned
  context hydration reads latest M rows at run start, the dashboard reads
  `m_states`, and cleanup prunes `m_states` after normal runs unless debug
  retention is enabled. The low-level orchestration boundary still allows
  `state_memory=None` for tests/injection, but SQLite-backed M is part of the
  product runtime shape.

## Gaps

- [x] **Architecture/code shape gap resolved:** docs describe explicit state
  transitions, and the implementation now exposes those transitions through a
  small coordinator plus focused lifecycle/action/persistence/session modules.

- [x] **Ownership gap resolved for terminal rendering:** orchestration emits
  debug events through the debug package, while Rich terminal rendering and
  debug helper code live under `src/face_of_agi/debug/`.

- [x] **Configuration/behavior gap resolved:** current configs keep Agent X
  tool calls disabled with `max_tool_calls: 0`, and orchestration keeps tool
  runtime separate from S/G model-role behavior.

- [x] **Data-boundary gap resolved for the game-loop hot path:** runtime
  execution now flows through `GameLoopSession` and `FrameTurnSnapshot`, while
  M-state persistence/debug record writes are handled at the persistence and
  memory-helper boundary.

- [x] **Naming gap resolved:** `_persist_observation_once()` no longer implies
  durable persistence because the misleading helper was removed.

- [ ] **Trace gap:** current terminal modes can print bulky model/updater data
  while omitting the concise turn-level facts a human likely wants.

# PR review cleaning tasks
- [x] **Reasoning summaries:** Agent X no longer requests or persists provider
  reasoning summaries or final-action reasoning text.
- [x] **Action coordinate space:** removed the separate action/bbox coordinate
  config knobs; model-native visual coordinate space now comes from explicit
  provider/model profiles in `vision_profiles.json`.

- [x] **Max responses + 2:** removed with the legacy Agent X live-tool response
  loop. Agent X now uses one shared provider `step(...)` loop where each step
  carries the final action schema; tool calls continue the loop and final
  structured output ends it.

- [x] **WORLD_TOOL_DESCRIPTION** and  **GOAL_TOOL_DESCRIPTION** : removed
  world/goal-specific Agent X live-tool descriptions and schemas.

- [x] **Repair and validate:** kept the shared repair loop and extracted the
  repeated provider repair callback plumbing used by world, goal, and updater.
  Agent X final-action repair now follows the same validation-error plus
  invalid-output repair shape through provider-specific repair messages.

- [x] **Finalize action:** replaced the separate finalize-action model phase
  with the shared Agent X step loop. The final action remains the structured
  committed Agent X output, but it is now returned as the terminal output of
  the same tool-capable step loop rather than through a separate finalizer call.

- [x] **State machine must read/write db**: the game loop now uses reusable
  `StateMemory` helpers for M-state context hydration, prewrite/complete, and
  model-input debug writes instead of scattering DB-shaped calls through the
  state-machine flow. The legacy generic state/experimental record paths were
  removed; dedicated M-state and debug tables remain.

- [x] **Same description parsing functions in goal and world model adapter**:
  actual parsing stays shared in contracts; provider/image adapter plumbing now lives in shared description helpers.

- [x] **Instructions in tooling.py:** removed the per-turn JSON
  `instructions` payload field from Agent X input. The Agent X system prompt
  was left unchanged.

- [x] **Tool runtime input to Agent X:** action history is now passed as a
  normal Agent X model input, not carried through the future tool-runtime
  boundary.

- [x] **Rename transition evidence:** renamed the shared runtime/persistence
  bundle to `TurnMetrics`, the SQLite column to `turn_metrics_json`, and the
  Agent updater subset to `AgentProgressFeedback`.
  
- [x] **Remove random and mock paths:** removed the random Agent X provider,
  mock/noop updater provider path, mocked S/G predictions, mock-only
  configs, and tests/docs that promoted those starter scaffold modes.

- [x] **Remove cheat_action_context:** removed the runtime config flag, game-dir
  override, shell context seeding path, local source parser helper, YAML
  entries, tests, and docs that treated parsed game-source action semantics as
  initial role context.

## Suggested Follow-Up

- [x] **Step 1: mechanically split the state machine.**
  Extracted named transition handlers from `GameLoopStateMachine.run()` and
  moved lifecycle, actions, helpers, session state, and persistence into
  focused modules without intentional behavior changes.

- [x] **Step 2: introduce a small run cursor.**
  `GameLoopSession` replaced the long list of `run()` local variables and now
  owns current run state, frame-buffer position, counters, refs, recent
  history, outputs, and terminal result.

- [x] **Step 3: rename or remove misleading helpers.**
  `_persist_observation_once()` was removed. Durable M-state writes now flow
  through `prewrite_frame_turn_source()` and `complete_frame_turn_state()`,
  which wrap the lower-level `prewrite_state()`/`complete_state()` calls for
  the game loop.

- [x] **Step 4: separate runtime DTOs from debug/replay DTOs.**
  `FrameTurnSnapshot` is now the immutable beginning-of-turn runtime input, and
  persistence/debug writes are assembled at the memory/persistence boundary
  instead of being mixed into the main loop.

- [x] **Step 5: decide Agent X live tool-call scope.**
  Keep S/G as model roles with prediction outputs for updater evidence, while
  preserving the tool-runtime framework for configured Agent X tools.

- [x] **Step 6: trim `FrameTurnContext`.**
  Removed unused `previous_observation` and `first_source_state_id` fields.
  `previous_source_state_id` remains as a tactical internal cursor for Agent X
  updater context until M/turn-ledger semantics are redesigned.

- [x] **Step 7: trim action history.**
  Model-facing action history is split from persisted/debug turn history, so
  the model prompt stays compact.

- [x] **Step 8: audit M-state metadata.**
  `metadata["update_input"]` was removed. The dashboard no longer depends on
  this broad updater snapshot and uses first-class M-state fields instead.

- [x] **Step 9: move debug presentation out of orchestration.**
  Moved Rich terminal rendering and shared debug helpers under `debug/` without
  changing terminal trace payloads. Orchestration now emits typed debug events
  through the debug bus, and `DebugTrace` renders those events as the terminal
  sink.

- [ ] **Step 10: redesign terminal views.**
  Replace volume-based trace modes with question-oriented views: quiet run
  result, turn summary, decisions, model I/O, and full JSONL/file output.

- [x] **Step 11: update tests after each accepted cleanup.**
  Existing tests assert current trace labels, action-history persistence, and
  M-state snapshots. The game-loop and SQLite-memory tests were updated for
  the accepted cleanup and the non-external suite passes.
