# Audit: Orchestrator Agent

## Sources

- `doc/architecture/software/models/roles.md`
- `doc/architecture/software/models/inputs.md`
- `doc/architecture/software/models/outputs.md`
- `doc/architecture/software/orchestration/overview.md`
- `doc/architecture/software/orchestration/responsibilities.md`
- `doc/architecture/software/orchestration/game_loop/state_machine.md`
- `doc/architecture/software/shared_contracts/contracts.md`
- `doc/run_runtime.md`
- `src/face_of_agi/contracts.py`
- `src/face_of_agi/models/orchestrator_agent/`
- `src/face_of_agi/orchestration/game_loop/state_machine.py`
- `src/face_of_agi/orchestration/orchestrator.py`
- `src/face_of_agi/orchestration/tool_runtime.py`
- `src/face_of_agi/orchestration/game_loop/post_decision_predictions.py`
- `src/face_of_agi/runtime/shell.py`
- `tests/test_runtime_smoke.py`
- `tests/test_orchestrator_agent_adapters.py`
- `tests/test_contracts.py`

Verification run:

```bash
uv run --group dev pytest tests/test_runtime_smoke.py tests/test_orchestrator_agent_adapters.py tests/test_contracts.py
```

Result: 32 passed.

## Findings

- The basic `X` model boundary is implemented. The architecture requires `X`
  to receive context, first/current observations, action space, and an
  `AgentToolRuntime`, then return one `DecisionResult` with a final action and
  `AgentTrace` (`doc/architecture/software/models/roles.md:21`,
  `doc/architecture/software/models/inputs.md:19`,
  `doc/architecture/software/models/outputs.md:5`). The protocol in
  `src/face_of_agi/models/orchestrator_agent/contracts.py:19` and
  `src/face_of_agi/models/orchestrator_agent/contracts.py:59` matches that
  shape.

- Random action selection now lives behind the configured `X` adapter, matching
  the orchestration responsibility rule that orchestration validates and
  applies `X` decisions rather than choosing randomly itself
  (`doc/architecture/software/orchestration/responsibilities.md:42`). The
  default `OrchestratorAgentAdapter` selects from the supplied action space and
  emits an `AgentTrace` (`src/face_of_agi/models/orchestrator_agent/adapter.py:49`,
  `src/face_of_agi/models/orchestrator_agent/adapter.py:61`).

- The OpenAI and Ollama `X` adapters implement a bounded native tool loop
  rather than only a random shell. They build prompt payloads with the composed
  `C^X` context, first/current observation metadata, visible refs, tool policy,
  and allowed actions (`src/face_of_agi/models/orchestrator_agent/tooling.py:43`);
  send first/current frame images (`src/face_of_agi/models/orchestrator_agent/tooling.py:78`);
  expose provider-native `world`, `goal`, and `submit_action` tools
  (`src/face_of_agi/models/orchestrator_agent/tooling.py:115`,
  `src/face_of_agi/models/orchestrator_agent/tooling.py:133`); invoke
  orchestration for tool calls; and return a trace with tool calls/results and
  backend metadata (`src/face_of_agi/models/orchestrator_agent/openai_adapter.py:49`,
  `src/face_of_agi/models/orchestrator_agent/openai_adapter.py:180`,
  `src/face_of_agi/models/orchestrator_agent/ollama_adapter.py:49`,
  `src/face_of_agi/models/orchestrator_agent/ollama_adapter.py:164`).

- Provider output validation is materially present. Tool-call parsing enforces
  known tool names, state/experimental observation refs, allowed final actions,
  and coordinate data for complex ARC actions
  (`src/face_of_agi/models/orchestrator_agent/tooling.py:151`,
  `src/face_of_agi/models/orchestrator_agent/tooling.py:230`,
  `src/face_of_agi/models/orchestrator_agent/tooling.py:243`).

- The frame-unrolled game loop calls `X` once per frame turn, supplies
  synthetic `NONE` on non-final frames, supplies real environment actions on
  final frames, passes the tool runtime, validates `NONE` and final action
  constraints, and steps the ARC environment only after a final controllable
  decision (`doc/architecture/software/orchestration/game_loop/state_machine.md:10`,
  `src/face_of_agi/orchestration/game_loop/state_machine.py:145`,
  `src/face_of_agi/orchestration/game_loop/state_machine.py:178`,
  `src/face_of_agi/orchestration/game_loop/state_machine.py:198`,
  `src/face_of_agi/orchestration/game_loop/state_machine.py:323`).

- Orchestration-owned tool routing and `E` persistence are implemented for
  agent-requested experiments. `OrchestrationAgentToolRuntime.invoke()` delegates
  to orchestration (`src/face_of_agi/orchestration/tool_runtime.py:98`), and
  `Orchestrator.invoke_tool_for_experiment()` resolves the source observation,
  routes to the configured world or goal tool, writes the output to rolling
  experimental memory, prunes by turn buffer, and returns an experimental
  `ObservationRef` (`src/face_of_agi/orchestration/orchestrator.py:97`,
  `src/face_of_agi/orchestration/orchestrator.py:117`,
  `src/face_of_agi/orchestration/orchestrator.py:130`).

- Post-decision world/goal predictions are separated from agent-requested tool
  calls, matching the docs that these committed predictions are not appended to
  `AgentTrace` and are not stored in `E`
  (`doc/architecture/software/orchestration/game_loop/state_machine.md:101`,
  `src/face_of_agi/orchestration/game_loop/post_decision_predictions.py:34`).
  The game loop passes them to the updater and persists them in `M`
  (`src/face_of_agi/orchestration/game_loop/state_machine.py:212`,
  `src/face_of_agi/orchestration/game_loop/state_machine.py:378`).

- Runtime wiring supports `random`, `openai`, and `ollama` agent backends,
  explicit world/goal backend selection, and config-driven tool-call budgets
  (`src/face_of_agi/runtime/shell.py:184`,
  `src/face_of_agi/runtime/shell.py:207`,
  `src/face_of_agi/runtime/shell.py:224`). The runtime docs cover these setup
  knobs (`doc/run_runtime.md:112`).

## Gaps

- The biggest code-vs-doc mismatch is animation-frame tool access. The
  architecture says tool handling exists on both controllable and
  non-controllable frames, and that during frame unrolling `X` may still run
  experiments even though it cannot affect the environment
  (`doc/architecture/software/orchestration/game_loop/state_machine.md:80`).
  The current implementation exposes no tools on non-controllable frames
  (`src/face_of_agi/orchestration/orchestrator.py:253`,
  `src/face_of_agi/orchestration/orchestrator.py:273`) and rejects runtime
  invocation when tools are disabled
  (`src/face_of_agi/orchestration/tool_runtime.py:106`). The prompt also tells
  `X` not to call world or goal tools on non-controllable frames
  (`src/face_of_agi/models/orchestrator_agent/instructions/system_prompt.md:10`),
  and `doc/run_runtime.md:167` documents the implemented behavior. Either the
  architecture docs are stale, or the runtime is under-implementing the target.

- Tool results are persisted and captured in the final trace, but the immediate
  provider feedback sent back into the active `X` reasoning loop is narrower
  than the docs imply. The architecture says orchestration returns a
  `ToolResult` plus an experimental ref and `X` may add that result and ref to
  active context (`doc/architecture/software/models/roles.md:27`). The adapter
  feedback JSON currently includes only `tool` and `observation_ref`, with the
  image appended separately when possible
  (`src/face_of_agi/models/orchestrator_agent/tooling.py:269`,
  `src/face_of_agi/models/orchestrator_agent/openai_adapter.py:190`,
  `src/face_of_agi/models/orchestrator_agent/ollama_adapter.py:174`). Tool
  explanations, metadata, source refs, and action data are available in stored
  trace objects but are not fully returned to the model context.

- State observation references are stable logical observation ids, not clearly
  persisted `M` record ids. `FrameTurnContext` uses
  `ObservationRef(memory="state", id=observation.id)` before the turn is
  persisted (`src/face_of_agi/orchestration/game_loop/state_machine.py:310`),
  and provider trace construction reconstructs refs from observation ids
  (`src/face_of_agi/models/orchestrator_agent/tooling.py:186`). State ref
  resolution scans `M` rows by stored observation payload id and special-cases
  the current live frame (`src/face_of_agi/orchestration/orchestrator.py:314`,
  `src/face_of_agi/orchestration/orchestrator.py:317`). This is workable for
  the current runtime, but it is weaker than the docs' language around exact
  persisted references from `M` and could become ambiguous if observation ids
  collide, are reused after reset, or need to identify one specific `m_states`
  row.

- The target language says `X` may call world and goal tools "any number of
  times" before returning a final action (`doc/architecture/software/models/roles.md:5`).
  Current provider adapters intentionally bound the loop with
  `max_tool_calls`, defaulting to two (`src/face_of_agi/models/orchestrator_agent/config.py:15`,
  `src/face_of_agi/models/orchestrator_agent/openai_adapter.py:79`,
  `src/face_of_agi/models/orchestrator_agent/ollama_adapter.py:78`). This is a
  sensible runtime safety constraint, but the architecture docs should say the
  tool chain is flexible within configured budget rather than literally
  unbounded.

- Updater handoff includes the full live trace, tool results by role, and
  post-decision predictions, but reward/update quantities are still empty
  placeholders (`src/face_of_agi/orchestration/game_loop/state_machine.py:402`).
  That means the agent-context updater boundary is structurally present, but
  not yet fulfilling the richer `Q^X` evidence expectation described by the
  architecture.

- Tests prove the contract and fake-backend behavior, but not live provider
  behavior. The OpenAI/Ollama adapter tests use fake clients and assert request
  shapes, tool-loop sequencing, repair, and trace capture. There is no
  automated live-model e2e check that an actual configured `X` reliably submits
  `NONE` for animation frames, calls tools with valid refs, or recovers from
  provider-specific malformed tool output.

## Suggested Follow-Up

- Decide whether non-controllable animation frames should allow experimental
  world/goal calls. If yes, change `_available_tool_names`,
  `tools_enabled`, the `X` prompt, and runtime tests. If no, update
  `doc/architecture/software/orchestration/game_loop/state_machine.md` and
  related architecture docs to make the current policy explicit.

- Expand `tool_result_feedback()` so `X` receives a fuller provider-safe
  `ToolResult` payload immediately: experimental ref, source ref, action when
  present, explanation, and safe metadata, plus the image payload when
  renderable.

- Normalize `ObservationRef` identity for `M`: either make state refs point to
  `m_states` row ids, or document that state refs are logical observation ids
  and add collision/reset safeguards.

- Update the architecture wording from "any number of times" to "zero or more
  times within the configured per-decision budget" unless truly unbounded tool
  chaining is still intended.

- Populate `RewardUpdateQuantities` from environment score/progress, prediction
  comparisons, and trace cost so the agent updater receives meaningful
  structured evidence.

- Add one optional live-provider smoke script or skipped integration test that
  exercises a full `X` decision with real tool schemas and validates final
  action submission, without making the normal unit suite depend on network or
  local model availability.
