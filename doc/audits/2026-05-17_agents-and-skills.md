# Audit: Agents And Skills

## Sources

- `AGENTS.md`
- `.agents/skills/architect/SKILL.md`
- `.agents/skills/backend-developer/SKILL.md`
- `.agents/skills/models-specialist/SKILL.md`
- `.agents/skills/doc-auditor/SKILL.md`
- `.agents/skills/dashboard-developer/SKILL.md`
- `README.md`
- `doc/architecture/software/overview.md`
- `doc/architecture/software/models/overview.md`
- `doc/architecture/software/orchestration/overview.md`
- `doc/architecture/software/updates/overview.md`
- `doc/architecture/software/memory/overview.md`
- `doc/architecture/software/runtime/overview.md`
- `doc/architecture/software/environment/overview.md`
- `src/face_of_agi/models/adapters.py`
- `src/face_of_agi/models/__init__.py`
- `src/face_of_agi/models/world/adapter.py`
- `src/face_of_agi/models/goal/adapter.py`
- `src/face_of_agi/orchestration/orchestrator.py`
- `src/face_of_agi/orchestration/tool_runtime.py`
- `src/face_of_agi/orchestration/game_loop/actions/post_decision_predictions.py`
- `debug/dashboard/**`

## Findings

- [x] **AGENTS.md is mostly aligned with the current codebase shape.**
  `AGENTS.md` correctly points agents to `doc/` as architecture context, warns
  against broad architecture changes without confirming the vision, preserves
  the no-external-API-test rule, and routes model-role work to
  `models/orchestrator_agent`, `models/world`, `models/goal`,
  `models/description`, and `models/updater`. That matches the current
  architecture docs, where orchestration owns the loop and world/goal are
  provider-neutral model roles, and it matches the current model registry and
  exports in `src/face_of_agi/models/adapters.py` and
  `src/face_of_agi/models/__init__.py`.

- [x] **Resolved: AGENTS.md now mentions the dashboard skill that exists in `.agents`.**
  The "Skills And Specialist Agents" section lists architect,
  models-specialist, backend-developer, dashboard-developer, and doc-auditor.
  The dashboard skill is present and
  relevant because the current repo contains a local Streamlit dashboard under
  `debug/dashboard/**`, and `README.md` documents the dashboard setup and run
  commands.

- [x] **Resolved: the architect skill no longer says S/G are tools.**
  `.agents/skills/architect/SKILL.md` now keeps role detail in the architecture
  docs and focuses on planning, module-boundary decisions, and specialist
  routing.

- [x] **Resolved: the models-specialist skill now uses current folders and role semantics.**
  `.agents/skills/models-specialist/SKILL.md` now scopes ownership to
  `models/orchestrator_agent`, `models/world`, `models/goal`,
  `models/description`, and `models/updater`. It names S/G as self-improving
  model roles and points agents back to the model architecture docs.

- [x] **The backend-developer skill is broadly relevant to current non-model code.**
  Its scope covers orchestration, runtime, memory, environment, contracts,
  updates, and backend tests. That aligns with the module map in
  `doc/architecture/software/overview.md` and with current source directories
  under `src/face_of_agi/`. Its operating rules also match the newer
  architecture: orchestration owns loop coordination and persistence decisions,
  environment stays thin, SQLite access stays in memory modules, and shared
  contracts remain provider-neutral.

- [x] **Resolved: the backend-developer skill names the tool-runtime boundary.**
  The backend skill now refers to tool runtime without restating model-role
  architecture.

- [x] **The doc-auditor skill matches this audit task and remains relevant.**
  The skill allows only new files in `doc/audits/`, forbids source and
  architecture-doc edits, requires reading `doc/architecture/software/`, and
  asks for findings with concrete doc and code paths. That behavior is still
  consistent with the repo's docs-as-source-of-context rule in `AGENTS.md`.

- [x] **The dashboard-developer skill matches the current dashboard boundary.**
  The current dashboard files live under `debug/dashboard/**`, and the skill
  correctly treats `src/**` as read-only by default so dashboard work does not
  accidentally alter runtime, model, orchestration, memory, environment, or
  shared-contract behavior. This scope is useful enough that it should be
  advertised from `AGENTS.md`.

## Gaps

- [x] **Resolved: S/G wording is consistent across agent instructions.**
  The current target is clear in `doc/architecture/software/overview.md`,
  `doc/architecture/software/models/overview.md`, and
  `doc/architecture/software/orchestration/overview.md`: S/G are
  self-improving model roles whose contexts feed X and P. The skills now point
  to those docs instead of repeating the architecture.

- [x] **Resolved: specialist routing covers dashboard work.**
  `.agents/skills/dashboard-developer/SKILL.md` exists and matches real code,
  and `AGENTS.md` now lists it for `debug/dashboard/**`.

- [x] **Resolved: the old model-tool path name was removed from specialist instructions.**
  The current tracked model role paths are `models/world` and `models/goal`.
  The model specialist skill now points to those paths and `models/description`.

- [x] **Resolved: tool-call language now stays out of S/G role routing.**
  Agent X still has generic tool-call scaffolding in
  `src/face_of_agi/models/orchestrator_agent` and an orchestration runtime hook,
  while S/G role semantics live in the model docs.

## Suggested Follow-Up

- [x] Update `.agents/skills/architect/SKILL.md` so it points agents to the
  architecture docs instead of repeating X/S/G details.

- [x] Update `.agents/skills/models-specialist/SKILL.md` to use the current
  `models/world` and `models/goal` paths, and to describe S/G as
  self-improving model roles.

- [x] Add `.agents/skills/dashboard-developer` to the `AGENTS.md` specialist routing
  list for work under `debug/dashboard/**`.

- [x] Tighten `.agents/skills/backend-developer/SKILL.md` from generic
  "tools" wording to the orchestration tool-runtime boundary.
