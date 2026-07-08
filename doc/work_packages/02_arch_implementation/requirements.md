# Work Package 02: Arch Implementation Requirements

## Goal

Implement the first functional framework boundaries from `arch.md` and
`techstack.md` while keeping the system provider-neutral and flexible.

This package should make the skeleton concrete enough that future work packages
can implement modules independently against shared contracts.

## Source References

- `doc/architecture/arch.md`
- `doc/architecture/techstack.md`
- `doc/project/arc-agi-3_concept.md`
- `doc/project/arc-agi-3_technicals.md`

## Requirements

- Add `arc-agi` as a runtime dependency.
- Add `pytest` as a dev dependency.
- Define minimal typed contracts using dataclasses and Protocols.
- Keep observation frames generic with `Any`.
- Use role-specific protocols for world, goal, agent, and updater models.
- Keep model backends replaceable and do not add any provider-specific code.
- Add SQLite-backed generic memory domains for state memory `M` and experimental memory `E`.
- Use separate generic tables for `M` and `E`; do not design the final normalized schema yet.
- Add a runtime boundary that accepts multiple games in one call through injected environment adapters.
- Implement a smoke flow that calls environment reset, writes memory records, calls a fake agent, stores the trace, and stops.
- Do not call `env.step` in the smoke flow.
- Do not call the updater in the smoke flow because there is no real next observation.
- Use temporary SQLite databases in tests.
- Update README with setup and check commands.

## Acceptance Criteria

- `face_of_agi` imports successfully.
- Contracts import successfully.
- SQLite initializes separate `state_records` and `experimental_records` tables.
- State and experimental memory can write and list generic records.
- Runtime smoke flow can process a fake game through reset-only orchestration.
- Tests verify no environment step is called.
- `uv run pytest` passes.
