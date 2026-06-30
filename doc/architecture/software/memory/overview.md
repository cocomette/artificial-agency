# Memory Overview

The memory module owns SQLite-backed storage primitives for runtime memory
domains. It exposes read and write operations to orchestration and keeps state
memory `M` separate from experimental memory `E`.

Memory does not decide what should be stored or reused. Orchestration is
responsible for choosing when model outputs, observations, traces, actions,
and updates are read, written, or returned to the agent as references.

## Current Shape

- `M`: committed, persistent run history that can be queried by reference.
- `m_states`: dedicated SQLite table for the current complete M state after
  each frame turn; normal run completion prunes this table to the latest row
  per game.
- `E`: rolling experimental buffer reserved for future tool-produced outputs.
- `e_experiments`: dedicated SQLite table for future tool output records.
- SQLite: embedded database backing both domains.
- Shared records: typed payloads for observations, traces, actions, context,
  metrics, and updates.
