# Memory Overview

Memory stores runtime state in SQLite.

- `m_states`: durable frame-turn state memory.
- `e_experiments`: rolling experimental records.
- `run_metadata`: run-level metadata.
- `model_input_debug_records`: captured model inputs for dashboard/debugging.

Orchestration decides what to write and when. Model adapters do not read or
write SQLite directly.
