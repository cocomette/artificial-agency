# Memory Overview

Runtime memory has two SQLite-backed domains:

- `M` state memory for committed frame turns and learned agent context.
- `E` experimental memory for generic Agent X tool outputs.

`M` stores the current observation, chosen action, Agent X trace, agent
context, turn metrics, metadata, and timestamps. The v1 schema also has
explicit tables for turn ledgers, candidate predictions, judge scores, goal
predictions, rewards, replay samples, and LoRA update attempts. Existing local
run databases using older table shapes are incompatible and should be reset.

`E` remains a rolling per-run tool-output store. Its rows are generic and are
not tied to a specific model role.
