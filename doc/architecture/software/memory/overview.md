# Memory Overview

Runtime memory has two SQLite-backed domains:

- `M` state memory for committed frame turns and learned agent context.
- `E` experimental memory for generic Agent X tool outputs.

`M` stores the current observation, chosen action, decision trace, agent
context, turn metrics, metadata, and timestamps. It no longer stores additional
role contexts or visual forecast artifacts. Existing local run databases using
the older table shape are incompatible and should be reset.

`E` remains a rolling per-run tool-output store for the dormant Agent X adapter
path. Its rows are generic and are not tied to a specific model role.
