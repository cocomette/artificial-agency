# Models Overview

The models module owns provider-neutral model role boundaries. It contains
model-specific folders, contracts, configs, and adapters, while keeping the
runtime orchestration boundary independent of the vLLM transport details.

The implemented real model roles are:

- orchestrator agent `X`
- transition change summary model
- agent context historizer
- updater `P`

The current real backend for every implemented model role is vLLM through its
OpenAI-compatible Chat Completions API. The `openai` Python SDK is used only as
the HTTP client for that vLLM endpoint; there is no OpenAI model provider.
Ollama, HuggingFace, and Diffusers provider paths are not part of the current
runtime.

## Target Folder Intent

The source tree keeps model roles separate:

- `models/orchestrator_agent`: decision model `X`
- `models/change`: transition change summary model
- `models/historizer`: agent context history summary model
- `models/updater`: updater model `P`
- `models/observation_text.py`: shared ARC-grid-to-text serialization

Each role exposes a provider-neutral contract to orchestration. vLLM adapters
live inside role-local `providers/` folders and translate between the framework
shape and the vLLM request shape. `models/providers/` is the final provider-call
layer: it owns OpenAI-compatible Chat Completions request assembly, the actual
vLLM call, and raw response normalization. It should not own role concepts such
as `AgentTrace`, updater tasks, or instruction-folder selection.

Provider-neutral contracts are intentionally kept even though the only real
backend is vLLM. That keeps orchestration from branching on concrete provider
objects and preserves clear role boundaries.

## Observation Contract

Implemented frame-consuming vLLM roles receive `ObservationText` plus cropped
PNG image attachments for the same ARC cells. The serializer accepts native 2D
ARC integer grids only, crops to original
coordinates `x=3..60` and `y=3..60`, optionally prints cropped rows with
original `0..63` coordinate labels and uppercase ARC symbols `0..F`, lists
4-connected same-symbol components unless disabled by config or the per-frame
overflow budget is exceeded, falls back from verbose component `runs=` geometry
to compact component fields before omitting components, optionally groups
same-symbol same-shape components, and emits component-level deltas for frame
bundles and change prompts.
Component IDs are frame-local; if either adjacent frame omits components, delta
text keeps only adjacent changed-cell counts and does not emit component IDs.
Observation-facing role instructions include a static ARC symbol color
glossary. Prompts still treat the serialized symbols and coordinates as
authoritative.

SQLite memory remains unchanged. Observation text is computed on demand while
building prompts, and cropped image payloads are computed at the vLLM adapter
boundary. Debug model-input capture stores the raw provider request including
image data URLs; terminal debug rendering sanitizes those data URLs.

Long retained transition bundles can be summarized in balanced overlapping
chunks. Each chunk sends images for the serialized frames it contains. When
multiple chunk summaries are produced, the optional final change reducer sees
ordered partial summaries, deterministic changed-cell metrics, action context,
row-only first/final/boundary keyframes, and cropped images for those selected
keyframes.

## Current Backend Notes

Each prompt-backed role call combines immutable role instructions with mutable
context. Agent `X` uses
`models/orchestrator_agent/instructions/system_prompt.md` as its immutable
instruction. Change, historizer, and updater roles use their role-local
instruction files. Mutable context remains ordinary text supplied through the
role contracts.

New Agent `X` ACTION6 outputs are validated against the active serialized crop:
with the default `crop_cells=3`, valid new coordinates are `x/y=3..60`.
Historical ACTION6 rows remain rendered as recorded original ARC grid data.
Cropped-frame changed counts are used for model-facing transition evidence and
no-change suppression. New ACTION6 outputs include both visible cropped ARC
coordinates and target text. Repeated zero-change ACTION6 attempts suppress
only the exact `x,y` coordinate in prompts; ACTION6 itself remains available.
