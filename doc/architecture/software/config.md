# Runtime Config

Runtime YAML is loaded by `face_of_agi.environment.config`.

## Required Keys

- `max_actions_per_level`
- one of `game_index`, `game_indices`, `game_ids`, or `game_selection`
- `agent`

Legacy `models:` configs are rejected.

## Agent Shape

```yaml
agent:
  backbone:
    backend: transformers
    model_family: qwen3_5_moe_multimodal
    model_path: /kaggle/input/face-of-agi-qwen36-35b-fp8-weights
    processor_path: null
    device: auto
    dtype: auto
    image_size: 224x224
    local_files_only: true
    representation_layer: image_tokens_mean
    model_kwargs:
      device_map: auto
  online:
    buffer_size: 512
    adapter_rank: 16
    ensemble_size: 5
    hidden_dim: 512
    learning_rate: 0.001
    batch_size: 32
  replay:
    max_updates_per_turn: 8
    max_seconds_per_turn: 0.5
    solved_level_updates: 32
  planner:
    horizon: 3
    candidate_count: 64
    coordinate_candidates: 16
    diagnostic_turns: 4
```

YAML accepts only `agent.backbone.backend: transformers`. Test code may inject
the deterministic fake backbone directly through dataclasses.

## Animation Keyframes

`animation_keyframe_pixel_threshold` controls how many changed raw frame
cells/pixels are needed before an intermediate animation frame becomes a
retained frame turn. It defaults to `8`; set it to `0` to keep every
non-duplicate frame. The final frame in each environment bundle is always
retained.

## Memory Compatibility

The current state-memory schema stores learner snapshots and learner traces.
Local SQLite run databases created by older runtime shapes should be reset
instead of migrated.
