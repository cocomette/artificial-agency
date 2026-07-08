# Runtime Config

Runtime YAML config is loaded by `face_of_agi.environment.config`.

## Required Runtime Keys

- `max_actions_per_level`
- one of `game_index`, `game_indices`, `game_ids`, or `game_selection` for most
  normal runs
- `models`

## Active Model Shape

```yaml
models:
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  agent:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  change:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  world:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  historizer:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  level_summary:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  agent_creator:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  agent_creator_role_author:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  updater:
    agent_probing:
      backend: vllm
      model: Qwen/Qwen3.6-35B-A3B-FP8
    agent_policy:
      backend: vllm
      model: Qwen/Qwen3.6-35B-A3B-FP8
    general:
      backend: vllm
      model: Qwen/Qwen3.6-35B-A3B-FP8
```

`models.change`, `models.world`, `models.historizer`, `models.level_summary`,
`models.updater.agent_probing`, `models.updater.agent_policy`, and
`models.updater.general` are required active game-loop slots. `models.agent`
is required by runtime assembly. `models.agent_creator` and
`models.agent_creator_role_author` are optional sidecar slots, but must be
configured together when used. They maintain learned roles in the creator
database; those roles are not proposed to the historizer and do not affect
updater selection. `models.shared_vlm` supplies defaults for local shared
Ollama/vLLM configs.

```yaml
agent_creator:
  base_learned_roles_file: data/agent_creator
  use_learned_roles: true
  batch_size: 8
  max_tool_calls: 4
  max_roles: 8
```

The top-level `agent_creator` section configures the sidecar creator store.
When creator model slots are configured, runtime startup writes a run-local
numbered `agent_creator_XX.sqlite` store near the memory database. If
`use_learned_roles` is true, startup copies from the latest numbered database
matching `base_learned_roles_file`; if no source exists, startup fails.

Supported active real backends are OpenAI, Ollama, and vLLM where implemented
by each role. Configurable and Hugging Face updater placeholders remain
development hooks and are not used by committed runtime configs.

## Action Windows

`probing_actions_window` and `policy_actions_window` configure exactly how
many ordered actions the corresponding agent updater must return in
`next_actions`. Both values are positive integers and default to `1`. A value
above `1` makes the runtime execute a fixed-length updater-planned action chain
while only running change summary and action-history compilation between middle
actions.

`probing_mode_cap_ratio` configures the deterministic probing cap applied after
the historizer proposes `probing`. The runtime computes
`(probing_actions_window + recent_probing_mode_count) / action_history_window`;
when the ratio is greater than `probing_mode_cap_ratio`, orchestration runs
policy instead. The ratio defaults to `0.35` and must be between `0` and `1`.

## Animation Bundles

Post-action animation bundles are passed to change summary as full ordered
frame arrays. The runtime advances directly to the final bundle frame as the
next controllable frame. Change-summary animation inputs are cropped normally,
then split into overlapping chunks of at most `max_frames_per_call` frames
before provider calls; the default is `10`, and each later chunk starts with
the previous chunk's final frame. Each chunk is then resized so it fits within
`animation_frame_budget_coefficient` configured input-frame areas on the change
model config. The coefficient defaults to `2` and values below `2` are clamped
to `2`. Change-summary configs may also set `gaussian_blur_kernel_size` and
`gaussian_noise_deviation`; kernel size `0` or `1` disables blur, while odd
values above `1` blur each prepared frame copy before noise. Deviation `0`
disables noise, while positive deviation adds independently sampled
zero-centered Gaussian RGB noise to each change-summary frame copy immediately
before the provider call. Change-summary configs may set `activate_diff_mask`
to insert black/white changed-pixel masks between each consecutive prepared
observation frame. vLLM change-summary configs may set
`frame_input_mode: video` and `video_fps` to send prepared transition frames as
one `data:video/jpeg` pre-extracted frame sequence with `media_io_kwargs`.
Change-summary configs may set `activate_components` to include deterministic
same-symbol connected-component facts in the prompt. `max_nb_components`
defaults to `50` and caps the cumulative number of listed components per frame
after grouped rows are sorted by larger shape first and lower duplicate count
first for same-size shapes. Each frame's component facts are extracted from the
full cropped frame.
World-model calls attach only the final current frame.

## Memory Compatibility

The current state-memory schema stores agent context only. Local SQLite run
databases created by older runtime shapes should be reset instead of migrated.
