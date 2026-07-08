# Runtime Config

Runtime YAML config is loaded by `face_of_agi.environment.config`.

## Required Runtime Keys

- `max_actions_per_level`
- one of `game_index`, `game_indices`, `game_ids`, or `game_selection` for most
  normal runs
- `models`

## Active Model Shape

The no-LoRA branch uses static `vllm` model-role inference. `models.shared_vlm`
supplies shared defaults for roles that use the same served FP8 model.

```yaml
models:
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  agent:
    backend: vllm
  change:
    backend: vllm
  memory:
    backend: vllm
  world:
    backend: vllm
  goal:
    backend: vllm
  interest:
    backend: vllm
  reward_judge:
    backend: vllm
```

`models.agent`, `models.change`, `models.memory`, `models.world`,
`models.goal`, `models.interest`, and `models.reward_judge` are required active
slots. `models.historizer` and `models.updater` are removed runtime keys and
fail config loading. Any `online_lora` block also fails config loading because
this branch has no adapter scheduler or training runtime.

The vLLM backend serves an OpenAI-compatible chat endpoint. Committed FP8 debug
configs use prefix caching, OpenAI chat-template content format, Qwen3
reasoning parsing, disabled thinking, and static high-concurrency settings.
They do not set runtime adapter flags.

Reward fields remain configurable. `reward_lp_weight_start` and
`reward_lp_weight_end` weight the immediate proxy learning-progress component,
which is Reward Judge prediction accuracy in this branch. This is text feedback
for later prompts, not measured model adaptation. Resource-cost fields are:
`reward_action_penalty`, `reward_trace_seconds_penalty`,
`reward_input_token_penalty_per_1k`, and
`reward_output_token_penalty_per_1k`.

## Animation Keyframes

`animation_keyframe_pixel_threshold` controls how many changed raw frame
cells/pixels are needed before an intermediate animation frame becomes a
retained frame turn. It defaults to `8`; set it to `0` to keep every
non-duplicate frame. The final frame in each environment bundle is always
retained.

## Memory Compatibility

The current schema requires a database reset. Local SQLite run databases
created by older runtime shapes should be reset instead of migrated.
