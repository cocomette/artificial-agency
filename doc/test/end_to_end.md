# End-To-End Checks

No provider-specific OpenAI, Ollama, HuggingFace, or Diffusers E2E runners are
kept in the repository. The supported real-model path is the runtime shell
against a vLLM OpenAI-compatible Chat Completions server.

Run commands from the repo root after starting vLLM separately:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

or choose one of the hardware-specific vLLM configs:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8.yaml
```

External-model E2E checks are manual because they depend on local hardware,
served model weights, and vLLM availability. Do not add them to the default
test suite, and do not run external API or live model tests unless explicitly
requested.
