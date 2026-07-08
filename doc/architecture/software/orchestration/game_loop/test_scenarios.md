# Game Loop Test Scenarios

Runtime tests should cover framework mechanics:

- active OpenAI/Ollama/vLLM config files load
- updater-produced actions advance controllable frame turns
- change summaries become action-history entries
- agent updater receives bounded action history and progress evidence
- state memory persists agent context and decision trace
- parallel runtime specs isolate per-game state

Tests should not assert deleted behavior or prompt wording.
