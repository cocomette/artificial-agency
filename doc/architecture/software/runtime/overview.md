# Runtime Overview

The runtime module is the program entry and assembly layer. It loads config,
constructs dependencies, selects startup behavior, and invokes orchestration.

Runtime does not own the real ARC game loop. Runtime starts orchestration;
orchestration owns the loop.

For configs with `game_indices`, runtime starts multiple isolated single-game
workers concurrently. Each worker gets its own environment, model adapters,
contexts, orchestrator, and SQLite memory file. Runtime shares only external
provider endpoints such as the vLLM server URL so provider-side request
batching can happen without moving game-loop ownership out of orchestration.
