# Runtime Overview

The runtime module is the program entry and assembly layer. It loads config,
constructs dependencies, selects startup behavior, and invokes orchestration.

Runtime does not own the real ARC game loop. Runtime starts orchestration;
orchestration owns the loop.

For configs with `game_indices`, runtime starts multiple isolated single-game
workers concurrently. Each worker gets its own environment, online learner,
frozen local backbone, and SQLite memory file. There is no shared model server
in the active Kaggle Transformers path.
