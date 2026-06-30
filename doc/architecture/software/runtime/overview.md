# Runtime Overview

The runtime module is the program entry and assembly layer. It loads config,
constructs dependencies, selects startup behavior, and invokes orchestration.

Runtime does not own the real ARC game loop. Runtime starts orchestration;
orchestration owns the loop.
