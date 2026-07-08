# Game Loop State Machine

`GameLoopStateMachine` is constructed with `StateMemory | None`,
`OnlineLearnerAgent`, and `DebugBus`.

The active turn sequence is:

1. load retained frame buffer;
2. build `FrameTurnContext`;
3. decide or synthesize `NONE`;
4. validate action payload;
5. resolve next frame through environment step or animation advance;
6. build `TransitionRecord`;
7. update learner and replay;
8. complete SQLite row;
9. advance session cursor.
