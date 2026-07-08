# Game Loop Interfaces

- `FrameTurnContext`: current observation, refs, frame index/count, control
  mode, and recent action history.
- `TransitionRecord`: observed action-conditioned transition.
- `LearnerTurnTrace`: decision, transition, replay stats, planner candidates,
  backbone metadata, and learner metadata.
- `MStateRecord`: durable SQLite row for complete turns.
