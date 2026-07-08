# Shared Contracts

Active shared contracts live in `face_of_agi.contracts`:

- `Observation`, `ObservationRef`, `EnvironmentInfo`: environment facts.
- `ActionSpec`, `FrameControlMode`: action and frame controllability.
- `AgentTrace`, `DecisionResult`: learner decision trace.
- `TransitionRecord`, `ReplayStats`, `PlannerCandidate`,
  `LearnerTurnTrace`: learner-oriented turn payloads.
- `MStateRecord`, `RunMetadataRecord`: SQLite row shapes.
- `GameRunResult`, `ParallelGameRunResult`: runtime outcomes.

Contracts are intentionally backend-neutral. The Transformers dependency stays
behind `online.backbone`.
