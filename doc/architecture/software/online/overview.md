# Online Learner

`src/face_of_agi/online/` owns the Kaggle-first learner:

- `backbone.py`: frozen local Transformers encoder plus deterministic fake
  encoder for tests.
- `learning.py`: prioritized transition buffer, ensemble latent dynamics,
  value head, and bounded replay trainer.
- `planner.py`: deterministic short-horizon planner with bounded `ACTION6`
  coordinate expansion.
- `agent.py`: resource controller for decision, transition observation,
  local updates, replay, and learner snapshots.
- `factory.py`: runtime assembly from `agent:` config.

The backbone is always frozen. Online updates are restricted to small local
state held by the learner for the current run.
