# System Architecture

FACE-OF-AGI is a Python runtime for ARC-AGI-3 experiments. The active system is
a Kaggle-first online learner with an in-process frozen Hugging Face
Transformers backbone and small trainable runtime components.

## Active Components

- `TransformersBackbone` loads local bundled model/processor files with
  `local_files_only=True`, runs in `eval()`, and never updates backbone
  weights.
- The online learner keeps a bounded prioritized transition buffer.
- A small ensemble latent dynamics model, value head, replay trainer, and
  short-horizon planner update from real transitions.
- Orchestration owns the environment loop, frame unrolling, deadlines,
  scorecard lifecycle, SQLite persistence, and debug events.

## Runtime Loop

For each frame turn:

1. Encode the current observation with the frozen backbone.
2. Plan over currently valid ARC actions, including bounded `ACTION6`
   coordinate candidates.
3. Submit one real action only on controllable frames; synthesize `NONE` for
   retained animation frames.
4. Build a transition record from the observed next frame.
5. Run bounded local update and replay.
6. Persist learner trace, learner snapshot, planner candidates, metrics, and
   run metadata.

## Memory

`m_states` stores committed frame turns. `learner_artifacts` stores optional
debug artifacts. Old prompt-role context/tool schemas are disposable and are
not migrated.

## Kaggle

Kaggle is the primary execution path. The notebook installs offline wheels,
loads model weights from Kaggle inputs, opens one scorecard, creates one
environment per selected game, and writes per-game SQLite files under
`/kaggle/working/runs`.
