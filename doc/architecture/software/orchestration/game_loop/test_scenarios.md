# Game Loop Test Scenarios

Runtime tests should cover framework mechanics:

- active vLLM config files load
- old `models.historizer` and `models.updater` keys fail fast
- fake Agent X decisions advance controllable frame turns
- initial Memory/Goal bootstrap occurs after reset
- two-stage candidate selection includes simple actions and coordinate
  proposals
- World predictions are persisted for candidates
- change summaries become action-history entries
- Reward Judge scores and separated reward components are persisted
- reward finalizes before Memory regeneration, using a reward-only Goal call
- reward and proxy learning-progress feedback reach next-turn action history
  and Memory ledger rows
- state memory persists agent context and Agent X trace
- parallel runtime specs isolate per-game state

Tests should not assert deleted behavior or prompt wording.
