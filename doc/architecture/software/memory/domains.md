# Memory Domains

`M` is the durable frame-turn ledger. Each complete row stores:

- current observation;
- chosen action;
- transition record;
- backbone metadata;
- replay stats;
- planner candidates;
- learner metrics and snapshot.

`learner_artifacts` stores optional diagnostics that do not belong in the main
turn ledger.
