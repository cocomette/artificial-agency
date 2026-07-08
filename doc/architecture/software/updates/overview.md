# Updates Overview

This branch has no online update system. World, Interest, Agent X, Memory,
Goal, Change Summary, and Reward Judge are static inference roles for the
duration of a run.

The runtime still computes rewards every turn. `TurnReward.learning_progress`
is retained as an immediate proxy equal to Reward Judge prediction accuracy,
then rendered into recent action history and Memory ledger rows. That gives the
static models textual feedback about which actions and transitions were judged
accurate or useful without changing model weights.

There are no replay samples, trainer workers, adapter roots, adapter versions,
or update-attempt persistence records in this branch.
