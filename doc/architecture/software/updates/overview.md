# Updates Overview

Updater P revises learned agent context. It is called only by orchestration.

Active tasks:

- agent probing game context after retained observed transitions
- agent policy game context after retained observed transitions
- agent general context at run end

The world-model call owns the world description and action effects. It runs
before the historizer and feeds the fresh world-model context to the historizer
and both agent updaters. The agent probing update emits an
`probing_strategy` and
chooses the next mechanics-learning action. The agent policy update emits
an `policy_strategy` and chooses the next goal-pursuing action. The
historizer selects the update mode from prior updater strategy snapshots and the
fresh world-model context. The general update uses run-level summary metadata.
