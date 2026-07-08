# Updates Overview

Updater P revises learned agent context. It is called only by orchestration.

Active tasks:

- agent game context after retained observed transitions

The compacter call owns the world description, action effects, previous actions
summary, and previous strategy summary. It runs before updater P and feeds the
fresh decomposed context to the agent updater. The updater emits
`current_strategy` and `next_actions`.
