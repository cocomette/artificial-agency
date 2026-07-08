# Updater Inputs

Updater P currently handles agent context updates.

The agent game updater receives:

- previous agent context
- current observation
- allowed/glossary actions
- bounded action history
- game memory
- agent context-history summary
- progress feedback
- context revision feedback
- action-outcome evidence

The general updater receives run-level stop metadata and the previous agent
context.
