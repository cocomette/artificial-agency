# Model Inputs

## Agent X

Receives active agent context, current observation, allowed actions, bounded
recent action history, first/current observation refs, action-outcome evidence,
and game memory.

## Change Summary

Receives transition frame evidence, the action, glossary actions, and previous
change elements.

## Historizer

Receives prior complete agent game contexts for the same run/game.

## Game Memory

Receives same-run action history, first game observation, current
post-transition observation, and non-identifying metadata.

## Updater P

The agent game updater receives previous agent context, current observation,
allowed actions, action history, game memory, historizer output, progress
feedback, revision feedback, and action-outcome evidence.

The general updater receives run-level metadata and previous agent context.
