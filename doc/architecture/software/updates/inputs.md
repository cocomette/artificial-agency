# Update Inputs

Updater `P` receives data only at orchestration-owned update boundaries. In a
simple step loop this is after a real environment step. In the frame-unrolled
loop this is after every observed frame decision.

Inputs are assembled by orchestration from live transition objects and memory
references. The updater does not read from SQLite directly.

## World Context Update Inputs

- previous world context `L^S_i,t`
- current observation reference
- actual next observation reference
- submitted real action or synthetic `NONE` action when relevant
- committed world post-decision prediction when available
- world tool results from the live agent trace when `X` requested them
- reward/update quantities
- prediction discrepancy when available

## Goal Context Update Inputs

- previous goal context `L^G_i,t`
- current observation reference
- actual next observation reference
- submitted real action or synthetic `NONE` action when relevant
- committed goal post-decision prediction when available
- goal tool results from the live agent trace when `X` requested them
- reward/update quantities
- goal discrepancy when available

## Agent Context Update Inputs

- previous agent context `L^X_i,t`
- full live agent trace
- current observation reference
- actual next observation reference
- submitted real action or synthetic `NONE` action when relevant
- tool calls and tool results carried by the trace
- committed post-decision prediction packet
- reward/update quantities
