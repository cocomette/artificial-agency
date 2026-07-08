# Orchestration Overview

Orchestration owns environment interaction and runtime sequencing. It does not
train backbone weights and does not write model-provider code.

Per frame turn it:

1. checks lifecycle/deadline state;
2. unrolls frame bundles into retained frame turns;
3. prewrites a source `m_states` row when memory is enabled;
4. asks `OnlineLearnerAgent` for a decision on controllable frames or
   synthesizes `NONE` for animation frames;
5. validates ARC actions and `ACTION6` coordinates;
6. steps the environment for real actions;
7. builds the transition record;
8. calls learner observation/replay;
9. completes persistence and emits debug events.
