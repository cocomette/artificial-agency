

# Description of intended framework

Change summarizer

Summarize frame transitions and animation bundles, provides ground-truth for world model.

Memory

Aggregates all change summaries and actions taken so far into a detailed text description of what the agent did so far and how the environment evolved. Maybe should get first and last frames.

World model

Has to provide a text prediction identical to change summarizer, but only has access to current frame and action. Reward is computed by comparing the prediction to the change summarizer. In the no-LoRA runtime this score is also the immediate proxy learning-progress signal.

Memory may be useful to provide to understand how the world works.

Goal model

Based on memory has to provide a detailed text prediction of what the goal is, including sub-goals.

It also has to provide an estimate of the number of steps it takes to get finish the level.

Agent

Gets memory, goal prediction, current frame, World predictions, Interest scores, and recent reward-bearing action history. It then selects an action. Without LoRA, reward and proxy learning-progress are feedback in the next prompts rather than parameter updates.

There could a be a 2-stage pipeline where first it proposes top N distinct actions (including various coordinate actions), the world model is run for each and appended to the context.

- in games without action6 we can simply sample the world model for all actions

Reward can be LP based on world model plus whether the predicted number of steps by the goal model has decreased (after the environment update).

- the weighting of these can be annealed based on number of steps, i.e. in the beginning LP is way more important, later goal following is more important.

intention: proxy learning-progress reward should make the agent take actions that expose parts of the environment the World model currently predicts well or poorly, while Goal delta provides the exploitation incentive. This branch does not measure true model improvement over adapter updates.
