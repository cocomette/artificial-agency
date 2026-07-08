

# Description of intended framework

Change summarizer

Summarize frame transitions and animation bundles, provides ground-truth for world model.

Memory

Aggregates all change summaries and actions taken so far into a detailed text description of what the agent did so far and how the environment evolved. Maybe should get first and last frames.

World model

Has to provide a text prediction identical to change summarizer, but only has access to current frame and action. Reward is computed by comparing the prediction to the change summarizer and doing lora.

Memory may be useful to provide to understand how the world works.

Goal model

Based on memory has to provide a detailed text prediction of what the goal is, including sub-goals. This can only be updated with lora once a level is solved, but that is not important for v1.

It also has to provide an estimate of the number of steps it takes to get finish the level.

Agent

Gets memory, goal prediction, and current frame and produces an action. It then observes a reward and is finetuned with lora.

There could a be a 2-stage pipeline where first it proposes top N distinct actions (including various coordinate actions), the world model is run for each and appended to the context.

- in games without action6 we can simply sample the world model for all actions

Reward can be LP based on world model plus whether the predicted number of steps by the goal model has decreased (after the environment update).

- the weighting of these can be annealed based on number of steps, i.e. in the beginning LP is way more important, later goal following is more important.

intention: learning progress reward and lora updates should properly make the agent take actions that maximize the rate of prediction improvement of the world model, i.e. take it to parts of the environment which are not yet well predicted but where the world model can learn fast. The goal delta reward should balance this and provide an exploitation incentive.