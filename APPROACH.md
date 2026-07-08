# Frozen Backbone + Small Online World-Model/Adapters + Replay

> Scope note: This is an approach-level description, not an implementation specification. It deliberately avoids ARC-specific hand-coded representations, symbolic transition languages, or environment-specific heuristics. ARC-AGI-3 is treated as one evaluation substrate for a more general class of resource-constrained interactive agents.


## One-sentence description

This approach uses a strong frozen local pretrained backbone for perception and general priors, then learns only small task-specific world-model components, adapters, memory, and policies online during evaluation, using replay and imagined practice to convert a small number of external actions into improved future action efficiency.

## Core thesis

The central bet is pragmatic:

> Do not train a complete interactive foundation agent from scratch. Use an existing local pretrained model as a perception and representation prior, freeze most of it, and adapt small components online to the current environment.

This is the lowest-offline-training serious approach among the non-symbolic families. It accepts that broad interactive pretraining may be expensive or unavailable, and instead relies on self-supervised learning from the current task's transition data.

The agent learns from every observed transition:

> I was in this observed state, I took this action, and the environment changed this way.

That transition is useful even if the agent did not receive explicit success feedback. The local world model becomes increasingly accurate for the current environment, and replay lets the agent practice internally without spending more external actions.

## Why this is fundamental rather than ARC-specific

The pattern is domain-general:

- In a game, actions transform visual states.
- In code, edits and commands transform repository/test states.
- In robotics, motor actions transform sensor states.
- In a UI, clicks and keystrokes transform interface states.
- In science, interventions transform measurement distributions.

The generic strategy is:

1. Use a pretrained backbone to understand observations.
2. Collect a small amount of task-specific interaction data.
3. Fit a small local predictive model.
4. Plan or improve behavior inside that local model.
5. Use replay to improve before spending more external actions.

No hand-coded ARC symbols are necessary. The environment-specific knowledge is learned online.

## Conceptual architecture

### 1. Frozen local backbone

The backbone supplies broad perceptual and representational priors. It may be visual, video-based, multimodal, or action-conditioned. The key property is that it is local and frozen during evaluation.

The backbone should provide useful representations for:

- visual structure;
- temporal changes;
- action-conditioned differences;
- object-like persistence if learned;
- spatial relations if learned;
- similarities to past experience;
- compact features for small online models.

Freezing the backbone controls compute, reduces overfitting risk, and makes online adaptation feasible within a single-machine budget.

### 2. Small online world model

A small trainable component learns the current environment's local dynamics. It predicts what will happen after actions, either in observation space, latent space, or both.

The model does not need to be universally correct. It only needs to be accurate enough over the states the agent is likely to visit and over the horizons needed for planning.

This is a practical distinction. A local online model can be narrow, fast, and task-specific while the frozen backbone remains general.

### 3. Adapter or fast-weight layer

Instead of changing the full backbone, the agent may maintain small adapter-like components that tune the representation to the current environment. Conceptually, this gives the system a limited form of plasticity without incurring the cost and instability of full fine-tuning.

The adapter's role is not to store a full solution. It should adjust the representation so that the current environment's action consequences, progress cues, and visual distinctions become easier for the small world model and policy to use.

### 4. Episodic transition buffer

The agent keeps a buffer of recent and important transitions. The buffer should prioritize:

- action consequences;
- surprising prediction errors;
- success/failure events;
- level transitions;
- irreversible changes;
- states near decisions;
- evidence about controllability;
- evidence about goals.

The buffer is the data source for online learning and replay.

### 5. Replay and imagined practice

Replay is the defining mechanism. After collecting real transitions, the agent can spend internal compute to improve its local model and policy before taking more external actions.

Replay can serve several purposes:

- consolidate the transition model;
- train a short-horizon policy for the current level;
- test candidate plans internally;
- identify which state/action pairs remain uncertain;
- prepare for the next level if levels share mechanics;
- reduce repeated mistakes.

The key point is that replay spends wall-clock compute but not environment actions.

### 6. Short-horizon planner or policy improver

The agent uses the learned local model to choose actions. It may plan over short action sequences or train a local policy through replay. At approach level, the important property is not the specific search algorithm. The important property is that action choice is informed by the current learned model rather than by blind exploration.

### 7. Resource-aware controller

The system needs a controller that decides when to:

- collect another real transition;
- perform more online updates;
- run replay;
- ask the planner for candidate actions;
- execute an external action;
- stop adapting and exploit.

This controller should be cost-sensitive. Online learning and replay are useful only if they are expected to reduce external action waste enough to justify their wall-clock cost.

## Evaluation-time behavior

The behavior pattern is:

1. Use the frozen backbone to encode the initial observation.
2. Initialize a small local model and task memory.
3. Take one or a few low-risk diagnostic actions.
4. Store transitions in the buffer.
5. Update the local world model and/or adapters.
6. Use the model to imagine candidate futures.
7. Choose an action that balances success probability, information gain, and risk.
8. Repeat until the level is solved or the model is shown to be wrong.
9. After a solved level, replay the trajectory and train the local model/policy to reproduce success with fewer actions.
10. Use the compressed learned dynamics in later levels.

The key pattern is interleaving:

> real transition → local update → replay → plan → real transition.

## Offline training requirement

This approach has the lowest custom offline training requirement among the serious non-symbolic approaches.

There are three possible levels.

### Level 0: No custom offline training

Use an existing local pretrained backbone, initialize small heads/adapters, and rely entirely on online transition learning. This is the least training-intensive version, but also the weakest.

### Level 1: Generic pretraining for small heads

Pretrain the online world-model heads and adapters on broad interactive tasks so they learn how to adapt quickly from small transition buffers. This is likely the best return on training effort.

### Level 2: Broad but lightweight interactive pretraining

Train the full online-adaptation loop on many generic environments. The frozen backbone may remain fixed, but the system learns how to allocate replay, how to probe, and how to avoid overfitting to bad models.

This level starts to approach the learned latent world-model family, but still keeps the frozen-backbone constraint.

## Why this can be action-efficient

The agent uses real actions only to gather necessary transition data. Once it has a few examples, it can train and test candidate strategies internally.

This is especially effective when:

- levels in the same environment share mechanics;
- the same action meanings recur;
- success trajectories can be improved through replay;
- early mistakes reveal useful dynamics;
- visual perception is mostly solved by the backbone.

The agent can spend compute between levels to reduce action count on later levels, which directly targets RHAE-style efficiency.

## Fit to a single-H100, no-internet, nine-hour evaluation

This approach is particularly compatible with the constraint because:

- the main model is local;
- the backbone is frozen;
- only small components are trained online;
- replay can be bounded;
- compute can be allocated adaptively;
- no hosted API is required.

The principal engineering risk is time allocation. Online updates must be cheap enough that the system does not waste the nine-hour budget. Easy environments should get little adaptation. Hard environments should get more replay only when the expected action savings are high.

## Strengths

### Lowest serious offline training requirement

This is the main advantage. It does not require training a complete agent from scratch.

### Practicality

A frozen backbone plus small online learners is much easier to build and iterate than a full recurrent meta-RL system.

### Generality

The learning target is generic action-conditioned prediction, not ARC-specific rule discovery.

### Test-time learning from sparse data

Every transition provides self-supervised signal. The agent does not need many explicit rewards.

### Replay advantage

The agent can improve between external actions or levels using internal compute.

### Good bridge architecture

This approach can later be upgraded into a full learned latent world-model system or flexible deliberation-token system.

## Weaknesses

### Limited prior over exploration

If the frozen backbone was not trained for interactive reasoning, the agent may still choose poor diagnostic actions.

### Model error from small data

The online world model may be trained on too few transitions and produce misleading predictions.

### Replay can amplify errors

If imagined rollouts are based on a wrong model, replay may make the policy worse.

### Frozen representations may be inadequate

A frozen backbone trained on passive perception may not expose the features needed for precise action-conditioned prediction.

### Goal inference remains hard

The local world model can learn dynamics, but it may not know what the agent is supposed to accomplish.

### Time budget trade-off

Too much online training wastes wall-clock time. Too little adaptation wastes external actions.

## Safeguards and design principles

### Keep online learning bounded

The system should never enter an uncontrolled training loop. Adaptation must have explicit time and compute budgets.

### Prefer conservative planning under uncertainty

The agent should not trust rollouts where the local model has high uncertainty or where candidate actions move into unfamiliar states.

### Use real observations to correct replay

Replay is useful only when repeatedly grounded by real transitions. The agent should compare imagined outcomes to observed outcomes and reduce trust after mismatch.

### Separate dynamics learning from goal inference

The agent should learn what actions do and separately estimate what states are valuable. Conflating the two can cause brittle behavior.

### Carry forward compressed knowledge, not raw overfit

After a level, the agent should preserve compact facts useful for future levels and discard noise specific to one trajectory.

## Best use cases

This approach is strongest when:

- the environment has repeated levels or repeated mechanics;
- visual structure is within the frozen backbone's competence;
- a small number of actions reveals useful dynamics;
- replay can improve execution efficiency;
- goals can be inferred from success/failure transitions or level structure.

It is weaker when:

- the first few actions must be nearly perfect;
- the task requires long-horizon hidden reasoning before any useful feedback;
- the environment is highly stochastic;
- the goal signal is extremely ambiguous;
- the frozen backbone does not parse the observation modality well.

## Relationship to the other approach families

Compared with learned active latent world modeling, this approach is cheaper and more practical but has a lower ceiling. It outsources broad perception to the frozen backbone and learns only the local dynamics online.

Compared with recurrent meta-RL, it requires much less offline training but more test-time adaptation. It does not assume the agent already has a learned internal learning algorithm.

Compared with flexible deliberation-token VLMs, it is less elegant but more controllable. The system can use a VLM/video backbone without requiring the entire reasoning process to be tokenized.

## ARC-AGI-3-specific relevance without ARC-specific assumptions

For ARC-AGI-3-style tasks, this approach would treat each environment as a new local interaction system. It would learn action effects, progress cues, and level-to-level regularities through online prediction and replay. It would not need a symbolic ARC representation.

The most promising pattern is:

- use early levels to learn local mechanics;
- replay solved levels to compress the successful strategy;
- apply the improved local model to later levels;
- minimize repeated exploratory actions.

This directly attacks the action-efficiency problem while respecting the no-internet, local-compute constraint.

## Evaluation criteria

The system should be evaluated on:

- success rate;
- external action count;
- wall-clock time spent adapting;
- prediction improvement per transition;
- replay usefulness;
- transfer from one level to later levels;
- robustness to wrong early models;
- ratio of real actions to internal imagined actions;
- calibration of uncertainty;
- performance when the frozen backbone is swapped or degraded.

The key diagnostic is whether online learning actually reduces future external actions enough to justify its compute cost.

## Research questions

1. How much can a frozen passive visual/video backbone support interactive control?
2. What is the smallest trainable online component that materially improves action efficiency?
3. How much replay is useful before model error dominates?
4. Can replay after one level improve later levels without explicit symbolic abstraction?
5. How should the agent decide when to stop adapting and act?
6. Can online adapters learn task-relevant representations from very few transitions?
7. Does broad pretraining of the adaptation loop give most of the benefit of full meta-RL at much lower cost?

## Bottom line

This is the best low-offline-training serious approach. It does not require a hand-authored symbolic language and does not require training a full interactive foundation agent from scratch. It uses a local frozen backbone for general perception, learns small task-specific dynamics online, and uses replay to convert scarce external interactions into improved action efficiency. Its ceiling is lower than a fully trained latent world-model or meta-RL agent, but its practicality is much higher.
