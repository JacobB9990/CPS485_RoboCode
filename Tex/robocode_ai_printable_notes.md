# Robocode DQN Q and A Prep Notes

This handout is focused on likely audience questions and practical technical answers.

Primary references:
1. [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py)
2. [Jacob/Jacob3_0/dqn_agent.py](Jacob/Jacob3_0/dqn_agent.py)
3. [Tex/robocode_ai_presentation.tex](Tex/robocode_ai_presentation.tex)

## 1) Why this state design?
Answer:
1. The 16 features are a compact set that captures self status, wall context, enemy geometry, and short-term motion changes.
2. It is intentionally small to keep online training stable and sample efficient.

Evidence:
1. State dimension and action count constants: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L31)
2. State encoding function: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L155)

Follow-up if asked about angle representation:
1. Heading and bearing use sin and cos to avoid wraparound discontinuity.
2. Angle wrapping is explicitly normalized to [-pi, pi].

Evidence:
1. Trig features: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L196)
2. Bearing wrap logic: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L172)

## 2) Why only 7 actions?
Answer:
1. DQN works best with low-cardinality discrete actions.
2. A larger action set would require more data and typically slows convergence.
3. Current weakness against Walls and SpinBot suggests adding one wall-escape action could help.

Evidence:
1. Action definitions: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L53)
2. Action execution map: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L216)

## 3) Why can win rate improve while average reward stays negative?
Answer:
1. Reward has dense penalties at many timesteps, not just terminal reward.
2. The bot can win while still collecting significant hit and wall penalties.

Equation:

$$
R_{episode}=\sum_{t=1}^{T} r_t + r_{terminal}
$$

Evidence:
1. Hit-by-bullet penalty: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L303)
2. Bullet-hit reward: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L308)
3. Wall penalty: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L311)
4. Terminal reward: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L246)

## 4) Is this Double DQN?
Answer:
1. Not exactly. Current target is standard DQN with target network.
2. True Double DQN would use policy net for action selection and target net for action evaluation.

Current target equation in this code path:

$$
y_t = r_t + \gamma (1-d_t) \max_{a'}Q_{\bar\theta}(s_{t+1},a')
$$

Evidence:
1. Target computation: [Jacob/Jacob3_0/dqn_agent.py](Jacob/Jacob3_0/dqn_agent.py#L149)
2. Train step docstring line: [Jacob/Jacob3_0/dqn_agent.py](Jacob/Jacob3_0/dqn_agent.py#L133)

## 5) How do you prove eval mode is not training?
Answer:
1. Eval mode disables training in the agent.
2. Even if loop code still calls transition push, the agent returns early.
3. Epsilon can be fixed at 0.0 for deterministic policy evaluation.

Evidence:
1. Eval flag setup in bot: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L96)
2. Training guard in agent: [Jacob/Jacob3_0/dqn_agent.py](Jacob/Jacob3_0/dqn_agent.py#L124)
3. Fixed epsilon in eval mode: [Jacob/Jacob3_0/dqn_agent.py](Jacob/Jacob3_0/dqn_agent.py#L194)

## 6) Why is Walls performance low?
Answer:
1. Likely curriculum mismatch plus limited wall-aware action expressivity.
2. Reward shaping may not penalize boundary trapping strongly enough.
3. Movement-heavy opponents expose this weakness more than stationary targets.

Evidence:
1. Results and interpretation in deck: [Tex/robocode_ai_presentation.tex](Tex/robocode_ai_presentation.tex#L234)

## 7) What should be changed first?
Suggested order:
1. Rebalance opponent sampling toward Walls and SpinBot.
2. Add one or two wall-escape actions.
3. Upgrade target to true Double DQN.
4. Retune reward coefficients after those changes.
5. Re-run held-out eval with epsilon fixed at 0.0.

Why this order:
1. Better data coverage and action expressivity usually give bigger gains than optimizer tweaks.

## 8) What metrics should you report in Q and A?
Minimum set:
1. Overall win rate.
2. Per-opponent win rate.
3. Mean and median reward.
4. Episode length distribution.
5. Eval-train gap per opponent.
6. Epsilon and steps_done at evaluation time.

Evidence for logging fields:
1. Episode log row creation: [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L271)

## 9) Snippets to quote quickly
Main loop snippet from [Jacob/Jacob3_0/dqn_bot.py](Jacob/Jacob3_0/dqn_bot.py#L133):

```python
state = self._encode_state()
action = self.agent.select_action(state)

if self.prev_state is not None and self.prev_action is not None:
    self.agent.push_transition(
        self.prev_state, self.prev_action, state, self.step_reward, done=False
    )
    self.episode_reward += self.step_reward
```

Training gate snippet from [Jacob/Jacob3_0/dqn_agent.py](Jacob/Jacob3_0/dqn_agent.py#L124):

```python
if not self.training_enabled:
    return
```

Target and loss snippet from [Jacob/Jacob3_0/dqn_agent.py](Jacob/Jacob3_0/dqn_agent.py#L149):

```python
with torch.no_grad():
    next_q_values = self.target_net(next_states).max(dim=1)[0]
    next_q_values[dones] = 0.0
    targets = rewards + self.gamma * next_q_values

loss = nn.SmoothL1Loss()(q_values, targets)
```

## 10) Tough questions to practice
1. What experiment would prove curriculum mismatch is the main cause of Walls failure?
2. Which single feature would you remove first in an ablation and why?
3. If reward remains negative while win rate improves, what behavior do you think is emerging?
4. How would true Double DQN change the target calculation in your code?
5. What is your exact criterion for saying a change generalizes and is not overfit?

## 11) Print checklist
1. Print [Tex/robocode_ai_printable_notes.pdf](Tex/robocode_ai_printable_notes.pdf).
2. Keep this markdown open for clickable code references during rehearsal.
3. Rebuild PDF after code or slide edits.
