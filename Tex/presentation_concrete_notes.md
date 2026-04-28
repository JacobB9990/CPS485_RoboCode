# Concrete Presentation Notes: Robocode AI Bots

These notes are meant for speaking from, not just reading. The emphasis is on the three bots I mainly worked on: `Jacob3_0`, `MeleeDQN`, and `NeuroEvoMelee`.

## 0. One-Sentence Project Framing

We compared several Robocode Tank Royale agents under the same battle harness: tabular SARSA, neural DQN variants, PPO actor-critic bots, heuristic melee controllers, and a neuroevolution bot. The key design question was how much battlefield information each bot should encode, and how that state representation connects to movement, targeting, firing, and reward.

Core reinforcement learning loop:

$$
s_t \rightarrow a_t \rightarrow r_t, s_{t+1}
$$

The bot observes a state vector, chooses an action, receives a reward from battle events, and updates its policy or value function.

---

# 1. Jacob3_0

## High-Level Description

`Jacob3_0` is the strongest final DQN-style bot in the project. It uses a compact continuous state vector and a small discrete action set. The design goal was sample efficiency: keep the input small enough that online DQN training could learn useful behavior from noisy Robocode rounds.

Files to know:

- `Jacob3_0/runtime/dqn_bot.py`
- `Jacob3_0/agent/dqn_agent.py`
- `Jacob3_0/checkpoints/dqn_weights.pt`
- `Jacob3_0/checkpoints/dqn_weights_curriculum.pt`

## State Space

State size:

$$
s_t \in \mathbb{R}^{16}
$$

The 16 features are normalized continuous values:

| Index | Feature | Meaning |
|---:|---|---|
| 0 | self energy / 100 | How much health the bot has left |
| 1 | speed / 8 | Current movement speed |
| 2 | sin heading | Direction without angle wrap discontinuity |
| 3 | cos heading | Direction without angle wrap discontinuity |
| 4 | x / arena width | Horizontal position |
| 5 | y / arena height | Vertical position |
| 6 | right wall distance / width | Wall awareness |
| 7 | top wall distance / height | Wall awareness |
| 8 | sin enemy bearing | Enemy angle relative to bot |
| 9 | cos enemy bearing | Enemy angle relative to bot |
| 10 | enemy distance / max distance | Range to scanned enemy |
| 11 | enemy energy / 100 | Target health |
| 12 | bearing delta / pi | Short-term enemy angle change |
| 13 | distance delta / max distance | Whether enemy is closing or separating |
| 14 | gun heat / 4 | Whether firing is currently possible |
| 15 | fresh scan flag | 1 if enemy scan is recent, else 0 |

Good thing to say:

The state is small but covers the four things a tank needs to know: my survivability, where I am relative to walls, where the enemy is, and whether my gun can fire.

Why sine/cosine matters:

Angles wrap at 360 degrees. A raw angle has an artificial jump from 359 to 0, but sine and cosine make that transition smooth.

$$
\theta = 359^\circ \text{ and } \theta = 0^\circ
$$

are far apart numerically as raw degrees, but close as:

$$
(\sin \theta, \cos \theta)
$$

## Action Space

Action count:

$$
a_t \in \{0,1,2,3,4,5,6\}
$$

| Action | Name | Command |
|---:|---|---|
| 0 | Strafe left | turn left 30 degrees, forward 80 |
| 1 | Strafe right | turn right 30 degrees, forward 80 |
| 2 | Forward | forward 100 |
| 3 | Backward | back 100 |
| 4 | Fire low | aim and fire power 1 |
| 5 | Fire medium | aim and fire power 2 |
| 6 | Fire high | aim and fire power 3 |

Good thing to say:

This is intentionally not a continuous-control bot. DQN estimates a value for every discrete action, so I kept the action space small enough that each action can get repeated experience.

## DQN Network

Architecture:

$$
16 \rightarrow 128 \rightarrow 128 \rightarrow 7
$$

The network outputs one Q-value per action:

$$
Q_\theta(s_t) =
\begin{bmatrix}
Q_\theta(s_t, a_0) \\
Q_\theta(s_t, a_1) \\
\cdots \\
Q_\theta(s_t, a_6)
\end{bmatrix}
$$

Action selection:

$$
a_t =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q_\theta(s_t,a), & \text{otherwise}
\end{cases}
$$

Epsilon decay:

$$
\epsilon_t = \epsilon_{\min} + (\epsilon_{\max} - \epsilon_{\min})e^{-t / d}
$$

In this implementation:

- learning rate: `1e-4`
- discount: `gamma = 0.99`
- epsilon starts at `0.9`
- epsilon ends at `0.05`
- epsilon decay steps: `2500`
- batch size: `128`
- replay memory: `10000`
- target soft update: `tau = 0.005`

## DQN Update Math

A transition is:

$$
(s_t, a_t, r_t, s_{t+1}, d_t)
$$

where:

- $d_t = 1$ if terminal
- $d_t = 0$ otherwise

Jacob3_0 uses a policy network and target network. The target value is:

$$
y_t = r_t + \gamma(1-d_t)\max_{a'}Q_{\bar{\theta}}(s_{t+1},a')
$$

The predicted value is:

$$
\hat{y}_t = Q_\theta(s_t,a_t)
$$

The loss is Huber loss:

$$
L(\theta) = \operatorname{Huber}(\hat{y}_t, y_t)
$$

Soft target update:

$$
\bar{\theta} \leftarrow \tau\theta + (1-\tau)\bar{\theta}
$$

Important nuance:

Jacob3_0 has a policy network and target network, but its target uses the target network max directly. That is standard target-network DQN. The separate `MeleeDQN` implementation uses the more explicit Double DQN target.

## Reward Design

Jacob3_0 gets dense event rewards:

| Event | Reward contribution |
|---|---:|
| bullet hits enemy | $$+0.02 \cdot \text{damage}$$ |
| hit by bullet | $$-0.025 \cdot \text{damage}$$ |
| wall hit | $$-0.05$$ |
| firing command | $$-0.01$$ |
| win round | $$+1.0$$ |
| die | $$-1.0$$ |

Bullet damage formula:

$$
\operatorname{damage}(p)=4p+\max(0,2(p-1))
$$

Episode return:

$$
R = \sum_{t=0}^{T} r_t
$$

Good thing to say:

The reward is not just win or lose. It nudges the bot toward dealing damage, avoiding damage, not wasting shots, and avoiding walls, while still giving a large terminal signal for winning.

## Why Jacob3_0 Performed Well

Key reasons:

- The state space is compact, so the DQN does not have to learn from a huge sparse representation.
- The action space has only seven choices, which makes value learning easier.
- The reward signal is frequent enough that the bot can improve before waiting for only terminal wins and losses.
- The bot logs enough information to compare reward, win rate, damage, placement, and state snapshots.

Presentation line:

Jacob3_0 was the best example of matching algorithm complexity to the amount of data we could realistically collect.

## Main Limitation

Jacob3_0 mostly tracks one recent scan, so in melee it does not explicitly encode multiple enemies the way `MeleeDQN` and `NeuroEvoMelee` do. It can still perform well, but it is not fully representing the whole battlefield.

---

# 2. MeleeDQN

## High-Level Description

`MeleeDQN` is the larger melee-focused DQN bot. It was designed for multiple opponents, not just one target. Compared with Jacob3_0, it has a bigger state vector, more actions, target selection, danger mapping, and stronger reward shaping.

Files to know:

- `MeleeDQN/runtime/melee_dqn_bot.py`
- `MeleeDQN/agent/dqn_agent.py`
- `MeleeDQN/checkpoints/melee_dqn_weights.pt`
- `MeleeDQN/training/train_melee_dqn_socket.py`

## State Space

State size:

$$
s_t \in \mathbb{R}^{48}
$$

The state is:

$$
48 = 16 + 4 \cdot 8
$$

That means:

- 16 global battlefield features
- 4 enemy summary blocks
- 8 features per enemy block

## Global Features

| Index | Feature | Meaning |
|---:|---|---|
| 0 | self energy / 100 | Current health |
| 1 | x signed/normalized | Position feature |
| 2 | y signed/normalized | Position feature |
| 3 | x / width | Left-to-right location |
| 4 | right wall distance / width | Wall spacing |
| 5 | y / height | Bottom-to-top location |
| 6 | top wall distance / height | Wall spacing |
| 7 | enemy count / 12 | Melee crowd size |
| 8 | closest enemy distance | Immediate threat range |
| 9 | average enemy distance | Overall spacing |
| 10 | recently hit flag | Defensive context |
| 11 | gun heat / 1.6 | Firing availability |
| 12 | crowding score | Local danger |
| 13 | close enemy count / 10 | Enemies within 200 |
| 14 | medium enemy count / 10 | Enemies from 200 to 400 |
| 15 | far enemy count / 10 | Remaining distant enemies |

## Enemy Blocks

There are four 8-feature enemy blocks:

1. nearest enemy
2. weakest enemy
3. most threatening enemy
4. current target

Each block is:

| Block Feature | Meaning |
|---:|---|
| 0 | sin relative bearing |
| 1 | cos relative bearing |
| 2 | distance / max distance |
| 3 | sin enemy heading |
| 4 | cos enemy heading |
| 5 | enemy velocity / 8 |
| 6 | enemy energy / 100 |
| 7 | scan age / 40 |

Good thing to say:

MeleeDQN explicitly represents the difference between the closest enemy, the weakest enemy, the biggest threat, and the currently selected target. That is a much richer melee representation than Jacob3_0.

## Action Space

Action count:

$$
a_t \in \{0,\ldots,14\}
$$

| Action | Name | Command |
|---:|---|---|
| 0 | ahead short | forward 80 |
| 1 | ahead medium | forward 160 |
| 2 | back short | back 80 |
| 3 | back medium | back 160 |
| 4 | turn left small | turn left 15, forward 40 |
| 5 | turn right small | turn right 15, forward 40 |
| 6 | turn left medium | turn left 35, forward 50 |
| 7 | turn right medium | turn right 35, forward 50 |
| 8 | strafe left | perpendicular movement around target |
| 9 | strafe right | perpendicular movement around target |
| 10 | head to open space | move toward safest sampled heading |
| 11 | flee cluster | move away from enemy cluster and walls |
| 12 | fire 1 | fire power 1 |
| 13 | fire 2 | fire power 2 |
| 14 | fire 3 | fire power 3 |

Action sanitization:

- If the DQN chooses fire with no valid target, hot gun, or bad aim, the bot converts it to movement.
- If forward movement would hit a wall, the bot converts it to `FLEE_CLUSTER`.
- If strafe is selected with no target, the bot goes to open space.

Good thing to say:

The DQN picks a high-level discrete action, but the runtime protects the bot from impossible or obviously bad commands. That keeps learning from being dominated by illegal shots and wall crashes.

## Network and DQN Math

Architecture:

$$
48 \rightarrow 256 \rightarrow 256 \rightarrow 15
$$

MeleeDQN uses Double DQN target selection:

$$
a^* = \arg\max_{a'}Q_\theta(s_{t+1},a')
$$

$$
y_t = r_t + \gamma(1-d_t)Q_{\bar{\theta}}(s_{t+1},a^*)
$$

This is different from standard DQN because the policy network chooses the next action, while the target network evaluates it.

Loss:

$$
L(\theta)=\operatorname{Huber}(Q_\theta(s_t,a_t), y_t)
$$

Soft update:

$$
\bar{\theta}\leftarrow \tau\theta+(1-\tau)\bar{\theta}
$$

Hyperparameters:

- learning rate: `1e-4`
- discount: `gamma = 0.985`
- epsilon starts at `0.95`
- epsilon ends at `0.05`
- epsilon decay steps: `15000`
- replay memory: `50000`
- batch size: `128`
- soft target update: `tau = 0.005`

## Target Selection

MeleeDQN scores enemies using distance, low energy, freshness, fire intent, and gun alignment.

The target score is approximately:

$$
\operatorname{score}(e)
=0.34D_e+0.22W_e+0.15F_e+0.15I_e+0.14G_e+\text{stickiness}
$$

where:

- $D_e$ is distance score
- $W_e$ is weakness score
- $F_e$ is scan freshness
- $I_e$ is enemy fire intent
- $G_e$ is gun alignment ease

It also uses hysteresis:

$$
\text{switch only if } \operatorname{score}(e_{new}) > \operatorname{score}(e_{current}) + 0.18
$$

and waits at least 10 ticks before switching again.

Good thing to say:

In melee, constantly switching targets is bad because the gun never finishes aiming. The selector uses stickiness so the bot can commit long enough to deal damage.

## Danger Map

Crowding score is based on inverse squared distance:

$$
\operatorname{crowding}
=\operatorname{clip}_{[0,1]}\left(25000\sum_e \frac{1}{\max(36,d_e)^2}\right)
$$

The bot also samples headings and chooses safer directions by scoring future positions.

Good thing to say:

The danger map gives the neural policy a safety prior. The DQN does not have to rediscover every wall and crowding rule from scratch.

## Reward Design

Dense reward terms:

| Event / behavior | Coefficient |
|---|---:|
| damage dealt | $+0.045 \cdot \text{damage}$ |
| damage taken | $-0.060 \cdot \text{damage}$ |
| kill | $+1.2$ |
| survival tick | $+0.003$ |
| wall hit | $-0.18$ |
| robot collision | $-0.14$ |
| dense enemy zone | $-0.08 \cdot \text{crowding}$ |
| inactivity | $-0.02$ |
| target switch | $-0.03$ |
| fire command | $-0.005$ |

Step rewards are clipped:

$$
r_t = \operatorname{clip}_{[-2,2]}(r_t)
$$

Terminal placement reward:

$$
\rho = 1 - \frac{\operatorname{placement}-1}{\operatorname{totalBots}-1}
$$

$$
r_T = 1.4\rho + \mathbb{1}[\text{alive at end}]
$$

Good thing to say:

MeleeDQN had the most carefully engineered melee reward. It rewards not just winning, but surviving longer, avoiding clusters, committing to targets, and converting damage into kills.

## Why MeleeDQN Was Important Even Though Results Were Weak

MeleeDQN is architecturally more ambitious than Jacob3_0:

- It tracks multiple enemies.
- It separates nearest, weakest, threatening, and current target.
- It has explicit anti-crowding and wall safety.
- It uses a 15-action melee action set.
- It uses Double DQN rather than the simpler target calculation.

However, the final benchmark showed that larger state/action spaces require much more training data and tuning. Its learned behavior struggled to turn the richer state into reliable offense.

Presentation line:

MeleeDQN taught us that more battlefield information is not automatically better. It gives the policy more expressive power, but it also makes the learning problem much harder.

---

# 3. NeuroEvoMelee

## High-Level Description

`NeuroEvoMelee` is not trained with gradient descent. It uses a fixed-topology neural network, but the weights are evolved with a genetic algorithm. The runtime loads a genome JSON file, runs a forward pass every tick, and turns four continuous outputs into movement, firing, and target-preference behavior.

Files to know:

- `NeuroEvoMelee/runtime/neuroevo_melee_bot.py`
- `NeuroEvoMelee/genome/feature_encoder.py`
- `NeuroEvoMelee/genome/genome_network.py`
- `NeuroEvoMelee/training/train_neuroevo_melee.py`
- `NeuroEvoMelee/data/current_genome.json`
- `NeuroEvoMelee/data/best_genome.json`

## State Space

State size:

$$
s_t \in \mathbb{R}^{20}
$$

The 20 inputs are:

| Index | Feature | Meaning |
|---:|---|---|
| 0 | self energy / 100 | Health |
| 1 | speed / 8 | Movement |
| 2 | sin heading | Direction |
| 3 | cos heading | Direction |
| 4 | wall proximity | 1 means close to wall |
| 5 | nearest distance | Nearest enemy range |
| 6 | nearest bearing sin | Nearest enemy angle |
| 7 | nearest bearing cos | Nearest enemy angle |
| 8 | nearest energy | Nearest enemy health |
| 9 | weakest distance | Weakest enemy range |
| 10 | weakest bearing sin | Weakest enemy angle |
| 11 | weakest bearing cos | Weakest enemy angle |
| 12 | weakest energy | Weakest enemy health |
| 13 | threat sum | Aggregate enemy pressure |
| 14 | max threat | Biggest single threat |
| 15 | close count / 8 | Crowd density |
| 16 | alive enemies / 10 | Melee size |
| 17 | gun heat / 1.6 | Firing availability |
| 18 | target alignment | Gun alignment to preferred target |
| 19 | target preference delta | Nearest pressure vs weakest vulnerability |

Good thing to say:

NeuroEvoMelee compresses the whole melee situation into 20 hand-designed features. It does not learn the representation; evolution only searches over the policy weights.

## Action Space

This bot has continuous neural outputs rather than a discrete action ID:

$$
o_t \in [-1,1]^4
$$

Network output meaning:

| Output | Meaning | Runtime conversion |
|---:|---|---|
| 0 | turn command | turn right $$35 \cdot o_0$$ degrees |
| 1 | movement command | ahead $$140 \cdot o_1$$ |
| 2 | fire intensity | convert from $$[-1,1]$$ to $$[0,1]$$ |
| 3 | target preference adjustment | updates nearest-vs-weakest bias |

Fire conversion:

$$
f = \frac{o_2 + 1}{2}
$$

The bot fires only if:

$$
f > 0.18
$$

and gun heat and energy allow it. Fire power is:

$$
p = \operatorname{clip}(3f,0.2,3.0)
$$

Target preference memory:

$$
b_t = \operatorname{clip}(0.85b_{t-1}+0.15o_3,-1,1)
$$

If the bias is positive, the bot prefers weakest enemy. If negative, it prefers nearest enemy.

Good thing to say:

The evolved policy has smoother control than DQN because it outputs continuous movement and firing intensity, but it also has less direct credit assignment than reinforcement learning.

## Genome Network

Current genome shape:

$$
20 \rightarrow 24 \rightarrow 4
$$

Hidden layer:

$$
h = \tanh(W_1s + b_1)
$$

Output layer:

$$
o = \tanh(W_2h + b_2)
$$

The genome stores:

- `w1`
- `b1`
- `w2`
- `b2`

## Evolution Math

Population:

$$
P = \{g_1,g_2,\ldots,g_N\}
$$

Each genome is evaluated and assigned a fitness:

$$
F(g_i)
$$

The trainer sorts by fitness, keeps elites, then creates children using crossover and mutation.

Crossover:

$$
w_j^{child} =
\begin{cases}
w_j^{A}, & \text{with probability } 0.5 \\
w_j^{B}, & \text{with probability } 0.5
\end{cases}
$$

Mutation:

$$
w_j' =
\begin{cases}
\operatorname{clip}(w_j+\mathcal{N}(0,\sigma),-2,2), & \text{with probability } \mu \\
w_j, & \text{otherwise}
\end{cases}
$$

Default evolution settings:

- population: `24`
- generations: `40`
- elite fraction: `0.2`
- mutation rate: `0.12`
- mutation standard deviation: `0.18`
- seed: `7`

Local fallback fitness in the trainer rewards movement, commitment, firing drive, and target preference:

$$
F = 0.35M + 0.25C + 0.20D + 0.20P
$$

where:

- $$M = 1 - |o_0|$$
- $$C = |o_1|$$
- $$D = \max(0,o_2)$$
- $$P = |o_3|$$

## Strengths and Weaknesses

Strengths:

- No replay buffer or gradient stability problems.
- Can optimize non-differentiable battle outcomes.
- Continuous action output is simple at runtime.
- Genome JSON is easy to save, load, mutate, and inspect.

Weaknesses:

- Fitness is noisy because Robocode battles are stochastic.
- Evolution needs many evaluations to beat gradient-based methods.
- Local heuristic fitness may not match actual battle wins.
- Final benchmark behavior did not show reliable damage output.

Presentation line:

NeuroEvoMelee was valuable because it gave us a completely different learning paradigm: instead of learning from transitions, it searches directly over policies.

---

# 4. MeleeSarsaBot

## High-Level Description

`MeleeSarsaBot` is a tabular baseline. It does not use a neural network. Instead, it buckets the game into a string state key and stores a Q-table.

Files:

- `MeleeDQN/runtime/melee_sarsa_bot.py`
- `MeleeDQN/agent/sarsa_table.py`

## State Space

The state is a discrete string key made from buckets:

- own energy bucket
- nearest enemy distance bucket
- nearest enemy bearing bucket
- weakest enemy distance bucket
- nearby enemy count bucket
- danger bucket
- wall proximity bucket
- gun-ready flag
- target energy bucket
- target distance bucket

So the state is not:

$$
s_t \in \mathbb{R}^n
$$

It is:

$$
s_t \in \mathcal{S}_{bucketed}
$$

Example style:

```text
me2|nd1|nb3|wd2|nn1|dg2|wp0|gr1|te1|td2
```

## Action Space

Action count:

$$
a_t \in \{0,\ldots,9\}
$$

| Action | Name |
|---:|---|
| 0 | orbit clockwise |
| 1 | orbit counterclockwise |
| 2 | retreat from cluster |
| 3 | advance to open space |
| 4 | radar sweep left |
| 5 | radar sweep right |
| 6 | fire low |
| 7 | fire medium |
| 8 | fire high |
| 9 | evade |

## SARSA Update

SARSA is on-policy. It learns from the action it actually takes next:

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\right]
$$

Key difference from DQN:

SARSA updates a table entry directly. DQN updates neural network weights.

Good thing to say:

SARSA is simpler and more interpretable. It cannot generalize across continuous states like DQN, but it is stable and gives a useful baseline.

---

# 5. PPOBot

## High-Level Description

`PPOBot` is an actor-critic policy gradient bot. It uses the same compact state/action setup as Jacob3_0, but learns a policy directly instead of learning Q-values.

Files:

- `PPOBot/runtime/PPO_Bot.py`
- `PPOBot/checkpoints/ppo_weights.pt`

## State Space

State size:

$$
s_t \in \mathbb{R}^{16}
$$

It uses the same kind of 16 features as Jacob3_0:

- self energy
- speed
- heading sine/cosine
- position
- wall distances
- enemy bearing sine/cosine
- enemy distance
- enemy energy
- bearing delta
- distance delta
- gun heat
- fresh scan flag

## Action Space

Action count:

$$
a_t \in \{0,\ldots,6\}
$$

Same seven actions as Jacob3_0:

- strafe left
- strafe right
- forward
- backward
- fire low
- fire medium
- fire high

## PPO Math

The actor outputs action probabilities:

$$
\pi_\theta(a_t|s_t)
$$

The critic estimates value:

$$
V_\theta(s_t)
$$

Advantage:

$$
A_t = R_t - V_\theta(s_t)
$$

Probability ratio:

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}
$$

Clipped PPO objective:

$$
L^{CLIP}(\theta)=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

Good thing to say:

DQN asks, "Which action has the highest predicted value?" PPO asks, "How should I adjust the action probabilities without changing the policy too violently?"

---

# 6. PPOBotAdvanced / PPO++

## High-Level Description

`PPOBotAdvanced` is the upgraded melee PPO path. It has a bigger observation builder and a multi-discrete action decoder, so it is closer to a real melee controller than the compact PPO bot.

Files:

- `PPOBotAdvanced/runtime/PPO_Bot.py`
- `PPOBotAdvanced/runtime/melee_env.py`
- `PPOBotAdvanced/agent/melee_ppo_agent.py`

## State Space

Observation size:

$$
s_t \in \mathbb{R}^{42}
$$

The observation includes:

- self state: energy, position, velocity, heading, gun heading, gun heat
- wall distances
- alive enemy count
- nearest enemy block
- weakest enemy block
- strongest threat block
- crowd density features
- danger score
- current target block

Each enemy block has:

- distance
- relative bearing
- velocity
- heading
- energy
- scan age

## Action Space

PPO++ uses a multi-discrete action:

$$
a_t = (a^{move}_t, a^{turn}_t, a^{fire}_t, a^{radar}_t)
$$

Branch sizes:

$$
(5,5,4,5)
$$

Total combinations:

$$
5 \cdot 5 \cdot 4 \cdot 5 = 500
$$

The branches mean:

- movement mode: hold, low-density movement, perpendicular left, perpendicular right, escape crowd
- body turn: negative large, negative small, zero, positive small, positive large
- fire power: none, low, medium, high
- radar turn: left large, left small, zero, right small, right large

Good thing to say:

The multi-discrete design avoids making one giant 500-class softmax. PPO learns separate categorical decisions that combine into a full Robocode command.

---

# 7. HybridMeleeBot

## High-Level Description

`HybridMeleeBot` is not a learned RL bot. It is a deterministic tactical hierarchy.

Files:

- `MeleeDQN/runtime/hybrid_melee_bot.py`

## State / Context

The bot builds a `BotContext` with:

- position
- energy
- velocity
- body heading
- gun heading
- radar heading
- arena size
- enemy count
- tracked enemies
- danger map

## Action Space

The top-level action is a tactical mode:

| Mode | Meaning |
|---|---|
| `SURVIVE` | low-energy survival behavior |
| `ENGAGE` | normal orbit and attack |
| `REPOSITION` | move away from wall or bad location |
| `FINISH_WEAK_TARGET` | pressure a low-health enemy |
| `ESCAPE_CROWD` | move to safer map region |

Good thing to say:

HybridMeleeBot is useful as a comparison because its behavior is interpretable. It shows what we can get from hand-coded target selection, danger maps, radar, and movement without learning.

---

# 8. MeleeOpponentModelBot

## High-Level Description

`MeleeOpponentModelBot` is a heuristic bot that models enemy behavior. It tracks firing frequency, movement style, aggression, accuracy, and threat level.

Files:

- `MeleeDQN/runtime/melee_opponent_model_bot.py`

## State / Model

Per enemy, it tracks:

- energy
- position
- distance
- bearing
- heading
- velocity
- lateral velocity
- closing velocity
- firing frequency
- target-me likelihood
- estimated accuracy
- movement style
- category
- threat score

Enemy category examples:

- spinner weak bot
- high accuracy threat
- close range aggressor
- passive survivor
- balanced

## Action Policy

This bot does not have a learned action space. It chooses movement, targeting, radar, and firing from threat models and adaptive bias variables.

Good thing to say:

Opponent modeling is a middle ground between pure rules and learning. The bot does not learn a neural policy, but it adapts its decisions based on observed enemy behavior.

---

# 9. Clean Comparison Table

| Bot | Learning type | State | Action space | Main idea |
|---|---|---:|---:|---|
| Jacob3_0 | DQN | 16 continuous | 7 discrete | compact DQN, strongest benchmark story |
| MeleeDQN | Double DQN | 48 continuous | 15 discrete | richer melee state, target selection, danger map |
| NeuroEvoMelee | genetic neural policy | 20 continuous | 4 continuous outputs | evolve policy weights directly |
| MeleeSarsaBot | tabular SARSA | bucketed string key | 10 discrete | stable interpretable baseline |
| PPOBot | PPO actor-critic | 16 continuous | 7 discrete | direct policy optimization |
| PPOBotAdvanced | melee PPO | 42 continuous | multi-discrete branches | richer PPO action decoder |
| HybridMeleeBot | heuristic | tactical context | tactical modes | rule-based melee controller |
| MeleeOpponentModelBot | heuristic model-based | enemy behavior model | adaptive commands | classify and react to opponents |

---

# 10. Presentation Math Cards

## DQN Bellman Target

$$
y_t = r_t + \gamma(1-d_t)\max_{a'}Q_{\bar{\theta}}(s_{t+1},a')
$$

## Double DQN Target

$$
a^* = \arg\max_{a'}Q_\theta(s_{t+1},a')
$$

$$
y_t = r_t + \gamma(1-d_t)Q_{\bar{\theta}}(s_{t+1},a^*)
$$

## SARSA Update

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\right]
$$

## PPO Ratio

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}
$$

## PPO Clipped Objective

$$
L^{CLIP}(\theta)=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

## Neuroevolution Forward Pass

$$
h = \tanh(W_1s+b_1)
$$

$$
o = \tanh(W_2h+b_2)
$$

## Mutation

$$
w' =
\begin{cases}
\operatorname{clip}(w+\mathcal{N}(0,\sigma),-2,2), & \text{with probability } \mu \\
w, & \text{otherwise}
\end{cases}
$$

---

# 11. Strong Speaking Points

## Why Jacob3_0 Beat More Complex Bots

Jacob3_0 had a better balance between representation size and training data. The 16-feature state was small enough for DQN to learn from, while still capturing energy, wall position, enemy geometry, and gun heat.

## Why MeleeDQN Struggled

MeleeDQN had a much richer model of the world, but that increased the learning burden. With 48 inputs and 15 actions, it needed more experience to learn which tactical actions actually produce damage and wins.

## Why NeuroEvoMelee Was Included

NeuroEvoMelee tested a different idea: instead of learning values or policy gradients from transitions, evolve the whole neural policy. It was useful for comparison, but noisy battle fitness and limited evaluations made it harder to reach strong battle behavior.

## Why Reward Can Be Negative Even When Win Rate Is Good

A bot can win but still take lots of damage, hit walls, or waste shots. The return is the sum of dense rewards and terminal reward:

$$
R = \sum_{t=0}^{T-1} r_t + r_T
$$

So win rate and reward measure different things.

## Biggest Design Lesson

The best bot was not simply the one with the largest state. The best bot was the one whose state, action space, reward design, and available training data matched each other.

---

# 12. Likely Questions and Answers

## What exactly was the state space for your main bot?

For Jacob3_0, the state was a 16-dimensional continuous vector: self energy, speed, heading sine/cosine, position, wall distances, enemy bearing sine/cosine, enemy distance, enemy energy, bearing change, distance change, gun heat, and a fresh-scan flag.

## Why not use raw angles?

Raw angles create discontinuities at wraparound. Using sine and cosine makes angle representation continuous.

$$
\theta \mapsto (\sin\theta,\cos\theta)
$$

## What was the action space?

Jacob3_0 used seven discrete actions: strafe left, strafe right, forward, backward, fire low, fire medium, and fire high.

MeleeDQN used fifteen actions by adding shorter and longer movement, open-space movement, flee-cluster behavior, strafing, and three fire powers.

NeuroEvoMelee used four continuous outputs: turn, move, fire intensity, and target preference.

## What is the difference between DQN and PPO here?

DQN learns:

$$
Q(s,a)
$$

and chooses the action with the highest value.

PPO learns:

$$
\pi(a|s)
$$

and updates action probabilities directly while clipping the update size.

## What is the difference between Jacob3_0 and MeleeDQN?

Jacob3_0 is compact: 16 features and 7 actions. It is easier to train.

MeleeDQN is richer: 48 features and 15 actions. It understands melee context better in theory, but needs more data and reward tuning.

## What is the difference between DQN and NeuroEvoMelee?

DQN learns from transitions and gradient descent:

$$
(s_t,a_t,r_t,s_{t+1})
$$

NeuroEvoMelee evolves complete policies by testing genomes, selecting high-fitness ones, and mutating/crossing them over.

## What would you improve next?

For Jacob3_0:

- add more explicit melee enemy summaries
- add a wall escape action
- use true Double DQN target calculation

For MeleeDQN:

- simplify the reward first
- train longer on staged melee curricula
- separate movement learning from firing learning

For NeuroEvoMelee:

- use real battle fitness instead of local heuristic fitness
- increase population and generations
- add elitism plus scenario diversity

---

# 13. Short Final Takeaway

Jacob3_0 showed that a compact DQN can perform very well when the state and action spaces are disciplined. MeleeDQN showed the cost of scaling to richer melee reasoning. NeuroEvoMelee showed a different policy-search path, useful for comparison but harder to tune with limited battle evaluations.
