# server.py
import socket
import numpy as np
from classifier import EnemyClassifier
from state_vector import StateBuilder
from dqn_agent import DQNAgent
from training_logger import TrainingLogger

HOST = "localhost"
PORT = 5000

STATE_DIM  = 8
ACTION_DIM = 4

agents = {
    cat: DQNAgent(STATE_DIM, ACTION_DIM, category=cat)
    for cat in ["DEFENSIVE", "RUSHER", "SNIPER", "DODGER", "UNKNOWN"]
}

for agent in agents.values():
    agent.load()

classifier    = EnemyClassifier(window_size=30)
state_builder = StateBuilder()
logger        = TrainingLogger(log_dir="logs")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(1)
print("Waiting for Robocode connection...")

conn, addr = s.accept()
print(f"Connected from {addr}")

episode        = 0
step           = 0
prev_state     = None
prev_action    = None
active_agent   = agents["UNKNOWN"]
episode_reward = 0.0

SAVE_EVERY = 10

try:
    while True:
        line = conn.recv(4096).decode().strip()
        if not line:
            print("Robot disconnected.")
            break

        if not line.startswith("STATE|"):
            continue

        payload = line.split("|")[1]
        parts   = payload.split(",")

        enemy_name = parts[0]
        reward     = float(parts[9])
        done       = int(parts[10])

        # ── Terminal state ─────────────────────────────────────────────
        if enemy_name == "terminal":
            if prev_state is not None:
                active_agent.remember(prev_state, prev_action, reward,
                                      np.zeros(STATE_DIM, dtype=np.float32), True)
                active_agent.train()

            outcome = "win" if reward > 0 else "loss"
            stats   = logger.log_episode(
                episode=episode,
                epsilon=active_agent.epsilon,
                outcome=outcome,
            )

            active_agent.decay_epsilon()
            classifier.reset_episode()
            prev_state     = None
            prev_action    = None
            episode_reward = 0.0
            episode       += 1
            step           = 0

            if episode % SAVE_EVERY == 0:
                for agent in agents.values():
                    agent.save()

            print(f"--- Episode {stats['episode']} | "
                  f"outcome={stats['outcome']} | "
                  f"reward={stats['total_reward']} | "
                  f"steps={stats['steps']} | "
                  f"avg_loss={stats['avg_loss']} | "
                  f"ε={stats['epsilon']} ---")

            conn.sendall(b"ACTION|0\n")
            continue

        # ── Normal state ───────────────────────────────────────────────
        state = {
            "enemy_name":   enemy_name,
            "distance":     float(parts[1]),
            "bearing":      float(parts[2]),
            "enemy_energy": float(parts[3]),
            "velocity":     float(parts[4]),
            "heading":      float(parts[5]),
            "my_x":         float(parts[6]),
            "my_y":         float(parts[7]),
            "my_energy":    float(parts[8]),
            "reward":       reward,
            "done":         done,
        }

        category     = classifier.update(state)
        active_agent = agents[category]
        state_vec    = state_builder.build(state)

        if prev_state is not None:
            active_agent.remember(prev_state, prev_action, reward, state_vec, False)
            loss = active_agent.train()
        else:
            loss = None

        action         = active_agent.act(state_vec)
        episode_reward += reward

        logger.log_step(
            reward=reward,
            loss=loss,
            category=category,
            epsilon=active_agent.epsilon,
            action=action,
        )

        print(f"[Ep {episode} Step {step}] "
              f"cat={category} "
              f"ε={active_agent.epsilon:.3f} "
              f"action={action} "
              f"reward={reward:.1f} "
              f"ep_reward={episode_reward:.1f} "
              f"loss={f'{loss:.4f}' if loss else 'buffering'}")

        prev_state  = state_vec
        prev_action = action
        step       += 1

        conn.sendall(f"ACTION|{action}\n".encode())

except Exception as e:
    print(f"Server error: {e}")
    raise

finally:
    for agent in agents.values():
        agent.save()
    conn.close()
    s.close()
    print("Server shut down. All weights saved.")