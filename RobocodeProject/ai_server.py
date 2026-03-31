import socket
import random
from classifier import EnemyClassifier

HOST = "localhost"
PORT = 5000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(1)

print("Waiting for Robocode connection...")

conn, addr = s.accept()
print(f"Connected from {addr}. Maintaining connection across all rounds.")

classifier = EnemyClassifier(window_size=20)
episode = 0
step = 0

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

        # ── Terminal state (death or win) ──────────────────────────────
        if enemy_name == "terminal":
            conn.sendall(f"ACTION|0\n".encode())
            classifier.reset_episode()
            episode += 1
            step = 0
            print(f"--- Episode {episode} ended (reward={reward}) ---")
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

        # ── Classify ───────────────────────────────────────────────────
        category = classifier.update(state)
        print(f"[Ep {episode} Step {step}] "
              f"dist={state['distance']:.1f}  "
              f"vel={state['velocity']:.1f}  "
              f"my_e={state['my_energy']:.1f}  "
              f"reward={reward}  "
              f"category={category}  "
              f"conf={classifier.summary()['confidence']:.2f}")

        # ── Action (random placeholder until DQN is wired in) ──────────
        action = random.randint(0, 3)
        conn.sendall(f"ACTION|{action}\n".encode())
        step += 1

except Exception as e:
    print(f"Server error: {e}")

finally:
    conn.close()
    s.close()
    print("Server shut down.")