# training_logger.py
import os
import csv
import json
from datetime import datetime


class TrainingLogger:
    """
    Writes one row per episode to a CSV and maintains a JSON summary
    file that gets updated every episode. Both files persist across
    sessions so training history accumulates over time.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # CSV — one row per episode, appends across sessions
        self.csv_path     = os.path.join(log_dir, "episodes.csv")
        self.summary_path = os.path.join(log_dir, "summary.json")

        self._init_csv()

        # In-memory state for the current episode
        self._reset_episode()

        # Running totals across all episodes (loaded from summary if exists)
        self.summary = self._load_summary()

    # ── Episode lifecycle ──────────────────────────────────────────────

    def _reset_episode(self):
        self.ep_rewards   = []
        self.ep_losses    = []
        self.ep_steps     = 0
        self.ep_category  = "UNKNOWN"
        self.ep_start     = datetime.now()

    def log_step(self,
                 reward:   float,
                 loss:     float | None,
                 category: str,
                 epsilon:  float,
                 action:   int) -> None:
        """Call every scan inside the normal state block."""
        self.ep_rewards.append(reward)
        self.ep_steps += 1
        self.ep_category = category
        if loss is not None:
            self.ep_losses.append(loss)

    def log_episode(self,
                    episode:  int,
                    epsilon:  float,
                    outcome:  str = "unknown") -> dict:
        """
        Call at episode end (terminal branch).
        outcome should be 'win' or 'loss' — pass this from Java's done signal.
        Returns the stats dict so server.py can print it.
        """
        duration    = (datetime.now() - self.ep_start).total_seconds()
        total_r     = sum(self.ep_rewards)
        avg_loss    = sum(self.ep_losses) / len(self.ep_losses) if self.ep_losses else 0.0
        avg_reward  = total_r / self.ep_steps if self.ep_steps > 0 else 0.0

        stats = {
            "episode":      episode,
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "category":     self.ep_category,
            "outcome":      outcome,
            "total_reward": round(total_r, 2),
            "avg_reward":   round(avg_reward, 4),
            "avg_loss":     round(avg_loss, 6),
            "steps":        self.ep_steps,
            "epsilon":      round(epsilon, 4),
            "duration_s":   round(duration, 1),
        }

        self._write_csv_row(stats)
        self._update_summary(stats)
        self._reset_episode()
        return stats

    # ── CSV ───────────────────────────────────────────────────────────

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "episode", "timestamp", "category", "outcome",
                    "total_reward", "avg_reward", "avg_loss",
                    "steps", "epsilon", "duration_s",
                ])
                writer.writeheader()

    def _write_csv_row(self, stats: dict):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writerow(stats)

    # ── JSON summary ──────────────────────────────────────────────────

    def _load_summary(self) -> dict:
        if os.path.exists(self.summary_path):
            with open(self.summary_path) as f:
                return json.load(f)
        return {
            "total_episodes": 0,
            "total_steps":    0,
            "wins":           0,
            "losses":         0,
            "best_reward":    float("-inf"),
            "recent_rewards": [],       # last 20 episode totals
            "by_category":    {},
        }

    def _update_summary(self, stats: dict):
        s = self.summary
        s["total_episodes"] += 1
        s["total_steps"]    += stats["steps"]

        if stats["outcome"] == "win":
            s["wins"] += 1
        elif stats["outcome"] == "loss":
            s["losses"] += 1

        if stats["total_reward"] > s["best_reward"]:
            s["best_reward"] = stats["total_reward"]

        s["recent_rewards"].append(stats["total_reward"])
        if len(s["recent_rewards"]) > 20:
            s["recent_rewards"].pop(0)

        # Per-category breakdown
        cat = stats["category"]
        if cat not in s["by_category"]:
            s["by_category"][cat] = {
                "episodes": 0, "wins": 0,
                "total_reward": 0.0, "avg_loss": 0.0
            }
        c = s["by_category"][cat]
        c["episodes"]     += 1
        c["total_reward"] += stats["total_reward"]
        c["avg_loss"]      = round(
            (c["avg_loss"] * (c["episodes"] - 1) + stats["avg_loss"]) / c["episodes"], 6
        )
        if stats["outcome"] == "win":
            c["wins"] += 1

        with open(self.summary_path, "w") as f:
            json.dump(s, f, indent=2)