# report.py  —  run this anytime: python report.py
import json
import csv
import os

LOG_DIR      = "logs"
CSV_PATH     = os.path.join(LOG_DIR, "episodes.csv")
SUMMARY_PATH = os.path.join(LOG_DIR, "summary.json")


def load_csv():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def load_summary():
    if not os.path.exists(SUMMARY_PATH):
        return None
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def win_rate(wins, total):
    if total == 0:
        return "n/a"
    return f"{wins / total * 100:.1f}%"


def trend(values, window=10):
    """Return 'up', 'down', or 'flat' based on last window vs previous window."""
    if len(values) < window * 2:
        return "not enough data"
    recent = sum(values[-window:])   / window
    prior  = sum(values[-window*2:-window]) / window
    diff   = recent - prior
    if diff > 0.5:
        return f"up   (+{diff:.2f})"
    if diff < -0.5:
        return f"down ({diff:.2f})"
    return f"flat ({diff:+.2f})"


def print_report():
    rows    = load_csv()
    summary = load_summary()

    if not rows or not summary:
        print("No training data found. Run some episodes first.")
        return

    sep = "─" * 52

    print(f"\n{'═' * 52}")
    print(f"  Training report")
    print(f"{'═' * 52}")

    # ── Overall ──
    total = summary["total_episodes"]
    wins  = summary["wins"]
    print(f"\n  Overall")
    print(sep)
    print(f"  Episodes        {total}")
    print(f"  Total steps     {summary['total_steps']}")
    print(f"  Win rate        {win_rate(wins, total)}")
    print(f"  Best reward     {summary['best_reward']:.2f}")

    recent = summary["recent_rewards"]
    if recent:
        avg_recent = sum(recent) / len(recent)
        print(f"  Avg reward      {avg_recent:.2f}  (last {len(recent)} eps)")

    # Reward trend
    all_rewards = [float(r["total_reward"]) for r in rows]
    print(f"  Reward trend    {trend(all_rewards)}")

    # Epsilon from last episode
    if rows:
        last = rows[-1]
        print(f"  Epsilon now     {last['epsilon']}")

    # ── Per-category breakdown ──
    print(f"\n  By category")
    print(sep)
    print(f"  {'Category':<12} {'Eps':>5} {'Win%':>7} {'Avg reward':>12} {'Avg loss':>10}")
    print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*12} {'─'*10}")

    for cat, data in summary["by_category"].items():
        eps    = data["episodes"]
        wr     = win_rate(data["wins"], eps)
        avg_r  = data["total_reward"] / eps if eps > 0 else 0
        avg_l  = data["avg_loss"]
        print(f"  {cat:<12} {eps:>5} {wr:>7} {avg_r:>12.2f} {avg_l:>10.6f}")

    # ── Recent episodes ──
    print(f"\n  Last 10 episodes")
    print(sep)
    print(f"  {'Ep':>4} {'Cat':<10} {'Result':<8} {'Reward':>8} {'Steps':>6} {'Loss':>10} {'ε':>7}")
    print(f"  {'─'*4} {'─'*10} {'─'*8} {'─'*8} {'─'*6} {'─'*10} {'─'*7}")

    for r in rows[-10:]:
        print(f"  {r['episode']:>4} "
              f"{r['category']:<10} "
              f"{r['outcome']:<8} "
              f"{float(r['total_reward']):>8.2f} "
              f"{r['steps']:>6} "
              f"{float(r['avg_loss']):>10.6f} "
              f"{float(r['epsilon']):>7.4f}")

    print(f"\n{'═' * 52}\n")


if __name__ == "__main__":
    print_report()