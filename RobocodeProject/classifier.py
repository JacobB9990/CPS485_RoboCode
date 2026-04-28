# classifier.py
import re
from collections import defaultdict, deque

CATEGORIES = ["DEFENSIVE", "RUSHER", "SNIPER", "DODGER", "UNKNOWN"]

NAME_TABLE = {
    "DEFENSIVE": [r"corners", r"walls", r"spinbot", r"spin", r"sittingduck"],
    "RUSHER":    [r"ramfire", r"fire", r"crazy", r"target"],
    "SNIPER":    [],
    "DODGER":    [r"tracker", r"aliens", r"alien"],
    "UNKNOWN":   [],
}

NAME_CONFIDENCE_THRESHOLD = 0.85


class EnemyClassifier:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self._reset()

    def _reset(self):
        self.category    = "UNKNOWN"
        self.confidence  = 0.0
        self.name_locked = False
        self.scan_count  = 0

        self._velocities = deque(maxlen=self.window_size)
        self._distances  = deque(maxlen=self.window_size)
        self._bearings   = deque(maxlen=self.window_size)
        self._headings   = deque(maxlen=self.window_size)

    def reset_episode(self):
        self._reset()

    # ── Public API ─────────────────────────────────────────────────────

    def update(self, state: dict) -> str:
        self.scan_count += 1

        if self.scan_count == 1:
            cat, conf = self._classify_by_name(state["enemy_name"])
            if conf >= NAME_CONFIDENCE_THRESHOLD:
                self.category    = cat
                self.confidence  = conf
                self.name_locked = True
                return self.category

        self._velocities.append(abs(state["velocity"]))
        self._distances.append(state["distance"])
        self._bearings.append(state["bearing"])
        self._headings.append(state["heading"])

        if not self.name_locked and self.scan_count >= 5:
            cat, conf       = self._classify_by_behavior()
            self.category   = cat
            self.confidence = conf

        return self.category

    def summary(self) -> dict:
        return {
            "category":    self.category,
            "confidence":  round(self.confidence, 2),
            "scans":       self.scan_count,
            "name_locked": self.name_locked,
        }

    # ── Stage 1: name lookup ───────────────────────────────────────────

    def _classify_by_name(self, raw_name: str) -> tuple[str, float]:
        name = raw_name.split(".")[-1].lower().strip()
        scores = defaultdict(float)
        for category, patterns in NAME_TABLE.items():
            if category == "UNKNOWN":
                continue
            for pattern in patterns:
                if re.search(pattern, name):
                    scores[category] += 1.0

        if not scores:
            return "UNKNOWN", 0.0

        best_cat  = max(scores, key=scores.get)
        best_conf = min(scores[best_cat], 1.0)
        return best_cat, best_conf

    # ── Stage 2: behavioral heuristics ────────────────────────────────

    def _classify_by_behavior(self) -> tuple[str, float]:
        scores = defaultdict(float)

        avg_vel         = sum(self._velocities) / len(self._velocities)
        avg_dist        = sum(self._distances)  / len(self._distances)
        dist_variance   = self._variance(self._distances)
        bearing_variance = self._variance(self._bearings)
        heading_variance = self._variance(self._headings)
        approach_rate   = self._approach_rate()

        # ── DEFENSIVE: slow or stationary, holds position ──────────────
        # Covers both old SITTER (spinning in place) and CAMPER (corner holder).
        # Key signal: low velocity AND stable distance to us.
        if avg_vel < 2.0:
            scores["DEFENSIVE"] += 2.0
        if dist_variance < 800:
            scores["DEFENSIVE"] += 1.0
        if avg_dist > 250:
            scores["DEFENSIVE"] += 0.5

        # ── RUSHER: fast, closing in aggressively ──────────────────────
        if avg_vel > 5.0:
            scores["RUSHER"] += 1.5
        if approach_rate < -10:
            scores["RUSHER"] += 1.0
        if avg_dist < 200:
            scores["RUSHER"] += 0.5

        # ── SNIPER: very slow, stays far, consistent range ─────────────
        if avg_vel < 1.0:
            scores["SNIPER"] += 2.0
        if avg_dist > 400:
            scores["SNIPER"] += 1.0
        if dist_variance < 500:
            scores["SNIPER"] += 0.5

        # ── DODGER: moves fast AND changes direction frequently ─────────
        # High velocity combined with high heading variance is the signature —
        # it's not just moving, it's changing direction unpredictably.
        if avg_vel > 3.0:
            scores["DODGER"] += 1.0
        if heading_variance > 0.3:          # direction changes a lot
            scores["DODGER"] += 2.0
        if dist_variance > 1500:            # distance bounces around = strafing
            scores["DODGER"] += 1.0
        if bearing_variance > 0.2:          # bearing shifts fast relative to us
            scores["DODGER"] += 0.5

        if not scores:
            return "UNKNOWN", 0.0

        best_cat   = max(scores, key=scores.get)
        total      = sum(scores.values())
        confidence = scores[best_cat] / total if total > 0 else 0.0
        return best_cat, confidence

    # ── Helpers ────────────────────────────────────────────────────────

    def _variance(self, window: deque) -> float:
        if len(window) < 2:
            return 0.0
        mean = sum(window) / len(window)
        return sum((x - mean) ** 2 for x in window) / len(window)

    def _approach_rate(self) -> float:
        dists = list(self._distances)
        if len(dists) < 2:
            return 0.0
        deltas = [dists[i+1] - dists[i] for i in range(len(dists)-1)]
        return sum(deltas) / len(deltas)