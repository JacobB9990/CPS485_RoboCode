import re
from collections import defaultdict, deque

# ── Category definitions ───────────────────────────────────────────────
CATEGORIES = ["SITTER", "RUSHER", "SNIPER", "CAMPER", "UNKNOWN"]

# Known Robocode sample/common bots mapped to categories.
# getName() returns the fully qualified name like "sample.Corners" or just "Corners"
# depending on how it's packaged — we strip the package prefix before matching.
NAME_TABLE = {
    "SITTER":  [r"corners", r"spinbot", r"walls", r"tracker"],
    "RUSHER":  [r"ramfire", r"fire", r"crazy", r"target"],
    "SNIPER":  [r"sitting\s*duck", r"sittingduck"],
    "CAMPER":  [r"corners", r"walls"],
    "UNKNOWN": [],
}

# Confidence required from name lookup to skip behavioral stage
NAME_CONFIDENCE_THRESHOLD = 0.85


class EnemyClassifier:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self._reset()

    def _reset(self):
        self.category = "UNKNOWN"
        self.confidence = 0.0
        self.name_locked = False          # True if name lookup gave a clean match
        self.scan_count = 0

        # Rolling windows for behavioral features
        self._velocities   = deque(maxlen=self.window_size)
        self._distances    = deque(maxlen=self.window_size)
        self._bearings     = deque(maxlen=self.window_size)

    def reset_episode(self):
        """Call this at the start of each new round."""
        self._reset()

    # ── Public API ─────────────────────────────────────────────────────

    def update(self, state: dict) -> str:
        """
        Feed one scan's worth of state and return the current category.

        Expected keys in state:
            enemy_name   (str)
            distance     (float)
            bearing      (float, radians)
            enemy_energy (float)
            velocity     (float)
        """
        self.scan_count += 1

        # Stage 1: name lookup (only runs once per episode)
        if self.scan_count == 1:
            cat, conf = self._classify_by_name(state["enemy_name"])
            if conf >= NAME_CONFIDENCE_THRESHOLD:
                self.category    = cat
                self.confidence  = conf
                self.name_locked = True
                return self.category

        # Stage 2: behavioral accumulator (always runs)
        self._velocities.append(abs(state["velocity"]))
        self._distances.append(state["distance"])
        self._bearings.append(state["bearing"])

        # Only reclassify after enough samples, and not if name-locked
        if not self.name_locked and self.scan_count >= 5:
            cat, conf = self._classify_by_behavior()
            self.category   = cat
            self.confidence = conf

        return self.category

    def summary(self) -> dict:
        return {
            "category":   self.category,
            "confidence": round(self.confidence, 2),
            "scans":      self.scan_count,
            "name_locked": self.name_locked,
        }

    # ── Stage 1: name lookup ───────────────────────────────────────────

    def _classify_by_name(self, raw_name: str) -> tuple[str, float]:
        # Strip package prefix: "sample.Corners" → "corners"
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
        """
        Score each category based on rolling behavioral features.
        Returns (category, confidence) where confidence is 0–1.
        """
        scores = defaultdict(float)

        avg_vel      = sum(self._velocities) / len(self._velocities)
        avg_dist     = sum(self._distances)  / len(self._distances)
        dist_variance = self._variance(self._distances)
        approach_rate = self._approach_rate()   # negative = closing in

        # ── RUSHER: fast, closing distance quickly, close range ──
        if avg_vel > 5.0:
            scores["RUSHER"] += 1.5
        if approach_rate < -10:        # closing fast
            scores["RUSHER"] += 1.0
        if avg_dist < 200:
            scores["RUSHER"] += 0.5

        # ── SNIPER: very slow or stationary, stays far away ──
        if avg_vel < 1.0:
            scores["SNIPER"] += 2.0
        if avg_dist > 400:
            scores["SNIPER"] += 1.0
        if dist_variance < 500:        # stays at consistent range
            scores["SNIPER"] += 0.5

        # ── CAMPER: low velocity, mid-far distance, low variance ──
        if avg_vel < 2.5:
            scores["CAMPER"] += 1.0
        if 250 < avg_dist < 500:
            scores["CAMPER"] += 1.0
        if dist_variance < 1000:
            scores["CAMPER"] += 0.5

        # ── SITTER: non-zero but low velocity, circling (bearing changes) ──
        bearing_variance = self._variance(self._bearings)
        if avg_vel < 4.0:
            scores["SITTER"] += 0.5
        if bearing_variance > 0.1:     # bearing is shifting = orbiting
            scores["SITTER"] += 1.0

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
        """Average change in distance per scan (negative = closing)."""
        dists = list(self._distances)
        if len(dists) < 2:
            return 0.0
        deltas = [dists[i+1] - dists[i] for i in range(len(dists)-1)]
        return sum(deltas) / len(deltas)