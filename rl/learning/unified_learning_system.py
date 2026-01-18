import json
import os
from typing import Dict

from ..config.time_manager import time_manager


class UnifiedLearningSystem:
    def __init__(self, stats_path: str):
        self.stats_path = stats_path
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        self.stats = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "weights": {},
            "total_samples": 0,
            "last_update_time": None,
        }

    def _save(self) -> None:
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def update(self, features: Dict, reward: float) -> None:
        weights = self.stats.get("weights", {})
        for k, v in features.items():
            weights[k] = weights.get(k, 0.0) + v * reward * 0.01
        self.stats["weights"] = weights
        self.stats["total_samples"] = self.stats.get("total_samples", 0) + 1
        self.stats["last_update_time"] = time_manager.now().isoformat()
        self._save()

    def get_weights(self) -> Dict:
        return self.stats.get("weights", {})

    def get_status(self) -> Dict:
        return {
            "total_samples": self.stats.get("total_samples", 0),
            "last_update_time": self.stats.get("last_update_time"),
            "weights": self.stats.get("weights", {}),
        }




