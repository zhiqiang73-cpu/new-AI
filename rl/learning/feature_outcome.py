import json
import os
from datetime import datetime
from typing import Dict


class FeatureOutcomeTracker:
    """
    Track outcome impact by feature group and adjust:
    - Entry position size multiplier
    - Exit decision threshold delta
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.entry_bias, self.exit_bias, self.history = self._load()

    def _default_bias(self) -> Dict[str, float]:
        return {}

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return (
                        data.get("entry_bias", self._default_bias()),
                        data.get("exit_bias", self._default_bias()),
                        data.get("history", []),
                    )
            except Exception:
                pass
        return self._default_bias(), self._default_bias(), []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "entry_bias": self.entry_bias,
                        "exit_bias": self.exit_bias,
                        "history": self.history[-200:],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

    def _compute_reward(self, pnl_percent: float, hold_minutes: float = None, fee_pct: float = 0.12) -> float:
        pnl_percent = float(pnl_percent or 0.0)
        fee_pct = float(fee_pct or 0.0)
        time_penalty = 0.0
        if hold_minutes is not None:
            time_penalty = max(0.0, hold_minutes - 15) * 0.01
        net_pnl = pnl_percent - fee_pct - time_penalty
        return max(-1.0, min(1.0, net_pnl / 2.0))

    def _update_bias(self, bias: Dict[str, float], features: Dict[str, float], reward: float, lr: float = 0.05):
        for key, value in features.items():
            v = float(value or 0.0)
            delta = lr * reward * v
            bias[key] = float(bias.get(key, 0.0)) + delta
            # Clamp to avoid extreme effects
            bias[key] = max(-1.0, min(1.0, bias[key]))

    def update_entry(self, features: Dict[str, float], pnl_percent: float, hold_minutes: float = None, fee_pct: float = 0.12):
        reward = self._compute_reward(pnl_percent, hold_minutes, fee_pct)
        self._update_bias(self.entry_bias, features, reward)
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": "entry",
                "reward": round(reward, 4),
                "features": {k: round(float(v or 0.0), 3) for k, v in features.items()},
            }
        )
        self._save()

    def update_exit(self, features: Dict[str, float], pnl_percent: float, hold_minutes: float = None, fee_pct: float = 0.12):
        reward = self._compute_reward(pnl_percent, hold_minutes, fee_pct)
        self._update_bias(self.exit_bias, features, reward)
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": "exit",
                "reward": round(reward, 4),
                "features": {k: round(float(v or 0.0), 3) for k, v in features.items()},
            }
        )
        self._save()

    def entry_size_factor(self, features: Dict[str, float]) -> float:
        if not features:
            return 1.0
        score = 0.0
        for key, value in features.items():
            score += float(value or 0.0) * float(self.entry_bias.get(key, 0.0))
        # Limit impact to +/-30%
        return max(0.7, min(1.3, 1.0 + score))

    def exit_threshold_delta(self, features: Dict[str, float]) -> float:
        if not features:
            return 0.0
        score = 0.0
        for key, value in features.items():
            score += float(value or 0.0) * float(self.exit_bias.get(key, 0.0))
        # Convert to threshold delta, limited to +/-10
        return max(-10.0, min(10.0, score * 10.0))

