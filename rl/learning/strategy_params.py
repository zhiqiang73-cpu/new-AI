import json
import os
from typing import Dict


class StrategyParamLearner:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.params = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        defaults = {
            "entry_threshold_bias": 0.0,
            "profit_lock_start": 0.5, # Tighter start (prev 0.6)
            "profit_lock_base_drop": 0.25, # Significantly tighter (prev 0.5)
            "profit_lock_slope": 0.05,
            "default_sl_pct": 0.3,
            "default_tp_pct": 3.5,
            "min_risk_reward": 1.5,
        }
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2)

    def get_entry_bias(self) -> float:
        return float(self.params.get("entry_threshold_bias", 0.0))

    def get_exit_params(self) -> Dict:
        return {
            "profit_lock_start": float(self.params.get("profit_lock_start", 0.6)),
            "profit_lock_base_drop": float(self.params.get("profit_lock_base_drop", 0.5)),
            "profit_lock_slope": float(self.params.get("profit_lock_slope", 0.05)),
        }

    def get_sl_tp_params(self) -> Dict:
        return {
            "default_sl_pct": float(self.params.get("default_sl_pct", 0.3)),
            "default_tp_pct": float(self.params.get("default_tp_pct", 3.5)),
            "min_risk_reward": float(self.params.get("min_risk_reward", 1.5)),
        }

    def update(self, reward: float, timing: Dict = None) -> Dict:
        bias = float(self.params.get("entry_threshold_bias", 0.0))
        profit_lock_start = float(self.params.get("profit_lock_start", 0.6))
        profit_lock_base_drop = float(self.params.get("profit_lock_base_drop", 0.5))
        profit_lock_slope = float(self.params.get("profit_lock_slope", 0.05))
        default_sl_pct = float(self.params.get("default_sl_pct", 0.3))
        default_tp_pct = float(self.params.get("default_tp_pct", 3.5))
        min_risk_reward = float(self.params.get("min_risk_reward", 1.5))

        if reward > 0:
            bias = max(-8.0, bias - 0.5)
            profit_lock_start = max(0.3, profit_lock_start - 0.02)
            profit_lock_base_drop = max(0.15, profit_lock_base_drop - 0.02) # Allow tightening further
            profit_lock_slope = min(0.2, profit_lock_slope + 0.01)
            default_sl_pct = max(0.15, default_sl_pct - 0.02)
            default_tp_pct = min(8.0, default_tp_pct + 0.1)
            min_risk_reward = min(3.0, min_risk_reward + 0.05)
        elif reward < 0:
            bias = min(8.0, bias + 0.5)
            profit_lock_start = min(1.5, profit_lock_start + 0.02)
            profit_lock_base_drop = min(0.4, profit_lock_base_drop + 0.02) # Cap looseness to 0.4 (prev 1.0)

            profit_lock_slope = max(0.01, profit_lock_slope - 0.01)
            default_sl_pct = min(1.0, default_sl_pct + 0.02)
            default_tp_pct = max(1.0, default_tp_pct - 0.1)
            min_risk_reward = max(1.2, min_risk_reward - 0.05)

        self.params["entry_threshold_bias"] = bias
        self.params["profit_lock_start"] = profit_lock_start
        self.params["profit_lock_base_drop"] = profit_lock_base_drop
        self.params["profit_lock_slope"] = profit_lock_slope
        self.params["default_sl_pct"] = default_sl_pct
        self.params["default_tp_pct"] = default_tp_pct
        self.params["min_risk_reward"] = min_risk_reward
        self.save()
        return self.params.copy()

