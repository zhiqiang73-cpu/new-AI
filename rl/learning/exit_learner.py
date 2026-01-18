"""
Exit Timing Learner - Learns optimal exit conditions through reinforcement
"""
import json
import os
from typing import Dict, List, Optional


class ExitTimingLearner:
    """
    Learns optimal exit timing based on:
    1. K-line patterns before/after exit
    2. Trade profitability
    3. Market regime at exit time
    """

    def __init__(self, data_dir: str = "rl_data"):
        self.data_dir = data_dir
        self.params_file = os.path.join(data_dir, "exit_params.json")
        self.min_trades = 20

        # Learnable parameters
        self.params = {
            # Profit-based exit thresholds
            "quick_profit_pct": 0.5,  # Quick profit target
            "normal_profit_pct": 1.0,  # Normal profit target
            "trailing_stop_trigger": 0.8,  # When to activate trailing stop (%)
            "trailing_stop_distance": 0.4,  # Trailing stop distance (%)

            # Time-based parameters
            "max_hold_minutes": 60,  # Maximum hold time
            "min_profit_after_time": 0.2,  # Minimum profit after max_hold_minutes

            # Pattern-based weights (learned)
            "weight_momentum_reversal": 0.3,  # Exit on momentum reversal
            "weight_volume_spike": 0.2,  # Exit on unusual volume
            "weight_sr_rejection": 0.5,  # Exit on S/R rejection

            # Learning rate and counters
            "lr": 0.08,  # Learning rate (higher for faster learning)
            "total_exits": 0,
            "profitable_exits": 0,
        }

        # Exit history for analysis
        self.exit_history = []
        self._load()

    def _load(self):
        """Load saved parameters"""
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, "r") as f:
                    saved = json.load(f)
                    self.params.update(saved)
            except Exception:
                pass

    def _save(self):
        """Save parameters"""
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            with open(self.params_file, "w") as f:
                json.dump(self.params, f, indent=2)
        except Exception:
            pass

    def analyze_exit_quality(
        self,
        exit_price: float,
        direction: str,
        klines_before: List[Dict],
        klines_after: List[Dict],
    ) -> Dict:
        """
        Analyze if exit timing was good
        Returns timing quality score: -1 (bad) to +1 (perfect)
        """
        if not klines_before or not klines_after:
            return {"quality": 0.0, "reason": "insufficient_data"}

        if direction == "LONG":
            # For long: good exit = price went down after
            prices_after = [k["close"] for k in klines_after[:5]]
            if not prices_after:
                return {"quality": 0.0, "reason": "no_after_data"}

            min_after = min(prices_after)
            max_after = max(prices_after)

            if min_after < exit_price * 0.995:
                # Price dropped after exit - good timing
                drop_pct = (exit_price - min_after) / exit_price * 100
                quality = min(1.0, drop_pct / 0.5)  # Normalize
                return {"quality": quality, "reason": "price_dropped", "drop_pct": drop_pct}
            elif max_after > exit_price * 1.005:
                # Price went up after exit - could have held longer
                rise_pct = (max_after - exit_price) / exit_price * 100
                quality = -min(1.0, rise_pct / 0.5)
                return {"quality": quality, "reason": "left_profit", "rise_pct": rise_pct}
        else:
            # For short: good exit = price went up after
            prices_after = [k["close"] for k in klines_after[:5]]
            if not prices_after:
                return {"quality": 0.0, "reason": "no_after_data"}

            max_after = max(prices_after)
            min_after = min(prices_after)

            if max_after > exit_price * 1.005:
                # Price rose after exit - good timing
                rise_pct = (max_after - exit_price) / exit_price * 100
                quality = min(1.0, rise_pct / 0.5)
                return {"quality": quality, "reason": "price_rose", "rise_pct": rise_pct}
            elif min_after < exit_price * 0.995:
                # Price went down after exit - could have held longer
                drop_pct = (exit_price - min_after) / exit_price * 100
                quality = -min(1.0, drop_pct / 0.5)
                return {"quality": quality, "reason": "left_profit", "drop_pct": drop_pct}

        return {"quality": 0.0, "reason": "neutral"}

    def update(
        self,
        pnl_pct: float,
        exit_timing_quality: float,
        hold_minutes: float,
        exit_reason: str,
    ):
        """
        Update learnable parameters based on trade outcome
        """
        self.params["total_exits"] += 1
        if pnl_pct > 0:
            self.params["profitable_exits"] += 1

        lr = self.params["lr"]

        # Combined reward: pnl + timing quality
        reward = (pnl_pct * 0.7 + exit_timing_quality * 0.3) * 0.5

        # Adjust parameters based on reward
        if reward > 0:
            # Good exit - reinforce current parameters direction
            if exit_reason == "TAKE_PROFIT":
                # If TP was hit and outcome good, current TP is appropriate
                pass
            elif exit_reason == "TIME_COST":
                # If time exit was good, maybe reduce max_hold_minutes
                if exit_timing_quality > 0.3:
                    self.params["max_hold_minutes"] *= (1 - lr * 0.1)
            elif exit_reason == "TRAILING_STOP":
                # Trailing stop worked well
                if exit_timing_quality > 0.5:
                    self.params["trailing_stop_distance"] *= (1 - lr * 0.05)
        else:
            # Bad exit - adjust parameters
            if exit_reason == "STOP_LOSS":
                # SL hit with bad outcome - maybe SL too tight
                if pnl_pct < -0.5:
                    # Actually good SL prevented bigger loss
                    pass
                else:
                    # Maybe noise - could widen SL slightly
                    pass
            elif exit_reason == "TAKE_PROFIT":
                # TP was hit but timing was bad - could have held longer
                if exit_timing_quality < -0.3:
                    self.params["normal_profit_pct"] *= (1 + lr * 0.1)

        # Ensure bounds
        self.params["quick_profit_pct"] = max(0.2, min(1.0, self.params["quick_profit_pct"]))
        self.params["normal_profit_pct"] = max(0.5, min(3.0, self.params["normal_profit_pct"]))
        self.params["max_hold_minutes"] = max(15, min(180, self.params["max_hold_minutes"]))
        self.params["trailing_stop_distance"] = max(0.2, min(1.0, self.params["trailing_stop_distance"]))

        self._save()

        return {
            "reward": reward,
            "updated_params": {
                "quick_profit_pct": self.params["quick_profit_pct"],
                "normal_profit_pct": self.params["normal_profit_pct"],
                "max_hold_minutes": self.params["max_hold_minutes"],
            },
        }

    def should_exit_now(
        self,
        pnl_pct: float,
        hold_minutes: float,
        momentum: float,
        near_sr: bool,
    ) -> Dict:
        """
        Decide if should exit now based on learned parameters
        """
        should_exit = False
        reason = None
        confidence = 0.0

        # Quick profit check
        if pnl_pct >= self.params["quick_profit_pct"]:
            if momentum < 0 or near_sr:
                should_exit = True
                reason = "QUICK_PROFIT"
                confidence = 0.7

        # Normal profit with momentum reversal
        if pnl_pct >= self.params["normal_profit_pct"]:
            should_exit = True
            reason = "TAKE_PROFIT"
            confidence = 0.85

        # Time-based exit
        if hold_minutes > self.params["max_hold_minutes"]:
            if pnl_pct < self.params["min_profit_after_time"]:
                should_exit = True
                reason = "TIME_COST"
                confidence = 0.6

        # Pattern-based exits (weighted)
        pattern_score = 0.0
        if momentum < -0.5:
            pattern_score += self.params["weight_momentum_reversal"]
        if near_sr:
            pattern_score += self.params["weight_sr_rejection"]

        if pattern_score > 0.5 and pnl_pct > 0:
            should_exit = True
            reason = "PATTERN_EXIT"
            confidence = pattern_score

        return {
            "should_exit": should_exit,
            "reason": reason,
            "confidence": confidence,
            "pnl_pct": pnl_pct,
        }

    def get_exit_params(self) -> Dict:
        """Get current exit parameters for display"""
        return {
            "quick_profit_pct": round(self.params["quick_profit_pct"], 2),
            "normal_profit_pct": round(self.params["normal_profit_pct"], 2),
            "max_hold_minutes": int(self.params["max_hold_minutes"]),
            "trailing_stop_distance": round(self.params["trailing_stop_distance"], 2),
            "total_exits": self.params["total_exits"],
            "profitable_exits": self.params["profitable_exits"],
            "win_rate": round(
                self.params["profitable_exits"] / max(1, self.params["total_exits"]) * 100, 1
            ),
        }



