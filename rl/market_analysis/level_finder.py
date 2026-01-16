import json
import os
import time
from typing import Dict, List, Tuple


DEFAULT_WEIGHTS = {
    "volume_density": 0.16,
    "touch_bounce_count": 0.17,
    "bounce_magnitude": 0.12,
    "failed_breakout_count": 0.17,
    "duration_days": 0.08,
    "multi_tf_confirm": 0.12,
    "orderbook_bid_wall": 0.06,
    "orderbook_ask_wall": 0.04,
    "orderbook_big_ratio": 0.05,
    "recent_volume_ratio": 0.03,
}

FEATURE_NAMES_CN = {
    "volume_density": "成交量密度",
    "touch_bounce_count": "触及与反弹次数",
    "bounce_magnitude": "反弹幅度",
    "failed_breakout_count": "失败突破次数",
    "duration_days": "持续时间",
    "multi_tf_confirm": "多周期触及确认",
    "orderbook_bid_wall": "订单簿买单墙厚度",
    "orderbook_ask_wall": "订单簿卖单墙厚度",
    "orderbook_big_ratio": "订单簿大单占比",
    "recent_volume_ratio": "近期成交量活跃度",
}


class LevelFeatureCalculator:
    def __init__(self, tolerance_pct: float = 0.005):
        # 提高容差到0.5%，更好识别S/R区域
        self.tolerance_pct = tolerance_pct

    def _near(self, price: float, level: float) -> bool:
        return abs(price - level) / level <= self.tolerance_pct

    def multi_tf_confirm(self, level: float, klines_by_tf: Dict[str, List[Dict]], tf_weights: Dict[str, float]) -> float:
        total_w = sum(tf_weights.values()) or 1.0
        score = 0.0
        for tf, klines in klines_by_tf.items():
            if not klines:
                continue
            touched = any(self._near(k["close"], level) for k in klines)
            if touched:
                score += tf_weights.get(tf, 0)
        return min(score / total_w, 1.0)

    def calculate(self, level: float, klines: List[Dict]) -> Dict:
        touches = 0
        bounces = 0
        bounce_magnitude = 0.0
        volumes = []
        failed_breakouts = 0
        first_ts = None
        last_ts = None
        max_volume = max([k.get("volume", 0) for k in klines], default=1)

        for i in range(1, len(klines)):
            k = klines[i]
            prev = klines[i - 1]
            price = k["close"]
            if self._near(price, level):
                touches += 1
                volumes.append(k.get("volume", 0))
                if first_ts is None:
                    first_ts = k.get("time")
                last_ts = k.get("time")

                if price > level and prev["close"] < level:
                    failed_breakouts += 1
                if price < level and prev["close"] > level:
                    failed_breakouts += 1

                # Bounce magnitude
                if i + 1 < len(klines):
                    next_close = klines[i + 1]["close"]
                    bounce = abs(next_close - level) / level
                    bounce_magnitude += bounce
                    if bounce > self.tolerance_pct:
                        bounces += 1

        volume_density = (sum(volumes) / len(volumes)) / max_volume if volumes else 0.0
        duration_days = 0.0
        if first_ts and last_ts and last_ts > first_ts:
            duration_days = (last_ts - first_ts) / 86400.0

        return {
            "volume_density": min(volume_density, 1.0),
            "touch_bounce_count": min((touches + bounces) / 10, 1.0),
            "bounce_magnitude": min(bounce_magnitude / max(1, bounces), 1.0),
            "failed_breakout_count": min(failed_breakouts / 5, 1.0),
            "duration_days": min(duration_days / 7, 1.0),
            "multi_tf_confirm": 0.0,
            "orderbook_bid_wall": 0.0,
            "orderbook_ask_wall": 0.0,
            "orderbook_big_ratio": 0.0,
            "recent_volume_ratio": 0.0,
        }


class BestLevelFinder:
    def __init__(self, stats_path: str):
        self.stats_path = stats_path
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        self.stats = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                weights = data.get("weights", DEFAULT_WEIGHTS.copy())
                for k, v in DEFAULT_WEIGHTS.items():
                    weights.setdefault(k, v)
                total = sum(weights.values())
                if total > 0:
                    for k in weights:
                        weights[k] = weights[k] / total
                data["weights"] = weights
                return data
        return {
            "weights": DEFAULT_WEIGHTS.copy(),
            "total_trades": 0,
            "effective_trades": 0,
            "weight_history": [],
        }

    def _save(self) -> None:
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def score_level(self, features: Dict) -> float:
        weights = self.stats.get("weights", DEFAULT_WEIGHTS)
        score = 0.0
        for key, weight in weights.items():
            score += float(features.get(key, 0)) * weight
        return score * 100

    def get_weights_display(self) -> List[Dict]:
        weights = self.stats.get("weights", DEFAULT_WEIGHTS)
        return [
            {"name": FEATURE_NAMES_CN.get(k, k), "weight": round(v * 100, 1), "key": k}
            for k, v in weights.items()
        ]

    def get_stats_summary(self) -> str:
        total = self.stats.get("total_trades", 0)
        effective = self.stats.get("effective_trades", 0)
        rate = (effective / total * 100) if total > 0 else 0
        return f"total={total}, effective={effective}, rate={rate:.1f}%"

    def get_feature_analysis(self) -> Dict:
        return {
            "weights": self.stats.get("weights", DEFAULT_WEIGHTS),
            "total_trades": self.stats.get("total_trades", 0),
            "effective_trades": self.stats.get("effective_trades", 0),
        }

    def get_learning_progress(self) -> Dict:
        total = self.stats.get("total_trades", 0)
        effective = self.stats.get("effective_trades", 0)
        return {
            "total_trades": total,
            "effective_trades": effective,
            "min_trades_needed": 30,
            "progress_percent": min(100, total / 30 * 100) if total < 30 else 100,
            "can_adjust": total >= 30,
            "effectiveness_rate": (effective / total * 100) if total > 0 else 0,
            "weight_adjustments": len(self.stats.get("weight_history", [])),
            "current_weights": self.stats.get("weights", DEFAULT_WEIGHTS),
        }

    def update_stats(self, effective: bool) -> None:
        self.stats["total_trades"] = self.stats.get("total_trades", 0) + 1
        if effective:
            self.stats["effective_trades"] = self.stats.get("effective_trades", 0) + 1
        self._save()

    def update_weights(self, features: Dict, reward: float, min_trades: int = 30) -> None:
        total = self.stats.get("total_trades", 0)
        if total < min_trades:
            return

        before = self.stats.get("weights", DEFAULT_WEIGHTS).copy()
        weights = before.copy()
        # Increased learning rate for faster adaptation
        lr = self.stats.get("learning_rate", 0.12)
        for k in weights:
            delta = lr * reward * (features.get(k, 0) - 0.5)
            weights[k] = max(0.01, weights[k] + delta)

        # Normalize
        s = sum(weights.values())
        if s > 0:
            for k in weights:
                weights[k] = weights[k] / s

        history = self.stats.get("weight_history", [])
        history.append({"ts": time.time(), "weights": weights})
        self.stats["weights"] = weights
        self.stats["weight_history"] = history[-200:]
        self._save()
        deltas = {k: round(weights[k] - before[k], 6) for k in weights}
        return {
            "reward": reward,
            "before": before,
            "after": weights,
            "delta": deltas,
            "total_trades": total,
        }

