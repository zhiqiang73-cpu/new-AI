import json
import os
from typing import Dict, List

from .level_finder import LevelFeatureCalculator, DEFAULT_WEIGHTS


class LevelDiscovery:
    def __init__(self, buckets: List[int] = None):
        self.buckets = buckets or [50, 100, 250, 500, 1000]

    def _round_level(self, price: float, bucket: int) -> float:
        return round(price / bucket) * bucket

    def _integer_levels(self, prices: List[float]) -> List[float]:
        levels = set()
        for bucket in self.buckets:
            for price in prices:
                levels.add(self._round_level(price, bucket))
        return list(levels)

    def _swing_levels(self, klines: List[Dict], window: int = 5) -> List[float]:
        # 增大窗口到5根K线，更准确识别局部高低点
        levels = set()
        for i in range(window, len(klines) - window):
            high = klines[i]["high"]
            low = klines[i]["low"]
            left = klines[i - window:i]
            right = klines[i + 1:i + 1 + window]
            if all(high >= k["high"] for k in left) and all(
                high >= k["high"] for k in right
            ):
                levels.add(high)
            if all(low <= k["low"] for k in left) and all(
                low <= k["low"] for k in right
            ):
                levels.add(low)
        return list(levels)

    def _fractal_levels(self, klines: List[Dict], window: int = 3) -> List[float]:
        # 分形高低点识别（更严格的高低点）
        levels = set()
        for i in range(window, len(klines) - window):
            high = klines[i]["high"]
            low = klines[i]["low"]
            left = klines[i - window:i]
            right = klines[i + 1:i + 1 + window]
            # 分形高点：中间K线的high严格高于左右所有K线的high
            if all(high > k["high"] for k in left) and all(high > k["high"] for k in right):
                levels.add(high)
            # 分形低点
            if all(low < k["low"] for k in left) and all(low < k["low"] for k in right):
                levels.add(low)
        return list(levels)

    def _consolidation_levels(self, klines: List[Dict], min_touches: int = 3) -> List[float]:
        # 识别价格盘整区域（多次触及的价格）
        # 使用$100精度，过滤微小波动噪音
        if len(klines) < 20:
            return []
        levels = set()
        price_touches = {}
        for k in klines:
            for p in [k["high"], k["low"], k["close"]]:
                rounded = round(p / 100) * 100  # $100 precision - 过滤噪音
                price_touches[rounded] = price_touches.get(rounded, 0) + 1
        for price, count in price_touches.items():
            if count >= min_touches:
                levels.add(price)
        return list(levels)

    def _volume_profile_levels(self, klines: List[Dict], bucket: int = 100) -> List[float]:
        # 成交量密集区（$100 精度）
        buckets = {}
        for k in klines:
            price = self._round_level(k["close"], bucket)
            vol = k.get("volume", 0)
            if vol is None:
                vol = 0
            buckets[price] = buckets.get(price, 0) + vol
        top = sorted(buckets.items(), key=lambda x: x[1], reverse=True)[:8]
        return [p for p, _ in top]

    def _recent_high_low(self, klines: List[Dict], lookback: int = 50) -> List[float]:
        # 最近N根K线的最高最低点（优先关注近期数据）
        if len(klines) < lookback:
            lookback = len(klines)
        recent = klines[-lookback:]
        highs = [k["high"] for k in recent]
        lows = [k["low"] for k in recent]
        return [max(highs), min(lows)]

    def discover_all(
        self,
        klines: List[Dict],
        current_price: float = None,
        atr: float = None,
        max_distance_pct: float = None,
    ) -> Dict:
        if not klines:
            return {"support": [], "resistance": []}

        prices = [k["close"] for k in klines]
        current_price = current_price or prices[-1]

        candidates = set()
        # 多种方法发现候选位（优先关注近期K线）
        # 1. 最近50根K线的局部高低点（最重要）
        recent_hl = self._recent_high_low(klines, lookback=50)
        candidates.update(recent_hl)
        # 2. 最近100根K线的成交量密集区
        recent_klines = klines[-100:] if len(klines) >= 100 else klines
        candidates.update(self._volume_profile_levels(recent_klines))
        # 3. 近期摆动点（缩小窗口，更敏感）
        candidates.update(self._swing_levels(recent_klines, window=3))
        candidates.update(self._fractal_levels(recent_klines, window=3))
        # 4. 近期整理区间（降低最小触及次数，更敏感）
        candidates.update(self._consolidation_levels(recent_klines, min_touches=2))
        # 5. 整数位（辅助，但权重低）
        candidates.update(self._integer_levels(prices[-100:]))

        # ========== Level Merging: 合并相近能级 ==========
        def merge_nearby(levels: List[float], tolerance_pct: float = 0.2) -> List[float]:
            """合并距离<0.2%的能级（避免密集堆积）"""
            if not levels:
                return []
            sorted_levels = sorted(levels)
            merged = []
            current_group = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                distance_pct = (sorted_levels[i] - current_group[-1]) / current_group[-1] * 100
                if distance_pct < tolerance_pct:
                    current_group.append(sorted_levels[i])
                else:
                    # 取平均作为代表
                    merged.append(sum(current_group) / len(current_group))
                    current_group = [sorted_levels[i]]
            
            merged.append(sum(current_group) / len(current_group))
            return merged
        
        candidates = merge_nearby(list(candidates), tolerance_pct=0.2)

        # Dynamic band（缩小搜索范围，更关注近期S/R）
        if max_distance_pct is None:
            if atr and current_price > 0:
                # 缩小到0.5%-1.0%，更贴近当前价格
                atr_pct = (atr / current_price) * 100
                max_distance_pct = min(max(atr_pct * 50, 0.5), 1.0)
            else:
                max_distance_pct = 0.5  # Default 0.5%（约475 USDT @ 95k）

        def within_band(level: float) -> bool:
            return abs(level - current_price) / current_price * 100 <= max_distance_pct

        filtered = [c for c in candidates if within_band(c)]
        
        # 选择最强的S/R（按后续评分排序），距离只是加分项
        support = sorted([c for c in filtered if c < current_price], reverse=True)[:3]  # 取3个最强支撑
        resistance = sorted([c for c in filtered if c > current_price])[:3]  # 取3个最强阻力

        # ========== 强制最小间距：确保交易空间 ==========
        # 降低到0.3%（用户要求，更适合短线）
        min_gap_pct = 0.3
        
        if support and resistance:
            best_support = support[-1]
            best_resistance = resistance[0]
            gap_pct = (best_resistance - best_support) / current_price * 100
            
            # 如果间距<0.3%，寻找更远的S/R
            if gap_pct < min_gap_pct:
                new_resistance = [r for r in resistance if (r - best_support) / current_price * 100 >= min_gap_pct]
                if new_resistance:
                    resistance = new_resistance
                else:
                    # 如果找不到，返回空
                    return {"support": [], "resistance": []}

        return {"support": support, "resistance": resistance}


class LevelScoring:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.weights = self._load_weights()
        self.feature_calc = LevelFeatureCalculator()

    def _load_weights(self) -> Dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                weights = data.get("weights", DEFAULT_WEIGHTS.copy())
                for k, v in DEFAULT_WEIGHTS.items():
                    weights.setdefault(k, v)
                total = sum(weights.values())
                if total > 0:
                    for k in weights:
                        weights[k] = weights[k] / total
                return weights
        return DEFAULT_WEIGHTS.copy()

    def save_weights(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"weights": self.weights}, f, indent=2)

    def score(self, level: float, klines: List[Dict]) -> float:
        features = self.feature_calc.calculate(level, klines)
        score = 0.0
        for k, w in self.weights.items():
            score += float(features.get(k, 0)) * w
        return score * 100

    def score_multi_tf(
        self,
        level: float,
        klines_by_tf: Dict[str, List[Dict]],
        tf_weights: Dict[str, float],
        extra_features: Dict[str, float] = None,
    ) -> Dict:
        combined = {}
        for tf, kl in klines_by_tf.items():
            features = self.feature_calc.calculate(level, kl)
            w = tf_weights.get(tf, 0)
            for k, v in features.items():
                combined[k] = combined.get(k, 0) + v * w

        combined["multi_tf_confirm"] = self.feature_calc.multi_tf_confirm(
            level, klines_by_tf, tf_weights
        )
        if extra_features:
            for k, v in extra_features.items():
                combined[k] = v

        score = 0.0
        for k, w in self.weights.items():
            score += float(combined.get(k, 0)) * w
        return {"score": score * 100, "features": combined}

    def get_features(self, level: float, klines: List[Dict]) -> Dict:
        return self.feature_calc.calculate(level, klines)

