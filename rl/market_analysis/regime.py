"""
Market Regime Detection - Trend vs Ranging
"""
from typing import Dict, List, Optional


class MarketRegimeDetector:
    """
    Detects market regime: TRENDING, RANGING, or VOLATILE
    Uses ATR ratio, ADX-like logic, and price action
    """

    def __init__(self):
        self.history = []  # Store recent regime for smoothing
        self.max_history = 10

    def detect(
        self,
        klines: List[Dict],
        atr: float,
        ema_short: float,
        ema_long: float,
    ) -> Dict:
        """
        Detect current market regime
        Returns: {regime, confidence, atr_ratio, trend_strength}
        """
        if not klines or len(klines) < 20:
            return {"regime": "UNKNOWN", "confidence": 0.0}

        current_price = klines[-1]["close"]

        # 1. ATR ratio (current ATR vs average ATR)
        atr_ratio = self._calculate_atr_ratio(klines, atr)

        # 2. Trend strength (EMA separation)
        trend_strength = abs(ema_short - ema_long) / current_price * 100

        # 3. Price range ratio (recent high-low vs price)
        range_ratio = self._calculate_range_ratio(klines[-20:], current_price)

        # 4. Direction consistency (how many candles in same direction)
        direction_score = self._calculate_direction_score(klines[-10:])

        # Decision logic
        regime = "NORMAL"
        confidence = 0.5

        if atr_ratio > 1.5:
            # High volatility
            if direction_score > 0.6:
                regime = "TRENDING_VOLATILE"
                confidence = min(0.9, 0.5 + direction_score * 0.4)
            else:
                regime = "VOLATILE"
                confidence = min(0.8, 0.4 + atr_ratio * 0.2)
        elif trend_strength > 0.5 and direction_score > 0.5:
            # Strong trend
            regime = "TRENDING"
            confidence = min(0.9, 0.5 + trend_strength * 0.3 + direction_score * 0.2)
        elif range_ratio < 0.015 and atr_ratio < 0.8:
            # Low volatility, tight range
            regime = "RANGING"
            confidence = min(0.85, 0.5 + (1 - atr_ratio) * 0.3)
        else:
            regime = "NORMAL"
            confidence = 0.5

        # Smooth with history
        self.history.append(regime)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        smoothed_regime = self._get_dominant_regime()

        return {
            "regime": smoothed_regime,
            "raw_regime": regime,
            "confidence": round(confidence, 2),
            "atr_ratio": round(atr_ratio, 3),
            "trend_strength": round(trend_strength, 3),
            "range_ratio": round(range_ratio, 4),
            "direction_score": round(direction_score, 2),
        }

    def _calculate_atr_ratio(self, klines: List[Dict], current_atr: float) -> float:
        """Calculate ATR ratio vs historical average"""
        if len(klines) < 50 or current_atr <= 0:
            return 1.0

        # Calculate historical ATRs
        atrs = []
        for i in range(-50, -14):
            high = klines[i]["high"]
            low = klines[i]["low"]
            prev_close = klines[i - 1]["close"]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            atrs.append(tr)

        avg_atr = sum(atrs) / len(atrs) if atrs else current_atr
        return current_atr / avg_atr if avg_atr > 0 else 1.0

    def _calculate_range_ratio(self, klines: List[Dict], current_price: float) -> float:
        """Calculate price range as percentage of current price"""
        if not klines:
            return 0.0
        high = max(k["high"] for k in klines)
        low = min(k["low"] for k in klines)
        return (high - low) / current_price if current_price > 0 else 0.0

    def _calculate_direction_score(self, klines: List[Dict]) -> float:
        """
        Calculate direction consistency score
        1.0 = all candles same direction, 0.0 = perfectly mixed
        """
        if not klines:
            return 0.5
        ups = sum(1 for k in klines if k["close"] > k["open"])
        return abs(ups / len(klines) - 0.5) * 2  # Scale to 0-1

    def _get_dominant_regime(self) -> str:
        """Get most common regime from history"""
        if not self.history:
            return "NORMAL"
        from collections import Counter
        return Counter(self.history).most_common(1)[0][0]

    def get_strategy_adjustments(self, regime: str) -> Dict:
        """
        Get strategy parameter adjustments based on regime
        """
        adjustments = {
            "TRENDING": {
                "entry_threshold_delta": -5,  # Lower threshold, more entries
                "sr_weight_multiplier": 0.7,  # Reduce S/R importance
                "tp_multiplier": 1.5,  # Wider take profit
                "sl_multiplier": 1.2,  # Slightly wider stop loss
                "prefer_breakout": True,
            },
            "TRENDING_VOLATILE": {
                "entry_threshold_delta": -3,
                "sr_weight_multiplier": 0.6,
                "tp_multiplier": 2.0,
                "sl_multiplier": 1.5,
                "prefer_breakout": True,
            },
            "RANGING": {
                "entry_threshold_delta": +10,  # Higher threshold, fewer entries
                "sr_weight_multiplier": 1.5,  # Increase S/R importance
                "tp_multiplier": 0.7,  # Tighter take profit
                "sl_multiplier": 0.8,  # Tighter stop loss
                "prefer_breakout": False,
            },
            "VOLATILE": {
                "entry_threshold_delta": +15,  # Much higher threshold
                "sr_weight_multiplier": 0.8,
                "tp_multiplier": 1.0,
                "sl_multiplier": 1.5,
                "prefer_breakout": False,
            },
            "NORMAL": {
                "entry_threshold_delta": 0,
                "sr_weight_multiplier": 1.0,
                "tp_multiplier": 1.0,
                "sl_multiplier": 1.0,
                "prefer_breakout": False,
            },
            "UNKNOWN": {
                "entry_threshold_delta": +5,
                "sr_weight_multiplier": 1.0,
                "tp_multiplier": 1.0,
                "sl_multiplier": 1.0,
                "prefer_breakout": False,
            },
        }
        return adjustments.get(regime, adjustments["NORMAL"])


class BreakoutDetector:
    """
    Detects and confirms breakouts of support/resistance levels
    """

    def __init__(self):
        self.pending_breakouts = {}  # level -> {direction, start_time, candles}

    def check_breakout(
        self,
        level: float,
        level_type: str,  # "support" or "resistance"
        klines: List[Dict],
        tolerance_pct: float = 0.003,
    ) -> Dict:
        """
        Check if a breakout is confirmed
        Returns: {is_breakout, is_confirmed, direction, strength, candles_below}
        """
        if not klines or len(klines) < 5:
            return {"is_breakout": False, "is_confirmed": False}

        current_price = klines[-1]["close"]
        tolerance = level * tolerance_pct

        # Check recent candles
        recent = klines[-5:]
        
        if level_type == "support":
            # Bearish breakout: price below support
            below_count = sum(1 for k in recent if k["close"] < level - tolerance)
            body_below = sum(1 for k in recent if k["open"] < level and k["close"] < level)
            
            if below_count >= 2:
                # Calculate strength
                avg_distance = sum(
                    (level - k["close"]) / level * 100
                    for k in recent if k["close"] < level
                ) / max(1, below_count)
                
                return {
                    "is_breakout": True,
                    "is_confirmed": below_count >= 3 or body_below >= 2,
                    "direction": "DOWN",
                    "strength": min(1.0, avg_distance / 0.5),  # Normalize
                    "candles_below": below_count,
                    "action": "CLOSE_LONG_AND_SHORT",  # Close long, open short
                }
        else:
            # Bullish breakout: price above resistance
            above_count = sum(1 for k in recent if k["close"] > level + tolerance)
            body_above = sum(1 for k in recent if k["open"] > level and k["close"] > level)
            
            if above_count >= 2:
                avg_distance = sum(
                    (k["close"] - level) / level * 100
                    for k in recent if k["close"] > level
                ) / max(1, above_count)
                
                return {
                    "is_breakout": True,
                    "is_confirmed": above_count >= 3 or body_above >= 2,
                    "direction": "UP",
                    "strength": min(1.0, avg_distance / 0.5),
                    "candles_above": above_count,
                    "action": "CLOSE_SHORT_AND_LONG",  # Close short, open long
                }

        return {"is_breakout": False, "is_confirmed": False}

    def check_false_breakout(
        self,
        level: float,
        level_type: str,
        klines: List[Dict],
        lookback: int = 10,
    ) -> Dict:
        """
        Check if recent price action shows a false breakout (price returned)
        This is valuable for learning - false breakouts strengthen the level
        """
        if not klines or len(klines) < lookback:
            return {"is_false_breakout": False}

        recent = klines[-lookback:]
        current_price = klines[-1]["close"]
        tolerance = level * 0.003

        if level_type == "support":
            # Check if price went below then came back above
            went_below = any(k["low"] < level - tolerance for k in recent[:-3])
            now_above = current_price > level + tolerance
            
            if went_below and now_above:
                return {
                    "is_false_breakout": True,
                    "level_strengthened": True,
                    "direction": "DOWN_THEN_UP",
                    "reward_signal": +0.5,  # Positive reward for S/R learning
                }
        else:
            # Check if price went above then came back below
            went_above = any(k["high"] > level + tolerance for k in recent[:-3])
            now_below = current_price < level - tolerance
            
            if went_above and now_below:
                return {
                    "is_false_breakout": True,
                    "level_strengthened": True,
                    "direction": "UP_THEN_DOWN",
                    "reward_signal": +0.5,
                }

        return {"is_false_breakout": False}



