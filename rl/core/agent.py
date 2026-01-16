import json
import os
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..execution.exit_manager import ExitDecision, ExitManager
from ..execution.sl_tp import PositionSizer, StopLossTakeProfit
from ..learning.dynamic_threshold import DynamicThresholdOptimizer
from ..learning.north_star import NorthStarOptimizer
from ..learning.strategy_params import StrategyParamLearner
from ..learning.exit_learner import ExitTimingLearner
from ..market_analysis.indicators import TechnicalAnalyzer
from ..market_analysis.level_finder import BestLevelFinder, FEATURE_NAMES_CN
from ..market_analysis.levels import LevelDiscovery, LevelScoring
from ..market_analysis.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from ..market_analysis.regime import MarketRegimeDetector, BreakoutDetector
from ..position.batch_position_manager import BatchPositionManager
from ..risk.risk_controller import RiskController
from .knowledge import EvolutionManager, KnowledgeBase, TradeLogger


class ThoughtChain:
    def __init__(self, max_records: int = 50):
        self.records = deque(maxlen=max_records)

    def add(self, data: Dict) -> None:
        self.records.append(data)

    def get_recent(self, n: int = 10) -> List[Dict]:
        return list(self.records)[-n:]


class TradingAgent:
    MAX_POSITIONS = 3

    def __init__(self, api_client, data_dir: str = "rl_data", leverage: int = 10):
        self.client = api_client
        self.data_dir = data_dir
        self.leverage = leverage
        self.base_leverage = leverage

        os.makedirs(data_dir, exist_ok=True)
        self.positions: List[Dict] = []
        self.position_states: Dict[str, Dict] = {}
        self.current_position = None

        self.analyzer = TechnicalAnalyzer()
        self.level_discovery = LevelDiscovery()
        self.level_scoring = LevelScoring(f"{data_dir}/levels.json")
        self.level_finder = BestLevelFinder(f"{data_dir}/level_stats.json")
        self.sl_tp = StopLossTakeProfit(self.level_scoring)
        self.position_sizer = PositionSizer(max_risk_percent=2.0)
        self.exit_manager = ExitManager(f"{data_dir}/exit_params.json")
        self.strategy = StrategyParamLearner(f"{data_dir}/strategy_params.json")
        self.exit_manager.update_params(self.strategy.get_exit_params())
        self.sl_tp.update_params(self.strategy.get_sl_tp_params())
        self.trade_logger = TradeLogger(f"{data_dir}/trades.db")
        self.knowledge = KnowledgeBase(f"{data_dir}/knowledge.json")
        self.evolution = EvolutionManager(self.trade_logger, self.knowledge)
        self.threshold = DynamicThresholdOptimizer()
        self.north_star = NorthStarOptimizer()
        self.multi_tf = MultiTimeframeAnalyzer()
        self.batch_manager = BatchPositionManager()
        self.risk = RiskController()
        self.thoughts = ThoughtChain()

        # New modules: Market regime & Breakout detection
        self.regime_detector = MarketRegimeDetector()
        self.breakout_detector = BreakoutDetector()
        self.exit_learner = ExitTimingLearner(data_dir)
        self.current_regime = None
        self.regime_adjustments = {}

        self.best_support = None
        self.best_resistance = None
        self.last_market = None
        self.last_level_scores: List[Dict] = []
        self.last_tf_weights: Dict[str, float] = {}
        self._last_entry_time = 0
        self.entry_cooldown = 15
        self.last_signal_state = {}
        self.last_entry_plan = []
        self.last_entry_signal = None

    def _get_entry_context(self, market: Dict) -> Dict:
        scores = self._score_entry(market)
        stats = self.trade_logger.get_stats()
        threshold = self.threshold.compute(
            stats.get("total_trades", 0), stats.get("win_rate", 0) / 100
        )
        north_star = self.north_star.evaluate(stats)
        aggressive_delta = north_star.get("aggressive_delta", 0)
        effective_threshold = threshold["threshold"] - aggressive_delta
        if north_star.get("mode") == "explore":
            effective_threshold -= 5
        effective_threshold += self.strategy.get_entry_bias()
        effective_threshold = max(15, effective_threshold)
        cooldown = self.entry_cooldown
        if north_star.get("mode") == "explore":
            cooldown = max(5, self.entry_cooldown - 5)
        return {
            "scores": scores,
            "threshold": threshold,
            "north_star": north_star,
            "effective_threshold": effective_threshold,
            "cooldown": cooldown,
            "trade_count": stats.get("total_trades", 0),
        }

        self._load_positions()

    def _save_positions(self) -> None:
        path = os.path.join(self.data_dir, "active_positions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"positions": self.positions}, f, indent=2)

    def _load_positions(self) -> None:
        path = os.path.join(self.data_dir, "active_positions.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", [])
            except Exception:
                self.positions = []

    def _score_level_multi_tf(
        self,
        level: float,
        kl_1m,
        kl_15m,
        kl_8h,
        kl_1w,
        tf_weights: Dict[str, float],
        extra_features: Dict[str, float] = None,
    ) -> Dict:
        result = self.level_scoring.score_multi_tf(
            level,
            {"1m": kl_1m, "15m": kl_15m, "8h": kl_8h, "1w": kl_1w},
            tf_weights,
            extra_features=extra_features,
        )
        return result

    def _get_tf_weights(self, current_price: float, atr_15m: float) -> Dict[str, float]:
        # 以15m为主导，降低1m噪音
        base = {"1m": 0.10, "15m": 0.55, "8h": 0.25, "1w": 0.10}
        high_noise = {"1m": 0.05, "15m": 0.60, "8h": 0.25, "1w": 0.10}
        if current_price <= 0:
            return base
        vol = atr_15m / current_price
        # 波动率越高，1m权重越低
        t = min(1.0, max(0.0, (vol - 0.003) / 0.006))
        weights = {}
        for k in base:
            weights[k] = base[k] * (1 - t) + high_noise[k] * t
        total = sum(weights.values()) or 1.0
        for k in weights:
            weights[k] = weights[k] / total
        return weights

    def _recent_volume_ratio(self, kl_1m: List[Dict]) -> float:
        if not kl_1m or len(kl_1m) < 10:
            return 0.0
        recent = kl_1m[-5:]
        base = kl_1m[-50:] if len(kl_1m) >= 50 else kl_1m
        recent_avg = sum(k.get("volume", 0) for k in recent) / max(1, len(recent))
        base_avg = sum(k.get("volume", 0) for k in base) / max(1, len(base))
        if base_avg <= 0:
            return 0.0
        ratio = recent_avg / base_avg
        return min(ratio, 2.0) / 2.0

    def _find_kline_index(self, klines: List[Dict], ts: int) -> Optional[int]:
        if not klines:
            return None
        best_idx = None
        best_diff = None
        for i, k in enumerate(klines):
            diff = abs(int(k.get("time", 0)) - ts)
            if best_diff is None or diff < best_diff:
                best_idx = i
                best_diff = diff
        return best_idx

    def _timing_feedback(
        self, position: Dict, exit_price: float, exit_time: datetime
    ) -> Dict:
        market = self.last_market or {}
        klines = market.get("klines_1m") or []
        if not klines:
            return {}

        try:
            entry_ts = int(datetime.fromisoformat(position["timestamp_open"]).timestamp())
        except Exception:
            return {}

        exit_ts = int(exit_time.timestamp())
        entry_idx = self._find_kline_index(klines, entry_ts)
        exit_idx = self._find_kline_index(klines, exit_ts)
        if entry_idx is None or exit_idx is None:
            return {}

        window = 10

        def _clamp01(value: float) -> float:
            return max(0.0, min(1.0, value))

        def _quality(idx: int, price: float, is_entry: bool) -> Optional[float]:
            start = max(0, idx - window)
            end = min(len(klines) - 1, idx + window)
            highs = [k["high"] for k in klines[start : end + 1]]
            lows = [k["low"] for k in klines[start : end + 1]]
            if not highs or not lows:
                return None
            max_high = max(highs)
            min_low = min(lows)
            span = max_high - min_low
            if span <= 0:
                return None
            direction = position.get("direction")
            if is_entry:
                if direction == "LONG":
                    score = 1 - (price - min_low) / span
                else:
                    score = 1 - (max_high - price) / span
            else:
                if direction == "LONG":
                    score = 1 - (max_high - price) / span
                else:
                    score = 1 - (price - min_low) / span
            return _clamp01(score)

        entry_quality = _quality(entry_idx, position.get("entry_price", 0), True)
        exit_quality = _quality(exit_idx, exit_price, False)
        if entry_quality is None or exit_quality is None:
            return {}

        timing_score = (entry_quality + exit_quality) / 2.0
        timing_reward = (timing_score - 0.5) * 2.0

        start_idx = min(entry_idx, exit_idx)
        end_idx = max(entry_idx, exit_idx)
        segment = klines[start_idx : end_idx + 1] if end_idx >= start_idx else []
        mfe = 0.0
        mae = 0.0
        if segment:
            entry_price = float(position.get("entry_price", 0))
            seg_high = max(k["high"] for k in segment)
            seg_low = min(k["low"] for k in segment)
            if position.get("direction") == "LONG":
                mfe = (seg_high - entry_price) / max(1e-9, entry_price) * 100
                mae = (entry_price - seg_low) / max(1e-9, entry_price) * 100
            else:
                mfe = (entry_price - seg_low) / max(1e-9, entry_price) * 100
                mae = (seg_high - entry_price) / max(1e-9, entry_price) * 100

        return {
            "entry_quality": round(entry_quality, 4),
            "exit_quality": round(exit_quality, 4),
            "timing_score": round(timing_score, 4),
            "timing_reward": round(timing_reward, 4),
            "mfe_pct": round(mfe, 4),
            "mae_pct": round(mae, 4),
            "window": window,
        }

    def _orderbook_features(self, level: float, orderbook: Optional[Dict]) -> Dict[str, float]:
        if not orderbook:
            return {
                "orderbook_bid_wall": 0.0,
                "orderbook_ask_wall": 0.0,
                "orderbook_big_ratio": 0.0,
            }
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids and not asks:
            return {
                "orderbook_bid_wall": 0.0,
                "orderbook_ask_wall": 0.0,
                "orderbook_big_ratio": 0.0,
            }
        tol = self.level_scoring.feature_calc.tolerance_pct
        total_bid = sum(q for _, q in bids) or 0.0
        total_ask = sum(q for _, q in asks) or 0.0
        near_bids = [(p, q) for p, q in bids if level > 0 and abs(p - level) / level <= tol]
        near_asks = [(p, q) for p, q in asks if level > 0 and abs(p - level) / level <= tol]
        near_bid_qty = sum(q for _, q in near_bids) or 0.0
        near_ask_qty = sum(q for _, q in near_asks) or 0.0
        bid_wall = near_bid_qty / total_bid if total_bid > 0 else 0.0
        ask_wall = near_ask_qty / total_ask if total_ask > 0 else 0.0

        avg_bid = total_bid / len(bids) if bids else 0.0
        avg_ask = total_ask / len(asks) if asks else 0.0
        big_bid = sum(q for _, q in near_bids if avg_bid > 0 and q >= avg_bid * 3)
        big_ask = sum(q for _, q in near_asks if avg_ask > 0 and q >= avg_ask * 3)
        near_total = near_bid_qty + near_ask_qty
        big_ratio = (big_bid + big_ask) / near_total if near_total > 0 else 0.0

        return {
            "orderbook_bid_wall": min(bid_wall, 1.0),
            "orderbook_ask_wall": min(ask_wall, 1.0),
            "orderbook_big_ratio": min(big_ratio, 1.0),
        }

    def _feature_breakdown(self, features: Dict) -> List[Dict]:
        weights = self.level_finder.stats.get("weights", {})
        total = 0.0
        rows = []
        for key, weight in weights.items():
            value = float(features.get(key, 0))
            contrib = value * weight
            total += contrib
            rows.append(
                {
                    "key": key,
                    "name": FEATURE_NAMES_CN.get(key, key),
                    "value": value,
                    "weight": weight,
                    "contribution": contrib,
                }
            )
        for row in rows:
            if total > 0:
                row["contribution_pct"] = row["contribution"] / total * 100
            else:
                row["contribution_pct"] = 0.0
        rows.sort(key=lambda x: x["contribution_pct"], reverse=True)
        return rows

    def analyze_market(
        self, kl_1m, kl_15m, kl_8h, kl_1w, orderbook: Optional[Dict] = None
    ) -> Optional[Dict]:
        if not kl_1m:
            return None
        analysis_1m = self.analyzer.analyze(kl_1m)
        analysis_15m = self.analyzer.analyze(kl_15m)
        analysis_8h = self.analyzer.analyze(kl_8h)
        analysis_1w = self.analyzer.analyze(kl_1w)

        tf = self.multi_tf.analyze(analysis_1m, analysis_15m, analysis_8h, analysis_1w)
        current_price = kl_1m[-1]["close"]
        atr_15m = analysis_15m.get("atr", 0)
        tf_weights = self._get_tf_weights(current_price, atr_15m)
        self.last_tf_weights = tf_weights
        recent_volume_ratio = self._recent_volume_ratio(kl_1m)

        levels_1m = self.level_discovery.discover_all(
            kl_1m, current_price=current_price, atr=atr_15m
        )
        levels_15m = self.level_discovery.discover_all(
            kl_15m, current_price=current_price, atr=atr_15m
        )
        levels_8h = self.level_discovery.discover_all(
            kl_8h, current_price=current_price, atr=atr_15m
        )
        levels_1w = self.level_discovery.discover_all(
            kl_1w, current_price=current_price, atr=atr_15m
        )

        candidates = set()
        for group in (levels_1m, levels_15m, levels_8h, levels_1w):
            candidates.update(group.get("support", []))
            candidates.update(group.get("resistance", []))

        best_support = None
        best_resistance = None

        level_scores = []
        for level in candidates:
            extra_features = self._orderbook_features(level, orderbook)
            extra_features["recent_volume_ratio"] = recent_volume_ratio
            result = self._score_level_multi_tf(
                level,
                kl_1m,
                kl_15m,
                kl_8h,
                kl_1w,
                tf_weights,
                extra_features=extra_features,
            )
            score = result["score"]
            features = result["features"]
            breakdown = self._feature_breakdown(features)
            level_scores.append(
                {
                    "price": level,
                    "score": score,
                    "features": features,
                    "breakdown": breakdown,
                }
            )
            if level <= current_price:
                if not best_support or score > best_support["score"]:
                    best_support = {
                        "price": level,
                        "score": score,
                        "features": features,
                        "feature_breakdown": breakdown,
                    }
            if level >= current_price:
                if not best_resistance or score > best_resistance["score"]:
                    best_resistance = {
                        "price": level,
                        "score": score,
                        "features": features,
                        "feature_breakdown": breakdown,
                    }

        candidates_count = {
            "support": len(levels_1m.get("support", [])) + len(levels_15m.get("support", [])),
            "resistance": len(levels_1m.get("resistance", [])) + len(levels_15m.get("resistance", [])),
        }

        self.best_support = best_support
        self.best_resistance = best_resistance
        self.last_level_scores = sorted(
            level_scores, key=lambda x: x["score"], reverse=True
        )[:12]

        # Detect market regime (TRENDING / RANGING / VOLATILE)
        ema_short = analysis_15m.get("ema_short", current_price)
        ema_long = analysis_15m.get("ema_long", current_price)
        regime_info = self.regime_detector.detect(
            kl_15m, atr_15m, ema_short, ema_long
        )
        self.current_regime = regime_info.get("regime", "NORMAL")
        self.regime_adjustments = self.regime_detector.get_strategy_adjustments(
            self.current_regime
        )

        # Check for breakouts of best S/R levels
        breakout_support = None
        breakout_resistance = None
        if best_support:
            breakout_support = self.breakout_detector.check_breakout(
                best_support["price"], "support", kl_1m
            )
        if best_resistance:
            breakout_resistance = self.breakout_detector.check_breakout(
                best_resistance["price"], "resistance", kl_1m
            )

        market = {
            "current_price": kl_1m[-1]["close"],
            "analysis_1m": analysis_1m,
            "analysis_15m": analysis_15m,
            "analysis_8h": analysis_8h,
            "analysis_1w": analysis_1w,
            "macro_trend": tf["macro_trend"],
            "micro_trend": tf["micro_trend"],
            "best_support": best_support,
            "best_resistance": best_resistance,
            "candidates_count": candidates_count,
            "level_scores": self.last_level_scores,
            "tf_weights": tf_weights,
            "klines_1m": kl_1m[-240:] if kl_1m else [],
            # New: regime and breakout info
            "regime": regime_info,
            "breakout_support": breakout_support,
            "breakout_resistance": breakout_resistance,
        }
        self.last_market = market
        return market

    def _score_entry(self, market: Dict) -> Dict:
        analysis = market["analysis_15m"]
        price = market["current_price"]
        long_score = 0
        short_score = 0
        breakout_signal = None

        # Check for breakout signals (highest priority)
        br_support = market.get("breakout_support", {})
        br_resistance = market.get("breakout_resistance", {})

        if br_support and br_support.get("is_confirmed"):
            # Support broken = bearish breakout = SHORT signal
            short_score += 30 + br_support.get("strength", 0) * 20
            breakout_signal = {
                "type": "SUPPORT_BREAK",
                "direction": "SHORT",
                "strength": br_support.get("strength", 0),
                "action": "CLOSE_LONG_AND_SHORT",
            }

        if br_resistance and br_resistance.get("is_confirmed"):
            # Resistance broken = bullish breakout = LONG signal
            long_score += 30 + br_resistance.get("strength", 0) * 20
            breakout_signal = {
                "type": "RESISTANCE_BREAK",
                "direction": "LONG",
                "strength": br_resistance.get("strength", 0),
                "action": "CLOSE_SHORT_AND_LONG",
            }

        # Regime-based adjustments
        regime = market.get("regime", {}).get("regime", "NORMAL")
        regime_multiplier = 1.0
        if regime == "TRENDING":
            regime_multiplier = 1.2  # Boost signals in trends
        elif regime == "RANGING":
            regime_multiplier = 0.8  # Reduce in range
        elif regime == "VOLATILE":
            regime_multiplier = 0.7  # Be cautious

        # Trend contribution (20 points base)
        if market["macro_trend"]["direction"] == "BULLISH":
            long_score += 20
        if market["macro_trend"]["direction"] == "BEARISH":
            short_score += 20

        # S/R score contribution (higher weight: score/2, max 35)
        if market.get("best_support"):
            distance = abs(price - market["best_support"]["price"]) / price * 100
            sr_mult = self.regime_adjustments.get("sr_weight_multiplier", 1.0)
            if distance < 3:
                long_score += min(35, market["best_support"]["score"] / 2 * sr_mult)
            elif distance < 5:
                long_score += min(20, market["best_support"]["score"] / 3 * sr_mult)

        if market.get("best_resistance"):
            distance = abs(price - market["best_resistance"]["price"]) / price * 100
            sr_mult = self.regime_adjustments.get("sr_weight_multiplier", 1.0)
            if distance < 3:
                short_score += min(35, market["best_resistance"]["score"] / 2 * sr_mult)
            elif distance < 5:
                short_score += min(20, market["best_resistance"]["score"] / 3 * sr_mult)

        # RSI contribution (10 points)
        rsi = analysis.get("rsi", 50)
        if rsi < 35:
            long_score += 10
        if rsi > 65:
            short_score += 10

        # MACD contribution (10 points)
        if analysis.get("macd_histogram", 0) > 0:
            long_score += 10
        if analysis.get("macd_histogram", 0) < 0:
            short_score += 10

        # Apply regime multiplier
        long_score = int(long_score * regime_multiplier)
        short_score = int(short_score * regime_multiplier)

        return {
            "long": long_score,
            "short": short_score,
            "rsi": rsi,
            "regime": regime,
            "breakout_signal": breakout_signal,
        }

    def should_enter(self, market: Dict) -> Optional[Dict]:
        if len(self.positions) >= self.MAX_POSITIONS:
            return None

        risk_ok = self.risk.can_trade()
        if not risk_ok.get("allowed"):
            return None
        entry_ctx = self._get_entry_context(market)
        scores = entry_ctx["scores"]
        threshold = entry_ctx["threshold"]
        effective_threshold = entry_ctx["effective_threshold"]
        cooldown = entry_ctx.get("cooldown", self.entry_cooldown)
        if time.time() - self._last_entry_time < cooldown:
            return None
        self.last_signal_state = entry_ctx

        # Check for breakout signal (priority over normal entry)
        breakout_signal = scores.get("breakout_signal")
        if breakout_signal and breakout_signal.get("strength", 0) > 0.3:
            direction = breakout_signal["direction"]
            # Breakout has higher priority
            return {
                "direction": direction,
                "reason": "breakout",
                "breakout_type": breakout_signal["type"],
                "scores": scores,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "strength": 80 + int(breakout_signal["strength"] * 20),  # 80-100
                "flip_action": breakout_signal.get("action"),  # For reversal logic
            }

        # Normal score-based entry
        if scores["long"] >= effective_threshold:
            return {
                "direction": "LONG",
                "reason": "score",
                "scores": scores,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "strength": scores["long"],
            }
        if scores["short"] >= effective_threshold:
            return {
                "direction": "SHORT",
                "reason": "score",
                "scores": scores,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "strength": scores["short"],
            }
        return None

    def should_flip_position(self, market: Dict, current_pos: Dict) -> Optional[Dict]:
        """
        Check if we should close current position and open opposite (breakout reversal)
        Called when breakout signal appears against current position
        """
        scores = self._score_entry(market)
        breakout_signal = scores.get("breakout_signal")

        if not breakout_signal:
            return None

        current_dir = current_pos.get("direction", "LONG")
        signal_dir = breakout_signal.get("direction")

        # Only flip if breakout is against current position
        if current_dir == signal_dir:
            return None  # Same direction, no flip needed

        # Check if breakout is strong enough to justify flip
        if breakout_signal.get("strength", 0) < 0.5:
            return None  # Not strong enough

        return {
            "should_flip": True,
            "close_direction": current_dir,
            "new_direction": signal_dir,
            "breakout_type": breakout_signal["type"],
            "strength": breakout_signal["strength"],
            "reason": f"Breakout reversal: {breakout_signal['type']}",
        }

    def execute_entry(self, market: Dict, signal: Dict) -> Dict:
        price = market["current_price"]
        atr = market["analysis_15m"].get("atr", 0)
        self.sl_tp.update_params(self.strategy.get_sl_tp_params())
        sltp = self.sl_tp.calculate(price, signal["direction"], atr)

        balance = 0.0
        try:
            balances = self.client.get_balance()
            usdt = next((b for b in balances if b.get("asset") == "USDT"), None)
            if usdt:
                balance = float(usdt.get("availableBalance", usdt.get("balance", 0)))
        except Exception:
            balance = 0.0

        base_qty = self.position_sizer.calculate_size(
            balance or 100.0, price, sltp["stop_loss"]
        )
        base_qty = max(0.001, round(base_qty, 3))
        base_qty = min(base_qty, 0.5)

        batches = self.batch_manager.plan_entries(signal.get("strength", 50))
        total_batches = max(1, len(batches))
        created = []
        for idx, batch in enumerate(batches):
            if len(self.positions) >= self.MAX_POSITIONS:
                break
            qty = round(base_qty * batch["ratio"], 3)
            notional = qty * price / max(1, self.leverage)
            if balance > 0 and notional > balance * 0.95:
                return {"error": "保证金不足，已跳过入场"}
            if total_batches > 1:
                tp_scale = 0.7 + 0.3 * (idx / (total_batches - 1))
            else:
                tp_scale = 1.0
            if signal["direction"] == "LONG":
                take_profit = price + (sltp["take_profit"] - price) * tp_scale
            else:
                take_profit = price - (price - sltp["take_profit"]) * tp_scale
            trade_id = str(uuid.uuid4())
            position = {
                "trade_id": trade_id,
                "direction": signal["direction"],
                "entry_price": price,
                "quantity": qty,
                "stop_loss": sltp["stop_loss"],
                "take_profit": take_profit,
                "leverage": self.leverage,
                "timestamp_open": datetime.now().isoformat(),
                "entry_score": signal.get("strength", 0),
                "entry_reason": signal.get("reason", ""),
                "batch_index": idx + 1,
                "batch_ratio": batch.get("ratio"),
                "level_features": (
                    market.get("best_support", {}).get("features")
                    if signal["direction"] == "LONG"
                    else market.get("best_resistance", {}).get("features")
                ),
            }
            try:
                self.client.place_order(
                    symbol="BTCUSDT",
                    side="BUY" if signal["direction"] == "LONG" else "SELL",
                    order_type="MARKET",
                    quantity=position["quantity"],
                )
            except Exception as exc:
                return {"error": str(exc)}

            self.positions.append(position)
            created.append(position)

        self._save_positions()
        self._last_entry_time = time.time()
        self.last_entry_plan = batches
        self.last_entry_signal = signal
        return created[0] if created else {"error": "no_position"}

    def check_exit_all(self, current_price: float, market: Dict) -> List[Tuple[Dict, ExitDecision]]:
        exits = []
        scores = self.get_current_scores(market)
        market_with_scores = dict(market)
        market_with_scores["entry_scores"] = {"long": scores["long"], "short": scores["short"]}
        market_with_scores["entry_threshold"] = scores.get("threshold", {})
        for pos in list(self.positions):
            state = self.position_states.get(pos.get("trade_id")) or {}
            decision = self.exit_manager.evaluate(pos, market_with_scores, current_price, state)
            if state:
                self.position_states[pos.get("trade_id")] = state
            if decision:
                exits.append((pos, decision))
        return exits

    def execute_exit_position(
        self, position: Dict, current_price: float, reason: str, confirmations: List[str], skip_api: bool = False
    ) -> Optional[Dict]:
        if not skip_api:
            side = "SELL" if position["direction"] == "LONG" else "BUY"
            try:
                self.client.place_order(
                    symbol="BTCUSDT",
                    side=side,
                    order_type="MARKET",
                    quantity=position["quantity"],
                    reduce_only=True,
                )
            except Exception:
                pass  # 继续清理本地持仓，即使API失败

        entry_price = position["entry_price"]
        pnl = (
            (current_price - entry_price) * position["quantity"]
            if position["direction"] == "LONG"
            else (entry_price - current_price) * position["quantity"]
        )
        pnl_percent = (
            (current_price - entry_price) / entry_price * 100
            if position["direction"] == "LONG"
            else (entry_price - current_price) / entry_price * 100
        )

        exit_time = datetime.now()
        trade = {
            "trade_id": position["trade_id"],
            "direction": position["direction"],
            "entry_price": entry_price,
            "exit_price": current_price,
            "quantity": position["quantity"],
            "leverage": position.get("leverage", self.leverage),
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "exit_reason": reason,
            "timestamp_open": position["timestamp_open"],
            "timestamp_close": exit_time.isoformat(),
            "stop_loss": position.get("stop_loss"),
            "take_profit": position.get("take_profit"),
        }
        self.trade_logger.log_trade(trade)
        self.risk.update_trade_result(pnl_percent)
        self.level_finder.update_stats(pnl > 0)
        features = position.get("level_features")
        if features:
            pnl_reward = max(-1.0, min(1.0, pnl_percent / 2.0))
            timing_feedback = self._timing_feedback(position, current_price, exit_time)
            timing_reward = float(timing_feedback.get("timing_reward", 0.0)) if timing_feedback else 0.0
            reward = max(-1.0, min(1.0, pnl_reward * 0.7 + timing_reward * 0.3))
            trade["reward"] = reward
            trade["decision_feedback"] = {
                "entry_score": position.get("entry_score"),
                "exit_reason": reason,
                "tf_weights": (self.last_market or {}).get("tf_weights"),
                "pnl_reward": round(pnl_reward, 4),
                "timing": timing_feedback,
            }
            learned = self.strategy.update(reward, timing_feedback)
            self.exit_manager.update_params(self.strategy.get_exit_params())
            self.sl_tp.update_params(self.strategy.get_sl_tp_params())
            trade["strategy_params"] = learned
            update = self.level_finder.update_weights(features, reward)
            if update:
                trade["weight_update"] = update
        self.positions = [p for p in self.positions if p["trade_id"] != position["trade_id"]]
        self._save_positions()
        return trade

    def get_current_scores(self, market: Dict) -> Dict:
        entry_ctx = self._get_entry_context(market)
        scores = entry_ctx["scores"]
        threshold = entry_ctx["threshold"]
        return {
            "long": scores["long"],
            "short": scores["short"],
            "min_score": entry_ctx["effective_threshold"],
            "threshold": threshold,
            "phase": "dynamic",
            "trade_count": entry_ctx.get("trade_count", 0),
            "north_star": entry_ctx.get("north_star", {}),
            "cooldown": entry_ctx.get("cooldown", self.entry_cooldown),
        }

    def get_ai_logic(self) -> Dict:
        if not self.last_market:
            return {}
        scores = self.get_current_scores(self.last_market)
        analysis = self.last_market["analysis_15m"]
        because = [
            f"大趋势: {self.last_market['macro_trend']['direction']}",
            f"小趋势: {self.last_market['micro_trend']['direction']}",
            f"RSI: {analysis.get('rsi', 50):.1f}",
            f"MACD柱: {analysis.get('macd_histogram', 0):.4f}",
        ]
        if self.best_support:
            because.append(
                f"支撑位 {self.best_support['price']:.0f} 评分 {self.best_support['score']:.0f}"
            )
        if self.best_resistance:
            because.append(
                f"阻力位 {self.best_resistance['price']:.0f} 评分 {self.best_resistance['score']:.0f}"
            )

        therefore = [
            f"多头分数: {scores['long']:.0f}",
            f"空头分数: {scores['short']:.0f}",
            f"阈值: {scores['min_score']:.0f}",
        ]

        conclusion = "观望"
        if scores["long"] >= scores["min_score"]:
            conclusion = "多头信号"
        elif scores["short"] >= scores["min_score"]:
            conclusion = "空头信号"

        positions = []
        current_price = self.last_market.get("current_price", 0)
        opportunity_delta = self.exit_manager.params.get("opportunity_delta", 0)
        min_profit = self.exit_manager.params.get("min_profit_pct", 0)
        max_hold = self.exit_manager.params.get("max_hold_minutes", 0)
        for pos in self.positions:
            hold_minutes = None
            try:
                opened = datetime.fromisoformat(pos.get("timestamp_open"))
                hold_minutes = (datetime.now() - opened).total_seconds() / 60
            except Exception:
                hold_minutes = None
            pnl_percent = 0.0
            entry_price = pos.get("entry_price", 0)
            if entry_price:
                if pos.get("direction") == "LONG":
                    pnl_percent = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_percent = (entry_price - current_price) / entry_price * 100
            if pos.get("direction") == "LONG":
                opp_score = scores.get("short", 0)
                opp_label = "空头"
            else:
                opp_score = scores.get("long", 0)
                opp_label = "多头"
            opp_need = scores.get("min_score", 0) + opportunity_delta
            decision = self.exit_manager.evaluate(
                pos,
                {
                    **self.last_market,
                    "entry_scores": {"long": scores["long"], "short": scores["short"]},
                    "entry_threshold": scores.get("threshold", {}),
                },
                current_price,
            )
            positions.append(
                {
                    "trade_id": pos.get("trade_id"),
                    "direction": pos.get("direction"),
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "stop_loss": pos.get("stop_loss"),
                    "take_profit": pos.get("take_profit"),
                    "entry_score": pos.get("entry_score", 0),
                    "hold_minutes": hold_minutes,
                    "pnl_percent": pnl_percent,
                    "exit_hint": decision.reason if decision else "",
                    "exit_notes": decision.confirmations if decision else [],
                    "opportunity": {
                        "label": opp_label,
                        "score": opp_score,
                        "need": opp_need,
                        "gap": opp_score - opp_need,
                    },
                    "time_cost": {
                        "hold_minutes": hold_minutes if hold_minutes else 0,
                        "max_hold": max_hold,
                        "current_profit": pnl_percent,
                        "min_profit": min_profit,
                    },
                }
            )

        entry_info = {
            "long_score": scores["long"],
            "short_score": scores["short"],
            "threshold": scores["min_score"],
            "phase": scores.get("phase", ""),
        }
        if scores.get("threshold"):
            entry_info["base_threshold"] = scores["threshold"].get("threshold")
        entry_info["cooldown"] = scores.get("cooldown", self.entry_cooldown)
        if self.last_entry_signal:
            entry_info["last_entry"] = {
                "direction": self.last_entry_signal.get("direction"),
                "strength": self.last_entry_signal.get("strength"),
                "reason": self.last_entry_signal.get("reason"),
            }

        batch_info = {"entry_batches": self.last_entry_plan} if self.last_entry_plan else {}

        # 策略参数（机会成本/时间成本阈值）
        strategy_params = {
            "opportunity_delta": opportunity_delta,
            "max_hold_minutes": max_hold,
            "min_profit_pct": min_profit,
        }

        # 市场状态识别
        regime_info = self.last_market.get("regime", {})
        regime_display = {
            "regime": regime_info.get("regime", "NORMAL"),
            "confidence": regime_info.get("confidence", 0),
            "atr_ratio": regime_info.get("atr_ratio", 1.0),
            "trend_strength": regime_info.get("trend_strength", 0),
            "adjustments": self.regime_adjustments,
        }

        # 突破信号
        breakout_info = None
        br_support = self.last_market.get("breakout_support", {})
        br_resistance = self.last_market.get("breakout_resistance", {})
        if br_support and br_support.get("is_breakout"):
            breakout_info = {
                "type": "SUPPORT_BREAK",
                "confirmed": br_support.get("is_confirmed", False),
                "direction": "DOWN",
                "strength": br_support.get("strength", 0),
                "action": "平多+做空" if br_support.get("is_confirmed") else "观察中",
            }
        elif br_resistance and br_resistance.get("is_breakout"):
            breakout_info = {
                "type": "RESISTANCE_BREAK",
                "confirmed": br_resistance.get("is_confirmed", False),
                "direction": "UP",
                "strength": br_resistance.get("strength", 0),
                "action": "平空+做多" if br_resistance.get("is_confirmed") else "观察中",
            }

        # 离场学习参数
        exit_learner_info = self.exit_learner.get_exit_params()

        return {
            "because": because,
            "therefore": therefore,
            "conclusion": conclusion,
            "entry": entry_info,
            "positions": positions,
            "batch": batch_info,
            "north_star": scores.get("north_star", {}),
            "tf_weights": (self.last_market or {}).get("tf_weights"),
            "strategy_params": strategy_params,
            "regime": regime_display,
            "breakout": breakout_info,
            "exit_learner": exit_learner_info,
        }

