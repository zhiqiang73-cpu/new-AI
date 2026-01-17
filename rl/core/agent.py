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
from ..leverage_optimizer import LeverageOptimizer
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
        self.base_margin_ratio = 0.08  # 8% of available margin
        # Limit order settings (maker-friendly, with adaptive re-quote)
        self.limit_offset_pct = 0.0003  # 0.03% maker offset
        self.limit_cross_offset_pct = 0.0002  # 0.02% cross offset
        self.limit_requote_seconds = 2
        self.limit_requote_attempts = 6
        self.limit_maker_attempts = 3

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
        self.leverage_optimizer = LeverageOptimizer()

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
        # 固定阈值50，不再动态调整
        effective_threshold = 50
        return {
            "scores": scores,
            "threshold": {"threshold": 50},
            "effective_threshold": effective_threshold,
            "cooldown": self.entry_cooldown,
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

    def _get_available_margin(self) -> float:
        try:
            balances = self.client.get_balance()
            usdt = next((b for b in balances if b.get("asset") == "USDT"), None)
            if usdt:
                return float(usdt.get("availableBalance", usdt.get("balance", 0)))
        except Exception:
            pass
        return 0.0

    def _smart_leverage(self, signal_strength: float, north_star_score: float, stats: Dict) -> int:
        leverage = float(self.base_leverage)
        signal_factor = 0.8 + (max(0.0, min(signal_strength, 100.0)) / 100.0) * 0.6
        ns_factor = 0.8 + (max(0.0, min(north_star_score, 100.0)) / 100.0) * 0.6
        win_rate = float(stats.get("win_rate", 0)) / 100.0
        if win_rate >= 0.6:
            win_factor = 1.1
        elif win_rate <= 0.35:
            win_factor = 0.8
        elif win_rate <= 0.45:
            win_factor = 0.9
        else:
            win_factor = 1.0
        leverage *= signal_factor * ns_factor * win_factor
        return max(3, min(50, int(round(leverage))))

    def _smart_position_size(
        self,
        price: float,
        stop_loss: float,
        signal_strength: float,
        north_star_score: float,
        stats: Dict,
    ) -> Tuple[float, int, float]:
        available_margin = self._get_available_margin()
        if available_margin <= 0:
            return 0.0, self.leverage, 0.0
        leverage = self._smart_leverage(signal_strength, north_star_score, stats)
        signal_factor = 0.6 + (max(0.0, min(signal_strength, 100.0)) / 100.0) * 0.4
        ns_factor = 0.6 + (max(0.0, min(north_star_score, 100.0)) / 100.0) * 0.4
        margin_ratio = self.base_margin_ratio * signal_factor * ns_factor
        margin_ratio = max(0.02, min(0.20, margin_ratio))
        margin_budget = available_margin * margin_ratio
        notional_budget = margin_budget * leverage
        qty_by_margin = notional_budget / price if price > 0 else 0.0
        qty_by_risk = self.position_sizer.calculate_size(
            available_margin, price, stop_loss
        )
        qty = min(qty_by_margin, qty_by_risk)
        return qty, leverage, margin_budget

    def _calc_limit_price(self, current_price: float, side: str, attempt: int = 0) -> float:
        use_cross = attempt >= self.limit_maker_attempts
        if side == "BUY":
            if use_cross:
                return current_price * (1 + self.limit_cross_offset_pct)
            return current_price * (1 - self.limit_offset_pct)
        if use_cross:
            return current_price * (1 - self.limit_cross_offset_pct)
        return current_price * (1 + self.limit_offset_pct)

    def _place_limit_with_requote(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        price = current_price
        for attempt in range(max(1, self.limit_requote_attempts)):
            limit_price = self._calc_limit_price(price, side, attempt=attempt)
            order = self.client.place_order(
                symbol=symbol,
                side=side,
                order_type="LIMIT",
                quantity=quantity,
                price=limit_price,
                time_in_force="GTC",
                reduce_only=reduce_only,
            )
            order_id = order.get("orderId") if isinstance(order, dict) else None
            time.sleep(self.limit_requote_seconds)
            if order_id:
                status = self.client.get_order(symbol, order_id=order_id)
                if isinstance(status, dict):
                    order_status = status.get("status", "")
                    if order_status == "FILLED":
                        return status
                    if order_status == "PARTIALLY_FILLED":
                        # Partially filled - still return it, record what we got
                        return status
                    # Try to cancel unfilled order
                    try:
                        self.client.cancel_order(symbol, order_id=order_id)
                    except Exception:
                        # Cancel failed - check status again, order may have filled
                        time.sleep(0.5)
                        recheck = self.client.get_order(symbol, order_id=order_id)
                        if isinstance(recheck, dict) and recheck.get("status") in ("FILLED", "PARTIALLY_FILLED"):
                            return recheck
            try:
                ticker = self.client.get_ticker_price(symbol)
                price = float(ticker.get("price", price))
            except Exception:
                pass
        return None

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

        # Trend contribution (micro > macro for short-term trading)
        macro_dir = market["macro_trend"]["direction"]
        micro_dir = market.get("micro_trend", {}).get("direction")
        # Micro trend (15m) has higher weight for short-term trading
        if micro_dir == "BULLISH":
            long_score += 25
        if micro_dir == "BEARISH":
            short_score += 25
        # Macro trend (8h/1w) has lower weight
        if macro_dir == "BULLISH":
            long_score += 10
        if macro_dir == "BEARISH":
            short_score += 10
        # If macro and micro diverge, trust micro more (short-term)
        if macro_dir == "BULLISH" and micro_dir == "BEARISH":
            long_score -= 10  # Penalize going against micro
            short_score += 5
        if macro_dir == "BEARISH" and micro_dir == "BULLISH":
            short_score -= 10  # Penalize going against micro
            long_score += 5

        # S/R score contribution (tight range for short-term trading)
        # 核心逻辑：
        # - 做多：靠近支撑位加分，靠近阻力位减分（上涨空间有限）
        # - 做空：靠近阻力位加分，靠近支撑位减分（下跌空间有限）
        
        support_price = 0
        resistance_price = 0
        support_distance_pct = 999
        resistance_distance_pct = 999
        
        if market.get("best_support"):
            support_price = market["best_support"]["price"]
            support_distance_pct = abs(price - support_price) / price * 100
            sr_mult = self.regime_adjustments.get("sr_weight_multiplier", 1.0)
            # 做多：靠近支撑位加分
            if support_distance_pct < 0.1:  # Within 0.1% (~$100)
                long_score += min(35, market["best_support"]["score"] / 2 * sr_mult)
            elif support_distance_pct < 0.3:  # Within 0.3% (~$300)
                long_score += min(20, market["best_support"]["score"] / 3 * sr_mult)
            # 做空：靠近支撑位减分（下跌空间有限）
            if support_distance_pct < 0.1:
                short_score -= 20  # 惩罚在支撑位附近做空

        if market.get("best_resistance"):
            resistance_price = market["best_resistance"]["price"]
            resistance_distance_pct = abs(price - resistance_price) / price * 100
            sr_mult = self.regime_adjustments.get("sr_weight_multiplier", 1.0)
            # 做空：靠近阻力位加分
            if resistance_distance_pct < 0.1:  # Within 0.1% (~$100)
                short_score += min(35, market["best_resistance"]["score"] / 2 * sr_mult)
            elif resistance_distance_pct < 0.3:  # Within 0.3% (~$300)
                short_score += min(20, market["best_resistance"]["score"] / 3 * sr_mult)
            # 做多：靠近阻力位减分（上涨空间有限）
            if resistance_distance_pct < 0.1:
                long_score -= 20  # 惩罚在阻力位附近做多
        
        # 额外检查：如果价格卡在支撑阻力之间的空间太小，两边都减分
        if support_price > 0 and resistance_price > 0:
            sr_range_pct = (resistance_price - support_price) / price * 100
            if sr_range_pct < 0.3:  # 支撑阻力间距 < 0.3%，空间太小
                long_score -= 15
                short_score -= 15

        # RSI contribution (10 points, 15m)
        rsi = analysis.get("rsi", 50)
        if rsi < 35:
            long_score += 10
        if rsi > 65:
            short_score += 10

        # MACD contribution (10 points, 15m)
        if analysis.get("macd_histogram", 0) > 0:
            long_score += 10
        if analysis.get("macd_histogram", 0) < 0:
            short_score += 10

        # 1m momentum boost to avoid long-bias in short-term downtrends
        analysis_1m = market.get("analysis_1m", {})
        macd_1m = analysis_1m.get("macd_histogram", 0)
        rsi_1m = analysis_1m.get("rsi", 50)
        if macd_1m > 0:
            long_score += 5
        if macd_1m < 0:
            short_score += 5
        if rsi_1m < 30:
            long_score += 4
        if rsi_1m > 70:
            short_score += 4

        # Apply regime multiplier and clamp to non-negative
        long_score = int(max(0, long_score) * regime_multiplier)
        short_score = int(max(0, short_score) * regime_multiplier)

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

        # Normal score-based entry with trend-aware tie-breaking
        # MICRO trend takes priority for short-term trading
        long_ok = scores["long"] >= effective_threshold
        short_ok = scores["short"] >= effective_threshold
        if long_ok and short_ok:
            macro_dir = market.get("macro_trend", {}).get("direction")
            micro_dir = market.get("micro_trend", {}).get("direction")
            score_gap = abs(scores["long"] - scores["short"])

            # Priority: 1) Micro trend  2) Score difference  3) Macro trend
            if micro_dir == "BULLISH":
                chosen = "LONG"
            elif micro_dir == "BEARISH":
                chosen = "SHORT"
            elif score_gap >= 5:
                chosen = "LONG" if scores["long"] > scores["short"] else "SHORT"
            elif macro_dir == "BULLISH":
                chosen = "LONG"
            elif macro_dir == "BEARISH":
                chosen = "SHORT"
            else:
                return None  # All signals unclear, don't enter

            return {
                "direction": chosen,
                "reason": "score_trend_tiebreak",
                "scores": scores,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "strength": scores["long"] if chosen == "LONG" else scores["short"],
            }

        if long_ok:
            return {
                "direction": "LONG",
                "reason": "score",
                "scores": scores,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "strength": scores["long"],
            }
        if short_ok:
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
        stats = self.trade_logger.get_stats()
        north_star = self.north_star.evaluate(stats)
        signal_strength = float(signal.get("strength", 50))
        available_margin = self._get_available_margin()
        if available_margin <= 0:
            return {"error": "无法获取可用保证金"}
        base_qty, leverage, margin_budget = self._smart_position_size(
            price, sltp["stop_loss"], signal_strength, float(north_star.get("score", 0)), stats
        )
        if base_qty <= 0:
            return {"error": "计算仓位为0，保证金可能不足"}
        base_qty = max(0.001, round(base_qty, 3))
        base_qty = min(base_qty, 0.5)
        filters = {}
        try:
            filters = self.client.get_symbol_filters("BTCUSDT")
        except Exception:
            filters = {}
        min_qty = float(filters.get("min_qty", 0) or 0)
        min_notional = float(filters.get("min_notional", 0) or 0)
        if min_qty > 0 and base_qty < min_qty:
            return {"error": f"下单数量过小: qty={base_qty}, minQty={min_qty}"}
        if min_notional > 0 and price * base_qty < min_notional:
            notional = price * base_qty
            return {"error": f"名义价值过小: notional={notional:.4f}, minNotional={min_notional}"}
        if leverage != self.leverage:
            try:
                self.client.set_leverage("BTCUSDT", leverage)
            except Exception:
                pass
            self.leverage = leverage

        batches = self.batch_manager.plan_entries(signal.get("strength", 50))
        total_batches = max(1, len(batches))
        created = []
        for idx, batch in enumerate(batches):
            if len(self.positions) >= self.MAX_POSITIONS:
                break
            qty = round(base_qty * batch["ratio"], 3)
            # Ensure batch qty meets minimum requirements
            if qty < min_qty:
                qty = min_qty
            if min_notional > 0 and price * qty < min_notional:
                continue  # Skip this batch if too small
            margin_estimate = qty * price / max(1, leverage)
            if available_margin > 0 and margin_estimate > available_margin * 0.9:
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
                "leverage": leverage,
                "margin_used": round(margin_estimate, 4),
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
                side = "BUY" if signal["direction"] == "LONG" else "SELL"
                order = self._place_limit_with_requote(
                    symbol="BTCUSDT",
                    side=side,
                    quantity=position["quantity"],
                    current_price=price,
                )
                if not order:
                    return {"error": "limit_order_unfilled"}
                executed_qty = float(order.get("executedQty", position["quantity"]))
                if executed_qty > 0:
                    position["quantity"] = round(executed_qty, 3)
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
        self,
        position: Dict,
        current_price: float,
        reason: str,
        confirmations: List[str],
        skip_api: bool = False,
        skip_order: bool = False,
    ) -> Optional[Dict]:
        if skip_order:
            skip_api = True
        if not skip_api:
            side = "SELL" if position["direction"] == "LONG" else "BUY"
            try:
                order = self._place_limit_with_requote(
                    symbol="BTCUSDT",
                    side=side,
                    quantity=position["quantity"],
                    current_price=current_price,
                    reduce_only=True,
                )
                if not order:
                    return None
            except Exception:
                pass  # 继续清理本地持仓，即使API失败

        entry_price = position["entry_price"]
        qty = position["quantity"]
        
        # 计算原始盈亏
        raw_pnl = (
            (current_price - entry_price) * qty
            if position["direction"] == "LONG"
            else (entry_price - current_price) * qty
        )
        
        # 计算手续费（限价单0.02%，开仓+平仓各一次）
        commission_rate = 0.0002  # 0.02% per side
        entry_commission = entry_price * qty * commission_rate
        exit_commission = current_price * qty * commission_rate
        total_commission = entry_commission + exit_commission
        
        # 扣除手续费后的净盈亏
        pnl = raw_pnl - total_commission
        
        # 计算百分比（基于入场价值）
        entry_value = entry_price * qty
        pnl_percent = (pnl / entry_value * 100) if entry_value > 0 else 0.0

        exit_time = datetime.now()
        
        # Calculate hold time
        hold_minutes = 0.0
        try:
            entry_dt = datetime.fromisoformat(position["timestamp_open"])
            hold_minutes = (exit_time - entry_dt).total_seconds() / 60.0
        except Exception:
            pass
        
        # Analyze exit timing quality using exit_learner
        exit_timing_quality = 0.0
        klines_before = []
        klines_after = []
        market = self.last_market or {}
        klines_1m = market.get("klines_1m", [])
        
        if klines_1m and len(klines_1m) >= 10:
            try:
                exit_ts = int(exit_time.timestamp())
                exit_idx = self._find_kline_index(klines_1m, exit_ts)
                if exit_idx is not None:
                    # Get 5 candles before and after
                    klines_before = klines_1m[max(0, exit_idx - 5):exit_idx]
                    klines_after = klines_1m[exit_idx + 1:min(len(klines_1m), exit_idx + 6)]
                    
                    timing_analysis = self.exit_learner.analyze_exit_quality(
                        current_price,
                        position["direction"],
                        klines_before,
                        klines_after,
                    )
                    exit_timing_quality = float(timing_analysis.get("quality", 0.0))
            except Exception:
                pass
        
        # Update exit learner
        try:
            exit_learner_result = self.exit_learner.update(
                pnl_percent,
                exit_timing_quality,
                hold_minutes,
                reason,
            )
            # Update exit_manager with learned parameters from exit_learner
            exit_params = self.exit_learner.get_exit_params()
            if self.exit_manager:
                # Sync learned parameters to exit_manager
                update_dict = {
                    "max_hold_minutes": exit_params.get("max_hold_minutes", 45),
                    "min_profit_pct": exit_params.get("min_profit_after_time", 0.2),
                }
                self.exit_manager.update_params(update_dict)
        except Exception as e:
            exit_learner_result = None
        
        trade = {
            "trade_id": position["trade_id"],
            "direction": position["direction"],
            "entry_price": entry_price,
            "exit_price": current_price,
            "quantity": position["quantity"],
            "leverage": position.get("leverage", self.leverage),
            "pnl": round(pnl, 4),
            "pnl_percent": round(pnl_percent, 4),
            "raw_pnl": round(raw_pnl, 4),  # 扣除手续费前的盈亏
            "commission": round(total_commission, 4),  # 总手续费
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
            # Combine exit_learner timing quality with existing timing feedback
            combined_timing = (exit_timing_quality * 0.6 + timing_reward * 0.4)
            reward = max(-1.0, min(1.0, pnl_reward * 0.7 + combined_timing * 0.3))
            trade["reward"] = reward
            trade["decision_feedback"] = {
                "entry_score": position.get("entry_score"),
                "exit_reason": reason,
                "tf_weights": (self.last_market or {}).get("tf_weights"),
                "pnl_reward": round(pnl_reward, 4),
                "timing": timing_feedback,
                "exit_timing_quality": round(exit_timing_quality, 3),
            }
            learned = self.strategy.update(reward, timing_feedback)
            self.exit_manager.update_params(self.strategy.get_exit_params())
            self.sl_tp.update_params(self.strategy.get_sl_tp_params())
            trade["strategy_params"] = learned
            if exit_learner_result:
                trade["exit_learner"] = exit_learner_result
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

