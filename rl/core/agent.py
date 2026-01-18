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
from ..learning.decision_learner import DecisionFeatureLearner
from ..learning.exit_decision_learner import ExitDecisionLearner
from ..learning.feature_outcome import FeatureOutcomeTracker
from ..leverage_optimizer import LeverageOptimizer
from ..market_analysis.indicators import TechnicalAnalyzer
from ..market_analysis.level_finder import BestLevelFinder, FEATURE_NAMES_CN
from ..market_analysis.levels import LevelDiscovery, LevelScoring
from ..market_analysis.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from ..market_analysis.regime import MarketRegimeDetector, BreakoutDetector
from ..market_analysis.pattern_detector import PatternDetector
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

    def __init__(self, api_client, data_dir: str = "rl_data", leverage: int = 18):
        self.client = api_client
        self.data_dir = data_dir
        self.leverage = leverage
        self.base_leverage = leverage
        self.base_margin_ratio = 0.12  # 12% of available margin（从8%提高到12%）
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
        self.pattern_detector = PatternDetector()
        self.sl_tp = StopLossTakeProfit(self.level_scoring)
        self.position_sizer = PositionSizer(max_risk_percent=2.0)
        # 先创建占位符，待exit_decision_learner初始化后更新
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
        self.decision_learner = DecisionFeatureLearner(
            os.path.join(data_dir, "decision_weights.json")
        )

        # New modules: Market regime & Breakout detection
        self.regime_detector = MarketRegimeDetector()
        self.breakout_detector = BreakoutDetector()
        self.exit_learner = ExitTimingLearner(data_dir)
        self.exit_decision_learner = ExitDecisionLearner(data_dir)  # 出场决策反向传播学习器
        self.feature_outcome = FeatureOutcomeTracker(os.path.join(data_dir, "feature_outcome.json"))
        # 将exit_decision_learner注入到exit_manager
        self.exit_manager.exit_decision_learner = self.exit_decision_learner
        self.current_regime = None
        self.regime_adjustments = {}

        self.best_support = None
        self.best_resistance = None
        self.last_market = None
        self.last_level_scores: List[Dict] = []
        self.last_tf_weights: Dict[str, float] = {}
        self._last_valid: Dict[str, float] = {}
        self._last_entry_time = 0
        self.entry_cooldown = 15
        self.last_signal_state = {}
        self.last_entry_plan = []
        self.last_entry_signal = None
        self.start_time = time.time()  # Agent startup time
        self.near_sr_threshold_pct = 0.1

    def _get_entry_context(self, market: Dict) -> Dict:
        scores = self._score_entry(market)
        stats = self.trade_logger.get_stats()
        
        # ========== 启用 DynamicThresholdOptimizer ==========
        # 根据交易数量和胜率动态调整阈值
        trade_count = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 50) / 100.0  # 转换为0-1
        threshold_data = self.threshold.compute(trade_count, win_rate)
        base_threshold = self._safe_float(threshold_data.get("threshold"), 50)
        
        # ========== 应用市场状态调整 ==========
        # 从 regime_adjustments 获取阈值调整
        regime_delta = self._safe_float(self.regime_adjustments.get("entry_threshold_delta"), 0)
        effective_threshold = max(20, min(80, base_threshold + regime_delta))
        
        return {
            "scores": scores,
            "threshold": threshold_data,
            "effective_threshold": effective_threshold,
            "cooldown": self.entry_cooldown,
            "trade_count": trade_count,
            "regime_delta": regime_delta,
        }

    def _init_positions(self) -> None:
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
        # 以15m为主导，但增加短线敏感度，降低长周期干扰
        base = {"1m": 0.20, "15m": 0.60, "8h": 0.15, "1w": 0.05}
        high_noise = {"1m": 0.12, "15m": 0.65, "8h": 0.15, "1w": 0.08}
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

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _safe_float_with_last(self, key: str, value, default: float = 0.0) -> float:
        if value is None:
            return float(self._last_valid.get(key, default))
        try:
            v = float(value)
        except Exception:
            v = float(self._last_valid.get(key, default))
        else:
            self._last_valid[key] = v
        return v

    def _smart_leverage(self, signal_strength: float, north_star_score: float, stats: Dict) -> int:
        """
        智能杠杆：15-50x动态范围
        目标：强信号+高胜率 → 高杠杆，弱信号+低胜率 → 低杠杆
        
        杠杆分布:
        - 15x: 弱信号(55分) + 低胜率(<45%)
        - 30x: 中等信号(70分) + 中等胜率(50%)
        - 50x: 强信号(85+) + 高胜率(>60%) + 趋势市
        """
        MIN_LEVERAGE = 15
        MAX_LEVERAGE = 50
        BASE_LEVERAGE = 25  # 基准杠杆
        
        leverage = float(BASE_LEVERAGE)
        
        # ========== 信号强度因子（55-100分 → 0.7-1.5倍）==========
        # 55分 → 0.7倍, 75分 → 1.0倍, 100分 → 1.5倍
        normalized_signal = (max(55.0, min(signal_strength, 100.0)) - 55.0) / 45.0  # 0-1
        signal_factor = 0.7 + normalized_signal * 0.8
        
        # ========== 胜率因子 ==========
        win_rate = float(stats.get("win_rate", 50)) / 100.0
        if win_rate >= 0.60:
            win_factor = 1.4   # 胜率>60%: 激进
        elif win_rate >= 0.55:
            win_factor = 1.2   # 胜率55-60%: 偏激进
        elif win_rate >= 0.50:
            win_factor = 1.0   # 胜率50-55%: 标准
        elif win_rate >= 0.45:
            win_factor = 0.85  # 胜率45-50%: 保守
        else:
            win_factor = 0.7   # 胜率<45%: 很保守
        
        # ========== 市场状态因子 ==========
        regime = self.current_regime or "NORMAL"
        if regime in ("TRENDING", "TRENDING_VOLATILE"):
            regime_factor = 1.2  # 趋势市更激进
        elif regime == "RANGING":
            regime_factor = 0.8  # 震荡市保守
        elif regime == "VOLATILE":
            regime_factor = 0.7  # 高波动保守
        else:
            regime_factor = 1.0
        
        # ========== 综合计算 ==========
        leverage *= signal_factor * win_factor * regime_factor
        
        # 限制在15-50x范围
        final_leverage = max(MIN_LEVERAGE, min(MAX_LEVERAGE, int(round(leverage))))
        
        return final_leverage

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
        
        # 仓位调整：基于信号强度动态调整
        signal_factor = 0.8 + (max(0.0, min(signal_strength, 100.0)) / 100.0) * 0.5
        margin_ratio = self.base_margin_ratio * signal_factor
        
        # 允许更大仓位（上限提高到30%），以达到盈利目标
        margin_ratio = max(0.05, min(0.30, margin_ratio))
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
                        # 【修复】取消剩余挂单，确保成交量确定
                        try:
                            self.client.cancel_order(symbol, order_id=order_id)
                        except Exception:
                            pass
                        # 重新查询最终成交状态
                        time.sleep(0.5)
                        final_status = self.client.get_order(symbol, order_id=order_id)
                        if isinstance(final_status, dict):
                            return final_status
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

        # Merge nearby levels across timeframes to avoid tight S/R clusters
        # 改进: 0.2% → 0.1% 提升合并精度 (~$100 @ $100k)
        def _merge_nearby(levels: List[float], tolerance_pct: float = 0.1) -> List[float]:
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
                    merged.append(sum(current_group) / len(current_group))
                    current_group = [sorted_levels[i]]
            merged.append(sum(current_group) / len(current_group))
            return merged

        candidates = set(_merge_nearby(list(candidates), tolerance_pct=0.1))

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
            base_score = result["score"]
            features = result["features"]
            breakdown = self._feature_breakdown(features)
            
            # ========== 距离当前价格更近的S/R获得加分（优先关注近期）==========
            distance_pct = abs(level - current_price) / current_price * 100
            # 距离<0.5%时，额外+15分；距离<1%时，额外+8分；距离<1.5%时，额外+3分
            proximity_bonus = 0.0
            if distance_pct < 0.5:
                proximity_bonus = 15.0
            elif distance_pct < 1.0:
                proximity_bonus = 8.0
            elif distance_pct < 1.5:
                proximity_bonus = 3.0
            score = base_score + proximity_bonus
            
            level_scores.append(
                {
                    "price": level,
                    "score": score,
                    "base_score": base_score,
                    "proximity_bonus": proximity_bonus,
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

        # Enforce minimum S/R gap across timeframes (0.3%)
        min_gap_pct = 0.3
        if best_support and best_resistance:
            gap_pct = (best_resistance["price"] - best_support["price"]) / current_price * 100
            if gap_pct < min_gap_pct:
                resistance_candidates = [
                    ls for ls in level_scores if ls["price"] >= current_price
                ]
                resistance_candidates = sorted(
                    resistance_candidates, key=lambda x: x["score"], reverse=True
                )
                min_resistance = best_support["price"] * (1 + min_gap_pct / 100)
                far_resistance = next(
                    (r for r in resistance_candidates if r["price"] >= min_resistance),
                    None,
                )
                if far_resistance:
                    best_resistance = {
                        "price": far_resistance["price"],
                        "score": far_resistance["score"],
                        "features": far_resistance["features"],
                        "feature_breakdown": far_resistance["breakdown"],
                    }
                else:
                    support_candidates = [
                        ls for ls in level_scores if ls["price"] <= current_price
                    ]
                    support_candidates = sorted(
                        support_candidates, key=lambda x: x["score"], reverse=True
                    )
                    max_support = best_resistance["price"] * (1 - min_gap_pct / 100)
                    far_support = next(
                        (s for s in support_candidates if s["price"] <= max_support),
                        None,
                    )
                    if far_support:
                        best_support = {
                            "price": far_support["price"],
                            "score": far_support["score"],
                            "features": far_support["features"],
                            "feature_breakdown": far_support["breakdown"],
                        }
                    else:
                        best_support = None
                        best_resistance = None

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

        sr_gap_pct = None
        sr_gap_valid = True
        if best_support and best_resistance:
            sr_gap_pct = (best_resistance["price"] - best_support["price"]) / current_price * 100
            sr_gap_valid = sr_gap_pct >= min_gap_pct

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
            "sr_gap_pct": sr_gap_pct,
            "sr_gap_valid": sr_gap_valid,
            "candidates_count": candidates_count,
            "level_scores": self.last_level_scores,
            "tf_weights": tf_weights,
            "klines_1m": kl_1m[-240:] if kl_1m else [],
            "klines_15m": kl_15m[-240:] if kl_15m else [],
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

        # ========== 趋势评分（压缩权重，避免虚高）==========
        macro_dir = market["macro_trend"]["direction"]
        micro_dir = market.get("micro_trend", {}).get("direction")
        # Micro trend (15m) 是短线核心，权重 20（从25降低）
        if micro_dir == "BULLISH":
            long_score += 20
        if micro_dir == "BEARISH":
            short_score += 20
        # Macro trend (8h/1w) 确认信号，权重 10（保持）
        if macro_dir == "BULLISH":
            long_score += 10
        if macro_dir == "BEARISH":
            short_score += 10
        # 如果宏观微观一致，额外奖励（双重确认）
        if macro_dir == "BULLISH" and micro_dir == "BULLISH":
            long_score += 5  # 双重确认奖励
        elif macro_dir == "BEARISH" and micro_dir == "BEARISH":
            short_score += 5
        # 如果背离，轻微惩罚（从-10改为-5）
        if macro_dir == "BULLISH" and micro_dir == "BEARISH":
            long_score -= 5
        if macro_dir == "BEARISH" and micro_dir == "BULLISH":
            short_score -= 5

        # ========== S/R 边界评分（权重最高，因为是核心优势）==========
        # 核心逻辑：做多靠近支撑，做空靠近阻力
        support_price = 0
        resistance_price = 0
        support_distance_pct = 999
        resistance_distance_pct = 999
        
        if market.get("best_support"):
            support_price = self._safe_float(market["best_support"].get("price"), 0.0)
            support_score = self._safe_float(market["best_support"].get("score"), 0.0)
            support_distance_pct = abs(price - support_price) / price * 100
            sr_mult = self._safe_float(self.regime_adjustments.get("sr_weight_multiplier"), 1.0)
            # 做多：靠近支撑位加分（距离阈值与S/R间距对齐）
            if support_distance_pct < 0.2:  # Within 0.2%（从0.1%放宽）
                long_score += min(40, support_score / 2 * sr_mult)
            elif support_distance_pct < 0.5:  # Within 0.5%（从0.3%放宽）
                long_score += min(25, support_score / 3 * sr_mult)
            # 做空：靠近支撑位减分
            if support_distance_pct < 0.2:
                short_score -= 18

        if market.get("best_resistance"):
            resistance_price = self._safe_float(market["best_resistance"].get("price"), 0.0)
            resistance_score = self._safe_float(market["best_resistance"].get("score"), 0.0)
            resistance_distance_pct = abs(price - resistance_price) / price * 100
            sr_mult = self._safe_float(self.regime_adjustments.get("sr_weight_multiplier"), 1.0)
            # 做空：靠近阻力位加分
            if resistance_distance_pct < 0.2:
                short_score += min(40, resistance_score / 2 * sr_mult)
            elif resistance_distance_pct < 0.5:
                short_score += min(25, resistance_score / 3 * sr_mult)
            # 做多：靠近阻力位减分
            if resistance_distance_pct < 0.2:
                long_score -= 18
        
        # 空间检查：移除（已由Clearway Filter处理）
        # S/R间距由levels.py和agent跨周期过滤保证≥0.3%

        # ========== 技术指标评分（辅助信号，权重压缩）==========
        rsi = self._safe_float_with_last("rsi_15m", analysis.get("rsi"), 50)
        if rsi < 35:
            long_score += 8  # 从10降到8
        if rsi > 65:
            short_score += 8

        # MACD (从10降到8)
        if self._safe_float_with_last("macd_15m", analysis.get("macd_histogram"), 0) > 0:
            long_score += 8
        if self._safe_float_with_last("macd_15m", analysis.get("macd_histogram"), 0) < 0:
            short_score += 8

        # 1m 动量（保持，因为权重已经很小）
        analysis_1m = market.get("analysis_1m", {})
        macd_1m = self._safe_float_with_last("macd_1m", analysis_1m.get("macd_histogram"), 0)
        rsi_1m = self._safe_float_with_last("rsi_1m", analysis_1m.get("rsi"), 50)
        if macd_1m > 0:
            long_score += 5
        if macd_1m < 0:
            short_score += 5
        if rsi_1m < 30:
            long_score += 4
        if rsi_1m > 70:
            short_score += 4

        # ========== K线形态评分（辅助确认，权重12）==========
        # 修复漏洞8：初始化pattern_boost防止NameError
        pattern_boost = {"long": 0, "short": 0}
        pattern_hits = []
        
        try:
            pattern_hits = self.pattern_detector.detect(market)
            market["patterns"] = pattern_hits if pattern_hits else []  # 修复漏洞2：确保不是None
            long_pattern_score = sum(
                self._safe_float(p.get("score"), 0.0)
                for p in pattern_hits
                if p.get("direction") == "LONG"
            )
            short_pattern_score = sum(
                self._safe_float(p.get("score"), 0.0)
                for p in pattern_hits
                if p.get("direction") == "SHORT"
            )
            long_score += min(12, int(long_pattern_score))
            short_score += min(12, int(short_pattern_score))
            pattern_boost = {"long": min(12, int(long_pattern_score)), "short": min(12, int(short_pattern_score))}
        except Exception as e:
            # 形态检测失败，不影响主流程
            market["patterns"] = []

        # Middle-ground penalty
        is_middle_ground = False
        if support_price > 0 and resistance_price > 0:
            if support_distance_pct > 0.3 and resistance_distance_pct > 0.3:
                is_middle_ground = True
        if is_middle_ground and not pattern_hits:
            long_score -= 10
            short_score -= 10
        
        # ========== 决策特征学习（自我进化核心）==========
        # 修复漏洞1：异常保护
        long_learning_score = 0.0
        short_learning_score = 0.0
        long_features = {}
        short_features = {}
        
        try:
            long_features = self.decision_learner.extract_features(market, "LONG")
            short_features = self.decision_learner.extract_features(market, "SHORT")
            long_learning_score = self.decision_learner.score(long_features)
            short_learning_score = self.decision_learner.score(short_features)
            
            long_score += int(round(long_learning_score))
            short_score += int(round(short_learning_score))
        except Exception as e:
            # 学习模块失败，不影响基础交易
            pass

        # Apply regime multiplier and clamp to non-negative/upper bound
        long_score = int(max(0, long_score) * regime_multiplier)
        short_score = int(max(0, short_score) * regime_multiplier)
        
        # ========== 市场偏向性检测：动态调整做空/做多难度 ==========
        # 基于多周期趋势判断市场整体方向
        trend_bullish_count = 0
        trend_bearish_count = 0
        
        if macro_dir == "BULLISH": trend_bullish_count += 2
        if macro_dir == "BEARISH": trend_bearish_count += 2
        if micro_dir == "BULLISH": trend_bullish_count += 1
        if micro_dir == "BEARISH": trend_bearish_count += 1
        
        # 计算市场偏向 (-3 到 +3)
        market_bias = trend_bullish_count - trend_bearish_count
        
        # 动态调整做空难度
        if market_bias >= 2:  # 强牛市：做空分数打7.5折
            short_score = int(short_score * 0.75)
        elif market_bias >= 1:  # 弱牛市：做空分数打8.5折
            short_score = int(short_score * 0.85)
        
        # 对称处理做多（熊市中做多更难）
        if market_bias <= -2:  # 强熊市：做多分数打7.5折
            long_score = int(long_score * 0.75)
        elif market_bias <= -1:  # 弱熊市：做多分数打8.5折
            long_score = int(long_score * 0.85)
        
        long_score = min(100, long_score)
        short_score = min(100, short_score)

        return {
            "long": long_score,
            "short": short_score,
            "rsi": rsi,
            "regime": regime,
            "breakout_signal": breakout_signal,
            "pattern_boost": pattern_boost,
            "patterns": pattern_hits,
            "decision_features": {"long": long_features, "short": short_features},
            "learning_boost": {"long": round(long_learning_score, 2), "short": round(short_learning_score, 2)},
        }

    def _predict_trade_outcome(self, entry_price: float, sl: float, tp: float, score: float, atr: float) -> Dict:
        """
        基于数学模型预测交易结果
        """
        try:
            # 1. 预测持仓时间 (基于ATR)
            dist_to_tp = abs(tp - entry_price)
            # 假设有效趋势波动为ATR的60%（扣除噪音）
            if atr > 0:
                effective_atr = atr * 0.6
            elif entry_price > 0:
                effective_atr = entry_price * 0.005
            else:
                effective_atr = 1.0 # 避免除零，给一个默认值
                
            candles_needed = dist_to_tp / effective_atr
            # 即使只差一点，至少也要1根K线(15m)
            duration_min = max(15, round(candles_needed * 15))
            
            # 2. 预测胜率 (基于模型置信度 + 盈亏比难度)
            # 基础胜率 50%
            base_win_rate = 50.0
            
            # AI评分修正: 分数每高出阈值(55) 1分，胜率+1.5%
            # 分数范围通常 50-90
            score_surplus = max(0, score - 55)
            score_impact = score_surplus * 1.5
            
            # 盈亏比修正: R/R越高，达成难度越大
            risk = abs(entry_price - sl)
            reward = abs(tp - entry_price)
            rr_ratio = reward / risk if risk > 0 else 1.0
            
            # 以1:1.5为基准。
            # 如果追求 1:3，胜率会自然下降。
            # 公式: 难度系数 = (RR / 1.5) ^ 0.5
            difficulty = (rr_ratio / 1.5) ** 0.5
            
            predicted_wr = (base_win_rate + score_impact) / (difficulty if difficulty > 0 else 1)
            
            # 极值限制
            predicted_wr = max(35.0, min(95.0, predicted_wr))
            
            return {
                "win_rate": predicted_wr,
                "duration": duration_min,
                "rr_ratio": rr_ratio
            }
        except Exception:
            # Fallback in case of any math error
            return {
                "win_rate": 50.0,
                "duration": 45,
                "rr_ratio": 1.5
            }

    def _build_signal_key(self, direction: str, reason: str, scores: Dict) -> str:
        patterns = scores.get("patterns") or []
        pattern_ids = []
        for p in patterns:
            name = p.get("name", "")
            p_dir = p.get("direction", "")
            if name or p_dir:
                pattern_ids.append(f"{name}:{p_dir}")
        pattern_key = ",".join(sorted(set(pattern_ids)))
        breakout = scores.get("breakout_signal") or {}
        breakout_type = breakout.get("type", "")
        return f"{direction}|{reason}|{breakout_type}|{pattern_key}"

    def _can_add_position(
        self,
        direction: str,
        current_price: float,
        signal_strength: Optional[float] = None,
        signal_key: Optional[str] = None,
    ) -> bool:
        """
        检查是否允许加仓（分批建仓过滤器）
        规则：
        1) 默认：必须满足 (时间间隔 > 3分钟) 或 (价格差距 > 0.2%)
        2) 同一信号：必须间隔足够时间且信号更强才允许加仓
        """
        same_dir_pos = [p for p in self.positions if p.get("direction") == direction]
        if not same_dir_pos:
            return True

        try:
            # 找到最近的一笔持仓
            latest = max(same_dir_pos, key=lambda p: p.get("timestamp_open", ""))
            
            last_time = datetime.fromisoformat(latest.get("timestamp_open"))
            time_diff = (datetime.now() - last_time).total_seconds()
            
            last_price = float(latest.get("entry_price", 0))
            price_diff_pct = 0.0
            if last_price > 0:
                price_diff_pct = abs(current_price - last_price) / last_price
            
            MIN_INTERVAL = 180  # 3分钟
            MIN_PRICE_GAP = 0.002 # 0.2%
            MIN_STRENGTH_DELTA = 5  # 同一信号需明显更强

            # 同一信号只开一笔：必须间隔足够时间且信号更强
            last_signal = self.last_entry_signal or {}
            if signal_key and last_signal and last_signal.get("direction") == direction:
                last_key = last_signal.get("signal_key")
                if last_key == signal_key:
                    last_strength = float(last_signal.get("strength") or 0)
                    if time_diff < MIN_INTERVAL:
                        return False
                    if signal_strength is None or signal_strength < last_strength + MIN_STRENGTH_DELTA:
                        return False
                    return True
            
            # 如果既没有在这个价格停留够久，价格也没有显著变化，则禁止加仓
            if time_diff < MIN_INTERVAL and price_diff_pct < MIN_PRICE_GAP:
                return False
                
            return True
        except Exception:
            return True

    def should_enter(self, market: Dict) -> Optional[Dict]:
        # Startup cooldown: wait 3 minutes before trading
        if time.time() - self.start_time < 180:
             return None

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
        # 改进：突破信号需要多维度确认，不能直接入场（防止假突破）
        breakout_signal = scores.get("breakout_signal")
        if breakout_signal and breakout_signal.get("strength", 0) > 0.3:
            direction = breakout_signal["direction"]
            
            # ========== 多维度确认 ==========
            analysis_15m = market.get("analysis_15m", {})
            analysis_1m = market.get("analysis_1m", {})
            
            # 确认1: MACD方向一致
            macd_15m = analysis_15m.get("macd_histogram", 0)
            macd_confirm = (direction == "LONG" and macd_15m > 0) or (direction == "SHORT" and macd_15m < 0)
            
            # 确认2: RSI未极端（避免追高杀低）
            rsi_15m = analysis_15m.get("rsi", 50)
            rsi_ok = (direction == "LONG" and rsi_15m < 75) or (direction == "SHORT" and rsi_15m > 25)
            
            # 确认3: 成交量放大（近5根vs前50根）
            klines_1m = market.get("klines_1m", [])
            volume_confirm = False
            if len(klines_1m) >= 10:
                recent_vol = sum(k.get("volume", 0) for k in klines_1m[-5:]) / 5
                avg_vol = sum(k.get("volume", 0) for k in klines_1m[-50:]) / max(1, len(klines_1m[-50:]))
                volume_confirm = recent_vol > avg_vol * 1.2  # 成交量放大20%
            
            # 确认4: 突破强度足够（至少0.5）
            strength_confirm = breakout_signal.get("strength", 0) >= 0.5
            
            # 确认计数
            confirms = sum([macd_confirm, rsi_ok, volume_confirm, strength_confirm])
            confirmations_list = []
            if macd_confirm: confirmations_list.append("MACD方向一致")
            if rsi_ok: confirmations_list.append("RSI未极端")
            if volume_confirm: confirmations_list.append("成交量放大")
            if strength_confirm: confirmations_list.append("突破强度足够")
            
            # 至少2项确认才允许入场（降低错过短线突破的概率）
            if confirms >= 2:
                signal_key = self._build_signal_key(direction, "breakout_confirmed", scores)
                if not self._can_add_position(
                    direction,
                    market["current_price"],
                    signal_strength=80 + int(breakout_signal["strength"] * 20),
                    signal_key=signal_key,
                ):
                    return None
                return {
                    "direction": direction,
                    "reason": "breakout_confirmed",
                    "breakout_type": breakout_signal["type"],
                    "scores": scores,
                    "threshold": threshold,
                    "effective_threshold": effective_threshold,
                    "strength": 80 + int(breakout_signal["strength"] * 20),  # 80-100
                    "flip_action": breakout_signal.get("action"),
                    "confirmations": confirmations_list,
                    "confirm_count": confirms,
                    "signal_key": signal_key,
                }
            # 确认不足，降级为普通评分入场（突破信号仍加分，但需要满足阈值）

        # Adjacent multi-signal filter: enforce minimum S/R gap
        if market.get("sr_gap_valid") is False:
            return None
        
        # Fee-aware edge filter: dynamic minimum edge based on ATR + fees
        price = market.get("current_price", 0)
        analysis_15m = market.get("analysis_15m", {})
        atr_15m = analysis_15m.get("atr", 0) or 0
        atr_pct = (atr_15m / price) * 100 if price > 0 else 0.0
        estimated_fee_pct = 0.12  # 0.12% round-trip fee + slippage buffer
        min_edge_pct = max(estimated_fee_pct * 1.8, min(0.8, atr_pct * 0.6 + estimated_fee_pct))
        best_support = market.get("best_support")
        best_resistance = market.get("best_resistance")
        profit_space_long = None
        profit_space_short = None
        if best_resistance and best_resistance.get("price") and price:
            profit_space_long = (best_resistance["price"] - price) / price * 100
        if best_support and best_support.get("price") and price:
            profit_space_short = (price - best_support["price"]) / price * 100

        # Normal score-based entry with trend-aware tie-breaking
        # MICRO trend takes priority for short-term trading
        long_ok = scores["long"] >= effective_threshold
        short_ok = scores["short"] >= effective_threshold
        if long_ok and profit_space_long is not None and profit_space_long < min_edge_pct:
            long_ok = False
        if short_ok and profit_space_short is not None and profit_space_short < min_edge_pct:
            short_ok = False
        # Scheme B for middle-ground: allow if strong score or pattern confirmation
        if scores.get("patterns"):
            for p in scores["patterns"]:
                if p.get("direction") == "LONG":
                    long_ok = long_ok or scores["long"] >= effective_threshold
                if p.get("direction") == "SHORT":
                    short_ok = short_ok or scores["short"] >= effective_threshold
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

            decision_features = scores.get("decision_features", {}).get(chosen.lower(), {})
            sr_snapshot = {
                "support": (market.get("best_support") or {}).get("price"),
                "resistance": (market.get("best_resistance") or {}).get("price"),
                "timestamp": datetime.now().isoformat(),
            }
            
            # 分批建仓检查（同一信号只开一笔）
            strength = scores["long"] if chosen == "LONG" else scores["short"]
            signal_key = self._build_signal_key(chosen, "score_trend_tiebreak", scores)
            if not self._can_add_position(
                chosen,
                market["current_price"],
                signal_strength=strength,
                signal_key=signal_key,
            ):
                return None

            return {
                "direction": chosen,
                "reason": "score_trend_tiebreak",
                "scores": scores,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "strength": strength,
                "patterns": scores.get("patterns"),
                "decision_features": decision_features,
                "signal_key": signal_key,
                "sr_snapshot": sr_snapshot,
            }

        if long_ok:
            sr_snapshot = {
                "support": (market.get("best_support") or {}).get("price"),
                "resistance": (market.get("best_resistance") or {}).get("price"),
                "timestamp": datetime.now().isoformat(),
            }
            signal_key = self._build_signal_key("LONG", "score", scores)
            if not self._can_add_position(
                "LONG",
                market["current_price"],
                signal_strength=scores["long"],
                signal_key=signal_key,
            ):
                return None
            return {
                "direction": "LONG",
                "reason": "score",
                "scores": scores,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "strength": scores["long"],
                "patterns": scores.get("patterns"),
                "decision_features": scores.get("decision_features", {}).get("long", {}),
                "signal_key": signal_key,
                "sr_snapshot": sr_snapshot,
            }
        if short_ok:
            sr_snapshot = {
                "support": (market.get("best_support") or {}).get("price"),
                "resistance": (market.get("best_resistance") or {}).get("price"),
                "timestamp": datetime.now().isoformat(),
            }
            signal_key = self._build_signal_key("SHORT", "score", scores)
            if not self._can_add_position(
                "SHORT",
                market["current_price"],
                signal_strength=scores["short"],
                signal_key=signal_key,
            ):
                return None
            return {
                "direction": "SHORT",
                "reason": "score",
                "scores": scores,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "strength": scores["short"],
                "patterns": scores.get("patterns"),
                "decision_features": scores.get("decision_features", {}).get("short", {}),
                "signal_key": signal_key,
                "sr_snapshot": sr_snapshot,
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
        # ====== Scheme C: 决策使用快照，下单前用最新SR再校验 ======
        current_market = self.last_market or market
        current_support = (current_market.get("best_support") or {}).get("price")
        current_resistance = (current_market.get("best_resistance") or {}).get("price")
        near_threshold_pct = float(getattr(self, "near_sr_threshold_pct", 0.1) or 0.1)
        if signal.get("direction") == "LONG" and current_resistance:
            dist_pct = abs(float(current_resistance) - price) / price * 100 if price > 0 else 0.0
            if dist_pct <= near_threshold_pct:
                return {"error": f"靠近阻力位({near_threshold_pct:.2f}%)，已阻止做多入场"}
        if signal.get("direction") == "SHORT" and current_support:
            dist_pct = abs(price - float(current_support)) / price * 100 if price > 0 else 0.0
            if dist_pct <= near_threshold_pct:
                return {"error": f"靠近支撑位({near_threshold_pct:.2f}%)，已阻止做空入场"}
        atr = self._safe_float_with_last(
            "atr_15m",
            market["analysis_15m"].get("atr"),
            0.0,
        )
        self.sl_tp.update_params(self.strategy.get_sl_tp_params())
        # 传递market和regime_adjustments给智能止盈计算
        sltp = self.sl_tp.calculate(price, signal["direction"], atr, market=market, regime_adjustments=self.regime_adjustments)
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
        
        # ========== 确保始终提取决策特征（在仓位计算前）==========
        decision_features = signal.get("decision_features", {})
        if not decision_features:
            # 如果信号中没有决策特征，从市场数据重新提取
            decision_features = self.decision_learner.extract_features(
                market, signal["direction"]
            )
        
        # Feature-driven position sizing (same-feature reinforcement)
        size_factor = self.feature_outcome.entry_size_factor(decision_features)
        base_qty = base_qty * size_factor
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

        # 强制单次信号只开一笔，禁用即时分批，交由 _can_add_position 控制节奏
        # batches = self.batch_manager.plan_entries(signal.get("strength", 50))
        qty = base_qty
        
        # Ensure qty meets minimum requirements
        if qty < min_qty:
            qty = min_qty
        if min_notional > 0 and price * qty < min_notional:
            return {"error": "仓位价值过小"}
            
        margin_estimate = qty * price / max(1, leverage)
        if available_margin > 0 and margin_estimate > available_margin * 0.9:
            return {"error": "保证金不足，已跳过入场"}

        take_profit = sltp["take_profit"]
        stop_loss = sltp["stop_loss"]
            
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
            "batch_index": 1,
            "batch_ratio": 1.0,
            "patterns": signal.get("patterns", []),
            "level_features": signal.get("level_features", {}),
            "level_finder_score": getattr(self.level_finder, 'last_score', 0),
            "decision_features": decision_features,
            "entry_size_factor": round(size_factor, 3),
            "external": False,
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
        self.trade_logger.log_trade(position)
        self._save_positions()
        
        self._last_entry_time = time.time()
        # self.last_entry_plan = batches # batches is commented out, so this line is likely not needed or needs re-evaluation
        self.last_entry_signal = signal
        return position

    def check_exit_all(self, current_price: float, market: Dict) -> List[Tuple[Dict, ExitDecision]]:
        exits = []
        scores = self.get_current_scores(market)
        market_with_scores = dict(market)
        market_with_scores["entry_scores"] = {"long": scores["long"], "short": scores["short"]}
        market_with_scores["entry_threshold"] = scores.get("threshold", {})
        market_with_scores["feature_outcome"] = self.feature_outcome
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
        fraction: float = 1.0,  # 平仓比例，默认全平
    ) -> Optional[Dict]:
        if skip_order:
            skip_api = True
        
        total_qty = position["quantity"]
        exit_qty = total_qty * fraction
        
        if not skip_api:
            side = "SELL" if position["direction"] == "LONG" else "BUY"
            try:
                order = self._place_limit_with_requote(
                    symbol="BTCUSDT",
                    side=side,
                    quantity=exit_qty,
                    current_price=current_price,
                    reduce_only=True,
                )
                if not order:
                    return None
            except Exception:
                pass  # 继续清理本地持仓，即使API失败

        entry_price = position["entry_price"]
        qty = exit_qty
        
        # 计算原始盈亏
        raw_pnl = (
            (current_price - entry_price) * qty
            if position["direction"] == "LONG"
            else (entry_price - current_price) * qty
        )
        
        # 计算手续费（保守估算按Taker 0.05%，开仓+平仓各一次）
        commission_rate = 0.0005  # 0.05% per side (Taker rate)
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
            "quantity": exit_qty,
            "leverage": position.get("leverage", self.leverage),
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_percent, 2),
            "raw_pnl": round(raw_pnl, 2),
            "commission": round(total_commission, 4),
            "exit_reason": f"{reason} (Partial {fraction*100:.0f}%)" if fraction < 1.0 else reason,
            "timestamp_open": position["timestamp_open"],
            "timestamp_close": exit_time.isoformat(),
            "is_win": 1 if pnl > 0 else 0,
            "hold_duration_minutes": round(hold_minutes, 2),
            "stop_loss": position.get("stop_loss"),
            "take_profit": position.get("take_profit"),
            "patterns": position.get("patterns", []),
            "strategies": position.get("strategies", []), 
            "market_state": (self.current_regime or (self.last_market or {}).get("regime", {}).get("regime") or "UNKNOWN"),
            "entry_reason": position.get("entry_reason", ""),
        }
        self.trade_logger.log_trade(trade)
        self.risk.update_trade_result(pnl_percent)
        self.level_finder.update_stats(pnl > 0)
        
        # ========== 决策特征学习（独立执行，不依赖level_features）==========
        decision_features = position.get("decision_features", {})
        if not decision_features and self.last_market:
            # 如果没有记录，尝试从当前市场重新提取
            direction = position.get("direction", "LONG")
            decision_features = self.decision_learner.extract_features(
                self.last_market, direction
            )
        
        is_external = bool(position.get("external"))
        # 始终尝试更新决策学习器
        if decision_features and not is_external:
            try:
                # 只有全平或主要利润时才更新学习，或者部分平仓也更新？
                # 更新为本次平仓的盈亏
                decision_update = self.decision_learner.update(
                    decision_features,
                    pnl_percent,
                    hold_minutes=hold_minutes,
                )
                trade["decision_learning"] = decision_update
            except Exception as e:
                pass  # 学习失败不影响交易记录
        
        # ========== K线形态学习 ==========
        patterns = position.get("patterns", [])
        is_win = pnl > 0
        pattern_updates = []
        for pattern in patterns:
            pattern_name = pattern.get("name") if isinstance(pattern, dict) else pattern
            if pattern_name:
                try:
                    if not is_external:
                        update = self.pattern_detector.update_pattern(pattern_name, pnl_percent, is_win)
                        if update:
                            pattern_updates.append(update)
                except Exception as e:
                    pass  # 形态学习失败不影响交易
        if pattern_updates:
            trade["pattern_learning"] = pattern_updates
        
        # ========== 出场决策反向传播学习 ==========
        if not is_external:
            try:
                market = self.last_market or {}
                exit_features = self.exit_decision_learner.extract_features(
                    position, market, current_price
                )
                exit_decision_update = self.exit_decision_learner.update(
                    exit_features,
                    exit_quality=exit_timing_quality,
                    pnl_percent=pnl_percent,
                    hold_minutes=hold_minutes,
                )
                trade["exit_decision_learning"] = exit_decision_update
            except Exception:
                pass  # 出场决策学习失败不影响交易

            # Feature outcome reinforcement for entry/exit controls
            try:
                if decision_features:
                    self.feature_outcome.update_entry(
                        decision_features,
                        pnl_percent=pnl_percent,
                        hold_minutes=hold_minutes,
                    )
                if exit_features:
                    self.feature_outcome.update_exit(
                        exit_features,
                        pnl_percent=pnl_percent,
                        hold_minutes=hold_minutes,
                    )
            except Exception:
                pass
        
        # Level特征学习（如果有的话）
        features = position.get("level_features")
        if features and not is_external:
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
        
        if fraction >= 1.0:
            # 全平：移除持仓
            self.positions = [p for p in self.positions if p["trade_id"] != position["trade_id"]]
            if position.get("trade_id") in self.position_states:
                del self.position_states[position.get("trade_id")]
        else:
            # 分批：更新剩余数量
            position["quantity"] = round(total_qty - exit_qty, 4)
            # 记录部分平仓事件以免重复触发？
            # 可以在 position_states 中记录 "secured_profit": True
            state = self.position_states.get(position.get("trade_id"), {})
            state["secured_profit"] = True
            self.position_states[position.get("trade_id")] = state
            
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
        current_price = self.last_market.get("current_price", 0)
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

        # 生成白话文解读
        narrative_parts = []
        
        # 1. 趋势解读
        macro = self.last_market['macro_trend']['direction']
        micro = self.last_market['micro_trend']['direction']
        trend_desc = "一致看涨" if macro == "BULLISH" and micro == "BULLISH" else \
                     "一致看跌" if macro == "BEARISH" and micro == "BEARISH" else \
                     "分歧(大势" + macro + "/小势" + micro + ")"
        narrative_parts.append(f"当前市场趋势{trend_desc}。")
        
        # 2. 指标解读
        rsi_val = analysis.get('rsi', 50)
        rsi_status = "超买" if rsi_val > 70 else "超卖" if rsi_val < 30 else "中性"
        narrative_parts.append(f"RSI({rsi_val:.1f}){rsi_status}。")
        
        # 3. 支撑阻力
        if self.best_support and self.best_resistance:
            cur_price = current_price or 0
            dist_s = (cur_price - self.best_support['price']) / cur_price * 100
            dist_r = (self.best_resistance['price'] - cur_price) / cur_price * 100
            narrative_parts.append(f"下方{dist_s:.2f}%处有支撑，上方{dist_r:.2f}%处有阻力。")
            
        # 4. 决策逻辑
        if conclusion == "多头信号":
             narrative_parts.append(f"因多头评分({scores['long']:.0f})突破阈值({scores['min_score']:.0f})，系统决定做多。")
        elif conclusion == "空头信号":
             narrative_parts.append(f"因空头评分({scores['short']:.0f})突破阈值({scores['min_score']:.0f})，系统决定做空。")
        else:
             high_score = max(scores['long'], scores['short'])
             narrative_parts.append(f"最高评分({high_score:.0f})未达阈值({scores['min_score']:.0f})，保持观望。")
             
        # 5. 预期 (基于数学推演)
        # 即使当前未开仓，也可以基于假设的入场（当前价格）进行推演
        calc_direction = "LONG" if scores["long"] > scores["short"] else "SHORT"
        calc_score = scores["long"] if calc_direction == "LONG" else scores["short"]
        if calc_score < scores["min_score"]:
            # 如果未达阈值，使用最高分做假设，但通过前缀说明是"假设"
            calc_direction = "LONG" if scores["long"] > scores["short"] else "SHORT"
            calc_score = max(scores["long"], scores["short"])
        
        # 动态计算 SL/TP
        atr = analysis.get("atr", current_price * 0.01)
        self.sl_tp.update_params(self.strategy.get_sl_tp_params())
        sltp = self.sl_tp.calculate(current_price, calc_direction, atr, market=self.last_market, regime_adjustments=self.regime_adjustments)
        
        prediction = self._predict_trade_outcome(
            current_price, 
            sltp["stop_loss"], 
            sltp["take_profit"], 
            calc_score, 
            atr
        )
        
        rr_str = f"1:{prediction['rr_ratio']:.1f}"
        win_str = f"{prediction['win_rate']:.1f}%"
        time_str = f"{prediction['duration']}分钟"
        
        narrative_parts.append(f"当前模型推演：盈亏比{rr_str}，预期胜率{win_str}，预计持仓{time_str}。")

        narrative = "".join(narrative_parts)

        # ... (Previous code)
        
        # 5. 预期 (基于数学推演)
        # ... (Prediction logic code) ...
        # ...

        positions = []
        # current_price already defined at top
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
            direction = pos.get("direction")
            
            if entry_price:
                if direction == "LONG":
                    pnl_percent = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_percent = (entry_price - current_price) / entry_price * 100
            
            # 【UI增强】计算SL/TP距离百分比
            try:
                sl = float(pos.get("stop_loss", 0))
                tp = float(pos.get("take_profit", 0))
                dist_sl = abs(current_price - sl) / current_price * 100 if sl > 0 else 0
                dist_tp = abs(tp - current_price) / current_price * 100 if tp > 0 else 0
            except Exception:
                dist_sl = 0
                dist_tp = 0

            # ... (Opportunity logic) ...
            
            # 【分批止盈】调用exit_manager评估
            # 传递更多状态以支持分批止盈判断
            # ...
            decision = self.exit_manager.evaluate(
                pos,
                {
                    **self.last_market,
                    "entry_scores": {"long": scores["long"], "short": scores["short"]},
                    "entry_threshold": scores.get("threshold", {}),
                },
                current_price,
            )
            
            # Decision might suggest partial exit
            
            positions.append(
                {
                    "trade_id": pos.get("trade_id"),
                    "direction": direction,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "dist_sl_pct": round(dist_sl, 2), # New
                    "dist_tp_pct": round(dist_tp, 2), # New
                    "entry_score": pos.get("entry_score", 0),
                    "hold_minutes": hold_minutes,
                    "pnl_percent": pnl_percent,
                    "exit_hint": decision.reason if decision else "",
                    "exit_notes": decision.confirmations if decision else [],
                    # ... (rest of dict)
                }
            )
        
        # ... (rest of function) ...


        entry_info = {
            "long_score": scores["long"],
            "short_score": scores["short"],
            "threshold": scores["min_score"],
            "phase": scores.get("phase", ""),
            "pattern_boost": self.last_signal_state.get("scores", {}).get("pattern_boost", {"long": 0, "short": 0}),
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
        
        # 学习状态增强
        learning_stats = self.strategy.get_learning_stats() if hasattr(self.strategy, 'get_learning_stats') else {}
        epsilon = learning_stats.get("epsilon", 0.1)
        samples = learning_stats.get("total_samples", 0)
        
        learning_info = {
            "mode": "EXPLORATION" if epsilon > 0.1 else "EXPLOITATION",
            "epsilon": round(epsilon, 3),
            "samples": samples,
            "focus": "优化震荡策略" if samples > 1000 else "基础特征积累", # 简化的动态焦点
            "recent_accuracy": learning_stats.get("recent_accuracy", 0.0),
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
            "narrative": narrative,
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

