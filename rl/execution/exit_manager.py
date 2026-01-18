from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..learning.exit_decision_learner import ExitDecisionLearner


@dataclass
class ExitDecision:
    reason: str
    confirmations: List[str]
    fraction: float = 1.0  # 1.0=全部平仓, 0.5=平一半


class PositionState:
    def __init__(self, position: dict):
        self.position = position
        
    def get_max_profit(self) -> float:
        return self.position.get("max_profit_pct", 0.0)
        
    def update_max_profit(self, current_profit: float):
        if current_profit > self.get_max_profit():
            self.position["max_profit_pct"] = current_profit


class ExitManager:
    def __init__(self, _params_path: str = None, exit_decision_learner: "ExitDecisionLearner" = None):
        self.exit_decision_learner = exit_decision_learner  # 反向传播学习器
        self.params = {
            "max_loss_pct": 1.0,
            "min_profit_pct": 0.8,                # 从0.3提升至0.8：确保覆盖0.1%双向手续费并留有余地
            "max_hold_minutes": 45,
            "min_hold_minutes": 5,
            "opportunity_delta": 25,              # 从15提升到25：新信号必须明显更强
            "switch_min_new_score": 70,           # 新增：新信号至少70分
            "switch_max_loss_pct": -0.1,          # 新增：亏损超过0.1%不允许切换
            "switch_min_hold_minutes": 15,        # 提升至15分钟：减少频繁切换
            "profit_lock_start": 0.6,
            "profit_lock_base_drop": 0.5,
            "profit_lock_slope": 0.05,
        }

    def update_params(self, params: dict) -> None:
        if not params:
            return
        for k, v in params.items():
            if k in self.params and v is not None:
                self.params[k] = v

    def _get_hold_minutes(self, position: dict) -> Optional[float]:
        ts = position.get("timestamp_open")
        if not ts:
            return None
        try:
            opened = datetime.fromisoformat(ts)
        except Exception:
            return None
        return (datetime.now() - opened).total_seconds() / 60

    def _get_signal_scores(self, market: dict) -> Tuple[float, float, float]:
        scores = market.get("entry_scores") or {}
        threshold = market.get("entry_threshold") or {}
        long_score = float(scores.get("long", 0))
        short_score = float(scores.get("short", 0))
        min_score = float(threshold.get("threshold", market.get("min_score", 0)))
        return long_score, short_score, min_score

    def _check_sr_exit(
        self, direction: str, current_price: float, pnl_pct: float, market: dict,
        state: Optional[dict] = None
    ) -> Optional[ExitDecision]:
        """
        紧贴支撑阻力位出场：
        - 做多：触及阻力位后回落时止盈（卖在高点）
        - 做空：触及支撑位后反弹时止盈（买在低点）
        
        使用 state 记录是否曾触及关键位置
        """
        if state is None:
            state = {}
            
        # 获取最佳支撑阻力位
        best_support = market.get("best_support")
        best_resistance = market.get("best_resistance")
        
        # 触及阈值：0.03%（约$30 @ $100k）- 必须非常接近才算"触及"
        touch_pct = 0.03
        # 回落阈值：0.05%（约$50）- 短线精细操作
        # 加上限价单偏移0.03%，实际范围约$80
        pullback_pct = 0.05
        
        if direction == "LONG":
            # 做多时，追踪最高价和是否触及阻力
            max_price = state.get("max_price", current_price)
            if current_price > max_price:
                max_price = current_price
                state["max_price"] = max_price
            
            if best_resistance and pnl_pct > 0.3:
                resistance_price = float(best_resistance.get("price", 0))
                if resistance_price > 0:
                    # 检查是否曾触及阻力位（最高价接近阻力）
                    max_to_resistance_pct = (resistance_price - max_price) / max_price * 100
                    touched_resistance = max_to_resistance_pct <= touch_pct or max_price >= resistance_price
                    
                    if touched_resistance:
                        state["touched_resistance"] = True
                        # 检查是否开始回落
                        pullback_from_max = (max_price - current_price) / max_price * 100
                        if pullback_from_max >= pullback_pct:
                            return ExitDecision(
                                "SR_RESISTANCE_PULLBACK",
                                [
                                    f"resistance={resistance_price:.0f}",
                                    f"max_price={max_price:.0f}",
                                    f"pullback={pullback_from_max:.2f}%",
                                    f"pnl={pnl_pct:.2f}%",
                                ],
                            )
                    
                    # 假突破：突破阻力后回落到阻力下方
                    if state.get("touched_resistance") and current_price < resistance_price:
                        if pnl_pct > 0.5:
                            return ExitDecision(
                                "SR_FALSE_BREAKOUT",
                                [
                                    f"resistance={resistance_price:.0f}",
                                    f"pnl={pnl_pct:.2f}%",
                                ],
                            )
        else:
            # 做空时，追踪最低价和是否触及支撑
            min_price = state.get("min_price", current_price)
            if current_price < min_price:
                min_price = current_price
                state["min_price"] = min_price
            
            if best_support and pnl_pct > 0.3:
                support_price = float(best_support.get("price", 0))
                if support_price > 0:
                    # 检查是否曾触及支撑位（最低价接近支撑）
                    min_to_support_pct = (min_price - support_price) / support_price * 100
                    touched_support = min_to_support_pct <= touch_pct or min_price <= support_price
                    
                    if touched_support:
                        state["touched_support"] = True
                        # 检查是否开始反弹
                        bounce_from_min = (current_price - min_price) / min_price * 100
                        if bounce_from_min >= pullback_pct:
                            return ExitDecision(
                                "SR_SUPPORT_BOUNCE",
                                [
                                    f"support={support_price:.0f}",
                                    f"min_price={min_price:.0f}",
                                    f"bounce={bounce_from_min:.2f}%",
                                    f"pnl={pnl_pct:.2f}%",
                                ],
                            )
                    
                    # 假跌破：跌破支撑后反弹到支撑上方
                    if state.get("touched_support") and current_price > support_price:
                        if pnl_pct > 0.5:
                            return ExitDecision(
                                "SR_FALSE_BREAKDOWN",
                                [
                                    f"support={support_price:.0f}",
                                    f"pnl={pnl_pct:.2f}%",
                                ],
                            )
        
        return None

    def evaluate(
        self, position: dict, market: dict, current_price: float, state: Optional[dict] = None
    ) -> Optional[ExitDecision]:
        direction = position.get("direction")
        entry = position.get("entry_price", 0)
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        if entry <= 0:
            return None

        if direction == "LONG":
            pnl_pct = (current_price - entry) / entry * 100
            if stop_loss and current_price <= stop_loss:
                return ExitDecision("STOP_LOSS", ["stop_loss_hit"])
            if take_profit and current_price >= take_profit:
                return ExitDecision("TAKE_PROFIT", ["take_profit_hit"])
        else:
            pnl_pct = (entry - current_price) / entry * 100
            if stop_loss and current_price >= stop_loss:
                return ExitDecision("STOP_LOSS", ["stop_loss_hit"])
            if take_profit and current_price <= take_profit:
                return ExitDecision("TAKE_PROFIT", ["take_profit_hit"])

        if pnl_pct <= -self.params["max_loss_pct"]:
            return ExitDecision("MAX_LOSS", ["max_loss"])

        # ========== 支撑阻力位动态出场 ==========
        # 紧贴支撑阻力位，抓住波段高低点
        sr_exit = self._check_sr_exit(direction, current_price, pnl_pct, market, state)
        if sr_exit:
            return sr_exit

        # ESTIMATED_FEE_PCT: 0.05% taker * 2 = 0.1% + slippage buffer -> 0.12%
        ESTIMATED_FEE_PCT = 0.12

        if state is not None:
            max_pnl = max(state.get("max_pnl_pct", pnl_pct), pnl_pct)
            state["max_pnl_pct"] = max_pnl
            lock_start = self.params.get("profit_lock_start", 0.6)
            base_drop = self.params.get("profit_lock_base_drop", 0.25) # Tightened from 0.5
            slope = self.params.get("profit_lock_slope", 0.05)
            
            if max_pnl >= lock_start:
                # Dynamic drop: tighter as profit increases
                # Ensure we lock in at least (max_pnl - drop) > ESTIMATED_FEE_PCT + small_profit
                drop = max(0.1, base_drop - max_pnl * slope)
                
                # Check if the potential exit price covers fees
                locked_profit = max_pnl - drop
                if locked_profit < ESTIMATED_FEE_PCT + 0.05:
                     # If the drop would result in < 0.05% net profit, tighten it if possible
                     # But don't make it impossible to breathe (min drop 0.1%)
                     pass 

                if pnl_pct <= max_pnl - drop:
                    return ExitDecision(
                        "PROFIT_LOCK",
                        [f"max_pnl={max_pnl:.2f}", f"drop={drop:.2f}", f"locked={max_pnl-drop:.2f}"],
                    )

        hold_minutes = self._get_hold_minutes(position)
        long_score, short_score, min_score = self._get_signal_scores(market)
        min_profit = self.params["min_profit_pct"]
        min_hold = self.params["min_hold_minutes"]

        # 6. 新增：分批止盈 (Secure Profit)
        # 如果当前盈利不错 (>0.6%)，但分数开始下降，先平一半落袋为安
        scores = {"long": long_score, "short": short_score}
        partial_decision = self._check_secure_profit(direction, pnl_pct, scores)
        if partial_decision:
            return partial_decision
        
        # ========== 反向传播学习器出场建议 ==========
        # 如果配置了学习器，使用学习到的特征权重判断出场时机
        if self.exit_decision_learner:
            try:
                exit_features = self.exit_decision_learner.extract_features(
                    position, market, current_price
                )
                threshold = 70.0
                feature_outcome = market.get("feature_outcome")
                if feature_outcome:
                    threshold += feature_outcome.exit_threshold_delta(exit_features)
                    threshold = max(40.0, min(80.0, threshold))
                suggestion = self.exit_decision_learner.get_exit_suggestion(
                    position, market, current_price, threshold=threshold, features_override=exit_features
                )
                if suggestion.get("should_exit") and suggestion.get("score", 0) >= threshold:
                    return ExitDecision(
                        f"LEARNED_EXIT ({suggestion.get('top_reason', 'AI')})",
                        [
                            f"exit_score={suggestion.get('score', 0):.1f}",
                            f"pnl={pnl_pct:.2f}%",
                            f"learned_threshold={threshold:.0f}",
                        ],
                    )
            except Exception:
                pass  # 学习器失败不影响其他出场逻辑

        return None

    def _check_secure_profit(self, direction: str, pnl_pct: float, scores: dict) -> Optional[ExitDecision]:
        """
        分批止盈逻辑:
        当收益达到一定程度(0.6%)，且AI评分转弱(<50)时，平仓50%
        """
        secure_threshold = 0.6  # 0.6% 利润
        weak_score_threshold = 50.0 # 评分低于50视为转弱
        
        current_score = scores.get("long", 0) if direction == "LONG" else scores.get("short", 0)
        
        if pnl_pct > secure_threshold and current_score < weak_score_threshold:
            return ExitDecision(
                reason="分批止盈(收益>0.6%且信号转弱)",
                confirmations=[
                    f"当前收益{pnl_pct:.2f}% > {secure_threshold}%",
                    f"评分{current_score:.0f} < {weak_score_threshold}",
                    "建议平仓50%锁定利润"
                ],
                fraction=0.5
            )
        return None

