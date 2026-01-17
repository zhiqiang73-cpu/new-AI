from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class ExitDecision:
    reason: str
    confirmations: List[str]


class PositionState:
    def __init__(self, position: dict):
        self.position = position


class ExitManager:
    def __init__(self, _params_path: str = None):
        self.params = {
            "max_loss_pct": 1.0,
            "min_profit_pct": 0.3,
            "max_hold_minutes": 45,
            "min_hold_minutes": 5,
            "opportunity_delta": 15,
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

        if state is not None:
            max_pnl = max(state.get("max_pnl_pct", pnl_pct), pnl_pct)
            state["max_pnl_pct"] = max_pnl
            lock_start = self.params.get("profit_lock_start", 0.6)
            base_drop = self.params.get("profit_lock_base_drop", 0.5)
            slope = self.params.get("profit_lock_slope", 0.05)
            if max_pnl >= lock_start:
                drop = max(0.15, base_drop - max_pnl * slope)
                if pnl_pct <= max_pnl - drop:
                    return ExitDecision(
                        "PROFIT_LOCK",
                        [f"max_pnl={max_pnl:.2f}", f"drop={drop:.2f}"],
                    )

        hold_minutes = self._get_hold_minutes(position)
        long_score, short_score, min_score = self._get_signal_scores(market)
        min_profit = self.params["min_profit_pct"]
        min_hold = self.params["min_hold_minutes"]
        opportunity_delta = self.params["opportunity_delta"]

        if hold_minutes is not None and hold_minutes >= min_hold and min_score > 0:
            if direction == "LONG":
                if short_score >= min_score + opportunity_delta and pnl_pct < min_profit:
                    return ExitDecision(
                        "OPPORTUNITY_SWITCH",
                        [
                            "better_short_signal",
                            f"short={short_score:.0f}",
                            f"threshold={min_score:.0f}",
                        ],
                    )
            else:
                if long_score >= min_score + opportunity_delta and pnl_pct < min_profit:
                    return ExitDecision(
                        "OPPORTUNITY_SWITCH",
                        [
                            "better_long_signal",
                            f"long={long_score:.0f}",
                            f"threshold={min_score:.0f}",
                        ],
                    )

        max_hold = self.params["max_hold_minutes"]
        if hold_minutes is not None and hold_minutes >= max_hold and abs(pnl_pct) < min_profit:
            return ExitDecision(
                "TIME_COST",
                ["time_cost", f"hold_minutes={hold_minutes:.1f}"],
            )

        if pnl_pct >= self.params["min_profit_pct"]:
            return None

        return None

