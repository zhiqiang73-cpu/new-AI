from typing import Dict


class StopLossTakeProfit:
    """
    智能止损止盈系统 - 始终锚定支撑阻力位
    
    核心理念：
    - 做多止损：设在支撑位下方（跌破支撑=防线失守=止损）
    - 做多止盈：设在阻力位附近（遇到阻力=入袋为安）
    - 做空止损：设在阻力位上方（突破阻力=防线失守=止损）
    - 做空止盈：设在支撑位附近（遇到支撑=入袋为安）
    - 兜底：确保盈亏比≥1.2:1
    """
    
    def __init__(self, level_scoring=None):
        self.level_scoring = level_scoring
        # 安全边距参数（更精确锚定S/R）
        self.sl_margin_pct = 0.0015    # 止损在S/R外侧0.15% (~$142 @ $95k)
        self.tp_margin_pct = 0.0005    # 止盈在S/R内侧0.05% (~$47 @ $95k)
        # 止损范围限制
        self.min_sl_pct = 0.002        # 最小止损0.2%
        self.max_sl_pct = 0.02         # 最大止损2.0%
        # 止盈范围限制
        self.min_tp_pct = 0.001        # 最小止盈0.1%
        self.max_tp_pct = 0.05         # 最大止盈5.0%
        # 备用参数（无S/R时使用）
        self.default_sl_pct = 0.01     # 1%
        self.default_tp_pct = 0.015    # 1.5%
        self.min_risk_reward = 1.2     # 最小盈亏比
        self.min_fee_profit_pct = 0.0025  # 最小盈利0.25%，覆盖手续费+滑点
        # 市场状态调整倍数
        self.tp_multiplier = 1.0
        self.sl_multiplier = 1.0
    
    def update_params(self, params: Dict) -> None:
        # 可以通过学习系统更新参数
        if not params:
            return
        if "sl_margin_pct" in params:
            self.sl_margin_pct = params["sl_margin_pct"]
        if "tp_margin_pct" in params:
            self.tp_margin_pct = params["tp_margin_pct"]
        # 市场状态调整倍数
        if "tp_multiplier" in params:
            self.tp_multiplier = params["tp_multiplier"]
        if "sl_multiplier" in params:
            self.sl_multiplier = params["sl_multiplier"]
    
    def calculate(self, price: float, direction: str, atr: float, market: Dict = None, regime_adjustments: Dict = None) -> Dict:
        """
        智能计算止损止盈 - 始终锚定支撑阻力位
        
        逻辑：
        1. 优先使用S/R位（市场结构）
        2. 兜底使用固定百分比
        3. 确保盈亏比合理
        4. 应用市场状态调整
        """
        # 应用市场状态调整
        if regime_adjustments:
            self.tp_multiplier = regime_adjustments.get("tp_multiplier", 1.0)
            self.sl_multiplier = regime_adjustments.get("sl_multiplier", 1.0)

        # 动态费损保护：结合ATR波动和手续费
        atr = atr if atr is not None else 0.0
        atr_pct = (atr / price) if price > 0 else 0.0
        estimated_fee_pct = 0.0012  # 0.12% round-trip fee + slippage buffer
        dynamic_min_tp_pct = max(
            self.min_fee_profit_pct,
            min(self.max_tp_pct, atr_pct * 0.6 + estimated_fee_pct),
        )
        
        if direction == "LONG":
            # ========== 做多：止损锚定支撑，止盈锚定阻力 ==========
            stop_loss, sl_pct = self._calculate_long_sl(price, market)
            take_profit, tp_pct = self._calculate_long_tp(price, market)
            
            # 应用市场状态乘数
            sl_distance = price - stop_loss
            tp_distance = take_profit - price
            sl_distance *= self.sl_multiplier
            tp_distance *= self.tp_multiplier
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
            sl_pct *= self.sl_multiplier
            tp_pct *= self.tp_multiplier
            
            # 费损保护：确保最小盈利覆盖手续费+滑点（动态）
            if tp_pct < dynamic_min_tp_pct:
                tp_pct = min(dynamic_min_tp_pct, self.max_tp_pct)
                take_profit = price * (1 + tp_pct)
            
            # 盈亏比检查
            sl_distance = price - stop_loss if price > stop_loss else price * 0.01
            tp_distance = take_profit - price if take_profit > price else price * 0.015
            
            if tp_distance < sl_distance * self.min_risk_reward:
                # 盈亏比不够，缩小止损
                max_allowed_sl_distance = tp_distance / self.min_risk_reward
                if max_allowed_sl_distance / price >= self.min_sl_pct:
                    stop_loss = price - max_allowed_sl_distance
                    sl_pct = max_allowed_sl_distance / price
        else:
            # ========== 做空：止损锚定阻力，止盈锚定支撑 ==========
            stop_loss, sl_pct = self._calculate_short_sl(price, market)
            take_profit, tp_pct = self._calculate_short_tp(price, market)
            
            # 应用市场状态乘数
            sl_distance = stop_loss - price
            tp_distance = price - take_profit
            sl_distance *= self.sl_multiplier
            tp_distance *= self.tp_multiplier
            stop_loss = price + sl_distance
            take_profit = price - tp_distance
            sl_pct *= self.sl_multiplier
            tp_pct *= self.tp_multiplier
            
            # 费损保护：确保最小盈利覆盖手续费+滑点（动态）
            if tp_pct < dynamic_min_tp_pct:
                tp_pct = min(dynamic_min_tp_pct, self.max_tp_pct)
                take_profit = price * (1 - tp_pct)
            
            sl_distance = stop_loss - price if stop_loss > price else price * 0.01
            tp_distance = price - take_profit if price > take_profit else price * 0.015
            
            if tp_distance < sl_distance * self.min_risk_reward:
                max_allowed_sl_distance = tp_distance / self.min_risk_reward
                if max_allowed_sl_distance / price >= self.min_sl_pct:
                    stop_loss = price + max_allowed_sl_distance
                    sl_pct = max_allowed_sl_distance / price
        
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
        }
    
    def _calculate_long_sl(self, price: float, market: Dict = None) -> tuple:
        """
        计算做多止损 - 锚定支撑位下方
        
        逻辑：支撑位是多头的防线，跌破支撑说明防线失守，应该止损
        """
        best_support = market.get("best_support") if market else None
        
        if best_support and price > 0:
            support_price = float(best_support["price"])
            
            # 只要支撑在当前价格下方，就应该锚定
            if support_price < price:
                # 止损设在支撑下方（安全边距）
                stop_loss = support_price * (1 - self.sl_margin_pct)
                sl_pct = (price - stop_loss) / price
                
                # 限制止损范围
                if sl_pct < self.min_sl_pct:
                    # 支撑太近，使用最小止损
                    stop_loss = price * (1 - self.min_sl_pct)
                    sl_pct = self.min_sl_pct
                elif sl_pct > self.max_sl_pct:
                    # 支撑太远，使用最大止损
                    stop_loss = price * (1 - self.max_sl_pct)
                    sl_pct = self.max_sl_pct
                
                return stop_loss, sl_pct
        
        # 备用：固定1%
        stop_loss = price * (1 - self.default_sl_pct) if price > 0 else 0
        return stop_loss, self.default_sl_pct
    
    def _calculate_short_sl(self, price: float, market: Dict = None) -> tuple:
        """
        计算做空止损 - 锚定阻力位上方
        
        逻辑：阻力位是空头的防线，突破阻力说明防线失守，应该止损
        """
        best_resistance = market.get("best_resistance") if market else None
        
        if best_resistance and price > 0:
            resistance_price = float(best_resistance["price"])
            
            # 只要阻力在当前价格上方，就应该锚定
            if resistance_price > price:
                # 止损设在阻力上方（安全边距）
                stop_loss = resistance_price * (1 + self.sl_margin_pct)
                sl_pct = (stop_loss - price) / price
                
                # 限制止损范围
                if sl_pct < self.min_sl_pct:
                    stop_loss = price * (1 + self.min_sl_pct)
                    sl_pct = self.min_sl_pct
                elif sl_pct > self.max_sl_pct:
                    stop_loss = price * (1 + self.max_sl_pct)
                    sl_pct = self.max_sl_pct
                
                return stop_loss, sl_pct
        
        # 备用：固定1%
        stop_loss = price * (1 + self.default_sl_pct) if price > 0 else 0
        return stop_loss, self.default_sl_pct
    
    def _calculate_long_tp(self, price: float, market: Dict = None) -> tuple:
        """
        计算做多止盈 - 锚定阻力位下方
        
        逻辑：阻力位是多头的障碍，接近阻力应该入袋为安
        """
        best_resistance = market.get("best_resistance") if market else None
        
        if best_resistance and price > 0:
            resistance_price = float(best_resistance["price"])
            
            # 只要阻力在当前价格上方，就应该锚定
            if resistance_price > price:
                # 止盈设在阻力下方一点（提前入袋）
                take_profit = resistance_price * (1 - self.tp_margin_pct)
                tp_pct = (take_profit - price) / price
                
                # 限制止盈范围
                if tp_pct < self.min_tp_pct:
                    # 阻力太近，使用最小止盈
                    take_profit = price * (1 + self.min_tp_pct)
                    tp_pct = self.min_tp_pct
                elif tp_pct > self.max_tp_pct:
                    # 阻力太远，使用最大止盈
                    take_profit = price * (1 + self.max_tp_pct)
                    tp_pct = self.max_tp_pct
                
                return take_profit, tp_pct
        
        # 备用：固定1.5%
        take_profit = price * (1 + self.default_tp_pct) if price > 0 else 0
        return take_profit, self.default_tp_pct
    
    def _calculate_short_tp(self, price: float, market: Dict = None) -> tuple:
        """
        计算做空止盈 - 锚定支撑位上方
        
        逻辑：支撑位是空头的障碍，接近支撑应该入袋为安
        """
        best_support = market.get("best_support") if market else None
        
        if best_support and price > 0:
            support_price = float(best_support["price"])
            
            # 只要支撑在当前价格下方，就应该锚定
            if support_price < price:
                # 止盈设在支撑上方一点（提前入袋）
                take_profit = support_price * (1 + self.tp_margin_pct)
                tp_pct = (price - take_profit) / price
                
                # 限制止盈范围
                if tp_pct < self.min_tp_pct:
                    take_profit = price * (1 - self.min_tp_pct)
                    tp_pct = self.min_tp_pct
                elif tp_pct > self.max_tp_pct:
                    take_profit = price * (1 - self.max_tp_pct)
                    tp_pct = self.max_tp_pct
                
                return take_profit, tp_pct
        
        # 备用：固定1.5%
        take_profit = price * (1 - self.default_tp_pct) if price > 0 else 0
        return take_profit, self.default_tp_pct


class PositionSizer:
    def __init__(self, max_risk_percent: float = 2.0):
        self.max_risk_percent = max_risk_percent

    def calculate_size(self, balance: float, entry_price: float, stop_loss: float) -> float:
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        risk_per_unit = abs(entry_price - stop_loss)
        max_risk = balance * (self.max_risk_percent / 100)
        if risk_per_unit <= 0:
            return 0.0
        return max_risk / risk_per_unit
