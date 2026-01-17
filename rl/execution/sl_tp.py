from typing import Dict


class StopLossTakeProfit:
    def __init__(self, level_scoring=None):
        self.level_scoring = level_scoring
        # 固定止损1%，止盈2%（风险回报比1:2）
        self.sl_pct = 0.01  # 1%
        self.tp_pct = 0.02  # 2%

    def update_params(self, params: Dict) -> None:
        # 不再动态调整，保持固定
        pass

    def calculate(self, price: float, direction: str, atr: float) -> Dict:
        sl_pct = self.sl_pct
        tp_pct = self.tp_pct

        if direction == "LONG":
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        else:
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
        }


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


