from typing import Dict


class RiskController:
    def __init__(self):
        self.max_daily_loss_pct = 10.0  # 学习阶段放宽到10%
        self.max_consecutive_losses = 10  # 连亏10次才停
        self.consecutive_losses = 0
        self.daily_pnl_pct = 0.0
        self._last_reset_date = None

    def _check_daily_reset(self) -> None:
        """每日重置日亏损计数"""
        from datetime import date
        today = date.today()
        if self._last_reset_date != today:
            self.daily_pnl_pct = 0.0
            self._last_reset_date = today

    def update_trade_result(self, pnl_percent: float) -> None:
        self._check_daily_reset()
        self.daily_pnl_pct += pnl_percent
        if pnl_percent < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def can_trade(self) -> Dict:
        self._check_daily_reset()
        if self.daily_pnl_pct <= -self.max_daily_loss_pct:
            return {"allowed": False, "reason": "daily_loss_limit"}
        if self.consecutive_losses >= self.max_consecutive_losses:
            return {"allowed": False, "reason": "consecutive_losses"}
        return {"allowed": True, "reason": "ok"}


