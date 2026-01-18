from typing import Dict

from ..config.time_manager import time_manager


class DynamicThresholdOptimizer:
    def __init__(self):
        self.history = []

    def compute(self, trade_count: int, win_rate: float) -> Dict:
        if trade_count < 30:
            base = 30
        elif trade_count < 100:
            base = 40
        else:
            base = 50

        adjust = 0
        if win_rate < 0.45:
            adjust += 5
        elif win_rate > 0.6:
            adjust -= 5

        threshold = max(20, min(70, base + adjust))
        self.history.append(
            {"timestamp": time_manager.now().isoformat(), "threshold": threshold}
        )
        return {"threshold": threshold, "base": base, "adjust": adjust}




