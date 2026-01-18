from typing import Dict, List


class BatchPositionManager:
    def __init__(self, max_batches: int = 3):
        self.max_batches = max_batches

    def plan_entries(self, signal_strength: float) -> List[Dict]:
        batches = []
        if signal_strength >= 80:
            splits = [0.5, 0.3, 0.2]
        elif signal_strength >= 60:
            splits = [0.6, 0.4]
        else:
            splits = [1.0]

        for i, ratio in enumerate(splits):
            batches.append({"batch": i + 1, "ratio": ratio})
        return batches

    def plan_exits(self, pnl_percent: float) -> List[Dict]:
        exits = []
        if pnl_percent >= 1.5:
            exits = [{"ratio": 0.5}, {"ratio": 0.5}]
        elif pnl_percent >= 0.8:
            exits = [{"ratio": 0.5}]
        return exits




