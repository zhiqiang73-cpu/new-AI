from typing import Dict


class NorthStarOptimizer:
    def __init__(self):
        self.targets = {
            "win_rate": 55.0,
            "profit_factor": 1.3,
            "sharpe_ratio": 1.0,
        }
        self.weights = {
            "win_rate": 0.4,
            "profit_factor": 0.3,
            "sharpe_ratio": 0.3,
        }

    def _score(self, value: float, target: float, cap: float = 2.0) -> float:
        if target <= 0:
            return 0.0
        ratio = max(0.0, min(cap, value / target))
        return ratio / cap * 100

    def evaluate(self, stats: Dict) -> Dict:
        trades = int(stats.get("total_trades", 0))
        win_rate = float(stats.get("win_rate", 0))
        profit_factor = float(stats.get("profit_factor", 0))
        sharpe_ratio = float(stats.get("sharpe_ratio", 0))

        score_win = self._score(win_rate, self.targets["win_rate"])
        score_pf = self._score(profit_factor, self.targets["profit_factor"])
        score_sharpe = self._score(sharpe_ratio, self.targets["sharpe_ratio"])

        composite = (
            score_win * self.weights["win_rate"]
            + score_pf * self.weights["profit_factor"]
            + score_sharpe * self.weights["sharpe_ratio"]
        )

        mode = "explore" if trades < 50 else "optimize"
        if composite >= 65:
            mode = "optimize"

        aggressive_delta = 0
        if mode == "explore":
            aggressive_delta = 10
        elif composite < 40:
            aggressive_delta = 6

        return {
            "trades": trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "score": round(composite, 1),
            "mode": mode,
            "aggressive_delta": aggressive_delta,
            "score_parts": {
                "win_rate": round(score_win, 1),
                "profit_factor": round(score_pf, 1),
                "sharpe_ratio": round(score_sharpe, 1),
            },
        }




