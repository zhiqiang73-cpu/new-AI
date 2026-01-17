import json
import os
import sqlite3
from typing import Dict, List


class TradeLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                leverage INTEGER,
                pnl REAL,
                pnl_percent REAL,
                exit_reason TEXT,
                timestamp_open TEXT,
                timestamp_close TEXT,
                stop_loss REAL,
                take_profit REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def log_trade(self, trade: Dict) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(
                """
                INSERT INTO trades (
                    trade_id, direction, entry_price, exit_price, quantity,
                    leverage, pnl, pnl_percent, exit_reason,
                    timestamp_open, timestamp_close, stop_loss, take_profit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.get("trade_id"),
                    trade.get("direction"),
                    trade.get("entry_price"),
                    trade.get("exit_price"),
                    trade.get("quantity"),
                    trade.get("leverage", 10),
                    trade.get("pnl", 0),
                    trade.get("pnl_percent", 0),
                    trade.get("exit_reason", ""),
                    trade.get("timestamp_open"),
                    trade.get("timestamp_close"),
                    trade.get("stop_loss"),
                    trade.get("take_profit"),
                ),
            )
        except sqlite3.IntegrityError:
            # Avoid crash on duplicate trade_id during bulk close/logging
            conn.close()
            return
        conn.commit()
        conn.close()

    def get_recent(self, limit: int = 30) -> List[Dict]:
        if not os.path.exists(self.db_path):
            return []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        columns = [desc[0] for desc in c.description]
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def get_stats(self, last_n: int = 100) -> Dict:
        trades = self.get_recent(last_n)
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_profit_percent": 0,
                "avg_loss_percent": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "volatility": 0,
            }

        pnl_list = [t.get("pnl", 0) for t in trades]
        pnl_pct = [t.get("pnl_percent", 0) for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]

        total_pnl = sum(pnl_list)
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0
        avg_profit_pct = (
            sum([p for p in pnl_pct if p > 0]) / max(1, len([p for p in pnl_pct if p > 0]))
        )
        avg_loss_pct = (
            sum([abs(p) for p in pnl_pct if p < 0])
            / max(1, len([p for p in pnl_pct if p < 0]))
        )

        profit_factor = (sum(wins) / abs(sum(losses))) if losses else 0
        volatility = (sum([(p - (sum(pnl_pct) / len(pnl_pct))) ** 2 for p in pnl_pct]) / max(1, len(pnl_pct))) ** 0.5
        sharpe_ratio = (sum(pnl_pct) / len(pnl_pct)) / volatility if volatility > 0 else 0

        max_drawdown = 0
        peak = 0
        cumulative = 0
        for p in pnl_list:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_profit_percent": round(avg_profit_pct, 2),
            "avg_loss_percent": round(avg_loss_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "max_drawdown": round(max_drawdown, 2),
            "volatility": round(volatility, 3),
        }


class KnowledgeBase:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._save({"generation": 0})

    def _save(self, data: Dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> Dict:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_generation(self) -> int:
        return int(self._load().get("generation", 0))

    def increment_generation(self) -> int:
        data = self._load()
        data["generation"] = int(data.get("generation", 0)) + 1
        self._save(data)
        return data["generation"]


class EvolutionManager:
    def __init__(self, trade_logger: TradeLogger, knowledge: KnowledgeBase):
        self.trade_logger = trade_logger
        self.knowledge = knowledge

    def should_evolve(self, min_trades: int = 30) -> bool:
        stats = self.trade_logger.get_stats()
        return stats.get("total_trades", 0) >= min_trades

    def evolve(self) -> Dict:
        if not self.should_evolve():
            return {"evolved": False}
        generation = self.knowledge.increment_generation()
        return {"evolved": True, "generation": generation}


