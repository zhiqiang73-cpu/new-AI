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
                is_win INTEGER,
                hold_duration_minutes REAL,
                stop_loss REAL,
                take_profit REAL,
                raw_pnl REAL,
                commission REAL,
                patterns TEXT,
                market_state TEXT,
                entry_reason TEXT
            )
            """
        )
        conn.commit()
        
        # Schema Migration: 检查并添加缺失列
        c.execute("PRAGMA table_info(trades)")
        columns = [row[1] for row in c.fetchall()]
        
        if "raw_pnl" not in columns:
            c.execute("ALTER TABLE trades ADD COLUMN raw_pnl REAL")
        if "commission" not in columns:
            c.execute("ALTER TABLE trades ADD COLUMN commission REAL")
        if "patterns" not in columns:
            c.execute("ALTER TABLE trades ADD COLUMN patterns TEXT")
        if "is_win" not in columns:
            c.execute("ALTER TABLE trades ADD COLUMN is_win INTEGER")
        if "hold_duration_minutes" not in columns:
            c.execute("ALTER TABLE trades ADD COLUMN hold_duration_minutes REAL")
        if "market_state" not in columns:
            c.execute("ALTER TABLE trades ADD COLUMN market_state TEXT")
        if "entry_reason" not in columns:
            c.execute("ALTER TABLE trades ADD COLUMN entry_reason TEXT")
            
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
                    timestamp_open, timestamp_close, is_win, hold_duration_minutes,
                    stop_loss, take_profit, raw_pnl, commission, patterns,
                    market_state, entry_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    trade.get("is_win"),
                    trade.get("hold_duration_minutes"),
                    trade.get("stop_loss"),
                    trade.get("take_profit"),
                    trade.get("raw_pnl", 0),
                    trade.get("commission", 0),
                    json.dumps(trade.get("patterns", []), ensure_ascii=False),
                    trade.get("market_state", ""),
                    trade.get("entry_reason", ""),
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

        def _safe_num(value, default=0.0):
            try:
                if value is None:
                    return float(default)
                return float(value)
            except Exception:
                return float(default)

        pnl_list = [_safe_num(t.get("pnl")) for t in trades]
        pnl_pct = [_safe_num(t.get("pnl_percent")) for t in trades]
        commission_list = [_safe_num(t.get("commission")) for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]

        total_pnl = sum(pnl_list)
        total_commission = sum(commission_list)
        net_profit_ratio = (total_pnl / total_commission) if total_commission > 0 else 0
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

        # Calculate average holding time
        from datetime import datetime
        durations = []
        for t in trades:
            # t[10]=timestamp_open, t[11]=timestamp_close (Based on schema in _init_db: id, trade_id, direction, entry, exit, qty, lev, pnl, pnl%, exit_reason, open, close)
            # Schema indices:
            # 0:id, 1:trade_id, 2:direction, 3:entry, 4:exit, 5:qty, 6:lev, 7:pnl, 8:pnl%, 9:reason, 10:open, 11:close
            ts_open = t.get("timestamp_open")
            ts_close = t.get("timestamp_close")
            
            if ts_open and ts_close:
                try:
                    t_open = datetime.fromisoformat(ts_open)
                    t_close = datetime.fromisoformat(ts_close)
                    durations.append((t_close - t_open).total_seconds() / 60)
                except Exception:
                    pass
        
        avg_holding_time = sum(durations) / len(durations) if durations else 0

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
            "avg_holding_time": round(avg_holding_time, 1),
            "net_profit_ratio": round(net_profit_ratio, 3),
            "total_commission": round(total_commission, 4),
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


