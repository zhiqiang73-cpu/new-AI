import json
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template, request

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from client import BinanceFuturesClient
from rl.core.agent import TradingAgent

DB_PATH = os.path.join(os.path.dirname(__file__), "trading.db")
RL_DATA_DIR = os.path.join(BASE_DIR, "rl_data")
LOG_FILE = os.path.join(RL_DATA_DIR, "agent.log")

app = Flask(__name__)
app.secret_key = os.urandom(24)


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

agent_state = {
    "running": False,
    "thread": None,
    "agent": None,
    "logs": deque(maxlen=200),
    "last_update": None,
    "last_stop_reason": None,
    "config": {"leverage": 10, "near_sr_threshold_pct": 0.1},
}


def add_log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    agent_state["logs"].append(
        {"time": timestamp, "level": level, "message": message}
    )
    agent_state["last_update"] = datetime.now().isoformat()
    try:
        os.makedirs(RL_DATA_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} [{level}] {message}\n")
    except Exception:
        pass


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY,
            api_key TEXT,
            api_secret TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def get_api_keys():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT api_key, api_secret FROM api_keys ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row


def save_api_keys(api_key: str, api_secret: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM api_keys")
    c.execute(
        "INSERT INTO api_keys (api_key, api_secret) VALUES (?, ?)",
        (api_key, api_secret),
    )
    conn.commit()
    conn.close()


def get_client():
    keys = get_api_keys()
    if not keys:
        return None
    client = BinanceFuturesClient.__new__(BinanceFuturesClient)
    client.base_url = "https://testnet.binancefuture.com"
    client.api_key = keys[0]
    client.api_secret = keys[1]
    client.session = requests.Session()
    client.session.headers.update({"X-MBX-APIKEY": client.api_key})
    client.time_offset = 0
    client._sync_time()
    return client


def get_mainnet_klines(symbol: str, interval: str, limit: int = 150):
    base_url = "https://fapi.binance.com"
    res = requests.get(
        f"{base_url}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    return res.json()


def get_mainnet_order_book(symbol: str, limit: int = 100):
    base_url = "https://fapi.binance.com"
    res = requests.get(
        f"{base_url}/fapi/v1/depth",
        params={"symbol": symbol, "limit": limit},
        timeout=10,
    )
    return res.json()


def convert_klines(klines):
    if not isinstance(klines, list):
        return []
    candles = []
    for k in klines:
        if not isinstance(k, (list, tuple)) or len(k) < 6:
            continue
        time_val = k[0]
        try:
            time_val = int(time_val) // 1000
        except (TypeError, ValueError):
            continue
        candles.append(
            {
                "time": time_val,
                "open": _safe_float(k[1]),
                "high": _safe_float(k[2]),
                "low": _safe_float(k[3]),
                "close": _safe_float(k[4]),
                "volume": _safe_float(k[5]),
            }
        )
    return candles


def convert_order_book(depth):
    if not isinstance(depth, dict):
        return {"bids": [], "asks": []}
    bids = [(_safe_float(p), _safe_float(q)) for p, q in depth.get("bids", [])]
    asks = [(_safe_float(p), _safe_float(q)) for p, q in depth.get("asks", [])]
    return {"bids": bids, "asks": asks}


def run_agent_loop():
    os.makedirs(RL_DATA_DIR, exist_ok=True)
    client = get_client()
    if not client:
        agent_state["last_stop_reason"] = "api_keys_missing"
        add_log("API keys not configured", "ERROR")
        agent_state["running"] = False
        return

    config = agent_state.get("config", {})
    leverage = config.get("leverage", 10)
    agent = TradingAgent(client, data_dir=RL_DATA_DIR, leverage=leverage)
    agent.near_sr_threshold_pct = float(config.get("near_sr_threshold_pct", 0.1) or 0.1)
    agent_state["agent"] = agent
    add_log("Agent已启动", "SUCCESS")
    
    # Force sync positions on startup
    try:
        binance_positions = client.get_positions()
        active = [
            p for p in binance_positions if abs(_safe_float(p.get("positionAmt"))) > 0
        ]
        exchange_total = sum(
            abs(_safe_float(p.get("positionAmt"))) for p in active
        )
        agent_total = sum(_safe_float(p.get("quantity")) for p in agent.positions)
        if abs(exchange_total - agent_total) > 0.0005:
            add_log(f"启动时检测到持仓不一致: 交易所={exchange_total:.4f}, AI={agent_total:.4f}, 开始同步...", "WARNING")
            # Sync will happen automatically on next /api/positions call
    except Exception as e:
        add_log(f"启动时持仓检查失败: {str(e)}", "ERROR")

    while agent_state["running"]:
        try:
            kl_1m = get_mainnet_klines("BTCUSDT", "1m", 150)
            kl_15m = get_mainnet_klines("BTCUSDT", "15m", 150)
            kl_8h = get_mainnet_klines("BTCUSDT", "8h", 150)
            kl_1w = get_mainnet_klines("BTCUSDT", "1w", 50)
            order_book = None
            try:
                depth = get_mainnet_order_book("BTCUSDT", 100)
                if isinstance(depth, dict):
                    order_book = convert_order_book(depth)
            except Exception:
                order_book = None

            market = agent.analyze_market(
                convert_klines(kl_1m),
                convert_klines(kl_15m),
                convert_klines(kl_8h),
                convert_klines(kl_1w),
                order_book,
            )

            if not market:
                add_log("市场分析失败，等待下一轮", "WARNING")
                time.sleep(5)
                continue

            price = market.get("current_price", 0)
            best_support = market.get("best_support")
            best_resistance = market.get("best_resistance")
            scores = agent.get_current_scores(market)
            tf_weights = market.get("tf_weights") or {}

            if best_support and best_support.get("price") is not None:
                add_log(
                    "支撑位 {price:.0f} (评分 {score:.0f})".format(
                        price=_safe_float(best_support.get("price")),
                        score=_safe_float(best_support.get("score")),
                    )
                )
            if best_resistance and best_resistance.get("price") is not None:
                add_log(
                    "阻力位 {price:.0f} (评分 {score:.0f})".format(
                        price=_safe_float(best_resistance.get("price")),
                        score=_safe_float(best_resistance.get("score")),
                    )
                )
            add_log(
                "逻辑链: 趋势={macro}/{micro} 多={long:.0f} 空={short:.0f} 阈值={th:.0f} "
                "S={s:.0f} R={r:.0f} TF(1m={w1:.0f}%,15m={w15:.0f}%,8h={w8:.0f}%,1w={w1w:.0f}%)".format(
                    macro=market.get("macro_trend", {}).get("direction", "-"),
                    micro=market.get("micro_trend", {}).get("direction", "-"),
                    long=scores.get("long", 0),
                    short=scores.get("short", 0),
                    th=scores.get("min_score", 0),
                    s=(best_support or {}).get("price", 0),
                    r=(best_resistance or {}).get("price", 0),
                    w1=(tf_weights.get("1m", 0) * 100),
                    w15=(tf_weights.get("15m", 0) * 100),
                    w8=(tf_weights.get("8h", 0) * 100),
                    w1w=(tf_weights.get("1w", 0) * 100),
                ),
                "INFO",
            )

            exits = agent.check_exit_all(price, market)
            for pos, decision in exits:
                fraction = getattr(decision, "fraction", 1.0)
                trade = agent.execute_exit_position(
                    pos, price, decision.reason, decision.confirmations, fraction=fraction
                )
                if trade:
                    outcome = "盈利" if trade["pnl"] >= 0 else "亏损"
                    add_log(
                        f"平仓 {trade['trade_id'][:8]} {outcome} PnL={trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%) 原因={trade['exit_reason']}",
                        "SUCCESS" if trade["pnl"] >= 0 else "WARNING",
                    )
                    update = trade.get("weight_update")
                    if update:
                        delta = update.get("delta", {})
                        lines = []
                        for k, v in delta.items():
                            if abs(v) >= 0.0001:
                                lines.append(f"{k}: {v:+.4f}")
                        strategy = trade.get("strategy_params") or {}
                        reward = trade.get("reward")
                        if lines or strategy or reward is not None:
                            parts = []
                            if reward is not None:
                                parts.append(f"reward={reward:+.2f}")
                            if strategy:
                                parts.append(
                                    "参数:入场偏移={bias:+.2f},锁利起点={start:.2f},回撤={drop:.2f},斜率={slope:.2f}".format(
                                        bias=strategy.get("entry_threshold_bias", 0),
                                        start=strategy.get("profit_lock_start", 0),
                                        drop=strategy.get("profit_lock_base_drop", 0),
                                        slope=strategy.get("profit_lock_slope", 0),
                                    )
                                )
                            if lines:
                                parts.append("权重变化:" + ", ".join(lines))
                            add_log(
                                "学习链: " + " | ".join(parts),
                                "INFO",
                            )

            signal = agent.should_enter(market)
            if signal:
                pos = agent.execute_entry(market, signal)
                if pos and "error" not in pos:
                    effective_threshold = signal.get("effective_threshold")
                    if effective_threshold is None:
                        effective_threshold = signal.get("threshold", {}).get("threshold")
                    add_log(
                        "入场 {direction} 价={price:.2f} 分数={score:.0f} 阈值={threshold:.0f} "
                        "SL={sl:.2f} TP={tp:.2f} 杠杆={lev}x".format(
                            direction=signal["direction"],
                            price=price,
                            score=signal.get("strength", 0),
                            threshold=effective_threshold or 0,
                            sl=pos.get("stop_loss", 0),
                            tp=pos.get("take_profit", 0),
                            lev=pos.get("leverage", leverage),
                        ),
                        "SUCCESS",
                    )
                elif pos and "error" in pos:
                    add_log(f"入场失败: {pos['error']}", "WARNING")

            time.sleep(10)
        except Exception as exc:
            add_log(f"Agent循环异常: {exc}", "ERROR")
            time.sleep(5)


def run_agent_loop_with_restart():
    retries = 0
    while agent_state["running"]:
        try:
            add_log(f"Agent循环启动 #{retries + 1}")
            run_agent_loop()
        except Exception as exc:
            retries += 1
            add_log(f"Agent异常重启: {exc}", "ERROR")
            time.sleep(min(60, 5 * retries))


@app.route("/")
def index():
    return render_template("index_v4.html")


@app.route("/v3")
def index_v3():
    return render_template("index.html")


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        data = request.json or {}
        save_api_keys(data.get("api_key", ""), data.get("api_secret", ""))
        return jsonify({"success": True})

    keys = get_api_keys()
    if keys:
        return jsonify({"has_keys": True, "api_key": keys[0][:8] + "..."})
    return jsonify({"has_keys": False})


@app.route("/api/account")
def account():
    client = get_client()
    if not client:
        return jsonify({"error": "API keys not configured"}), 400
    try:
        account_info = client.get_account()
        balances = client.get_balance()
        if not isinstance(balances, list):
            balances = []
        result = []
        for b in balances:
            if _safe_float(b.get("balance")) > 0:
                result.append(
                    {
                        "asset": b["asset"],
                        "available": _safe_float(b.get("availableBalance")),
                        "total": _safe_float(b.get("balance")),
                    }
                )
        margin = {}
        if isinstance(account_info, dict):
            margin = {
                "walletBalance": _safe_float(account_info.get("totalWalletBalance")),
                "marginBalance": _safe_float(account_info.get("totalMarginBalance")),
                "availableBalance": _safe_float(account_info.get("totalAvailableBalance")),
                "initialMargin": _safe_float(account_info.get("totalInitialMargin")),
                "maintMargin": _safe_float(account_info.get("totalMaintMargin")),
                "unrealizedProfit": _safe_float(account_info.get("totalUnrealizedProfit")),
            }
        return jsonify({"balances": result, "margin": margin})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/positions")
def positions():
    client = get_client()
    if not client:
        return jsonify({"error": "API keys not configured"}), 400

    try:
        binance_positions = client.get_positions()
        if not isinstance(binance_positions, list):
            return jsonify({"error": "Invalid positions response"}), 400
        active = [
            p for p in binance_positions if abs(_safe_float(p.get("positionAmt"))) > 0
        ]
        ticker_price = None
        try:
            ticker = client.get_ticker_price("BTCUSDT")
            ticker_price = _safe_float(ticker.get("price")) if ticker else None
        except Exception:
            ticker_price = None

        agent = agent_state.get("agent")
        agent_positions = []
        if agent and agent.positions:
            agent_positions = agent.positions
        else:
            pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
            if os.path.exists(pos_file):
                try:
                    with open(pos_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        agent_positions = file_data.get("positions", [])
                except Exception:
                    agent_positions = []

        def _sync_external_positions(active_positions, agent_positions_list, agent_obj):
            """
            Sync positions between exchange and AI records.
            Handles two cases:
            1. Exchange > AI: Create external entries
            2. AI > Exchange: Remove excess AI entries (orphaned records)
            """
            updated = False
            if not active_positions:
                if agent_positions_list:
                    agent_positions_list.clear()
                    updated = True
                    add_log("同步清理: 交易所无持仓，已清空AI开仓明细", "WARNING")
                if updated:
                    if agent_obj and getattr(agent_obj, "positions", None) is not None:
                        agent_obj.positions = []
                        try:
                            agent_obj._save_positions()
                        except Exception:
                            pass
                    else:
                        try:
                            pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
                            with open(pos_file, "w", encoding="utf-8") as f:
                                json.dump({"positions": []}, f, indent=2)
                        except Exception:
                            pass
                return agent_positions_list

            # External position sync disabled: remove any existing external entries only
            cleaned = [p for p in agent_positions_list if not p.get("external")]
            if len(cleaned) != len(agent_positions_list):
                agent_positions_list[:] = cleaned
                updated = True
                add_log("已移除EXTERNAL持仓同步记录", "INFO")
                if agent_obj and getattr(agent_obj, "positions", None) is not None:
                    agent_obj.positions = list(agent_positions_list)
                    try:
                        agent_obj._save_positions()
                    except Exception:
                        pass
                else:
                    try:
                        pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
                        with open(pos_file, "w", encoding="utf-8") as f:
                            json.dump({"positions": agent_positions_list}, f, indent=2)
                    except Exception:
                        pass
            return agent_positions_list

        # Force sync positions (handles both exchange > AI and AI > exchange cases)
        agent_positions = _sync_external_positions(active, agent_positions, agent)
        agent_positions = [p for p in agent_positions if not p.get("external")]

        result = []
        for p in active:
            amt = _safe_float(p.get("positionAmt"))
            direction = "LONG" if amt > 0 else "SHORT"
            entry_price = _safe_float(p.get("entryPrice"))
            mark_price = _safe_float(p.get("markPrice", entry_price))
            pnl = _safe_float(p.get("unRealizedProfit"))
            notional = abs(amt) * mark_price
            leverage = int(p.get("leverage", 10))
            margin_used = _safe_float(
                p.get("positionInitialMargin")
                or p.get("isolatedMargin")
                or 0
            )
            if margin_used <= 0 and leverage > 0:
                margin_used = notional / leverage
            pnl_percent = (
                (mark_price - entry_price) / entry_price * 100
                if direction == "LONG"
                else (entry_price - mark_price) / entry_price * 100
            )

            trade_id = None
            stop_loss = None
            take_profit = None
            closest = None
            for ap in agent_positions:
                if ap.get("direction") != direction:
                    continue
                diff = abs(_safe_float(ap.get("entry_price")) - entry_price)
                if closest is None or diff < closest["diff"]:
                    closest = {"diff": diff, "pos": ap}
            if closest and closest["diff"] < 50:
                ap = closest["pos"]
                trade_id = ap.get("trade_id")
                stop_loss = ap.get("stop_loss")
                take_profit = ap.get("take_profit")

            if stop_loss is None or take_profit is None:
                sl_pct = 0.003
                tp_pct = 0.035
                if direction == "LONG":
                    stop_loss = entry_price * (1 - sl_pct)
                    take_profit = entry_price * (1 + tp_pct)
                else:
                    stop_loss = entry_price * (1 + sl_pct)
                    take_profit = entry_price * (1 - tp_pct)

            result.append(
                {
                    "symbol": p["symbol"],
                    "tradeId": trade_id,
                    "side": direction,
                    "amount": abs(amt),
                    "entryPrice": entry_price,
                    "markPrice": mark_price,
                    "pnl": pnl,
                    "pnlPercent": round(pnl_percent, 2),
                    "leverage": leverage,
                    "notional": round(notional, 2),
                    "marginUsed": round(margin_used, 4),
                    "stopLoss": stop_loss,
                    "takeProfit": take_profit,
                "liquidationPrice": _safe_float(p.get("liquidationPrice")),
                    "timestampOpen": None,
                }
            )
        agent_entries = []
        for ap in agent_positions:
            entry_price = _safe_float(ap.get("entry_price"))
            qty = _safe_float(ap.get("quantity"))
            direction = ap.get("direction")
            mark_price = ticker_price or entry_price
            pnl = 0.0
            pnl_percent = 0.0
            if entry_price > 0 and mark_price:
                if direction == "LONG":
                    pnl = (mark_price - entry_price) * qty
                    pnl_percent = (mark_price - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - mark_price) * qty
                    pnl_percent = (entry_price - mark_price) / entry_price * 100
            leverage = int(ap.get("leverage", 10))
            notional = abs(qty) * mark_price if mark_price else 0.0
            margin_used = _safe_float(ap.get("margin_used") or 0)
            if margin_used <= 0 and leverage > 0:
                margin_used = notional / leverage
            closest_exchange = None
            for ex in active:
                ex_direction = "LONG" if _safe_float(ex.get("positionAmt")) > 0 else "SHORT"
                if ex_direction != direction:
                    continue
                diff = abs(_safe_float(ex.get("entryPrice")) - entry_price)
                if closest_exchange is None or diff < closest_exchange["diff"]:
                    closest_exchange = {"diff": diff, "pos": ex}
            if closest_exchange and closest_exchange["diff"] < 50:
                leverage = int(closest_exchange["pos"].get("leverage", leverage))
            agent_entries.append(
                {
                    "tradeId": ap.get("trade_id"),
                    "side": direction,
                    "amount": qty,
                    "entryPrice": entry_price,
                    "markPrice": mark_price,
                    "pnl": pnl,
                    "pnlPercent": round(pnl_percent, 2),
                    "leverage": leverage,
                    "notional": round(notional, 2),
                    "marginUsed": round(margin_used, 4),
                    "stopLoss": ap.get("stop_loss"),
                    "takeProfit": ap.get("take_profit"),
                    "liquidationPrice": None,
                    "timestampOpen": ap.get("timestamp_open"),
                    "source": "external" if ap.get("external") else "agent",
                }
            )
        exchange_qty = sum(abs(_safe_float(p.get("positionAmt"))) for p in active)
        agent_qty = sum(_safe_float(p.get("quantity")) for p in agent_positions)
        qty_diff = exchange_qty - agent_qty
        pct_diff = (qty_diff / exchange_qty * 100) if exchange_qty else 0.0
        # 禁用总量不一致提醒：仅展示当前持仓数据
        has_mismatch = False
        note = ""
        exchange_margin_used = sum(_safe_float(p.get("marginUsed") or 0) for p in result)
        agent_margin_used = sum(_safe_float(p.get("marginUsed") or 0) for p in agent_entries)
        summary = {
            "exchange_qty": round(exchange_qty, 6),
            "agent_qty": round(agent_qty, 6),
            "qty_diff": round(qty_diff, 6),
            "pct_diff": round(pct_diff, 2),
            "has_mismatch": has_mismatch,
            "note": note,
            "exchange_margin_used": round(exchange_margin_used, 4),
            "agent_margin_used": round(agent_margin_used, 4),
        }
        return jsonify(
            {"positions": result, "agent_entries": agent_entries, "summary": summary}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/klines_all/<symbol>")
def klines_all(symbol):
    try:
        intervals = ["1m", "15m", "8h", "1w"]
        limits = {"1m": 500, "15m": 300, "8h": 150, "1w": 100}
        result = {}
        for interval in intervals:
            data = get_mainnet_klines(symbol, interval, limits.get(interval, 200))
            if not isinstance(data, list) or len(data) == 0:
                continue
            candles = []
            volumes = []
            for k in data:
                if not isinstance(k, (list, tuple)) or len(k) < 6:
                    continue
                try:
                    ts = int(k[0]) // 1000
                except (TypeError, ValueError):
                    continue
                o, h, l, c = (
                    _safe_float(k[1]),
                    _safe_float(k[2]),
                    _safe_float(k[3]),
                    _safe_float(k[4]),
                )
                vol = _safe_float(k[5])
                if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                    continue
                candles.append({"time": ts, "open": o, "high": h, "low": l, "close": c})
                volumes.append(
                    {
                        "time": ts,
                        "value": vol,
                        "color": "#26a69a" if c >= o else "#ef5350",
                    }
                )
            if candles:
                result[interval] = {"candles": candles, "volumes": volumes}
        if not result:
            return jsonify({"error": "No data"}), 400
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/agent/start", methods=["POST"])
def start_agent():
    if agent_state["running"]:
        return jsonify({"error": "Agent already running"}), 400
    agent_state["running"] = True
    agent_state["logs"].clear()
    agent_state["last_stop_reason"] = None
    add_log("Starting agent")
    thread = threading.Thread(target=run_agent_loop_with_restart, daemon=True)
    thread.start()
    agent_state["thread"] = thread
    return jsonify({"success": True, "message": "Agent started"})


@app.route("/api/agent/stop", methods=["POST"])
def stop_agent():
    if not agent_state["running"]:
        return jsonify({"error": "Agent not running"}), 400
    agent_state["running"] = False
    agent_state["last_stop_reason"] = "manual_stop"
    add_log("Stopping agent")
    return jsonify({"success": True, "message": "Agent stopped"})


@app.route("/api/agent/status")
def agent_status():
    agent = agent_state.get("agent")
    status = {
        "running": agent_state["running"],
        "last_update": agent_state["last_update"],
        "last_stop_reason": agent_state.get("last_stop_reason"),
    }
    if agent:
        stats = agent.trade_logger.get_stats()
        learning = agent.level_finder.get_learning_progress()
        status.update(
            {
                "generation": agent.knowledge.get_generation(),
                "total_trades": stats.get("total_trades", 0),
                "win_rate": stats.get("win_rate", 0),
                "total_pnl": stats.get("total_pnl", 0),
                "profit_factor": stats.get("profit_factor", 0),
                "best_support": agent.best_support,
                "best_resistance": agent.best_resistance,
                "learning": learning,
                "ai_logic": agent.get_ai_logic(),
            }
        )
    else:
        try:
            from rl.market_analysis.level_finder import BestLevelFinder

            level_finder = BestLevelFinder(os.path.join(RL_DATA_DIR, "level_stats.json"))
            status["learning"] = level_finder.get_learning_progress()
        except Exception:
            pass
    return jsonify(status)


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "GET":
        return jsonify(agent_state.get("config", {}))

    data = request.get_json(silent=True) or {}
    config = agent_state.get("config", {})

    if "near_sr_threshold_pct" in data:
        try:
            val = float(data.get("near_sr_threshold_pct"))
            # 合理范围：0.01% - 0.50%
            if val < 0.01:
                val = 0.01
            if val > 0.5:
                val = 0.5
            config["near_sr_threshold_pct"] = val
        except Exception:
            pass

    agent_state["config"] = config
    agent = agent_state.get("agent")
    if agent is not None:
        agent.near_sr_threshold_pct = float(config.get("near_sr_threshold_pct", 0.1) or 0.1)

    return jsonify(config)


@app.route("/api/agent/logs")
def agent_logs():
    return jsonify({"logs": list(agent_state["logs"])})


@app.route("/api/agent/levels")
def agent_levels():
    agent = agent_state.get("agent")
    if agent:
        return jsonify(
            {
                "best_support": agent.best_support,
                "best_resistance": agent.best_resistance,
                "score_records": agent.last_level_scores,
                "weights_display": agent.level_finder.get_weights_display(),
                "learning": agent.level_finder.get_learning_progress(),
                "stats_summary": agent.level_finder.get_stats_summary(),
            }
        )
    return jsonify({"best_support": None, "best_resistance": None})


@app.route("/api/agent/patterns")
def agent_patterns():
    """K线形态统计API"""
    agent = agent_state.get("agent")
    if agent:
        return jsonify(agent.pattern_detector.get_stats())
    return jsonify({"long": [], "short": [], "total": {"count": 0, "pnl": 0.0}})


@app.route("/api/agent/learning")
def agent_learning():
    """决策特征学习API"""
    agent = agent_state.get("agent")
    if agent:
        return jsonify({
            "weights": agent.decision_learner.get_weights(),
            "weights_cn": agent.decision_learner.get_feature_names_cn(),
            "history": agent.decision_learner.get_history(),
        })
    return jsonify({"weights": {}, "history": []})


@app.route("/api/agent/trades")
def agent_trades():
    agent = agent_state.get("agent")
    trades = []
    if agent:
        trades = agent.trade_logger.get_recent(50)
    else:
        try:
            from rl.core.knowledge import TradeLogger

            trades = TradeLogger(os.path.join(RL_DATA_DIR, "trades.db")).get_recent(50)
        except Exception:
            trades = []

    active_ids = set()
    if agent and agent.positions:
        active_ids = {p.get("trade_id") for p in agent.positions if p.get("trade_id")}
    else:
        pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
        if os.path.exists(pos_file):
            try:
                with open(pos_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    active_ids = {
                        p.get("trade_id")
                        for p in data.get("positions", [])
                        if p.get("trade_id")
                    }
            except Exception:
                active_ids = set()

    formatted = []
    for t in trades:
        trade_id = t.get("trade_id")
        if isinstance(trade_id, str) and trade_id.startswith("EXTERNAL"):
            continue
        # Parse patterns from JSON string if stored
        patterns_data = t.get("patterns", [])
        if isinstance(patterns_data, str):
            try:
                patterns_data = json.loads(patterns_data)
            except:
                patterns_data = []
        
        formatted.append(
            {
                "trade_id": t.get("trade_id"),
                "direction": t.get("direction"),
                "entry_price": _safe_float(t.get("entry_price")),
                "exit_price": _safe_float(t.get("exit_price")) if t.get("exit_price") is not None else None,
                "quantity": _safe_float(t.get("quantity")),
                "leverage": int(t.get("leverage", 10)),
                "pnl": _safe_float(t.get("pnl")),
                "pnl_percent": _safe_float(t.get("pnl_percent")),
                "raw_pnl": _safe_float(t.get("raw_pnl")),
                "commission": _safe_float(t.get("commission")),
                "exit_reason": t.get("exit_reason", ""),
                "entry_time": t.get("timestamp_open"),
                "exit_time": t.get("timestamp_close"),
                "is_active": t.get("trade_id") in active_ids,
                "patterns": patterns_data,
            }
        )
    # Add active positions as trades (for markers)
    active_positions = []
    current_price = 0
    if agent and agent.positions:
        active_positions = [p for p in agent.positions if not p.get("external")]
        if agent.last_market:
            current_price = agent.last_market.get("current_price", 0) or 0
    else:
        # Read from file when agent is stopped
        pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
        if os.path.exists(pos_file):
            try:
                with open(pos_file, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    active_positions = [p for p in file_data.get("positions", []) if not p.get("external")]
            except Exception:
                active_positions = []
        # Get current price from API
        try:
            ticker = client.get_ticker_price("BTCUSDT") if client else None
            if ticker:
                current_price = _safe_float(ticker.get("price"))
        except Exception:
            current_price = 0

    existing_trade_ids = {t.get("trade_id") for t in formatted}
    for pos in active_positions:
        # Skip if already in formatted from trade_logger
        if pos.get("trade_id") in existing_trade_ids:
            continue
        entry_price = _safe_float(pos.get("entry_price"))
        pnl = 0.0
        pnl_percent = 0.0
        if current_price and entry_price:
            if pos.get("direction") == "LONG":
                pnl = (current_price - entry_price) * _safe_float(pos.get("quantity"))
                pnl_percent = (current_price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - current_price) * _safe_float(pos.get("quantity"))
                pnl_percent = (entry_price - current_price) / entry_price * 100
        formatted.append(
            {
                "trade_id": pos.get("trade_id"),
                "direction": pos.get("direction"),
                "entry_price": entry_price,
                "exit_price": None,
                "quantity": _safe_float(pos.get("quantity")),
                "leverage": int(pos.get("leverage", 10)),
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "exit_reason": "",
                "entry_time": pos.get("timestamp_open"),
                "exit_time": None,
                "is_active": True,
            }
        )
    return jsonify({"trades": formatted})


@app.route("/api/daily_report")
def daily_report():
    date_str = request.args.get("date")
    if not date_str:
        date_str = datetime.now().date().isoformat()

    db_path = os.path.join(RL_DATA_DIR, "trades.db")
    if not os.path.exists(db_path):
        return jsonify({"date": date_str, "has_data": False, "summary": {}, "loss_lessons": []})

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in c.fetchall()}
        has_commission = "commission" in columns
        has_patterns = "patterns" in columns
        has_market_state = "market_state" in columns
        has_exit_reason = "exit_reason" in columns

        commission_expr = "commission" if has_commission else "0"
        patterns_expr = "patterns" if has_patterns else "''"
        market_state_expr = "market_state" if has_market_state else "''"
        exit_reason_expr = "exit_reason" if has_exit_reason else "''"

        c.execute(
            f"""
            SELECT
                trade_id,
                pnl,
                pnl_percent,
                {commission_expr} AS commission,
                {exit_reason_expr} AS exit_reason,
                {market_state_expr} AS market_state,
                {patterns_expr} AS patterns,
                timestamp_open,
                timestamp_close
            FROM trades
            WHERE (timestamp_close LIKE ?)
               OR ((timestamp_close IS NULL OR length(timestamp_close) = 0) AND (timestamp_open LIKE ?))
            """,
            (f"{date_str}%", f"{date_str}%"),
        )
        rows = c.fetchall()
    finally:
        conn.close()

    if not rows:
        return jsonify({"date": date_str, "has_data": False, "summary": {}, "loss_lessons": []})

    trades = []
    for r in rows:
        trades.append(
            {
                "trade_id": r[0],
                "pnl": _safe_float(r[1]),
                "pnl_percent": _safe_float(r[2]),
                "commission": _safe_float(r[3]),
                "exit_reason": r[4] or "",
                "market_state": r[5] or "",
                "patterns": r[6] or "",
                "timestamp_open": r[7],
                "timestamp_close": r[8],
            }
        )

    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gross_pnl = sum(t["pnl"] for t in trades)
    total_commission = sum(t["commission"] for t in trades)
    net_pnl = gross_pnl - total_commission
    win_rate = (len(wins) / total_trades) * 100 if total_trades else 0
    avg_pnl = gross_pnl / total_trades if total_trades else 0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    fee_ratio = (total_commission / abs(gross_pnl)) if gross_pnl != 0 else 0

    def _top_items(items, top_n=3):
        items.sort(key=lambda x: (-x["count"], x["avg_pnl"]))
        return items[:top_n]

    loss_lessons = []
    if losses:
        reason_map = {}
        for t in losses:
            key = t.get("exit_reason") or "UNKNOWN"
            data = reason_map.setdefault(key, {"count": 0, "pnl_sum": 0.0})
            data["count"] += 1
            data["pnl_sum"] += t["pnl"]
        exit_items = [
            {"name": k, "count": v["count"], "avg_pnl": v["pnl_sum"] / v["count"]}
            for k, v in reason_map.items()
        ]
        loss_lessons.append({"type": "exit_reason", "items": _top_items(exit_items)})

        state_map = {}
        for t in losses:
            key = t.get("market_state") or "UNKNOWN"
            data = state_map.setdefault(key, {"count": 0, "pnl_sum": 0.0})
            data["count"] += 1
            data["pnl_sum"] += t["pnl"]
        state_items = [
            {"name": k, "count": v["count"], "avg_pnl": v["pnl_sum"] / v["count"]}
            for k, v in state_map.items()
        ]
        loss_lessons.append({"type": "market_state", "items": _top_items(state_items)})

        pattern_map = {}
        for t in losses:
            raw_patterns = t.get("patterns") or ""
            patterns = []
            if isinstance(raw_patterns, str):
                try:
                    patterns = json.loads(raw_patterns)
                except Exception:
                    patterns = []
            elif isinstance(raw_patterns, list):
                patterns = raw_patterns
            for p in patterns:
                name = p.get("name") if isinstance(p, dict) else p
                if not name:
                    continue
                data = pattern_map.setdefault(name, {"count": 0, "pnl_sum": 0.0})
                data["count"] += 1
                data["pnl_sum"] += t["pnl"]
        if pattern_map:
            pattern_items = [
                {"name": k, "count": v["count"], "avg_pnl": v["pnl_sum"] / v["count"]}
                for k, v in pattern_map.items()
            ]
            loss_lessons.append({"type": "pattern", "items": _top_items(pattern_items)})

    summary = {
        "date": date_str,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "gross_pnl": round(gross_pnl, 4),
        "total_commission": round(total_commission, 4),
        "net_pnl": round(net_pnl, 4),
        "avg_pnl": round(avg_pnl, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "fee_ratio": round(fee_ratio, 4),
        "net_gt_fee": net_pnl > total_commission,
    }

    return jsonify(
        {
            "date": date_str,
            "has_data": True,
            "summary": summary,
            "loss_lessons": loss_lessons,
        }
    )


@app.route("/api/close", methods=["POST"])
def close_position():
    client = get_client()
    if not client:
        return jsonify({"error": "API keys not configured"}), 400
    data = request.json or {}
    try:
        side = "SELL" if data.get("side") == "LONG" else "BUY"
        order = client.place_order(
            symbol=data.get("symbol", "BTCUSDT"),
            side=side,
            order_type="MARKET",
            quantity=_safe_float(data.get("quantity")),
            reduce_only=True,
        )
        # 同步AI持仓与交易记录
        agent = agent_state.get("agent")
        trade_id = data.get("tradeId")
        if agent and trade_id:
            price = None
            try:
                price = _safe_float(client.get_ticker_price("BTCUSDT").get("price"))
            except Exception:
                price = None
            for pos in list(agent.positions):
                if pos.get("trade_id") != trade_id:
                    continue
                exit_price = price or pos.get("entry_price", 0)
                trade = agent.execute_exit_position(
                    pos, exit_price, "MANUAL_CLOSE", ["manual_close"], skip_api=True
                )
                if trade:
                    add_log(
                        f"手动平仓 {trade['trade_id'][:8]} PnL={trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)",
                        "INFO",
                    )
                break
        return jsonify({"success": True, "order": order})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/close_all", methods=["POST"])
def close_all_positions():
    client = get_client()
    if not client:
        return jsonify({"error": "API keys not configured"}), 400
    try:
        positions = client.get_positions()
        if not isinstance(positions, list):
            return jsonify({"error": "Invalid positions response"}), 400
        active = [p for p in positions if abs(_safe_float(p.get("positionAmt"))) > 0]
        closed = 0
        
        # Close exchange positions
        if active:
            for p in active:
                amt = _safe_float(p.get("positionAmt"))
                side = "SELL" if amt > 0 else "BUY"
                try:
                    client.place_order(
                        symbol=p["symbol"],
                        side=side,
                        order_type="MARKET",
                        quantity=abs(amt),
                        reduce_only=True,
                    )
                    closed += 1
                except Exception as e:
                    add_log(f"平仓失败: {str(e)}", "ERROR")
        
        # Get current price for logging
        price = None
        try:
            price = _safe_float(client.get_ticker_price("BTCUSDT").get("price"))
        except Exception:
            pass
        
        # Clear ALL AI positions and log them
        agent = agent_state.get("agent")
        logged_count = 0
        
        # Get positions from agent or file
        agent_positions = []
        if agent and agent.positions:
            agent_positions = list(agent.positions)
        else:
            pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
            if os.path.exists(pos_file):
                try:
                    with open(pos_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        agent_positions = data.get("positions", [])
                except Exception:
                    agent_positions = []
        
        # Log and clear all AI positions
        for pos in agent_positions:
            exit_price = price or pos.get("entry_price", 0)
            if agent:
                try:
                    trade = agent.execute_exit_position(
                        pos, exit_price, "MANUAL_CLOSE_ALL", ["manual_close_all"], skip_order=True
                    )
                    if trade:
                        logged_count += 1
                        add_log(
                            f"一键清仓记录 {trade['trade_id'][:8]} PnL={trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)",
                            "INFO",
                        )
                except Exception as e:
                    add_log(f"记录清仓失败 {pos.get('trade_id', 'unknown')}: {str(e)}", "ERROR")
        
        # Force clear all positions from memory and file
        if agent and hasattr(agent, "positions"):
            agent.positions = []
            try:
                agent._save_positions()
            except Exception:
                pass
        
        # Clear file
        pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
        try:
            with open(pos_file, "w", encoding="utf-8") as f:
                json.dump({"positions": []}, f, indent=2)
        except Exception:
            pass
        
        add_log(f"一键清仓完成: 交易所平仓{closed}笔, AI记录清理{logged_count}笔", "INFO")
        
        return jsonify({
            "success": True,
            "closed": closed,
            "logged": logged_count,
            "message": f"已平仓{closed}笔, 已清理{logged_count}条AI记录"
        })
    except Exception as exc:
        add_log(f"一键清仓错误: {str(exc)}", "ERROR")
        return jsonify({"error": str(exc)}), 400


@app.route("/api/commission")
def get_commission():
    """获取手续费记录"""
    client = get_client()
    if not client:
        return jsonify({"error": "API keys not configured"}), 400
    try:
        # Get recent trades with commission info
        trades = client.get_account_trades("BTCUSDT", limit=50)
        if not isinstance(trades, list):
            trades = []
        
        # Get commission income history
        income = client.get_income_history(income_type="COMMISSION", limit=50)
        if not isinstance(income, list):
            income = []
        
        # Calculate totals
        total_commission = sum(abs(_safe_float(t.get("commission"))) for t in trades)
        commission_asset = trades[0].get("commissionAsset", "USDT") if trades else "USDT"
        
        # Format recent trades
        recent = []
        for t in trades[:20]:
            recent.append({
                "time": t.get("time"),
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "price": _safe_float(t.get("price")),
                "qty": _safe_float(t.get("qty")),
                "commission": _safe_float(t.get("commission")),
                "commissionAsset": t.get("commissionAsset", "USDT"),
                "realizedPnl": _safe_float(t.get("realizedPnl")),
            })
        
        return jsonify({
            "total_commission": round(total_commission, 4),
            "commission_asset": commission_asset,
            "recent_trades": recent,
            "income_history": income[:20],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/sync_positions", methods=["POST"])
def sync_positions():
    """
    Force sync positions between exchange and AI records.
    This will:
    1. Remove orphaned AI entries (AI > Exchange)
    2. Create external entries (Exchange > AI)
    """
    client = get_client()
    if not client:
        return jsonify({"error": "API keys not configured"}), 400
    
    try:
        # Get exchange positions
        binance_positions = client.get_positions()
        if not isinstance(binance_positions, list):
            return jsonify({"error": "Invalid positions response"}), 400
        active = [
            p for p in binance_positions if abs(_safe_float(p.get("positionAmt"))) > 0
        ]
        
        # Get AI positions
        agent = agent_state.get("agent")
        agent_positions = []
        if agent and agent.positions:
            agent_positions = list(agent.positions)
        else:
            pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
            if os.path.exists(pos_file):
                try:
                    with open(pos_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        agent_positions = data.get("positions", [])
                except Exception:
                    agent_positions = []
        
        # Calculate before sync
        exchange_long = sum(
            abs(_safe_float(p.get("positionAmt")))
            for p in active
            if _safe_float(p.get("positionAmt")) > 0
        )
        exchange_short = sum(
            abs(_safe_float(p.get("positionAmt")))
            for p in active
            if _safe_float(p.get("positionAmt")) < 0
        )
        exchange_total = exchange_long + exchange_short
        
        agent_long = sum(_safe_float(p.get("quantity")) for p in agent_positions if p.get("direction") == "LONG")
        agent_short = sum(_safe_float(p.get("quantity")) for p in agent_positions if p.get("direction") == "SHORT")
        agent_total = agent_long + agent_short
        
        before_diff = exchange_total - agent_total
        
        # Sync
        def _sync_external_positions(active_positions, agent_positions_list, agent_obj):
            """External sync disabled: remove any existing external entries only."""
            cleaned = [p for p in agent_positions_list if not p.get("external")]
            removed_count = len(agent_positions_list) - len(cleaned)
            if removed_count > 0:
                agent_positions_list[:] = cleaned
                add_log("已移除EXTERNAL持仓同步记录", "INFO")
                if agent_obj and getattr(agent_obj, "positions", None) is not None:
                    agent_obj.positions = list(agent_positions_list)
                    try:
                        agent_obj._save_positions()
                    except Exception:
                        pass
                else:
                    try:
                        pos_file = os.path.join(RL_DATA_DIR, "active_positions.json")
                        with open(pos_file, "w", encoding="utf-8") as f:
                            json.dump({"positions": agent_positions_list}, f, indent=2)
                    except Exception:
                        pass
            return agent_positions_list, removed_count
        
        synced_positions, removed = _sync_external_positions(active, agent_positions, agent)
        
        # Calculate after sync
        agent_long_after = sum(
            _safe_float(p.get("quantity"))
            for p in synced_positions
            if p.get("direction") == "LONG"
        )
        agent_short_after = sum(
            _safe_float(p.get("quantity"))
            for p in synced_positions
            if p.get("direction") == "SHORT"
        )
        agent_total_after = agent_long_after + agent_short_after
        after_diff = exchange_total - agent_total_after
        
        add_log(f"强制同步完成: 删除{removed}条多余记录, 差值从{before_diff:.4f}调整为{after_diff:.4f}", "INFO")
        
        return jsonify({
            "success": True,
            "removed": removed,
            "before": {
                "exchange": round(exchange_total, 4),
                "agent": round(agent_total, 4),
                "diff": round(before_diff, 4),
            },
            "after": {
                "exchange": round(exchange_total, 4),
                "agent": round(agent_total_after, 4),
                "diff": round(after_diff, 4),
            },
        })
    except Exception as exc:
        add_log(f"同步失败: {str(exc)}", "ERROR")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    init_db()
    # 禁用自动重启，避免开发模式下反复重载导致“假崩溃”
    app.run(debug=False, use_reloader=False, port=5000)

