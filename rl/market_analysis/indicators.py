from typing import Dict, List


def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    ema_values = [values[0]]
    for v in values[1:]:
        ema_values.append(v * k + ema_values[-1] * (1 - k))
    return ema_values


def rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(values: List[float]) -> Dict:
    ema_fast = ema(values, 12)
    ema_slow = ema(values, 26)
    if len(ema_fast) != len(ema_slow):
        return {"macd": 0, "signal": 0, "histogram": 0}
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, 9)
    histogram = macd_line[-1] - signal_line[-1] if macd_line and signal_line else 0
    return {"macd": macd_line[-1], "signal": signal_line[-1], "histogram": histogram}


def atr(klines: List[Dict], period: int = 14) -> float:
    if len(klines) < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        high = klines[i]["high"]
        low = klines[i]["low"]
        prev_close = klines[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / period if trs else 0.0


class TechnicalAnalyzer:
    def analyze(self, klines: List[Dict]) -> Dict:
        if not klines:
            return {}
        closes = [k["close"] for k in klines]
        volumes = [k.get("volume", 0) for k in klines]

        ema_7 = ema(closes, 7)[-1]
        ema_25 = ema(closes, 25)[-1]
        ema_99 = ema(closes, 99)[-1] if len(closes) >= 99 else ema_25
        rsi_val = rsi(closes, 14)
        macd_data = macd(closes)
        atr_val = atr(klines, 14)

        volume_ratio = 0.0
        if len(volumes) >= 20:
            volume_ratio = volumes[-1] / (sum(volumes[-20:]) / 20)

        trend = "NEUTRAL"
        if ema_7 > ema_25 > ema_99:
            trend = "BULLISH"
        elif ema_7 < ema_25 < ema_99:
            trend = "BEARISH"

        return {
            "ema_7": ema_7,
            "ema_25": ema_25,
            "ema_99": ema_99,
            "rsi": rsi_val,
            "macd": macd_data["macd"],
            "macd_signal": macd_data["signal"],
            "macd_histogram": macd_data["histogram"],
            "atr": atr_val,
            "volume_ratio": volume_ratio,
            "trend": trend,
        }




