"""
出场决策反向传播学习器 - ExitDecisionLearner

核心思想：
- 每笔交易后，根据出场时机质量调整出场特征权重
- 好的出场时机（接近最佳点）→ 奖励促成该出场的特征
- 差的出场时机（过早或过晚）→ 惩罚促成该出场的特征
- 特征影响下次出场决策

学习目标：
- 最大化出场价格相对于入场后最优价格的比例
- 最小化持仓时间与盈利的比值（时间效率）
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class ExitDecisionLearner:
    """
    出场决策反向传播学习器
    
    特征维度：
    1. profit_momentum: 盈利动量（盈利速度）
    2. sr_proximity: 支撑阻力接近度
    3. reversal_signal: 反转信号强度
    4. time_pressure: 时间压力
    5. trend_exhaustion: 趋势衰竭度
    6. volume_divergence: 成交量背离
    """
    
    FEATURE_NAMES_CN = {
        "profit_momentum": "盈利动量",
        "sr_proximity": "支撑阻力接近度",
        "reversal_signal": "反转信号强度",
        "time_pressure": "时间压力",
        "trend_exhaustion": "趋势衰竭度",
        "volume_divergence": "成交量背离",
    }
    
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "exit_decision_weights.json")
        os.makedirs(data_dir, exist_ok=True)
        self.weights = self._load()
        self.history = []
        self.momentum = {}  # 梯度动量
        
    def _default_weights(self) -> Dict[str, float]:
        """初始权重（均匀分布）"""
        return {
            "profit_momentum": 0.20,
            "sr_proximity": 0.25,
            "reversal_signal": 0.20,
            "time_pressure": 0.10,
            "trend_exhaustion": 0.15,
            "volume_divergence": 0.10,
        }
    
    def _load(self) -> Dict[str, float]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                    self.momentum = data.get("momentum", {})
                    weights = data.get("weights", {})
                    merged = self._default_weights()
                    for k, v in weights.items():
                        if k in merged:
                            merged[k] = float(v)
                    # 归一化
                    total = sum(abs(v) for v in merged.values()) or 1.0
                    return {k: v / total for k, v in merged.items()}
            except:
                pass
        return self._default_weights()
    
    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({
                    "weights": self.weights,
                    "history": self.history[-100:],
                    "momentum": self.momentum,
                }, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def extract_features(self, position: Dict, market: Dict, current_price: float) -> Dict[str, float]:
        """
        提取出场决策特征
        
        输入：
        - position: 当前持仓信息
        - market: 市场分析数据
        - current_price: 当前价格
        
        输出：
        - 6个归一化特征 [-1, 1]
        """
        direction = position.get("direction", "LONG")
        entry_price = position.get("entry_price", current_price)
        
        # 计算当前盈亏
        if direction == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        # ========== 特征1: 盈利动量 ==========
        # 基于盈亏百分比，范围[-1, 1]
        profit_momentum = max(-1.0, min(1.0, pnl_pct / 2.0))
        
        # ========== 特征2: 支撑阻力接近度 ==========
        sr_proximity = 0.0
        best_support = market.get("best_support")
        best_resistance = market.get("best_resistance")
        
        if direction == "LONG" and best_resistance:
            # 做多时，接近阻力位 = 应该出场
            dist = (best_resistance["price"] - current_price) / current_price * 100
            sr_proximity = max(0.0, 1 - dist / 0.5)  # 距离<0.5%时分值高
        elif direction == "SHORT" and best_support:
            # 做空时，接近支撑位 = 应该出场
            dist = (current_price - best_support["price"]) / current_price * 100
            sr_proximity = max(0.0, 1 - dist / 0.5)
        
        # ========== 特征3: 反转信号强度 ==========
        reversal_signal = 0.0
        analysis_15m = market.get("analysis_15m", {})
        macd = analysis_15m.get("macd_histogram", 0)
        rsi = analysis_15m.get("rsi", 50)
        
        if direction == "LONG":
            # 做多时，MACD转负/RSI超买 = 反转信号
            if macd < 0:
                reversal_signal += 0.5
            if rsi > 70:
                reversal_signal += 0.5
        else:
            # 做空时，MACD转正/RSI超卖 = 反转信号
            if macd > 0:
                reversal_signal += 0.5
            if rsi < 30:
                reversal_signal += 0.5
        
        reversal_signal = min(1.0, reversal_signal)
        
        # ========== 特征4: 时间压力 ==========
        time_pressure = 0.0
        ts_open = position.get("timestamp_open")
        if ts_open:
            try:
                opened = datetime.fromisoformat(ts_open)
                hold_minutes = (datetime.now() - opened).total_seconds() / 60
                # 超过30分钟开始有时间压力
                time_pressure = max(0.0, min(1.0, (hold_minutes - 30) / 30))
            except:
                pass
        
        # ========== 特征5: 趋势衰竭度 ==========
        trend_exhaustion = 0.0
        micro_trend = market.get("micro_trend", {}).get("direction")
        macro_trend = market.get("macro_trend", {}).get("direction")
        
        if direction == "LONG":
            # 做多时，趋势转弱 = 衰竭
            if micro_trend == "BEARISH":
                trend_exhaustion += 0.5
            if macro_trend == "BEARISH":
                trend_exhaustion += 0.5
        else:
            if micro_trend == "BULLISH":
                trend_exhaustion += 0.5
            if macro_trend == "BULLISH":
                trend_exhaustion += 0.5
        
        trend_exhaustion = min(1.0, trend_exhaustion)
        
        # ========== 特征6: 成交量背离 ==========
        volume_divergence = 0.0
        klines_1m = market.get("klines_1m", [])
        if len(klines_1m) >= 10:
            recent_vol = sum(k.get("volume", 0) for k in klines_1m[-5:]) / 5
            avg_vol = sum(k.get("volume", 0) for k in klines_1m[-20:]) / max(1, len(klines_1m[-20:]))
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                # 成交量萎缩 = 趋势可能结束
                if vol_ratio < 0.7:
                    volume_divergence = (0.7 - vol_ratio) / 0.7
        
        return {
            "profit_momentum": round(profit_momentum, 4),
            "sr_proximity": round(sr_proximity, 4),
            "reversal_signal": round(reversal_signal, 4),
            "time_pressure": round(time_pressure, 4),
            "trend_exhaustion": round(trend_exhaustion, 4),
            "volume_divergence": round(volume_divergence, 4),
        }
    
    def score(self, features: Dict[str, float]) -> float:
        """
        计算出场决策分数
        
        返回：0-100分，分数越高越应该出场
        """
        weighted_sum = 0.0
        for key, value in features.items():
            weight = self.weights.get(key, 0.0)
            weighted_sum += value * weight
        
        # 权重和约为1，特征范围[0, 1]，所以weighted_sum范围约[0, 1]
        # 放大到0-100
        return max(0, min(100, weighted_sum * 100))
    
    def should_exit(self, features: Dict[str, float], threshold: float = 60.0) -> bool:
        """
        基于学习的特征权重判断是否应该出场
        """
        return self.score(features) >= threshold
    
    def update(
        self,
        features: Dict[str, float],
        exit_quality: float,
        pnl_percent: float,
        hold_minutes: float = None,
        fee_pct: float = 0.12,
    ) -> Dict:
        """
        反向传播更新：根据出场质量调整权重
        
        exit_quality: 出场时机质量 [-1, 1]
            - +1: 完美出场（接近最高点/最低点）
            - 0: 一般出场
            - -1: 糟糕出场（过早或过晚）
        
        pnl_percent: 盈亏百分比（用于加权学习）
        """
        # 学习率
        lr = 0.12
        
        # 奖励信号：结合出场质量和盈亏
        # 盈利+好时机 = 强奖励
        # 盈利+差时机 = 弱奖励（时机不好但还是赚了）
        # 亏损+好时机 = 弱惩罚（时机好但方向错）
        # 亏损+差时机 = 强惩罚
        pnl_percent = float(pnl_percent or 0.0)
        fee_pct = float(fee_pct or 0.0)
        time_penalty = 0.0
        if hold_minutes is not None:
            time_penalty = max(0.0, hold_minutes - 15) * 0.01
        net_pnl = pnl_percent - fee_pct - time_penalty
        pnl_signal = max(-1.0, min(1.0, net_pnl / 2.0))
        combined_reward = exit_quality * 0.6 + pnl_signal * 0.4
        
        before = self.weights.copy()
        beta = 0.9  # 动量系数
        
        # ========== 反向传播更新 ==========
        for key in self.weights:
            feature_value = features.get(key, 0.0)
            
            # 梯度：如果特征值高且奖励为正，增加权重
            gradient = combined_reward * feature_value
            
            # 应用动量
            prev_momentum = self.momentum.get(key, 0)
            new_momentum = beta * prev_momentum + (1 - beta) * gradient
            self.momentum[key] = new_momentum
            
            # 更新权重
            self.weights[key] += lr * new_momentum
            self.weights[key] = max(-1.0, min(1.0, self.weights[key]))
        
        # 归一化（保持权重和为正）
        total_abs = sum(abs(v) for v in self.weights.values()) or 1.0
        for k in self.weights:
            self.weights[k] = self.weights[k] / total_abs
        
        # 记录历史
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "exit_quality": round(exit_quality, 4),
            "pnl_percent": round(pnl_percent, 4),
            "net_pnl": round(net_pnl, 4),
            "time_penalty": round(time_penalty, 4),
            "fee_pct": round(fee_pct, 4),
            "combined_reward": round(combined_reward, 4),
            "features": {k: round(v, 3) for k, v in features.items()},
            "weights_after": {k: round(v, 4) for k, v in self.weights.items()},
        })
        
        self.save()
        
        delta = {k: round(self.weights[k] - before[k], 4) for k in self.weights}
        
        return {
            "before": before,
            "after": self.weights.copy(),
            "delta": {k: v for k, v in delta.items() if abs(v) > 0.001},
            "combined_reward": round(combined_reward, 4),
            "exit_quality": round(exit_quality, 4),
        }
    
    def get_weights(self) -> Dict[str, float]:
        return self.weights
    
    def get_history(self) -> List[Dict]:
        return self.history[-50:]
    
    def get_feature_names_cn(self) -> Dict[str, str]:
        return self.FEATURE_NAMES_CN
    
    def get_exit_suggestion(
        self,
        position: Dict,
        market: Dict,
        current_price: float,
        threshold: float = 60.0,
        features_override: Dict[str, float] = None,
    ) -> Dict:
        """
        获取出场建议
        
        返回：
        - should_exit: 是否应该出场
        - score: 出场分数 (0-100)
        - features: 各特征值
        - reason: 主要原因
        """
        features = features_override or self.extract_features(position, market, current_price)
        score = self.score(features)
        
        # 找出最高权重的特征作为主要原因
        weighted_features = [(k, features[k] * self.weights.get(k, 0)) for k in features]
        weighted_features.sort(key=lambda x: x[1], reverse=True)
        top_feature = weighted_features[0][0] if weighted_features else "unknown"
        
        return {
            "should_exit": score >= threshold,
            "score": round(score, 1),
            "features": features,
            "top_reason": self.FEATURE_NAMES_CN.get(top_feature, top_feature),
            "threshold": threshold,
        }


