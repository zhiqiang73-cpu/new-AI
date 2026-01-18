import json
import os
from datetime import datetime
from typing import Dict, List


class DecisionFeatureLearner:
    """
    决策特征学习器 - 精选6个核心特征的自我进化系统
    
    核心思想：
    - 每笔交易后，根据盈亏调整特征权重
    - 盈利 → 奖励促成该交易的特征
    - 亏损 → 惩罚促成该交易的特征
    - 特征影响下次入场评分
    """
    
    FEATURE_NAMES_CN = {
        "sr_edge_proximity": "边界优势度",
        "profit_space_ratio": "利润空间比",
        "trend_consensus": "趋势共识度",
        "pattern_strength": "形态强度",
        "rsi_momentum": "RSI动量",
        "market_regime_score": "市场状态",
    }
    
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.weights = self._load()
        self.history = []
        
    def _default_weights(self) -> Dict[str, float]:
        """初始权重（均匀分布，让AI自己学）"""
        return {
            "sr_edge_proximity": 0.2,      # 边界优势
            "profit_space_ratio": 0.2,     # 空间比
            "trend_consensus": 0.2,        # 趋势一致性
            "pattern_strength": 0.15,      # 形态强度
            "rsi_momentum": 0.15,          # RSI动量
            "market_regime_score": 0.1,    # 市场状态
        }
    
    def _load(self) -> Dict[str, float]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                    weights = data.get("weights", {})
                    merged = self._default_weights()
                    for k, v in weights.items():
                        if k in merged:
                            merged[k] = float(v)
                    # 归一化权重（确保和为1）
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
                    "history": self.history[-100:]
                }, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def extract_features(self, market: Dict, direction: str) -> Dict[str, float]:
        """
        提取6个核心特征（全部归一化到[-1, 1]或[0, 1]）
        """
        price = market.get("current_price", 0)
        if price <= 0:
            return {k: 0.0 for k in self._default_weights()}
        
        best_support = market.get("best_support")
        best_resistance = market.get("best_resistance")
        analysis_15m = market.get("analysis_15m", {})
        regime = market.get("regime", {}).get("regime", "NORMAL")
        patterns = market.get("patterns", [])
        
        # ========== 特征1: 边界优势度 ==========
        sr_edge_proximity = 0.0
        if direction == "LONG" and best_support:
            dist = abs(price - best_support["price"]) / price * 100
            sr_edge_proximity = max(0.0, 1 - dist / 0.3)  # 距离<0.3%时，分值>0
        elif direction == "SHORT" and best_resistance:
            dist = abs(price - best_resistance["price"]) / price * 100
            sr_edge_proximity = max(0.0, 1 - dist / 0.3)
        
        # ========== 特征2: 利润空间比 ==========
        profit_space_ratio = 1.0  # 默认中性
        if best_support and best_resistance:
            support_price = float(best_support["price"])
            resistance_price = float(best_resistance["price"])
            if direction == "LONG":
                profit_space = (resistance_price - price) / price * 100
                risk_space = (price - support_price) / price * 100
            else:
                profit_space = (price - support_price) / price * 100
                risk_space = (resistance_price - price) / price * 100
            
            if risk_space > 0:
                ratio = profit_space / risk_space
                profit_space_ratio = min(1.0, ratio / 3.0)  # 3:1时满分
        
        # ========== 特征3: 趋势共识度 ==========
        macro_dir = market.get("macro_trend", {}).get("direction")
        micro_dir = market.get("micro_trend", {}).get("direction")
        analysis_1m = market.get("analysis_1m", {})
        macd_1m = analysis_1m.get("macd_histogram", 0)
        
        if direction == "LONG":
            votes = 0
            if macro_dir == "BULLISH": votes += 1
            if micro_dir == "BULLISH": votes += 1
            if macd_1m > 0: votes += 1
            trend_consensus = (votes - 1.5) / 1.5  # 归一化到[-1, 1]
        else:
            votes = 0
            if macro_dir == "BEARISH": votes += 1
            if micro_dir == "BEARISH": votes += 1
            if macd_1m < 0: votes += 1
            trend_consensus = (votes - 1.5) / 1.5
        
        # ========== 特征4: 形态强度 ==========
        pattern_strength = 0.0
        if patterns:
            my_patterns = [p for p in patterns if p.get("direction") == direction]
            if my_patterns:
                total_score = sum(p.get("score", 0) for p in my_patterns)
                pattern_strength = min(1.0, total_score / 15.0)  # 15分为满分
        
        # ========== 特征5: RSI动量 ==========
        rsi = analysis_15m.get("rsi", 50)
        rsi_momentum = (rsi - 50) / 50  # 归一化到[-1, 1]
        if direction == "SHORT":
            rsi_momentum = -rsi_momentum  # 做空时反转
        
        # ========== 特征6: 市场状态评分 ==========
        market_regime_score = 0.0
        if regime == "TRENDING":
            market_regime_score = 1.0
        elif regime == "RANGING":
            market_regime_score = 0.0
        elif regime == "VOLATILE":
            market_regime_score = -0.5
        else:
            market_regime_score = 0.5
        
        return {
            "sr_edge_proximity": round(sr_edge_proximity, 4),
            "profit_space_ratio": round(profit_space_ratio, 4),
            "trend_consensus": round(trend_consensus, 4),
            "pattern_strength": round(pattern_strength, 4),
            "rsi_momentum": round(rsi_momentum, 4),
            "market_regime_score": round(market_regime_score, 4),
        }
    
    def score(self, features: Dict[str, float]) -> float:
        """
        根据特征和权重计算额外分数
        返回：-10 到 +10（额外加分/减分）
        """
        weighted_sum = 0.0
        for key, value in features.items():
            weight = self.weights.get(key, 0.0)
            weighted_sum += value * weight
        
        # 权重和约为1，特征范围[-1, 1]，所以weighted_sum范围约[-1, 1]
        # 放大到[-10, 10]作为额外评分
        return weighted_sum * 10.0
    
    def update(
        self,
        features: Dict[str, float],
        pnl_percent: float,
        hold_minutes: float = None,
        fee_pct: float = 0.12,
    ) -> Dict:
        """
        强化学习更新：根据交易结果调整权重
        
        pnl_percent: 盈亏百分比（如 +2.5% 或 -1.0%）
        """
        # 学习率：0.12（适中，既不太激进也不太保守）
        lr = 0.12
        
        # 奖励信号：净利润 - 手续费 - 时间惩罚
        pnl_percent = float(pnl_percent or 0.0)
        fee_pct = float(fee_pct or 0.0)
        time_penalty = 0.0
        if hold_minutes is not None:
            time_penalty = max(0.0, hold_minutes - 15) * 0.01
        net_pnl = pnl_percent - fee_pct - time_penalty
        # 盈利2%时reward≈1, 亏损2%时reward≈-1
        reward = max(-1.0, min(1.0, net_pnl / 2.0))
        
        before = self.weights.copy()
        
        # ========== 反向传播：梯度下降更新 ==========
        # 如果reward为正，增强正特征的权重，削弱负特征
        # 如果reward为负，削弱正特征的权重，增强负特征
        for key in self.weights:
            feature_value = features.get(key, 0.0)
            # 梯度：reward * feature_value
            # 如果feature=1且reward=1，权重增加
            # 如果feature=1且reward=-1，权重减少
            gradient = lr * reward * feature_value
            self.weights[key] += gradient
            # 限制权重范围（避免极端值）
            self.weights[key] = max(-2.0, min(2.0, self.weights[key]))
        
        # 重新归一化（保持权重和为1的量纲）
        total_abs = sum(abs(v) for v in self.weights.values()) or 1.0
        for k in self.weights:
            self.weights[k] = self.weights[k] / total_abs
        
        # 记录学习历史
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "pnl_percent": round(pnl_percent, 4),
            "net_pnl": round(net_pnl, 4),
            "time_penalty": round(time_penalty, 4),
            "fee_pct": round(fee_pct, 4),
            "reward": round(reward, 4),
            "features_active": {k: round(v, 3) for k, v in features.items() if abs(v) > 0.1},
            "weights_after": {k: round(v, 4) for k, v in self.weights.items()},
        })
        
        self.save()
        
        delta = {k: round(self.weights[k] - before[k], 4) for k in self.weights}
        
        return {
            "before": before,
            "after": self.weights.copy(),
            "delta": {k: v for k, v in delta.items() if abs(v) > 0.001},
            "reward": round(reward, 4),
        }
    
    def get_weights(self) -> Dict[str, float]:
        return self.weights
    
    def get_history(self) -> List[Dict]:
        return self.history[-50:]
    
    def get_feature_names_cn(self) -> Dict[str, str]:
        return self.FEATURE_NAMES_CN
