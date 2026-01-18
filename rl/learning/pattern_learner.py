"""
K线形态强化学习器

功能：
1. 跟踪每个形态的出现次数、胜率、盈亏
2. 根据交易结果动态调整形态权重
3. 提供形态统计数据用于UI展示
"""
import json
import os
from typing import Dict, List, Optional


class PatternLearner:
    """K线形态强化学习器"""
    
    # 默认形态配置
    DEFAULT_PATTERNS = {
        # 做多形态
        "HAMMER": {"base": 10, "direction": "LONG", "name_cn": "锤子线"},
        "BULLISH_ENGULF": {"base": 12, "direction": "LONG", "name_cn": "看涨吞没"},
        "MORNING_STAR": {"base": 14, "direction": "LONG", "name_cn": "晨星"},
        "BIG_MARUBOZU_BULL": {"base": 8, "direction": "LONG", "name_cn": "大阳线"},
        # 做空形态
        "SHOOTING_STAR": {"base": 10, "direction": "SHORT", "name_cn": "流星线"},
        "BEARISH_ENGULF": {"base": 12, "direction": "SHORT", "name_cn": "看跌吞没"},
        "EVENING_STAR": {"base": 14, "direction": "SHORT", "name_cn": "黄昏星"},
        "BIG_MARUBOZU_BEAR": {"base": 8, "direction": "SHORT", "name_cn": "大阴线"},
    }
    
    def __init__(self, data_path: str):
        self.path = data_path
        self.weights = self._load_weights()
        
    def _load_weights(self) -> Dict:
        """加载形态权重"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("weights", self._init_weights())
            except Exception:
                pass
        return self._init_weights()
    
    def _init_weights(self) -> Dict:
        """初始化形态权重"""
        weights = {}
        for name, config in self.DEFAULT_PATTERNS.items():
            weights[name] = {
                "base": config["base"],
                "direction": config["direction"],
                "name_cn": config["name_cn"],
                "weight": 1.0,      # 当前权重倍数 (0.5 - 1.5)
                "count": 0,         # 出现次数
                "wins": 0,          # 盈利次数
                "losses": 0,        # 亏损次数
                "total_pnl": 0.0,   # 总盈亏百分比
                "last_pnl": 0.0,    # 最近一次盈亏
            }
        return weights
    
    def _save(self) -> None:
        """保存权重到文件"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "weights": self.weights,
            "updated_at": __import__("datetime").datetime.now().isoformat()
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def update(self, pattern_name: str, pnl_percent: float, is_win: bool) -> Optional[Dict]:
        """
        根据交易结果更新形态权重
        
        Args:
            pattern_name: 形态名称
            pnl_percent: 盈亏百分比
            is_win: 是否盈利
            
        Returns:
            更新后的权重信息
        """
        if pattern_name not in self.weights:
            return None
            
        w = self.weights[pattern_name]
        w["count"] += 1
        w["total_pnl"] += pnl_percent
        w["last_pnl"] = pnl_percent
        
        if is_win:
            w["wins"] += 1
        else:
            w["losses"] += 1
            
        # 计算形态胜率
        win_rate = w["wins"] / w["count"] if w["count"] > 0 else 0.5
        
        # 权重调整策略：
        # - 胜率 > 60%: 逐步加权 (每次 +0.05, 最大 1.5x)
        # - 胜率 < 40%: 逐步减权 (每次 -0.05, 最小 0.5x)
        # - 需要至少5次交易才开始调整
        if w["count"] >= 5:
            if win_rate > 0.6:
                w["weight"] = min(1.5, w["weight"] + 0.05)
            elif win_rate < 0.4:
                w["weight"] = max(0.5, w["weight"] - 0.05)
        
        self._save()
        
        return {
            "pattern": pattern_name,
            "count": w["count"],
            "win_rate": round(win_rate * 100, 1),
            "weight": round(w["weight"], 2),
            "pnl_change": round(pnl_percent, 3),
        }
    
    def get_pattern_score(self, pattern_name: str) -> float:
        """
        获取形态加权后的分数
        
        Args:
            pattern_name: 形态名称
            
        Returns:
            加权后的分数
        """
        w = self.weights.get(pattern_name)
        if not w:
            # 未知形态使用默认分数
            default = self.DEFAULT_PATTERNS.get(pattern_name, {"base": 10})
            return default["base"]
        return w["base"] * w["weight"]
    
    def get_stats(self) -> Dict:
        """
        返回所有形态的统计信息
        
        Returns:
            {
                "long": [...],  # 做多形态统计
                "short": [...], # 做空形态统计
                "total": {...}  # 总计
            }
        """
        long_patterns = []
        short_patterns = []
        total_count = 0
        total_wins = 0
        total_pnl = 0.0
        
        for name, data in self.weights.items():
            win_rate = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
            avg_pnl = data["total_pnl"] / data["count"] if data["count"] > 0 else 0
            
            stat = {
                "name": name,
                "name_cn": data.get("name_cn", name),
                "count": data["count"],
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(data["total_pnl"], 3),
                "avg_pnl": round(avg_pnl, 3),
                "weight": round(data["weight"], 2),
                "score": round(data["base"] * data["weight"], 1),
                "base_score": data["base"],
            }
            
            total_count += data["count"]
            total_wins += data["wins"]
            total_pnl += data["total_pnl"]
            
            if data["direction"] == "LONG":
                long_patterns.append(stat)
            else:
                short_patterns.append(stat)
        
        # 按出现次数排序
        long_patterns.sort(key=lambda x: -x["count"])
        short_patterns.sort(key=lambda x: -x["count"])
        
        return {
            "long": long_patterns,
            "short": short_patterns,
            "total": {
                "count": total_count,
                "wins": total_wins,
                "win_rate": round(total_wins / total_count * 100, 1) if total_count > 0 else 0,
                "pnl": round(total_pnl, 3),
            }
        }
    
    def get_weight(self, pattern_name: str) -> float:
        """获取形态当前权重"""
        w = self.weights.get(pattern_name)
        return w["weight"] if w else 1.0
