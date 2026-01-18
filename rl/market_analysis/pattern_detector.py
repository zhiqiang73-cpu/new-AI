import json
from typing import Dict, List, Optional

from ..learning.pattern_learner import PatternLearner


class PatternDetector:
    """K线形态检测器 - 作为辅助确认信号整合进打分系统"""

    PATTERN_NAMES_CN = {
        "HAMMER": "锤子线",
        "SHOOTING_STAR": "流星线",
        "BULLISH_ENGULF": "看涨吞没",
        "BEARISH_ENGULF": "看跌吞没",
        "MORNING_STAR": "晨星",
        "EVENING_STAR": "黄昏星",
        "BIG_MARUBOZU_BULL": "大阳线",
        "BIG_MARUBOZU_BEAR": "大阴线",
    }

    def __init__(self, data_dir: str = "rl_data"):
        self.pattern_learner = PatternLearner(f"{data_dir}/pattern_weights.json")

    def _last_n(self, klines: List[Dict], n: int = 3) -> List[Dict]:
        if not klines or len(klines) < n:
            return []
        return klines[-n:]

    def _body(self, k: Dict) -> float:
        return abs(k["close"] - k["open"])

    def _range(self, k: Dict) -> float:
        return max(1e-9, k["high"] - k["low"])

    def _upper_shadow(self, k: Dict) -> float:
        return k["high"] - max(k["open"], k["close"])

    def _lower_shadow(self, k: Dict) -> float:
        return min(k["open"], k["close"]) - k["low"]

    def _is_bull(self, k: Dict) -> bool:
        return k["close"] > k["open"]

    def _is_bear(self, k: Dict) -> bool:
        return k["close"] < k["open"]

    def detect(self, market: Dict) -> List[Dict]:
        """检测K线形态，返回形态列表"""
        patterns = []
        kl_1m = market.get("klines_1m", [])
        
        # 优先使用1m的最后3根K线进行形态识别
        base = self._last_n(kl_1m, 3)
        if len(base) < 3:
            return patterns

        k1, k2, k3 = base[-1], base[-2], base[-3]

        # 计算当前K线的特征
        body1 = self._body(k1)
        rng1 = self._range(k1)
        upper1 = self._upper_shadow(k1)
        lower1 = self._lower_shadow(k1)

        # ========== 单根K线形态 ==========
        # 锤子线：下影线 >= 2倍实体，上影线很小（看涨反转）
        if lower1 >= body1 * 2 and upper1 <= rng1 * 0.1:
            patterns.append({
                "name": "HAMMER",
                "name_cn": self.PATTERN_NAMES_CN["HAMMER"],
                "direction": "LONG",
                "score": self._get_learned_score("HAMMER"),
            })
        
        # 流星线：上影线 >= 2倍实体，下影线很小（看跌反转）
        if upper1 >= body1 * 2 and lower1 <= rng1 * 0.1:
            patterns.append({
                "name": "SHOOTING_STAR",
                "name_cn": self.PATTERN_NAMES_CN["SHOOTING_STAR"],
                "direction": "SHORT",
                "score": self._get_learned_score("SHOOTING_STAR"),
            })

        # ========== 双K线形态 ==========
        # 看涨吞没：当前阳线吞没前一阴线
        if (self._is_bull(k1) and self._is_bear(k2) and 
            k1["close"] > k2["open"] and k1["open"] < k2["close"]):
            patterns.append({
                "name": "BULLISH_ENGULF",
                "name_cn": self.PATTERN_NAMES_CN["BULLISH_ENGULF"],
                "direction": "LONG",
                "score": self._get_learned_score("BULLISH_ENGULF"),
            })
        
        # 看跌吞没：当前阴线吞没前一阳线
        if (self._is_bear(k1) and self._is_bull(k2) and 
            k1["close"] < k2["open"] and k1["open"] > k2["close"]):
            patterns.append({
                "name": "BEARISH_ENGULF",
                "name_cn": self.PATTERN_NAMES_CN["BEARISH_ENGULF"],
                "direction": "SHORT",
                "score": self._get_learned_score("BEARISH_ENGULF"),
            })

        # ========== 三K线形态 ==========
        # 晨星：阴线 + 小实体 + 阳线（看涨反转）
        if (self._is_bear(k3) and self._body(k2) <= self._range(k2) * 0.3 and 
            self._is_bull(k1)):
            patterns.append({
                "name": "MORNING_STAR",
                "name_cn": self.PATTERN_NAMES_CN["MORNING_STAR"],
                "direction": "LONG",
                "score": self._get_learned_score("MORNING_STAR"),
            })
        
        # 黄昏星：阳线 + 小实体 + 阴线（看跌反转）
        if (self._is_bull(k3) and self._body(k2) <= self._range(k2) * 0.3 and 
            self._is_bear(k1)):
            patterns.append({
                "name": "EVENING_STAR",
                "name_cn": self.PATTERN_NAMES_CN["EVENING_STAR"],
                "direction": "SHORT",
                "score": self._get_learned_score("EVENING_STAR"),
            })

        # ========== 大阳线/大阴线 ==========
        # 大阳线：实体占比 >= 90%（强烈看涨）
        if body1 / rng1 >= 0.9 and self._is_bull(k1):
            patterns.append({
                "name": "BIG_MARUBOZU_BULL",
                "name_cn": self.PATTERN_NAMES_CN["BIG_MARUBOZU_BULL"],
                "direction": "LONG",
                "score": self._get_learned_score("BIG_MARUBOZU_BULL"),
            })
        
        # 大阴线：实体占比 >= 90%（强烈看跌）
        if body1 / rng1 >= 0.9 and self._is_bear(k1):
            patterns.append({
                "name": "BIG_MARUBOZU_BEAR",
                "name_cn": self.PATTERN_NAMES_CN["BIG_MARUBOZU_BEAR"],
                "direction": "SHORT",
                "score": self._get_learned_score("BIG_MARUBOZU_BEAR"),
            })

        return patterns

    def get_stats(self) -> Dict:
        """统计K线形态的表现"""
        return self.pattern_learner.get_stats()
    
    def update_pattern(self, pattern_name: str, pnl_percent: float, is_win: bool) -> Optional[Dict]:
        """更新形态权重（交易结束时调用）"""
        return self.pattern_learner.update(pattern_name, pnl_percent, is_win)
    
    def _get_learned_score(self, pattern_name: str) -> float:
        """获取学习后的动态分数"""
        return self.pattern_learner.get_pattern_score(pattern_name)

