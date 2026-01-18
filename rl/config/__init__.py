"""
配置模块
包含统一配置和时区管理
"""
from .config_v4 import (
    TIMEFRAME_WEIGHTS,
    FEATURE_LEARNING,
    DYNAMIC_THRESHOLD,
    POSITION_MANAGEMENT,
    RISK_CONTROL,
    TIME_CONFIG,
    DATA_RECORDING,
    SYSTEM_INFO,
    get_config_summary,
)
from .time_manager import TimeManager, time_manager, now, timestamp, format_time, get_duration

__all__ = [
    # 配置
    'TIMEFRAME_WEIGHTS',
    'FEATURE_LEARNING',
    'DYNAMIC_THRESHOLD',
    'POSITION_MANAGEMENT',
    'RISK_CONTROL',
    'TIME_CONFIG',
    'DATA_RECORDING',
    'SYSTEM_INFO',
    'get_config_summary',
    # 时间管理
    'TimeManager',
    'time_manager',
    'now',
    'timestamp',
    'format_time',
    'get_duration',
]




