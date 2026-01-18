# 🔍 币安期货交易系统 - 完整逻辑链与思维树分析

> **分析时间**: 2026-01-15  
> **目的**: CT扫描式精准分析，找出不赚钱的根本原因  
> **状态**: 仅分析，不修改代码

---

## 📊 系统整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        币安期货强化学习交易系统 v4.0                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   主循环 (run_agent_loop)      │
                    │   频率: 每10秒执行一次          │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐      ┌───────────────────────┐
        │   市场分析模块          │      │   交易执行模块         │
        │  (analyze_market)      │      │  (should_enter/exit)  │
        └───────────────────────┘      └───────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐      ┌───────────────────────┐
        │   学习反馈模块          │      │   风险控制模块         │
        │  (Learning Systems)   │      │  (RiskController)     │
        └───────────────────────┘      └───────────────────────┘
```

---

## 🌳 完整决策树 - 入场流程

### 第1层：主循环触发

```
【主循环】run_agent_loop() [web/app.py:198]
│
├─ 影响点: 循环频率 (10秒)
│  └─ 问题: 10秒可能错过快速信号，也可能产生噪音
│
├─ 步骤1: 获取K线数据
│  ├─ get_mainnet_klines("BTCUSDT", "1m", 150)
│  ├─ get_mainnet_klines("BTCUSDT", "15m", 150)
│  ├─ get_mainnet_klines("BTCUSDT", "8h", 150)
│  └─ get_mainnet_klines("BTCUSDT", "1w", 50)
│     │
│     └─ 影响点: API限流、网络延迟、数据质量
│
└─ 步骤2: 调用 agent.analyze_market()
   │
   └─ 进入市场分析模块
```

---

### 第2层：市场分析 (analyze_market)

```
【市场分析】agent.analyze_market() [rl/core/agent.py:467]
│
├─ 2.1 技术指标分析
│  ├─ TechnicalAnalyzer.analyze(kl_1m)   → analysis_1m
│  ├─ TechnicalAnalyzer.analyze(kl_15m)  → analysis_15m
│  ├─ TechnicalAnalyzer.analyze(kl_8h)   → analysis_8h
│  └─ TechnicalAnalyzer.analyze(kl_1w)   → analysis_1w
│     │
│     ├─ 计算指标: RSI, MACD, EMA, ATR, 布林带
│     │
│     └─ 影响点: 指标参数 (RSI周期14, MACD参数12/26/9)
│        └─ 问题: 参数固定，未根据市场状态调整
│
├─ 2.2 多周期趋势分析
│  ├─ MultiTimeframeAnalyzer.analyze()
│  │  ├─ 宏观趋势 (macro_trend): 8h + 1w 加权
│  │  │  └─ 权重: 8h(20%) + 1w(10%) = 30%
│  │  │
│  │  └─ 微观趋势 (micro_trend): 1m + 15m 加权
│  │     └─ 权重: 1m(30%) + 15m(40%) = 70%
│  │
│  └─ 影响点: 权重配置 [config_v4.py:7-31]
│     └─ 问题: 权重固定，未根据市场波动性动态调整
│
├─ 2.3 支撑阻力位发现
│  ├─ LevelDiscovery.discover_all(kl_1m)   → levels_1m
│  ├─ LevelDiscovery.discover_all(kl_15m)  → levels_15m
│  ├─ LevelDiscovery.discover_all(kl_8h)   → levels_8h
│  └─ LevelDiscovery.discover_all(kl_1w)   → levels_1w
│     │
│     ├─ 发现方法: 局部高低点、成交量密集区、整数关口
│     │
│     └─ 影响点: 容差设置、最小触及次数
│        └─ 问题: 可能产生过多候选位，需要合并
│
├─ 2.4 候选位合并与过滤
│  ├─ _merge_nearby() [agent.py:503]
│  │  └─ 容差: 0.2% (相近价位合并)
│  │
│  └─ 影响点: 合并容差设置
│     └─ 问题: 0.2%可能过宽，导致关键位被合并
│
├─ 2.5 多周期评分
│  ├─ 对每个候选位调用 _score_level_multi_tf()
│  │  ├─ LevelScoring.score_multi_tf()
│  │  │  ├─ 6个特征维度评分:
│  │  │  │  ├─ volume_density (成交量密集度) - 权重20%
│  │  │  │  ├─ touch_bounce_count (触及反弹次数) - 权重20%
│  │  │  │  ├─ bounce_magnitude (反弹幅度) - 权重15%
│  │  │  │  ├─ failed_breakout_count (假突破次数) - 权重20%
│  │  │  │  ├─ duration_days (持续天数) - 权重10%
│  │  │  │  └─ multi_tf_confirm (多周期确认) - 权重15%
│  │  │  │
│  │  │  └─ 多周期权重: [agent.py:152-167]
│  │  │     ├─ 基础权重: 1m(10%), 15m(55%), 8h(25%), 1w(10%)
│  │  │     └─ 高波动时: 1m(5%), 15m(60%), 8h(25%), 1w(10%)
│  │  │
│  │  └─ 影响点: 特征权重、多周期权重
│  │     └─ 问题: 特征权重通过学习更新，但初始值可能不合理
│  │
│  └─ 选择最佳支撑/阻力位
│     ├─ best_support: 价格下方评分最高
│     └─ best_resistance: 价格上方评分最高
│
├─ 2.6 支撑阻力间距检查
│  ├─ 最小间距: 0.3% [agent.py:566]
│  │
│  └─ 影响点: 间距过小则清空best_support/resistance
│     └─ 问题: 0.3%可能过严格，导致无有效价位
│
├─ 2.7 市场状态识别
│  ├─ MarketRegimeDetector.detect()
│  │  ├─ TRENDING: 趋势市
│  │  ├─ RANGING: 震荡市
│  │  └─ VOLATILE: 高波动市
│  │
│  └─ 影响点: 不同市场状态使用不同策略调整
│     └─ 问题: 市场状态识别可能不准确
│
├─ 2.8 突破检测
│  ├─ BreakoutDetector.check_breakout()
│  │  ├─ 支撑位突破 → 看空信号
│  │  └─ 阻力位突破 → 看多信号
│  │
│  └─ 影响点: 突破信号优先级最高
│
└─ 输出: market字典
   ├─ current_price
   ├─ macro_trend, micro_trend
   ├─ best_support, best_resistance
   ├─ analysis_1m, analysis_15m, analysis_8h, analysis_1w
   ├─ regime (市场状态)
   └─ breakout_support, breakout_resistance
```

---

### 第3层：入场评分 (_score_entry)

```
【入场评分】agent._score_entry() [rl/core/agent.py:676]
│
├─ 初始化: long_score = 0, short_score = 0
│
├─ 3.1 突破信号检查 (最高优先级)
│  ├─ 支撑位突破 → short_score += 30 + strength×20
│  ├─ 阻力位突破 → long_score += 30 + strength×20
│  │
│  └─ 影响点: 突破信号直接入场，跳过其他评分
│     └─ 问题: 假突破风险高，需要确认
│
├─ 3.2 市场状态调整系数
│  ├─ TRENDING: ×1.2 (趋势市加分)
│  ├─ RANGING: ×0.8 (震荡市减分)
│  └─ VOLATILE: ×0.7 (高波动减分)
│     │
│     └─ 影响点: 最终分数会乘以系数
│
├─ 3.3 趋势评分 (35分)
│  ├─ 微观趋势 (micro_trend): 20分
│  │  ├─ BULLISH → long_score += 20
│  │  └─ BEARISH → short_score += 20
│  │
│  ├─ 宏观趋势 (macro_trend): 10分
│  │  ├─ BULLISH → long_score += 10
│  │  └─ BEARISH → short_score += 10
│  │
│  └─ 双重确认奖励: +5分
│     └─ 宏观+微观一致时额外加分
│
├─ 3.4 支撑阻力位评分 (最高40分)
│  ├─ 做多靠近支撑位:
│  │  ├─ 距离 < 0.2%: +min(40, score/2 × sr_mult)
│  │  └─ 距离 < 0.5%: +min(25, score/3 × sr_mult)
│  │
│  ├─ 做空靠近阻力位:
│  │  ├─ 距离 < 0.2%: +min(40, score/2 × sr_mult)
│  │  └─ 距离 < 0.5%: +min(25, score/3 × sr_mult)
│  │
│  ├─ 反向惩罚:
│  │  ├─ 做多靠近阻力位: long_score -= 18
│  │  └─ 做空靠近支撑位: short_score -= 18
│  │
│  └─ 影响点: 距离阈值 (0.2%, 0.5%)
│     └─ 问题: 阈值固定，未考虑ATR动态调整
│
├─ 3.5 技术指标评分 (25分)
│  ├─ RSI (15分):
│  │  ├─ RSI < 35: long_score += 8
│  │  └─ RSI > 65: short_score += 8
│  │
│  ├─ MACD (8分):
│  │  ├─ MACD > 0: long_score += 8
│  │  └─ MACD < 0: short_score += 8
│  │
│  └─ 1m动量 (9分):
│     ├─ MACD_1m > 0: long_score += 5
│     ├─ RSI_1m < 30: long_score += 4
│     └─ RSI_1m > 70: short_score += 4
│
├─ 3.6 K线形态评分 (12分)
│  ├─ PatternDetector.detect()
│  │  ├─ 识别形态: 锤子线、吞没、三只乌鸦等
│  │  └─ 根据形态方向加分
│  │
│  └─ 影响点: 形态识别准确性
│
├─ 3.7 中间地带惩罚
│  ├─ 距离支撑和阻力都 > 0.3% → 中间地带
│  └─ 无形态确认时: long_score -= 10, short_score -= 10
│
├─ 3.8 决策特征学习评分
│  ├─ DecisionFeatureLearner.extract_features()
│  ├─ DecisionFeatureLearner.score()
│  │
│  └─ 影响点: 学习器基于历史交易结果优化
│     └─ 问题: 初期数据不足时效果有限
│
├─ 3.9 市场偏向性调整
│  ├─ 强牛市 (bias >= 2): short_score × 0.75
│  ├─ 弱牛市 (bias >= 1): short_score × 0.85
│  ├─ 强熊市 (bias <= -2): long_score × 0.75
│  └─ 弱熊市 (bias <= -1): long_score × 0.85
│     │
│     └─ 影响点: 逆势交易难度增加
│
└─ 输出: scores字典
   ├─ long: 0-100分
   ├─ short: 0-100分
   ├─ breakout_signal (如果有)
   └─ decision_features
```

---

### 第4层：入场决策 (should_enter)

```
【入场决策】agent.should_enter() [rl/core/agent.py:980]
│
├─ 4.1 启动冷却检查
│  ├─ 启动后3分钟内禁止交易 [agent.py:982]
│  └─ 影响点: 避免启动时立即交易
│
├─ 4.2 仓位数量检查
│  ├─ MAX_POSITIONS = 3
│  └─ 已有3个仓位 → 返回None
│
├─ 4.3 风险控制检查
│  ├─ RiskController.can_trade()
│  │  ├─ 检查: 单日亏损、最大回撤、连续亏损
│  │  └─ 不允许 → 返回None
│  │
│  └─ 影响点: 风险参数 [risk_controller.py]
│
├─ 4.4 获取入场上下文
│  ├─ _get_entry_context()
│  │  ├─ 获取评分
│  │  ├─ 获取阈值 (固定55分) [agent.py:108]
│  │  └─ 获取交易数量统计
│  │
│  └─ 影响点: 阈值固定为55，未使用动态阈值
│     └─ 问题: DynamicThresholdOptimizer已实现但未使用！
│
├─ 4.5 冷却时间检查
│  ├─ entry_cooldown = 15秒
│  └─ 距离上次入场 < 15秒 → 返回None
│
├─ 4.6 突破信号优先处理
│  ├─ 如果有突破信号且强度 > 0.3
│  │  └─ 直接返回入场信号 (优先级最高)
│  │
│  └─ 影响点: 突破信号可能产生假突破
│
├─ 4.7 支撑阻力间距检查
│  ├─ sr_gap_valid == False → 返回None
│  └─ 影响点: 间距 < 0.3%时禁止入场
│
├─ 4.8 评分阈值检查
│  ├─ long_ok = long_score >= 55
│  ├─ short_ok = short_score >= 55
│  │
│  └─ 影响点: 阈值55可能过高或过低
│
├─ 4.9 同分处理 (long_ok && short_ok)
│  ├─ 优先级1: 微观趋势方向
│  ├─ 优先级2: 分数差距 >= 5分
│  ├─ 优先级3: 宏观趋势方向
│  └─ 全部不满足 → 返回None
│
├─ 4.10 分批建仓检查
│  ├─ _can_add_position()
│  │  ├─ 检查: 时间间隔 > 3分钟 或 价格差距 > 0.2%
│  │  └─ 不满足 → 返回None
│  │
│  └─ 影响点: 防止频繁加仓
│
└─ 输出: signal字典 或 None
   ├─ direction: "LONG" / "SHORT"
   ├─ strength: 分数
   ├─ reason: 入场原因
   └─ decision_features: 决策特征
```

---

### 第5层：执行入场 (execute_entry)

```
【执行入场】agent.execute_entry() [rl/core/agent.py:1126]
│
├─ 5.1 计算止损止盈
│  ├─ StopLossTakeProfit.calculate()
│  │  ├─ 输入: 价格、方向、ATR、market
│  │  ├─ 计算: stop_loss, take_profit
│  │  └─ 策略: 基于支撑阻力位 + ATR缓冲
│  │
│  └─ 影响点: SL/TP参数通过学习优化
│
├─ 5.2 计算仓位大小
│  ├─ _smart_position_size()
│  │  ├─ _smart_leverage() [agent.py:349]
│  │  │  ├─ 基础杠杆: 18x
│  │  │  ├─ 信号强度因子: 0.8-1.4倍
│  │  │  ├─ 胜率因子: 0.85-1.3倍
│  │  │  └─ 最终杠杆: 15-30x
│  │  │
│  │  ├─ 保证金比例: base_margin_ratio = 12%
│  │  │  └─ 信号强度调整: 0.8-1.3倍
│  │  │
│  │  └─ 风险限制: PositionSizer.calculate_size()
│  │     └─ max_risk_percent = 2.0%
│  │
│  └─ 影响点: 杠杆和仓位大小直接影响盈亏
│
├─ 5.3 下单执行
│  ├─ _place_limit_with_requote()
│  │  ├─ 限价单偏移: 0.03% (maker)
│  │  ├─ 重报价次数: 最多6次
│  │  └─ 等待成交: 每次2秒
│  │
│  └─ 影响点: 限价单可能不成交
│
├─ 5.4 记录持仓
│  ├─ 添加到 self.positions
│  ├─ TradeLogger.log_trade()
│  └─ _save_positions()
│
└─ 输出: position字典
   ├─ trade_id, direction, entry_price
   ├─ quantity, leverage, stop_loss, take_profit
   └─ decision_features (用于学习)
```

---

## 🌳 完整决策树 - 出场流程

### 第1层：出场检查触发

```
【出场检查】agent.check_exit_all() [rl/core/agent.py:1236]
│
├─ 遍历所有持仓 (self.positions)
│
└─ 对每个持仓调用 ExitManager.evaluate()
```

---

### 第2层：出场决策 (ExitManager.evaluate)

```
【出场决策】ExitManager.evaluate() [rl/execution/exit_manager.py:171]
│
├─ 2.1 基础止损止盈检查
│  ├─ 价格 <= stop_loss → 返回 "STOP_LOSS"
│  ├─ 价格 >= take_profit → 返回 "TAKE_PROFIT"
│  │
│  └─ 影响点: SL/TP设置是否合理
│
├─ 2.2 最大亏损检查
│  ├─ pnl_pct <= -1.0% → 返回 "MAX_LOSS"
│  └─ 影响点: 参数 max_loss_pct = 1.0%
│
├─ 2.3 支撑阻力动态出场
│  ├─ _check_sr_exit()
│  │  ├─ 做多触及阻力位后回落0.05% → 止盈
│  │  ├─ 做空触及支撑位后反弹0.05% → 止盈
│  │  ├─ 假突破检测 → 止盈
│  │  │
│  │  └─ 影响点: 触及阈值0.03%，回落阈值0.05%
│  │     └─ 问题: 阈值可能过小，容易被噪音触发
│  │
│  └─ 优先级: 高于机会成本切换
│
├─ 2.4 利润锁定
│  ├─ 最大盈利 >= 0.6% 时启动
│  ├─ 回撤公式: drop = max(0.15, 0.5 - max_pnl × 0.05)
│  ├─ 当前盈利 <= max_pnl - drop → 平仓
│  │
│  └─ 影响点: 参数 profit_lock_start = 0.6%
│
├─ 2.5 机会成本切换
│  ├─ 条件1: 持仓 >= 15分钟
│  ├─ 条件2: 新信号分数 >= 阈值 + 25分
│  ├─ 条件3: 新信号分数 >= 70分
│  ├─ 条件4: 当前亏损 >= -0.1%
│  ├─ 条件5: 当前盈利 < 0.8%
│  │
│  └─ 全部满足 → 返回 "OPPORTUNITY_SWITCH"
│     │
│     └─ 影响点: 参数 opportunity_delta = 25分
│        └─ 问题: 切换条件可能过严格，错过机会
│
├─ 2.6 分批止盈
│  ├─ _check_secure_profit()
│  │  ├─ 盈利 > 0.6% 且 评分 < 50 → 平50%
│  │  │
│  │  └─ 影响点: 参数 secure_threshold = 0.6%
│  │
│  └─ 问题: 部分平仓逻辑可能不完整
│
├─ 2.7 时间成本
│  ├─ 持仓 >= 45分钟 且 盈利 < 0.8% → 平仓
│  └─ 影响点: 参数 max_hold_minutes = 45
│
└─ 输出: ExitDecision 或 None
   ├─ reason: 平仓原因
   ├─ confirmations: 确认项列表
   └─ fraction: 平仓比例 (1.0=全平, 0.5=平一半)
```

---

### 第3层：执行出场 (execute_exit_position)

```
【执行出场】agent.execute_exit_position() [rl/core/agent.py:1251]
│
├─ 3.1 下单平仓
│  ├─ _place_limit_with_requote()
│  └─ reduce_only = True
│
├─ 3.2 计算盈亏
│  ├─ 原始盈亏: raw_pnl
│  ├─ 手续费: 0.05% × 2 (开仓+平仓)
│  └─ 净盈亏: pnl = raw_pnl - commission
│     │
│     └─ 影响点: 手续费影响实际收益
│
├─ 3.3 学习反馈
│  ├─ 决策特征学习更新
│  │  ├─ DecisionFeatureLearner.update()
│  │  └─ 基于 pnl_percent 更新权重
│  │
│  ├─ 形态学习更新
│  │  ├─ PatternDetector.update_pattern()
│  │  └─ 更新形态胜率统计
│  │
│  ├─ 特征权重学习更新
│  │  ├─ LevelFinder.update_weights()
│  │  └─ 基于 reward 更新特征权重
│  │
│  ├─ 策略参数学习更新
│  │  ├─ StrategyParamLearner.update()
│  │  └─ 更新: entry_threshold_bias, profit_lock参数等
│  │
│  └─ 出场时机学习更新
│     ├─ ExitTimingLearner.update()
│     └─ 学习最佳出场时机
│
├─ 3.4 更新持仓状态
│  ├─ 全平: 从 positions 移除
│  └─ 部分平: 更新 quantity
│
└─ 输出: trade字典
   ├─ trade_id, pnl, pnl_percent
   ├─ exit_reason, exit_timing_quality
   └─ learning_updates (各种学习更新)
```

---

## 🔗 关键影响点分析

### 1. 阈值设置问题 ⚠️

```
问题位置: agent.py:108
当前状态: effective_threshold = 55 (硬编码)
应该使用: DynamicThresholdOptimizer (已实现但未使用)

影响链:
  阈值55 (固定)
    ↓
  入场条件: score >= 55
    ↓
  可能过严格 → 错过机会
  或过宽松 → 垃圾交易
    ↓
  影响胜率和交易频率
    ↓
  影响整体收益
```

### 2. 多周期权重问题 ⚠️

```
问题位置: agent.py:152-167
当前状态: 权重根据波动率动态调整，但基础权重固定

影响链:
  1m权重10% (低波动) / 5% (高波动)
    ↓
  1分钟噪音影响小
    ↓
  但可能错过超短期机会
    ↓
  15m权重55-60% (主导)
    ↓
  15分钟趋势判断主导入场
    ↓
  如果15m判断错误，整体错误
```

### 3. 支撑阻力评分问题 ⚠️

```
问题位置: LevelScoring.score_multi_tf()
当前状态: 6个特征权重，通过学习更新

影响链:
  特征权重初始值 [config_v4.py:36-43]
    ↓
  初期学习数据不足
    ↓
  权重可能不合理
    ↓
  支撑阻力位评分不准确
    ↓
  入场位置选择错误
    ↓
  影响盈亏
```

### 4. 出场时机问题 ⚠️

```
问题位置: ExitManager.evaluate()
当前状态: 多个出场条件，优先级复杂

影响链:
  支撑阻力动态出场 (优先级高)
    ↓
  触及阈值0.03% (约$30 @ $100k)
    ↓
  可能被噪音触发
    ↓
  过早平仓
    ↓
  错过更大利润
    ↓
  或正确止盈
    ↓
  需要统计验证
```

### 5. 学习系统问题 ⚠️

```
问题位置: 多个学习模块
当前状态: 学习器已实现，但需要数据积累

影响链:
  初期交易数据少
    ↓
  学习效果有限
    ↓
  权重/参数可能不合理
    ↓
  决策质量差
    ↓
  产生亏损交易
    ↓
  学习数据质量差
    ↓
  形成负循环
```

---

## 🎯 不赚钱的可能原因分析

### 原因1: 阈值设置不合理

```
症状:
  - 交易频率过高 → 阈值55可能过低
  - 交易频率过低 → 阈值55可能过高
  - 胜率低 → 阈值55未过滤垃圾交易

验证方法:
  - 统计历史交易的score分布
  - 分析score与盈亏的关系
  - 检查是否应该使用DynamicThresholdOptimizer
```

### 原因2: 支撑阻力位识别不准确

```
症状:
  - 入场后价格未按预期反弹/回落
  - 止损频繁触发
  - 止盈很少触发

验证方法:
  - 检查best_support/resistance的评分
  - 分析入场位置与支撑阻力的距离
  - 统计支撑阻力位的有效性
```

### 原因3: 出场时机过早或过晚

```
症状:
  - 盈利单过早平仓 (利润锁定太早)
  - 亏损单过晚平仓 (止损太宽)
  - 机会成本切换不工作

验证方法:
  - 统计各出场原因的盈亏
  - 分析MFE/MAE (最大浮动盈利/亏损)
  - 检查出场时机质量评分
```

### 原因4: 仓位管理问题

```
症状:
  - 杠杆过高 → 单笔亏损大
  - 杠杆过低 → 单笔盈利小
  - 仓位大小不合理

验证方法:
  - 统计杠杆分布
  - 分析杠杆与盈亏的关系
  - 检查仓位大小计算逻辑
```

### 原因5: 市场状态识别错误

```
症状:
  - 趋势市按震荡市处理
  - 震荡市按趋势市处理
  - 策略调整系数错误

验证方法:
  - 检查regime识别准确性
  - 分析不同regime下的盈亏
  - 验证regime调整系数
```

### 原因6: 学习系统未生效

```
症状:
  - 交易数据积累不足
  - 权重/参数未更新
  - 学习效果不明显

验证方法:
  - 检查学习更新频率
  - 分析权重变化趋势
  - 验证学习是否改善表现
```

---

## 📈 建议的检查清单

### 数据层面检查

- [ ] 检查历史交易数据 (trades.db)
  - 总交易数
  - 胜率
  - 平均盈亏
  - 盈亏比

- [ ] 分析评分分布
  - 入场score分布
  - score与盈亏的关系
  - 阈值55是否合理

- [ ] 检查支撑阻力有效性
  - 入场位置分布
  - 支撑阻力位评分
  - 支撑阻力位有效性统计

### 逻辑层面检查

- [ ] 验证阈值设置
  - 是否应该使用DynamicThresholdOptimizer
  - 阈值55的来源和合理性

- [ ] 检查出场逻辑
  - 各出场原因的触发频率
  - 各出场原因的盈亏统计
  - 出场时机质量评分

- [ ] 验证学习系统
  - 学习更新是否执行
  - 权重/参数是否变化
  - 学习是否改善表现

### 参数层面检查

- [ ] 多周期权重
  - 当前权重配置
  - 是否需要调整

- [ ] 支撑阻力参数
  - 距离阈值 (0.2%, 0.5%)
  - 合并容差 (0.2%)
  - 最小间距 (0.3%)

- [ ] 出场参数
  - 利润锁定参数
  - 机会成本切换参数
  - 时间成本参数

---

## 🔍 下一步行动建议

1. **数据统计分析**
   - 导出历史交易数据
   - 分析score分布和盈亏关系
   - 找出关键问题点

2. **逻辑验证**
   - 检查DynamicThresholdOptimizer是否应该启用
   - 验证出场逻辑是否正确执行
   - 确认学习系统是否工作

3. **参数优化**
   - 根据统计数据调整参数
   - 测试不同参数组合
   - 回测验证改进效果

---

**分析完成时间**: 2026-01-15  
**分析范围**: 完整系统逻辑链  
**下一步**: 根据此分析进行数据统计和问题定位


