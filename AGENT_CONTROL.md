# Agent控制说明

## 界面已更新

现在Web界面顶部有Agent控制按钮了！

### 按钮位置

```
顶部导航栏：
[BTC Trading System v4.0]  [Start Agent/Stop Agent]  [Agent Status]  [API Status]  [Balance]  [PnL]
```

### 按钮状态

#### Agent停止时
- 显示：绿色 "Start Agent" 按钮
- Agent Status: 灰点 + "Stopped"

#### Agent运行时
- 显示：红色 "Stop Agent" 按钮
- Agent Status: 绿点 + "Running"

---

## 使用方法

### 启动Agent

1. 刷新页面：http://localhost:5000/
2. 查看顶部，应该看到绿色 "Start Agent" 按钮
3. 点击 "Start Agent"
4. 等待几秒，按钮变为红色 "Stop Agent"
5. Agent Status变为绿点 "Running"

### 停止Agent

1. 点击红色 "Stop Agent" 按钮
2. 确认对话框点击"确定"
3. 按钮变为绿色 "Start Agent"
4. Agent Status变为灰点 "Stopped"

---

## 状态指示器

### Agent Status（Agent状态）
- 绿点 + "Running" = Agent正在运行，自动交易中
- 灰点 + "Stopped" = Agent已停止，不会自动交易

### API Status（API状态）
- 绿点 + "Connected" = API连接正常
- 灰点 + "Disconnected" = API连接失败

---

## Agent运行后会发生什么

### 自动执行的操作

1. **市场分析**
   - 每60秒分析一次市场
   - 计算技术指标
   - 识别支撑阻力位
   - 多周期趋势分析

2. **入场检查**
   - 根据AI动态阈值判断
   - 满足条件时自动开仓
   - 分批建仓（根据信号强度）

3. **持仓管理**
   - 每10秒检查持仓状态
   - 自动止损止盈
   - 分批止盈
   - 跟踪止损

4. **数据记录**
   - 记录所有交易
   - 更新学习数据
   - 调整特征权重

### 实时显示

- **交易日志**：实时显示Agent的思考过程
- **AI决策逻辑**：显示当前评分和阈值
- **持仓信息**：实时更新
- **K线图**：1秒刷新

---

## 注意事项

### 首次启动

第一次点击"Start Agent"可能需要10-20秒启动时间，因为需要：
- 初始化所有模块
- 加载历史数据
- 同步交易所持仓
- 计算初始指标

### 网络要求

Agent运行需要稳定的网络连接：
- 获取实时K线数据
- 获取账户信息
- 执行交易订单

### 资金安全

**重要**：Agent会自动交易！
- 确保在测试网测试
- 设置合理的风险参数
- 定期检查持仓
- 随时可以点击"Stop Agent"停止

---

## 故障排查

### 问题1：点击"Start Agent"没反应

**原因**：API密钥未配置

**解决**：
1. 确保.env文件存在
2. 配置正确的API密钥
3. 刷新页面重试

### 问题2：Agent启动后立即停止

**原因**：初始化失败

**解决**：
1. 查看浏览器控制台（F12）
2. 查看服务器命令行窗口的错误信息
3. 检查网络连接
4. 验证API密钥有效性

### 问题3：Agent运行但不交易

**可能原因**：
1. 入场条件未满足（评分低于阈值）
2. 已达到最大持仓数量（3个）
3. 风险控制限制
4. 市场条件不佳

**查看方法**：
- 查看"AI决策逻辑"面板
- 查看"交易日志"
- 确认评分和阈值

---

## 手动控制 vs 自动交易

### 手动模式
- Agent停止状态
- 只显示市场数据
- 不会自动交易
- 可以手动平仓

### 自动模式（Agent运行）
- Agent自动分析市场
- 自动开仓平仓
- 执行风险管理
- 记录学习数据

---

## 监控Agent运行

### 关键指标

**交易日志**（实时）：
```
[10:23:15] LONG entry @ 91,200 (Score 72)
[10:23:10] AI Threshold: 55 (dynamic)
[10:23:08] Support found @ 91,200 (98pts)
```

**AI决策逻辑**：
```
Because:
  1m trend: BULLISH (weight 30%, score 21)
  ...
Therefore:
  Long score: 68/100
  AI threshold: 55
Conclusion: Monitoring
```

**持仓状态**：
- 实时盈亏
- 止损止盈距离
- 持仓时间

---

## 最佳实践

### 启动前检查

1. 确认API密钥配置正确
2. 确认账户有足够余额
3. 确认网络连接稳定
4. 查看当前市场状态

### 运行中监控

1. 定期查看交易日志
2. 监控持仓盈亏
3. 关注风险指标
4. 检查Agent状态

### 异常处理

1. 发现异常立即点击"Stop Agent"
2. 手动检查并平仓（如需要）
3. 查看错误日志
4. 修复问题后重新启动

---

## 快速操作指南

### 日常使用流程

```
1. 打开浏览器 -> http://localhost:5000/
2. 检查Agent Status（应该是Stopped）
3. 检查API Status（应该是Connected）
4. 检查Balance（确认有余额）
5. 点击 "Start Agent"
6. 观察交易日志
7. 需要停止时点击 "Stop Agent"
```

### 紧急停止

```
方法1：点击界面上的 "Stop Agent" 按钮
方法2：点击 "CLOSE ALL POSITIONS" 清仓（但Agent仍运行）
方法3：关闭命令行窗口（停止整个系统）
```

---

**现在刷新页面（Ctrl+R），就能看到Agent控制按钮了！**




