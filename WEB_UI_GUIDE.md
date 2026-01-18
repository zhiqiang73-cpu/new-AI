# Web UI v4.0 使用指南

## 快速启动

### 1. 启动Web服务器

```bash
cd "d:\MyAI\My work team\deeplearning no2\binance-futures-trading"
python web/app.py
```

### 2. 访问新界面

在浏览器中打开：
```
http://localhost:5000/index_v4.html
```

或者直接修改web/app.py，将默认路由指向新界面：

```python
@app.route("/")
def index():
    return render_template("index_v4.html")  # 改为v4版本
```

---

## 界面功能说明

### 顶部状态栏

- API Status: 连接状态（绿点=已连接，灰点=未连接）
- Balance: 账户余额
- Unrealized PnL: 未实现盈亏（绿色=盈利，红色=亏损）

### 主要区域

#### 1. K线图区域（左侧2x2网格）

**1m图和15m图**：
- 显示实时K线
- 底部显示支撑位（S）和阻力位（R）
- 绿色=上涨，红色=下跌
- 1秒刷新一次

**8h图和1w图**：
- 显示长周期趋势
- 不显示支撑阻力位
- 1秒刷新一次

#### 2. 右侧面板

**持仓信息**：
- 显示所有活跃持仓
- 入场价格、杠杆、数量
- 实时盈亏
- 止损止盈价格
- 单个平仓按钮
- 一键清仓按钮（红色）

**AI决策逻辑**：
- 因为（Because）：各周期趋势、指标分析
- 所以（Therefore）：做多/做空评分、AI阈值
- 结论（Conclusion）：当前状态和下一步行动
- 学习进度：交易数量和优化次数

#### 3. 底部面板

**交易日志**（左侧）：
- 实时追加最新日志
- 颜色标记：
  - 灰色边框 = INFO
  - 绿色边框 = SUCCESS
  - 黄色边框 = WARNING
  - 红色边框 = ERROR
- 显示最近20条
- 自动滚动

**交易历史**（右侧）：
- 最近10笔交易
- 显示：
  - 交易ID（前8位）
  - 方向（LONG/SHORT）
  - 入场/出场价格
  - 盈亏金额和百分比
  - 平仓原因
- 3秒刷新一次

---

## 实时更新机制

### 更新频率

- K线数据：1秒
- 持仓信息：1秒
- 支撑阻力：2秒
- AI决策逻辑：1秒
- 交易日志：1秒
- 交易历史：3秒
- API状态：5秒

### 数据一致性保证

所有数据均从后端API获取：

- `/api/klines_all/BTCUSDT` - K线数据
- `/api/account` - 账户余额
- `/api/positions` - 持仓信息
- `/api/agent/levels` - 支撑阻力位
- `/api/agent/status` - Agent状态和AI逻辑
- `/api/agent/logs` - 交易日志
- `/api/agent/trades` - 交易历史

---

## 交互功能

### 一键清仓

1. 点击右上角红色"CLOSE ALL"按钮
2. 确认操作
3. 系统自动：
   - 获取所有持仓
   - 逐个平仓
   - 更新Agent持仓记录
   - 显示结果

### 单个持仓平仓

1. 在持仓列表中找到目标持仓
2. 点击"Close"按钮
3. 自动执行平仓
4. 更新显示

---

## 调试技巧

### 1. 查看控制台

按F12打开浏览器控制台，查看：
- API请求状态
- 错误信息
- 数据格式

### 2. 网络监控

在浏览器DevTools的Network标签中：
- 查看API响应时间
- 检查数据格式
- 验证刷新频率

### 3. 检查后端日志

Web服务器控制台会显示：
- API请求
- 错误信息
- Agent状态

---

## 常见问题

### 1. K线不显示

**原因**：API密钥未配置或网络问题

**解决**：
```bash
# 检查.env文件是否存在
# 确认API密钥正确
# 测试网络连接
python test_system.py
```

### 2. 持仓信息不更新

**原因**：Agent未运行或数据不同步

**解决**：
- 检查Agent是否在运行
- 检查`/api/positions`返回的数据
- 清空浏览器缓存

### 3. 日志不显示

**原因**：Agent未运行

**解决**：
- 启动Agent
- 检查`/api/agent/logs`端点

### 4. AI逻辑显示"Agent not running"

**原因**：Agent确实未运行

**解决**：
- 通过原界面或API启动Agent
- 或者添加启动按钮到新界面

---

## 性能优化

### 减少刷新频率

如果性能不足，可调整更新间隔：

```javascript
// 在index_v4.html的startUpdateLoop()函数中修改

// 从1秒改为2秒
setInterval(() => {
    updateKlines();
    updatePositions();
}, 2000);  // 改为2000毫秒
```

### 减少K线数量

在后端`/api/klines_all`中调整limit参数。

---

## 下一步优化

可以考虑添加：

1. Agent启动/停止按钮
2. 杠杆设置界面
3. 手动下单功能（可选）
4. 性能统计图表
5. 学习系统权重可视化

---

## 技术栈

- 后端：Flask (Python)
- 前端：原生JavaScript + HTML5 + CSS3
- 图表：LightweightCharts v4.1.0
- 实时更新：定时轮询（1秒）

---

## 注意事项

1. 不要在生产环境使用debug模式
2. 确保API密钥安全
3. 定期备份rl_data目录
4. 监控服务器资源使用
5. 测试网环境先验证所有功能

---

**界面已就绪！现在可以启动Web服务器并开始使用。**




