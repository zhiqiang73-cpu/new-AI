# 系统功能测试指南

## 步骤1: 配置API密钥

1. 访问币安测试网: https://testnet.binancefuture.com/
2. 登录并生成API密钥
3. 在项目根目录创建 `.env` 文件:

```bash
# 创建.env文件
cd "d:\MyAI\My work team\deeplearning no2\binance-futures-trading"
echo BINANCE_API_KEY=你的API密钥 > .env
echo BINANCE_API_SECRET=你的API密钥密文 >> .env
```

或者手动创建 `.env` 文件，内容如下:

```
BINANCE_API_KEY=你的API密钥
BINANCE_API_SECRET=你的API密钥密文
```

## 步骤2: 安装依赖

```bash
pip install python-dotenv requests numpy pandas ta-lib
```

## 步骤3: 运行验证脚本

```bash
python test_system.py
```

## 测试项目

脚本会依次测试以下9个功能:

1. API连接测试
   - 时间同步
   - 账户余额
   - 价格获取

2. K线数据获取
   - 1m (200根)
   - 15m (200根)
   - 8h (150根)
   - 1w (100根)

3. 技术指标计算
   - EMA (7, 25, 99)
   - RSI
   - MACD
   - ADX
   - 布林带

4. 支撑阻力位识别
   - 1m周期
   - 15m周期
   - 评分排序

5. 多周期趋势分析 (v4.0新功能)
   - 综合趋势方向
   - 趋势强度
   - 多周期共振

6. AI动态阈值 (v4.0新功能)
   - 基础阈值
   - 市场调整
   - 表现调整

7. 交易历史读取
   - 最近5笔交易
   - 胜率统计
   - 盈亏数据

8. 学习系统状态
   - 支撑阻力权重
   - 止损止盈学习
   - 入场决策学习

9. 智能分批建仓 (v4.0新功能)
   - Kelly公式杠杆
   - 分批入场计划
   - 分批止盈计划

## 预期输出

```
################################################################################
#                                                                              #
#                  BTC Trading System v4.0 - Function Test                    #
#                                                                              #
################################################################################

================================================================================
  TEST 1: API Connection
================================================================================
Server Time: 2026-01-15 10:30:45
Time Offset: 123ms
Account Balance: $10,000.00 USDT
BTC Price: $91,500.00
Result: PASS

... (其他测试) ...

================================================================================
  TEST SUMMARY
================================================================================

  Total Tests: 9
  Passed: 9
  Failed: 0
  Pass Rate: 100.0%

  Details:
    [OK] API Connection                PASS
    [OK] K-line Fetching               PASS
    [OK] Technical Indicators          PASS
    [OK] Support/Resistance            PASS
    [OK] Multi-Timeframe Analysis      PASS
    [OK] Dynamic Threshold             PASS
    [OK] Trade History                 PASS
    [OK] Learning System               PASS
    [OK] Batch Position Manager        PASS

================================================================================
All tests passed! System is ready for Web UI development.
```

## 如果测试失败

### 问题1: API连接失败
- 检查.env文件是否正确
- 检查API密钥是否有效
- 检查网络连接

### 问题2: 新模块未找到
部分v4.0新功能可能尚未整合到agent.py中,这是正常的。
测试会显示 "SKIP - New module not integrated yet"

### 问题3: 数据文件不存在
如果是首次运行,部分学习数据文件可能不存在,这是正常的。
运行几笔交易后会自动生成。

## 下一步

测试通过后,即可开始Web界面开发。




