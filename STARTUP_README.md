# 启动程序使用说明

## 三种启动方式

### 1. 标准启动 (推荐首次使用)

**Windows用户**：
```
双击 START.bat
```

**特点**：
- 自动检查Python环境
- 检查API密钥配置
- 检查数据目录
- 可选择运行系统测试
- 启动Web服务器
- 自动打开浏览器

**启动菜单**：
```
1. Full Start - 完整启动（包含系统测试）
2. Quick Start - 快速启动（跳过测试）
3. Test Only - 仅运行测试
4. Exit - 退出
```

---

### 2. 快速启动 (已配置好系统)

**Windows用户**：
```
双击 QUICK_START.bat
```

**特点**：
- 跳过所有检查
- 直接启动Web服务器
- 自动打开浏览器
- 最快启动速度

**适用场景**：
- 系统已经配置完成
- 之前成功运行过
- 不需要检查环境

---

### 3. 命令行启动 (高级用户)

**直接启动Python脚本**：
```bash
cd "d:\MyAI\My work team\deeplearning no2\binance-futures-trading"
python start.py
```

**直接启动Web服务器**：
```bash
python web/app.py
```

然后手动打开浏览器访问：
```
http://localhost:5000/templates/index_v4.html
```

---

## 启动流程说明

### 标准启动流程

```
[1/5] Checking environment
      - Python版本检查
      - 必要模块检查 (flask, requests, numpy)

[2/5] Checking API keys
      - .env文件检查
      - API密钥配置验证

[3/5] Checking data directories
      - rl_data目录
      - web/static目录
      - web/templates目录

[4/5] Running system test (可选)
      - API连接测试
      - BTC价格获取测试

[5/5] Starting web server
      - 启动Flask服务器
      - 自动打开浏览器
```

---

## 首次启动准备

### 1. 确保Python已安装

检查Python版本：
```bash
python --version
```

要求：Python 3.8+

如果未安装，下载地址：
https://www.python.org/downloads/

安装时记得勾选"Add Python to PATH"

### 2. 安装必要模块

```bash
pip install flask requests numpy python-dotenv
```

或者使用requirements.txt：
```bash
pip install -r requirements.txt
```

### 3. 配置API密钥

创建`.env`文件（与START.bat在同一目录）：
```
BINANCE_API_KEY=你的API密钥
BINANCE_API_SECRET=你的API密钥密文
```

获取API密钥：https://testnet.binancefuture.com/

---

## 故障排查

### 问题1: "Python is not installed or not in PATH"

**原因**：Python未安装或未添加到系统PATH

**解决**：
1. 重新安装Python
2. 安装时勾选"Add Python to PATH"
3. 或手动添加Python到系统PATH

### 问题2: "Missing modules"

**原因**：缺少必要的Python模块

**解决**：
```bash
pip install flask requests numpy python-dotenv
```

### 问题3: "API keys not configured"

**原因**：.env文件不存在或配置错误

**解决**：
1. 检查.env文件是否在正确位置
2. 检查API密钥格式是否正确
3. 确保没有多余空格

### 问题4: "System test failed"

**原因**：网络问题或API密钥无效

**解决**：
1. 检查网络连接
2. 验证API密钥是否有效
3. 尝试直接访问：https://testnet.binancefuture.com/
4. 可以选择跳过测试（Quick Start）

### 问题5: "Port 5000 already in use"

**原因**：端口5000被其他程序占用

**解决方案1** - 关闭占用端口的程序：
```bash
netstat -ano | findstr :5000
taskkill /PID 进程号 /F
```

**解决方案2** - 修改端口：
编辑`web/app.py`最后一行：
```python
app.run(debug=True, port=5001)  # 改为5001
```

### 问题6: 浏览器未自动打开

**原因**：浏览器设置或安全软件拦截

**解决**：
手动打开浏览器，访问：
```
http://localhost:5000/templates/index_v4.html
```

---

## 停止服务器

### 方法1：在启动窗口中
```
按 Ctrl+C
```

### 方法2：关闭命令行窗口
```
直接关闭黑色的CMD窗口
```

### 方法3：结束进程
```bash
taskkill /F /IM python.exe
```
(注意：这会关闭所有Python进程)

---

## 高级选项

### 修改启动端口

编辑`start.py`：
```python
start_web_server(port=5001)  # 改为5001
```

### 禁用自动打开浏览器

编辑`start.py`：
```python
start_web_server(port=5000, open_browser=False)
```

### 添加自定义检查

在`start.py`中添加自定义检查函数。

---

## 日常使用建议

**首次使用**：
1. 使用START.bat
2. 选择"Full Start"
3. 运行完整测试

**日常使用**：
1. 使用QUICK_START.bat
2. 直接启动，节省时间

**调试模式**：
1. 使用命令行启动
2. 查看详细错误信息

---

## 文件说明

```
binance-futures-trading/
├── START.bat           # Windows标准启动（推荐）
├── QUICK_START.bat     # Windows快速启动
├── start.py            # Python启动脚本
├── web/app.py          # Flask Web服务器
└── .env                # API密钥配置（需手动创建）
```

---

## 常见使用场景

### 场景1：每天早上启动系统

```
1. 双击 QUICK_START.bat
2. 等待浏览器打开
3. 检查系统状态
4. 开始交易
```

### 场景2：首次部署到新电脑

```
1. 安装Python 3.8+
2. 安装依赖：pip install -r requirements.txt
3. 创建.env文件，配置API密钥
4. 双击 START.bat
5. 选择"1. Full Start"
6. 查看测试结果
```

### 场景3：系统更新后首次启动

```
1. 双击 START.bat
2. 选择"3. Test Only"
3. 验证所有功能正常
4. 再次运行，选择"1. Full Start"
```

---

**现在可以双击START.bat启动系统了！**




