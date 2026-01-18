"""
BTC Trading System v4.0 - 启动程序
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def print_banner():
    """打印启动横幅"""
    print("\n" + "="*80)
    print("                  BTC Trading System v4.0 - Launcher")
    print("="*80 + "\n")


def check_environment():
    """检查运行环境"""
    print("[1/5] Checking environment...")
    
    # 检查Python版本
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("    [ERROR] Python 3.8+ required, current:", f"{py_version.major}.{py_version.minor}")
        return False
    print(f"    [OK] Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    # 检查必要的模块
    required_modules = ['flask', 'requests', 'numpy']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"    [OK] {module} installed")
        except ImportError:
            missing_modules.append(module)
            print(f"    [ERROR] {module} not installed")
    
    if missing_modules:
        print(f"\n    Missing modules: {', '.join(missing_modules)}")
        print("    Run: pip install " + " ".join(missing_modules))
        return False
    
    return True


def check_api_keys():
    """检查API密钥配置"""
    print("\n[2/5] Checking API keys...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("    [WARNING] .env file not found")
        print("    Create .env file with:")
        print("        BINANCE_API_KEY=your_key")
        print("        BINANCE_API_SECRET=your_secret")
        return False
    
    # 读取.env文件
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    has_key = 'BINANCE_API_KEY=' in content
    has_secret = 'BINANCE_API_SECRET=' in content
    
    if has_key and has_secret:
        print("    [OK] API keys configured")
        return True
    else:
        print("    [ERROR] API keys not properly configured in .env")
        return False


def check_data_directories():
    """检查数据目录"""
    print("\n[3/5] Checking data directories...")
    
    directories = ['rl_data', 'web/static', 'web/templates']
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"    [WARN] {directory} not found, creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"    [OK] {directory} exists")
    
    return True


def run_system_test():
    """运行系统测试（快速版）"""
    print("\n[4/5] Running system test...")
    
    try:
        # 只测试API连接
        from client import BinanceFuturesClient
        
        client = BinanceFuturesClient()
        ticker = client.get_ticker_price("BTCUSDT")
        price = float(ticker['price'])
        
        print(f"    [OK] API connection successful")
        print(f"    [OK] BTC Price: ${price:,.2f}")
        return True
        
    except Exception as e:
        print(f"    [ERROR] System test failed: {e}")
        return False


def start_web_server(port=5000, open_browser=True):
    """启动Web服务器"""
    print("\n[5/5] Starting web server...")
    
    try:
        print(f"    Starting Flask on port {port}...")
        print(f"    URL: http://localhost:{port}/")
        
        if open_browser:
            # 延迟打开浏览器
            time.sleep(2)
            url = f"http://localhost:{port}/"
            print(f"    Opening browser: {url}")
            webbrowser.open(url)
        
        # 启动Flask
        os.environ['FLASK_ENV'] = 'development'
        subprocess.run([sys.executable, 'web/app.py'], check=True)
        
    except KeyboardInterrupt:
        print("\n\n    Web server stopped by user")
    except Exception as e:
        print(f"    [ERROR] Failed to start web server: {e}")
        return False
    
    return True


def main():
    """主启动流程"""
    print_banner()
    
    # 检查环境
    if not check_environment():
        print("\n[FAILED] Environment check failed. Please fix the issues above.")
        input("\nPress Enter to exit...")
        return
    
    # 检查API密钥
    api_ok = check_api_keys()
    if not api_ok:
        print("\n[WARNING] API keys not configured. Some features may not work.")
        # 不中断启动，直接继续
    
    # 检查数据目录
    check_data_directories()
    
    # 直接启动（不再选择阶段1-4）
    print("\n[4/5] Skipping system test...")
    start_web_server(port=5000, open_browser=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

