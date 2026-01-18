"""Fix all space issues in Python files"""
import re

def fix_file(filepath):
    """Fix spaces in a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        print(f"[ERROR] Cannot read {filepath}")
        return False
    
    # Fix if/else spacing
    content = re.sub(r'(\d)if\s+', r'\1 if ', content)
    content = re.sub(r'\)if\s+', r') if ', content)
    content = re.sub(r'else(\d)', r' else \1', content)
    content = re.sub(r'(\d)else(\d)', r'\1 else \2', content)
    content = re.sub(r'(\))else(\w)', r'\1 else \2', content)
    
    # Write back
    try:
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"[ERROR] Cannot write {filepath}: {e}")
        return False

files = [
    "web/app.py",
    "rl/core/agent.py",
    "rl/core/knowledge.py",
    "rl/execution/exit_manager.py",
    "rl/execution/sl_tp.py",
    "rl/learning/dynamic_threshold.py",
    "rl/learning/unified_learning_system.py",
    "rl/market_analysis/indicators.py",
    "rl/market_analysis/level_finder.py",
    "rl/market_analysis/multi_timeframe_analyzer.py",
    "rl/position/batch_position_manager.py",
    "rl/risk/risk_controller.py",
    "rl/config/time_manager.py",
]

for f in files:
    print(f"Fixing {f}...")
    if fix_file(f):
        print("  OK")

print("\nRunning autopep8...")
import subprocess
for f in files:
    subprocess.run(["autopep8", "--in-place", "--aggressive", "--aggressive", f], capture_output=True)

print("Done!")




