"""Batch fix indentation for all Python files"""
import subprocess
import os
from pathlib import Path

# List of all broken files
broken_files = [
    "rl/config.py",
    "rl/config/time_manager.py",
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
    "web/app.py",
]

print("=" * 60)
print("Fixing indentation in all Python files")
print("=" * 60)

for filepath in broken_files:
    if not os.path.exists(filepath):
        print(f"[SKIP] {filepath} - not found")
        continue
    
    print(f"\n[FIX] {filepath}")
    
    try:
        # Use autopep8 with aggressive mode
        result = subprocess.run(
            ["autopep8", "--in-place", "--aggressive", "--aggressive", "--aggressive", filepath],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Verify the file compiles
            verify_result = subprocess.run(
                ["python", "-m", "py_compile", filepath],
                capture_output=True,
                text=True
            )
            
            if verify_result.returncode == 0:
                print(f"  [OK] Fixed successfully")
            else:
                print(f"  [WARN] Fixed but still has errors:")
                print(f"    {verify_result.stderr[:200]}")
        else:
            print(f"  [ERROR] autopep8 failed")
            
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n" + "=" * 60)
print("Batch fix complete")
print("=" * 60)




