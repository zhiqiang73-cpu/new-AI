"""Aggressively fix Python indentation by rebuilding indent structure"""
import re

def fix_python_file(filepath):
    """Fix indentation in a Python file"""
    print(f"Fixing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    indent_stack = [0]  # Stack of indentation levels
    
    for i, line in enumerate(lines):
        # Preserve empty lines
        if not line.strip():
            fixed_lines.append('\n')
            continue
        
        stripped = line.strip()
        
        # Handle dedent keywords
        if re.match(r'^(except|elif|else|finally)(\s|:)', stripped):
            if len(indent_stack) > 1:
                indent_stack.pop()
        
        # Handle complete dedent (class/function at module level)
        if re.match(r'^(class |def |@)', stripped) and len(indent_stack) > 1:
            indent_stack = [0]
        
        # Apply current indentation
        current_indent = indent_stack[-1]
        fixed_line = '    ' * current_indent + stripped + '\n'
        fixed_lines.append(fixed_line)
        
        # Handle indent keywords
        if stripped.endswith(':'):
            indent_stack.append(current_indent + 1)
        
        # Handle single-line blocks (return, pass, etc.)
        if re.match(r'^(return|pass|break|continue|raise)\s', stripped):
            # These might end a block, but we'll let the next line's dedent handle it
            pass
    
    # Write fixed file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"  Fixed {len(fixed_lines)} lines")

# Fix all broken files
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
print("Aggressively fixing indentation...")
print("=" * 60)

for filepath in broken_files:
    try:
        fix_python_file(filepath)
    except Exception as e:
        print(f"  ERROR: {e}")

print("=" * 60)
print("Done! Verifying...")
print("=" * 60)

# Verify each file
import subprocess
for filepath in broken_files:
    result = subprocess.run(
        ["python", "-m", "py_compile", filepath],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"[OK] {filepath}")
    else:
        error_line = result.stderr.split('\n')[0] if result.stderr else 'Unknown error'
        print(f"[FAIL] {filepath}: {error_line}")




