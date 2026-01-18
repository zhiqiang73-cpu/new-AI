"""Final fix - parse and rebuild Python files with correct indentation"""
import tokenize
import io
import token as token_module

def rebuild_python_file(filepath):
    """Rebuild Python file with correct indentation using tokenizer"""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        # Tokenize to understand structure
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except:
        # If tokenization fails, use line-by-line rebuild
        return rebuild_by_lines(filepath, source)
    
    # Rebuild from tokens
    lines = []
    current_line = []
    indent_level = 0
    
    for i, tok in enumerate(tokens):
        if tok.type == token_module.INDENT:
            indent_level += 1
        elif tok.type == token_module.DEDENT:
            indent_level = max(0, indent_level - 1)
        elif tok.type == token_module.NEWLINE or tok.type == token_module.NL:
            if current_line:
                lines.append('    ' * indent_level + ''.join(current_line))
                current_line = []
        elif tok.type != token_module.ENCODING:
            current_line.append(tok.string)
    
    result = '\n'.join(lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result)
    
    return True

def rebuild_by_lines(filepath, source):
    """Fallback: rebuild line by line"""
    lines = source.split('\n')
    fixed = []
    indent = 0
    
    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            fixed.append(s)
            continue
        
        # Dedent
        if s.startswith(('except', 'elif', 'else', 'finally')):
            indent = max(0, indent - 1)
        
        # Module level
        if s.startswith(('class ', 'def ', '@', 'import ', 'from ')):
            if 'def ' in s or 'class ' in s:
                if not any(x in s for x in ['(self', 'cls)']):
                    indent = 0
        
        fixed.append('    ' * indent + s)
        
        # Indent
        if s.endswith(':'):
            indent += 1
    
    result = '\n'.join(fixed)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result)
    
    return True

# Fix all files
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

import subprocess
for f in files:
    print(f"Fixing {f}...")
    try:
        rebuild_python_file(f)
        r = subprocess.run(["python", "-m", "py_compile", f], capture_output=True)
        print(f"  {'OK' if r.returncode == 0 else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {e}")




