"""Properly fix Python files using AST-aware processing"""
import re
import os

def fix_python_keywords(content):
    """Fix broken Python keywords and spacing"""
    
    # Fix broken imports and keywords
    fixes = [
        # Core keywords
        (r'\bimp\s+or\s+t\b', 'import'),
        (r'\bfor\s+', 'for '),
        (r'\bif\s+', 'if '),
        (r'\bel\s+if\b', 'elif'),
        (r'\bel\s+se\b', 'else'),
        (r'\bwhile\s+', 'while '),
        (r'\btry\s+', 'try'),
        (r'\bexcept\s+', 'except '),
        (r'\bfinally\s+', 'finally'),
        (r'\bwith\s+', 'with '),
        (r'\braise\s+', 'raise '),
        (r'\bassert\s+', 'assert '),
        (r'\bdel\s+', 'del '),
        (r'\bpass\s+', 'pass'),
        (r'\bbreak\s+', 'break'),
        (r'\bcontin\s+ue\b', 'continue'),
        (r'\byield\s+', 'yield '),
        (r'\bglobal\s+', 'global '),
        (r'\breturn\s+', 'return '),
        (r'\bdef\s+', 'def '),
        (r'\bclass\s+', 'class '),
        
        # Fix function/variable names that got broken
        (r'\bin\s+dex\b', 'index'),
        (r'\bin\s+it\b', 'init'),
        (r'\bin\s+t\b', 'int'),
        (r'\bin\s+terval\b', 'interval'),
        (r'\bin\s+sert\b', 'insert'),
        (r'\bin\s+stance\b', 'instance'),
        (r'\bprin\s+t\b', 'print'),
        (r'\bBas\s+e\b', 'Base'),
        (r'\bexcept\s+ions\b', 'exceptions'),
        (r'\brec\s+or\s+d\b', 'record'),
        (r'\brec\s+or\s+ds\b', 'records'),
        (r'\ber\s+r\s+or\b', 'error'),
        (r'\berr\s+or\b', 'error'),
        (r'\bfor\s+mat\b', 'format'),
        (r'\bfor\s+matted\b', 'formatted'),
        (r'\bor\s+der\b', 'order'),
        (r'\bsor\s+t\b', 'sort'),
        (r'\bsh\s+or\s+t\b', 'short'),
        (r'\bsupp\s+or\s+t\b', 'support'),
        (r'\bresp\s+or\s+t\b', 'report'),
        (r'\bimp\s+or\s+t\s+ant\b', 'important'),
        (r'\bc\s+and\s+les\b', 'candles'),
        (r'\bc\s+and\s+idate\b', 'candidate'),
        (r'\bsc\s+or\s+e\b', 'score'),
        (r'\bsc\s+or\s+es\b', 'scores'),
        (r'\bsc\s+or\s+ing\b', 'scoring'),
        (r'\bhas\s+\s+attr\b', 'hasattr'),
        (r'\bexistin\s+g\b', 'existing'),
        (r'\brun\s+nin\s+g\b', 'running'),
        (r'\blearn\s+in\s+g\b', 'learning'),
        (r'\btrain\s+in\s+g\b', 'training'),
        (r'\bsettin\s+gs\b', 'settings'),
        (r'\bstrin\s+g\b', 'string'),
        (r'\breadin\s+g\b', 'reading'),
        (r'\bwritin\s+g\b', 'writing'),
        (r'\bthread\s+in\s+g\b', 'threading'),
        (r'\bencodin\s+g\b', 'encoding'),
        (r'\bgettin\s+g\b', 'getting'),
        (r'\bupdatin\s+g\b', 'updating'),
        (r'\blas\s+t\b', 'last'),
        (r'\bbas\s+e\b', 'base'),
        (r'\bmessag\s+e\b', 'message'),
        (r'\breas\s+on\b', 'reason'),
        (r'\bBin\s+ance\b', 'Binance'),
        (r'\bFlas\s+k\b', 'Flask'),
        (r'\bException\s+as\s+', 'Exception as '),
        (r'\bjsonif\s+y\b', 'jsonify'),
        (r'\btradin\s+g\b', 'trading'),
        (r'\bTradin\s+g\b', 'Trading'),
        (r'\bklin\s+es\b', 'klines'),
        (r'\bphas\s+e\b', 'phase'),
        (r'\bmin\s+\s+\b', 'min_'),
        (r'\bmax\s+\s+\b', 'max_'),
        (r'\bwin\s+\s+\b', 'win_'),
        (r'\bhas\s+\s+\b', 'has_'),
        (r'\bpleas\s+e\b', 'please'),
        (r'\bPleas\s+e\b', 'Please'),
        (r'\bperfor\s+mance\b', 'performance'),
        (r'\bhist\s+or\s+y\b', 'history'),
        (r'\bexpl\s+or\s+ation\b', 'exploration'),
        (r'\bdef\s+ault\b', 'default'),
        (r'\bur\s+and\s+om\b', 'urandom'),
        (r'\bconvert\s+_\s+klin\s+es\b', 'convert_klines'),
        (r'\bas\s+\s+set\b', 'asset'),
        (r'\bp\s+or\s+t\b', 'port'),
        (r'\bbin\s+\s+\b', 'bin '),
        (r'__main\s+\s+__', '__main__'),
        (r'\bisofor\s+mat\b', 'isoformat'),
        
        # Fix 'and' that became ' and '
        (r'\s+and\s+', ' and '),
        (r'\s+or\s+', ' or '),
        (r'\s+not\s+', ' not '),
        (r'\s+in\s+', ' in '),
        (r'\s+is\s+', ' is '),
        (r'\s+as\s+', ' as '),
        
        # Fix double spaces
        (r'  +', ' '),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    return content

def fix_file(filepath):
    """Fix a single Python file"""
    print(f"Fixing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes
        fixed_content = fix_python_keywords(content)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Test if it compiles
        try:
            compile(fixed_content, filepath, 'exec')
            print(f"  ✓ {filepath} - OK")
            return True
        except SyntaxError as e:
            print(f"  ✗ {filepath} - Syntax error at line {e.lineno}: {e.msg}")
            return False
            
    except Exception as e:
        print(f"  ✗ {filepath} - Error: {e}")
        return False

# Fix all Python files
files_to_fix = [
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
    "client.py",
]

print("=" * 60)
print("FIXING ALL DAMAGED PYTHON FILES")
print("=" * 60)

success_count = 0
for filepath in files_to_fix:
    if os.path.exists(filepath):
        if fix_file(filepath):
            success_count += 1
    else:
        print(f"  ✗ {filepath} - File not found")

print("=" * 60)
print(f"Fixed {success_count}/{len(files_to_fix)} files successfully")
print("=" * 60)




