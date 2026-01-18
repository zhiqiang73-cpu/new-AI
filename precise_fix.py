"""Precisely fix the damaged Python files"""
import re

def fix_content(content):
    """Apply precise fixes"""
    
    # Fix 'import' variants  
    content = re.sub(r'(\w)imp\s+or\s+t\s+', r'\1import ', content)  # wordimport
    content = re.sub(r'^imp\s+or\s+t\s+', 'import ', content, flags=re.MULTILINE)  # line start
    
    # Fix common broken words
    replacements = {
        'Bin ance': 'Binance',
        'Flas k': 'Flask',
        'flas k': 'flask',
        'collection s': 'collections',
        'datetime ': 'datetime',
        'Futures Client': 'FuturesClient',
        'for mat': 'format',
        'prin t': 'print',
        'runnin g': 'running',
        'tradin g': 'trading',
        'Tradin g': 'Trading',
        'learn in g': 'learning',
        'settin gs': 'settings',
        'in dex': 'index',
        'in it': 'init',
        'in terval': 'interval',
        'in t': 'int',
        'in sert': 'insert',
        'in stance': 'instance',
        'Bas e': 'Base',
        'bas e': 'base',
        'rec or d': 'record',
        'rec or ds': 'records',
        'err or ': 'error',
        'or der': 'order',
        'sh or t': 'short',
        'supp or t': 'support',
        'c and les': 'candles',
        'c and idate': 'candidate',
        'sc or e': 'score',
        'sc or es': 'scores',
        'las t': 'last',
        'existin g': 'existing',
        'encodin g': 'encoding',
        'gettin g': 'getting',
        'updatin g': 'updating',
        'has  attr': 'hasattr',
        'has _': 'has_',
        'win _': 'win_',
        'min _': 'min_',
        'max _': 'max_',
        'reas on': 'reason',
        'klin es': 'klines',
        'def ault': 'default',
        'isofor mat': 'isoformat',
        '__main __': '__main__',
        'ur and om': 'urandom',
        'jsonif y': 'jsonify',
        'contin ue': 'continue',
        'Pleas e': 'Please',
        'pleas e': 'please',
        'p or t': 'port',
        'as set': 'asset',
        'phas e': 'phase',
        'thread in g': 'threading',
        'train in g': 'training',
        'readin g': 'reading',
        'writin g': 'writing',
        'except ions': 'exceptions',
        'perfor mance': 'performance',
        'hist or y': 'history',
        'expl or ation': 'exploration',
        'messag e': 'message',
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    return content

def process_file(filepath):
    """Process a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed = fix_content(content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed)
        
        # Try to compile
        try:
            compile(fixed, filepath, 'exec')
            print(f"[OK] {filepath}")
            return True
        except SyntaxError as e:
            print(f"[SYNTAX ERROR] {filepath}:{e.lineno} - {e.msg}")
            return False
    except Exception as e:
        print(f"[ERROR] {filepath} - {e}")
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

print("Fixing files...")
success = 0
for f in files:
    if process_file(f):
        success += 1

print(f"\nFixed {success}/{len(files)} files")




