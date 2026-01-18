"""Comprehensive fix for all damaged files"""
import re

def comprehensive_fix(content):
    """Apply all necessary fixes"""
    
    # Fix compound words that got broken (must come first)
    word_fixes = {
        # Import statements
        r'collections\s*import': 'collections import',
        r'datetime\s*import': 'datetime import',  
        r'config\s*import': 'config import',
        r'client\s*import': 'client import',
        r'flask\s*import': 'flask import',
        r'requests\s*import': 'requests import',
        r'sqlite3\s*import': 'sqlite3 import',
        r'os\s*import': 'os import',
        r'sys\s*import': 'sys import',
        r'time\s*import': 'time import',
        r'json\s*import': 'json import',
        r'uuid\s*import': 'uuid import',
        r'threading\s*import': 'threading import',
        r'traceback\s*import': 'traceback import',
        
        # Binance
        r'Binance\s*Futures\s*Client': 'BinanceFuturesClient',
        
        # Common patterns with spaces
        r'\bisoformat\(\)': 'isoformat()',
        r'\burandom\(': 'urandom(',
        r'\bjsonify\(': 'jsonify(',
        
        # Fix broken identifiers with underscores
        r'\bhas_\s*keys': 'has_keys',
        r'\bwin_\s*rate': 'win_rate',
        r'\bmin_\s*': 'min_',
        r'\bmax_\s*': 'max_',
        r'\blast_\s*': 'last_',
        r'\bbase_\s*': 'base_',
        r'\bexit_\s*reason': 'exit_reason',
        r'\bentry_\s*price': 'entry_price',
        r'\btrade_\s*id': 'trade_id',
        r'\bstop_\s*loss': 'stop_loss',
        r'\btake_\s*profit': 'take_profit',
        
        # Module names
        r'\brl\s*\.\s*knowledge': 'rl.knowledge',
        r'\brl\s*\.\s*exit_manager': 'rl.exit_manager',
        r'from\s+rl\s+import': 'from rl import',
        
        # __main__
        r'__main\s*__': '__main__',
        
        # Spaces in numbers followed by if/else
        r'(\d+)\s*if\s+': r'\1 if ',
        r'(\d+)\s*else\s+': r'\1 else ',
        r'\)\s*if\s+': r') if ',
        r'\)\s*else\s+': r') else ',
    }
    
    for pattern, replacement in word_fixes.items():
        content = re.sub(pattern, replacement, content)
    
    return content

# Process all files
files = [
    "web/app.py",
]

for filepath in files:
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixed = comprehensive_fix(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed)
    
    print(f"  Written")

print("\nRunning autopep8 for indentation...")
import subprocess
subprocess.run(["autopep8", "--in-place", "--aggressive", "--aggressive", "--aggressive", "web/app.py"])

print("Testing...")
result = subprocess.run(["python", "-m", "py_compile", "web/app.py"], capture_output=True, text=True)
if result.returncode == 0:
    print("[SUCCESS] web/app.py is now valid!")
else:
    print("[FAILED] Still has errors:")
    print(result.stderr[:500])




