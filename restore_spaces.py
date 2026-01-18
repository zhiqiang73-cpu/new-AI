"""Restore spaces in Python code"""
import re
import keyword

def add_spaces_after_keywords(text):
    """Add spaces after Python keywords"""
    keywords_list = list(keyword.kwlist) + ['import', 'from', 'as', 'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'raise', 'assert', 'del', 'pass', 'break', 'continue', 'yield', 'lambda', 'global', 'nonlocal']
    
    for kw in keywords_list:
        # Add space after keyword if followed by letter/digit
        text = re.sub(rf'\b{kw}([a-zA-Z0-9_])', rf'{kw} \1', text)
    
    return text

def restore_python_code(text):
    """Restore spaces in Python code"""
    # Add space after 'from' and 'import'
    text = re.sub(r'from([a-zA-Z])', r'from \1', text)
    text = re.sub(r'import([a-zA-Z])', r'import \1', text)
    
    # Add space after 'def' and 'class'
    text = re.sub(r'def([a-zA-Z_])', r'def \1', text)
    text = re.sub(r'class([a-zA-Z_])', r'class \1', text)
    
    # Add space around operators
    text = re.sub(r'([a-zA-Z0-9_])=([a-zA-Z0-9_"\'{])', r'\1 = \2', text)
    text = re.sub(r'([a-zA-Z0-9_])==([a-zA-Z0-9_"\'])', r'\1 == \2', text)
    text = re.sub(r'([a-zA-Z0-9_])!=([a-zA-Z0-9_"\'])', r'\1 != \2', text)
    
    # Add space after 'if', 'elif', 'while', 'for'
    text = re.sub(r'if([a-zA-Z0-9_])', r'if \1', text)
    text = re.sub(r'elif([a-zA-Z0-9_])', r'elif \1', text)
    text = re.sub(r'while([a-zA-Z0-9_])', r'while \1', text)
    text = re.sub(r'for([a-zA-Z0-9_])', r'for \1', text)
    text = re.sub(r'in([a-zA-Z0-9_])', r'in \1', text)
    
    # Add space after 'return', 'raise', 'yield'
    text = re.sub(r'return([a-zA-Z0-9_"\'{[])', r'return \1', text)
    text = re.sub(r'raise([a-zA-Z0-9_])', r'raise \1', text)
    text = re.sub(r'yield([a-zA-Z0-9_])', r'yield \1', text)
    
    # Add space after 'try', 'except', 'finally', 'with', 'as'
    text = re.sub(r'try:', r'try:', text)
    text = re.sub(r'except([a-zA-Z0-9_])', r'except \1', text)
    text = re.sub(r'as([a-zA-Z0-9_])', r'as \1', text)
    text = re.sub(r'with([a-zA-Z0-9_])', r'with \1', text)
    
    # Fix common patterns
    text = re.sub(r'(\w)and(\w)', r'\1 and \2', text)
    text = re.sub(r'(\w)or(\w)', r'\1 or \2', text)
    text = re.sub(r'(\w)not(\w)', r'\1 not \2', text)
    
    return text

# Process all broken files
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

for filepath in files:
    print(f"Restoring {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        restored = restore_python_code(content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(restored)
        
        print(f"  Done")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nRestoration complete!")




