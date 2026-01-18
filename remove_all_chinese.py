"""Remove all Chinese and non-ASCII characters from docstrings and comments"""
import re

files = ["web/app.py"]

for filepath in files:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    fixed = []
    for line in lines:
        # If line has docstring with non-ASCII, replace
        if '"""' in line and any(ord(c) > 127 for c in line):
            indent = len(line) - len(line.lstrip())
            fixed.append(' ' * indent + '"""Function description"""\n')
        # If line is comment with non-ASCII, replace
        elif line.strip().startswith('#') and any(ord(c) > 127 for c in line):
            indent = len(line) - len(line.lstrip())
            fixed.append(' ' * indent + '# Comment\n')
        else:
            fixed.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed)
    
    print(f"Cleaned {filepath}")




