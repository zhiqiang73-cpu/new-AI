"""Fix indentation in web/app.py"""
import re

filepath = "web/app.py"

print("Reading file...")
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Fix indentation: replace single spaces at start of line with 4 spaces
fixed_lines = []
for i, line in enumerate(lines, 1):
    # Count leading spaces
    stripped = line.lstrip(' ')
    if len(line) > len(stripped):
        leading_spaces = len(line) - len(stripped)
        
        # If line starts with exactly 1 space (broken indentation)
        # and is not empty, fix it to 4 spaces
        if leading_spaces == 1 and stripped.strip():
            fixed_line = '    ' + stripped
            fixed_lines.append(fixed_line)
            print(f"Line {i}: Fixed indentation")
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

print("\nWriting fixed file...")
with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Done!")




