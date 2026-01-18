"""Clean broken Chinese comments"""
import re

file = "web/app.py"

with open(file, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace broken docstrings with simple English ones
content = re.sub(r'"""[^"]*?[��?][^"]*?"""', '"""Function description"""', content)

# Replace broken comments
content = re.sub(r'#[^\n]*?[��?][^\n]*', '# Comment', content)

with open(file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Cleaned!")




