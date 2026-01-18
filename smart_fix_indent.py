"""Smart indentation fix for Python files"""
import tokenize
import io

def fix_indentation(filepath):
    """Fix indentation by rewriting the file with proper Python indentation"""
    print(f"Reading {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    indent_level = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Empty lines or comments
        if not stripped or stripped.startswith('#'):
            fixed_lines.append(line if not line.strip() else '    ' * indent_level + stripped)
            continue
        
        # Dedent before certain keywords
        if stripped.startswith(('except', 'elif', 'else', 'finally')):
            indent_level = max(0, indent_level - 1)
        
        # Decrease indent for closing brackets/dedent
        if stripped.startswith(('return', 'break', 'continue', 'pass', 'raise')) and indent_level > 0:
            # Don't change indent for these
            pass
        
        # Apply current indent
        fixed_line = '    ' * indent_level + stripped
        fixed_lines.append(fixed_line)
        
        # Increase indent after certain patterns
        if stripped.endswith(':'):
            indent_level += 1
        
        # Decrease indent after return/break/continue/pass/raise at end of block
        # (will be handled by next line's dedent keyword)
    
    # Write fixed content
    fixed_content = '\n'.join(fixed_lines)
    
    print(f"Writing fixed {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Done!")

if __name__ == "__main__":
    fix_indentation("web/app.py")




