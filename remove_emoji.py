"""Remove all emoji from code files"""
import os
import re

# List of files to process
files_to_process = [
    "rl/core/agent.py",
    "rl/learning/unified_learning_system.py",
    "rl/learning/dynamic_threshold.py",
    "rl/position/batch_position_manager.py",
    "rl/market_analysis/multi_timeframe_analyzer.py",
    "rl/execution/sl_tp.py",
    "rl/config.py",
    "rl/config/time_manager.py",
    "rl/config/config_v4.py",
    "rl/risk/risk_controller.py",
    "rl/execution/exit_manager.py",
    "rl/market_analysis/indicators.py",
    "rl/core/knowledge.py",
    "rl/market_analysis/level_finder.py",
    "web/app.py",
    "web/templates/index.html",
]

# Common emoji pattern
emoji_pattern = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # Miscellaneous Symbols and Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\u2600-\u26FF"          # Miscellaneous Symbols
    "\u2700-\u27BF"          # Dingbats
    "\u2B50"                 # Star
    "]+", 
    flags=re.UNICODE
)

def remove_emoji_from_file(filepath):
    """Remove emoji from a single file"""
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_lines = content.count('\n')
        
        # Remove emoji
        new_content = emoji_pattern.sub('', content)
        
        # Remove extra spaces left by emoji removal
        new_content = re.sub(r'  +', ' ', new_content)
        
        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"[UPDATED] {filepath}")
        else:
            print(f"[OK] No emoji found in {filepath}")
            
    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")

def main():
    print("=" * 60)
    print("Removing emoji from all code files...")
    print("=" * 60)
    
    count = 0
    for filepath in files_to_process:
        remove_emoji_from_file(filepath)
        count += 1
    
    print("=" * 60)
    print(f"Processed {count} files")
    print("=" * 60)

if __name__ == "__main__":
    main()




