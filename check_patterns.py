import sqlite3
conn = sqlite3.connect('rl_data/trades.db')
c = conn.cursor()
c.execute("SELECT pattern_tags, pattern_details, pnl_percent FROM trades WHERE pattern_tags IS NOT NULL AND pattern_tags != '' LIMIT 10")
rows = c.fetchall()
print(f'Found {len(rows)} trades with patterns')
for r in rows:
    print(r)
    
# 统计所有形态
c.execute("SELECT pattern_tags FROM trades WHERE pattern_tags IS NOT NULL AND pattern_tags != ''")
all_patterns = c.fetchall()
pattern_counts = {}
for row in all_patterns:
    tags = row[0]
    if tags:
        for tag in tags.split(','):
            tag = tag.strip()
            if tag:
                pattern_counts[tag] = pattern_counts.get(tag, 0) + 1
                
print("\n形态统计:")
for name, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
    print(f"  {name}: {count}次")
