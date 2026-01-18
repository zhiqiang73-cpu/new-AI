"""分析交易记录"""
import sqlite3
import os

# 检查两个可能的数据库位置
db_paths = [
    'rl_data/trades.db',
    'data/trades.db'
]

for db_path in db_paths:
    if os.path.exists(db_path):
        print(f"\n{'='*60}")
        print(f"数据库: {db_path}")
        print('='*60)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"表: {tables}")
        
        for table in tables:
            print(f"\n--- {table} ---")
            
            # 获取表结构
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            col_names = [c[1] for c in columns]
            print(f"列: {col_names}")
            
            # 获取记录数
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"记录数: {count}")
            
            if count > 0 and table == 'trades':
                # 获取最近10条交易
                cursor.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 20")
                rows = cursor.fetchall()
                print(f"\n最近20条交易:")
                for row in rows:
                    print('-'*40)
                    for i, col in enumerate(col_names):
                        print(f"  {col}: {row[i]}")
                        
                # 统计信息
                cursor.execute("SELECT direction, COUNT(*), AVG(pnl_percent), SUM(pnl_percent) FROM trades GROUP BY direction")
                stats = cursor.fetchall()
                print(f"\n按方向统计:")
                for s in stats:
                    print(f"  {s[0]}: {s[1]}笔, 平均盈亏: {s[2]:.2f}%, 总盈亏: {s[3]:.2f}%")
                
                # 盈亏分布
                cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl_percent > 0")
                wins = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl_percent <= 0")
                losses = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM trades")
                total = cursor.fetchone()[0]
                
                print(f"\n胜率统计:")
                print(f"  盈利: {wins}笔")
                print(f"  亏损: {losses}笔")
                if total > 0:
                    print(f"  胜率: {wins/total*100:.1f}%")
                    
                # 盈亏比
                cursor.execute("SELECT AVG(pnl_percent) FROM trades WHERE pnl_percent > 0")
                avg_win = cursor.fetchone()[0] or 0
                cursor.execute("SELECT AVG(pnl_percent) FROM trades WHERE pnl_percent < 0")
                avg_loss = cursor.fetchone()[0] or -1
                
                print(f"\n盈亏比:")
                print(f"  平均盈利: {avg_win:.2f}%")
                print(f"  平均亏损: {avg_loss:.2f}%")
                if avg_loss != 0:
                    print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}")
                    
                # 最大单笔
                cursor.execute("SELECT MAX(pnl_percent), MIN(pnl_percent) FROM trades")
                max_pnl, min_pnl = cursor.fetchone()
                print(f"\n极值:")
                print(f"  最大盈利: {max_pnl:.2f}%")
                print(f"  最大亏损: {min_pnl:.2f}%")
                
        conn.close()
