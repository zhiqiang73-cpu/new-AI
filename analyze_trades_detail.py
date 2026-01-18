"""详细交易分析报告"""
import sqlite3
import os
from datetime import datetime
import sys

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')

db_path = 'rl_data/trades.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 70)
print("实盘交易详细分析报告")
print("=" * 70)

# 总体统计
cursor.execute("SELECT COUNT(*) FROM trades")
total = cursor.fetchone()[0]
print(f"\n总交易数: {total}")

# 时间范围
cursor.execute("SELECT MIN(timestamp_open), MAX(timestamp_close) FROM trades")
start_time, end_time = cursor.fetchone()
print(f"交易时间范围: {start_time} ~ {end_time}")

# 胜率统计
cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl_percent > 0")
wins = cursor.fetchone()[0]
total_wins = wins
cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl_percent <= 0")
losses = cursor.fetchone()[0]
print(f"\n胜率: {wins}/{total} = {wins/total*100:.1f}%")
print(f"   盈利交易: {wins}")
print(f"   亏损交易: {losses}")

# 盈亏比
cursor.execute("SELECT AVG(pnl_percent) FROM trades WHERE pnl_percent > 0")
avg_win = cursor.fetchone()[0] or 0
cursor.execute("SELECT AVG(pnl_percent) FROM trades WHERE pnl_percent < 0")
avg_loss = cursor.fetchone()[0] or -1
print(f"\n盈亏比: {abs(avg_win/avg_loss):.2f}")
print(f"   平均盈利: +{avg_win:.4f}%")
print(f"   平均亏损: {avg_loss:.4f}%")

# 总盈亏
cursor.execute("SELECT SUM(pnl), SUM(pnl_percent) FROM trades")
total_pnl, total_pnl_pct = cursor.fetchone()
print(f"\n总盈亏: ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")

# 按方向统计
print("\n按方向统计:")
cursor.execute("""
    SELECT direction, 
           COUNT(*), 
           SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
           AVG(pnl_percent), 
           SUM(pnl_percent),
           SUM(pnl)
    FROM trades 
    GROUP BY direction
""")
for row in cursor.fetchall():
    direction, count, wins, avg_pnl, sum_pnl_pct, sum_pnl = row
    win_rate = wins/count*100 if count > 0 else 0
    print(f"   {direction}: {count}bi, win_rate={win_rate:.1f}%, avg={avg_pnl:.3f}%, total=${sum_pnl:.2f}")

# 按出场原因统计
print("\n按出场原因统计:")
cursor.execute("""
    SELECT exit_reason, 
           COUNT(*), 
           SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
           AVG(pnl_percent), 
           SUM(pnl)
    FROM trades 
    GROUP BY exit_reason
    ORDER BY COUNT(*) DESC
""")
for row in cursor.fetchall():
    reason, count, wins, avg_pnl, sum_pnl = row
    win_rate = wins/count*100 if count > 0 else 0
    print(f"   {reason}: {count}bi, win_rate={win_rate:.1f}%, avg={avg_pnl:.3f}%, total=${sum_pnl:.2f}")

# 极值分析
print("\n极值分析:")
cursor.execute("SELECT trade_id, direction, pnl_percent, pnl, entry_price, exit_price FROM trades ORDER BY pnl_percent DESC LIMIT 3")
print("   最大盈利交易 (Top 3 wins):")
for row in cursor.fetchall():
    trade_id, direction, pnl_pct, pnl, entry, exit = row
    print(f"      {direction}: +{pnl_pct:.3f}% (${pnl:.2f}) entry${entry:.2f} -> exit${exit:.2f}")

cursor.execute("SELECT trade_id, direction, pnl_percent, pnl, entry_price, exit_price FROM trades ORDER BY pnl_percent ASC LIMIT 3")
print("   最大亏损交易 (Top 3 losses):")
for row in cursor.fetchall():
    trade_id, direction, pnl_pct, pnl, entry, exit = row
    print(f"      {direction}: {pnl_pct:.3f}% (${pnl:.2f}) entry${entry:.2f} -> exit${exit:.2f}")

# 持仓时间分析
print("\n持仓时间分析:")
cursor.execute("""
    SELECT 
        AVG((julianday(timestamp_close) - julianday(timestamp_open)) * 24 * 60) as avg_minutes,
        MIN((julianday(timestamp_close) - julianday(timestamp_open)) * 24 * 60) as min_minutes,
        MAX((julianday(timestamp_close) - julianday(timestamp_open)) * 24 * 60) as max_minutes
    FROM trades
    WHERE timestamp_close IS NOT NULL
""")
avg_hold, min_hold, max_hold = cursor.fetchone()
print(f"   平均持仓: {avg_hold:.1f}min")
print(f"   最短持仓: {min_hold:.1f}min")
print(f"   最长持仓: {max_hold:.1f}min")

# 杠杆分析
print("\n杠杆分析:")
cursor.execute("SELECT AVG(leverage), MIN(leverage), MAX(leverage) FROM trades")
avg_lev, min_lev, max_lev = cursor.fetchone()
print(f"   平均杠杆: {avg_lev:.1f}x")
print(f"   杠杆范围: {min_lev}x ~ {max_lev}x")

# 按杠杆分组的表现
cursor.execute("""
    SELECT 
        CASE 
            WHEN leverage <= 10 THEN '1-10x'
            WHEN leverage <= 15 THEN '11-15x'
            ELSE '16+x'
        END as lev_group,
        COUNT(*),
        AVG(pnl_percent),
        SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
    FROM trades
    GROUP BY lev_group
    ORDER BY lev_group
""")
print("   按杠杆组表现:")
for row in cursor.fetchall():
    group, count, avg_pnl, win_rate = row
    print(f"      {group}: {count}bi, win_rate={win_rate:.1f}%, avg={avg_pnl:.3f}%")

# 日度表现
print("\n日度表现 (最近10天):")
cursor.execute("""
    SELECT 
        DATE(timestamp_open) as trade_date,
        COUNT(*) as trades,
        SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
        SUM(pnl) as daily_pnl,
        SUM(pnl_percent) as daily_pnl_pct
    FROM trades
    GROUP BY DATE(timestamp_open)
    ORDER BY trade_date DESC
    LIMIT 10
""")
for row in cursor.fetchall():
    date, trades, wins, pnl, pnl_pct = row
    win_rate = wins/trades*100 if trades > 0 else 0
    sign = '+' if pnl >= 0 else ''
    print(f"   {date}: {trades}bi, win_rate={win_rate:.0f}%, pnl={sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)")

# 连续亏损分析
print("\n连续亏损分析:")
cursor.execute("SELECT pnl_percent FROM trades ORDER BY id")
all_pnls = [r[0] for r in cursor.fetchall()]
max_losing_streak = 0
current_streak = 0
for pnl in all_pnls:
    if pnl <= 0:
        current_streak += 1
        max_losing_streak = max(max_losing_streak, current_streak)
    else:
        current_streak = 0
print(f"   最长连续亏损: {max_losing_streak}bi")

# 回撤分析
print("\n回撤分析:")
cumulative = 0
peak = 0
max_drawdown = 0
for pnl in all_pnls:
    cumulative += pnl
    peak = max(peak, cumulative)
    drawdown = peak - cumulative
    max_drawdown = max(max_drawdown, drawdown)
print(f"   最大回撤: -{max_drawdown:.2f}%")
print(f"   累计盈亏: {cumulative:.2f}%")

# 期望值计算
print("\n期望值分析:")
global_win_rate = total_wins / total
global_loss_rate = 1 - global_win_rate
expected_value = (global_win_rate * avg_win) + (global_loss_rate * avg_loss)
print(f"   期望值: {expected_value:.4f}% per trade")
print(f"   (胜率 * 平均盈利) + (败率 * 平均亏损) = ({win_rate:.2f} * {avg_win:.4f}) + ({loss_rate:.2f} * {avg_loss:.4f})")

print("\n" + "=" * 70)
print("分析完成!")
print("=" * 70)

conn.close()
