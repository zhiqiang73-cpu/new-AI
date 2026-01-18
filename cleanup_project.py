"""
项目清理脚本
删除不需要的备份文件和临时文件
"""
import os
import shutil
from pathlib import Path
from datetime import datetime


def get_folder_size(path):
    """获取文件夹大小（MB）"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_folder_size(entry.path)
    except:
        pass
    return total / (1024 * 1024)  # 转换为MB


def cleanup_old_backups():
    """清理旧备份"""
    base_dir = Path(__file__).parent
    
    # 要删除的备份文件夹
    backup_dirs = [
        "rl_data_backup_20260112_232239",
        "rl_data_backup_20260112_232258",
        "rl_data_backups",
        "rl_data_test",
    ]
    
    print("="*60)
    print("清理旧备份文件夹")
    print("="*60)
    
    total_saved = 0
    for dir_name in backup_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            size = get_folder_size(dir_path)
            print(f"\n📁 {dir_name}")
            print(f"   大小: {size:.2f} MB")
            
            response = input(f"   删除？(y/n): ")
            if response.lower() == 'y':
                try:
                    shutil.rmtree(dir_path)
                    print(f"   ✅ 已删除，节省 {size:.2f} MB")
                    total_saved += size
                except Exception as e:
                    print(f"   ❌ 删除失败: {e}")
            else:
                print(f"   ⏭️ 跳过")
        else:
            print(f"\n⚠️ {dir_name} 不存在")
    
    print(f"\n总计节省: {total_saved:.2f} MB")


def cleanup_root_scripts():
    """清理根目录的临时脚本"""
    base_dir = Path(__file__).parent
    
    # 根目录的临时脚本（可能需要移动到scripts/文件夹）
    temp_scripts = [
        "check_db.py",
        "check_last_trades.py",
        "diagnose_no_trades.py",
        "fix_min_score.py",
        "inspect_nlm.py",
    ]
    
    print("\n" + "="*60)
    print("整理根目录脚本")
    print("="*60)
    print("\n这些脚本建议移动到 scripts/ 文件夹:")
    
    scripts_dir = base_dir / "scripts"
    
    found_any = False
    for script in temp_scripts:
        script_path = base_dir / script
        if script_path.exists():
            found_any = True
            print(f"  - {script}")
    
    if not found_any:
        print("  ✅ 没有找到临时脚本")
        return
    
    response = input("\n是否创建 scripts/ 文件夹并移动这些脚本？(y/n): ")
    if response.lower() == 'y':
        scripts_dir.mkdir(exist_ok=True)
        
        for script in temp_scripts:
            script_path = base_dir / script
            if script_path.exists():
                try:
                    shutil.move(str(script_path), str(scripts_dir / script))
                    print(f"  ✅ {script} → scripts/")
                except Exception as e:
                    print(f"  ❌ 移动失败: {e}")


def organize_docs():
    """整理文档文件夹"""
    base_dir = Path(__file__).parent
    docs_dir = base_dir / "docs"
    
    print("\n" + "="*60)
    print("文档文件夹整理")
    print("="*60)
    
    if not docs_dir.exists():
        print("⚠️ docs/ 文件夹不存在")
        return
    
    # 统计文档数量
    doc_files = list(docs_dir.glob("*.md"))
    print(f"\n当前有 {len(doc_files)} 个文档文件")
    
    # 建议的文档分类
    categories = {
        "archive": [  # 历史文档（可以归档）
            "backtest_fix_20260109.md",
            "bug_fix_20260113.md",
            "bugfix_20260109.md",
            "bugfix_feature_names.md",
            "backtest_zero_trades_fix.md",
            "zero_trades_diagnosis.md",
            "chart_improvements.md",
            "y_axis_adjustment_test.md",
            "CHANGELOG_20260113.md",
        ],
        "analysis": [  # 分析文档
            "data_persistence_analysis.md",
            "learning_system_analysis.md",
            "MATH_RIGOR_ANALYSIS.md",
            "SYSTEM_ANALYSIS_MIND_TREE.md",
            "SYSTEM_DIAGNOSIS_SUMMARY.md",
        ],
        "guides": [  # 指南文档
            "backtest_training_guide.md",
            "sl_tp_training_guide.md",
            "stability_improvements_guide.md",
            "QUICK_FIX_GUIDE.md",
            "FILE_REORGANIZATION_GUIDE.md",
        ],
    }
    
    print("\n建议创建子文件夹分类:")
    print("  - docs/archive/   (历史文档)")
    print("  - docs/analysis/  (分析文档)")
    print("  - docs/guides/    (指南文档)")
    
    response = input("\n是否创建子文件夹并整理文档？(y/n): ")
    if response.lower() == 'y':
        for category, files in categories.items():
            category_dir = docs_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for filename in files:
                file_path = docs_dir / filename
                if file_path.exists():
                    try:
                        shutil.move(str(file_path), str(category_dir / filename))
                        print(f"  ✅ {filename} → docs/{category}/")
                    except Exception as e:
                        print(f"  ❌ 移动失败: {e}")


def check_reorganization_backup():
    """检查重组备份"""
    base_dir = Path(__file__).parent
    backup_dir = base_dir / "rl_backup_before_reorganize"
    
    print("\n" + "="*60)
    print("重组备份检查")
    print("="*60)
    
    if not backup_dir.exists():
        print("✅ 没有重组备份（已清理或未执行重组）")
        return
    
    size = get_folder_size(backup_dir)
    print(f"\n📁 rl_backup_before_reorganize/")
    print(f"   大小: {size:.2f} MB")
    print(f"   说明: 文件重组前的备份")
    print(f"\n如果系统运行正常，可以删除这个备份")
    
    response = input(f"   删除备份？(y/n): ")
    if response.lower() == 'y':
        try:
            shutil.rmtree(backup_dir)
            print(f"   ✅ 已删除，节省 {size:.2f} MB")
        except Exception as e:
            print(f"   ❌ 删除失败: {e}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║           项目清理脚本 - 删除多余文件                     ║
╚═══════════════════════════════════════════════════════════╝

将清理以下内容:
1. 旧的备份文件夹 (rl_data_backup_*)
2. 根目录的临时脚本
3. 整理docs/文档
4. 重组备份（可选）

注意: 每个操作都会确认后再执行
""")
    
    response = input("是否继续？(y/n): ")
    if response.lower() != 'y':
        print("取消清理")
        return
    
    # 1. 清理旧备份
    cleanup_old_backups()
    
    # 2. 整理根目录脚本
    cleanup_root_scripts()
    
    # 3. 整理文档
    organize_docs()
    
    # 4. 检查重组备份
    check_reorganization_backup()
    
    print("\n" + "="*60)
    print("✅ 清理完成！")
    print("="*60)
    print("\n项目现在更整洁了 🎨")


if __name__ == "__main__":
    main()




