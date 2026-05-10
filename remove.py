#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_suffix_rename.py

用途：在数据集下的 hazy/ 和 mask/ 目录中删除文件名末尾的指定后缀（例如 "_fog_v1"）
并重命名文件。支持 dry-run 模式及重命名冲突自动处理。

用法示例（先模拟）：
  python remove_suffix_rename.py --root ./datasets --suffix _fog_v1 --dry-run

执行实际重命名：
  python remove_suffix_rename.py --root ./datasets --suffix _fog_v1

参数:
  --root    : 数据根目录（默认 ./datasets）
  --dirs    : 要处理的子目录，逗号分隔（默认 hazy,mask）
  --suffix  : 要删除的后缀（必须精确匹配文件 stem 的末尾），例如 "_fog_v1"
  --dry-run : 只打印将要做的操作，不实际重命名
  --log     : 重命名日志文件路径（默认 ./rename_log.csv）
  --verbose : 打印更详细信息
"""
import argparse
from pathlib import Path
import csv
import datetime
import os
import sys

def unique_path(dst_path: Path):
    """如果 dst_path 存在，则在文件名后加 _1/_2... 直至不冲突，返回新的 Path"""
    if not dst_path.exists():
        return dst_path
    parent = dst_path.parent
    stem = dst_path.stem
    suffix = dst_path.suffix  # 包含点，如 .png
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def process_dir(dirp: Path, suffix: str, dry_run: bool, log_rows: list, verbose: bool):
    if not dirp.exists() or not dirp.is_dir():
        if verbose:
            print(f"[WARN] 目录不存在：{dirp}，跳过。")
        return 0, 0
    files = sorted([p for p in dirp.iterdir() if p.is_file()])
    total = 0
    changed = 0
    for p in files:
        total += 1
        stem = p.stem
        ext = p.suffix
        if not stem.endswith(suffix):
            if verbose:
                print(f"[跳过] {p.name} (stem 不以 {suffix} 结尾)")
            continue
        new_stem = stem[:-len(suffix)]
        if new_stem == "":
            # 防止变成空名
            if verbose:
                print(f"[跳过] {p.name} -> new stem 为空，跳过")
            continue
        new_path = dirp / (new_stem + ext)
        # 若目标已存在，生成唯一路径
        if new_path.exists():
            unique_new_path = unique_path(new_path)
            if unique_new_path != new_path and verbose:
                print(f"[冲突] 目标已存在 {new_path.name}，将使用 {unique_new_path.name}")
            new_path = unique_new_path
        print(f"[重命名] {p.name}  ->  {new_path.name}")
        log_rows.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'src': str(p.resolve()),
            'dst': str(new_path.resolve()),
        })
        if not dry_run:
            try:
                os.rename(p, new_path)
            except Exception as e:
                print(f"[ERROR] 无法重命名 {p} -> {new_path}: {e}")
                # 将该行标注错误
                log_rows[-1]['error'] = str(e)
                continue
        changed += 1
    return total, changed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./datasets', help='数据集根目录，包含 hazy/ clear/ mask/')
    parser.add_argument('--dirs', type=str, default='hazy,mask', help='要处理的子目录，逗号分隔（默认 hazy,mask）')
    parser.add_argument('--suffix', type=str, required=True, help='要删除的后缀（例如 _fog_v1）')
    parser.add_argument('--dry-run', action='store_true', help='仅模拟，不实际重命名')
    parser.add_argument('--log', type=str, default='./rename_log.csv', help='重命名日志文件路径（CSV）')
    parser.add_argument('--verbose', action='store_true', help='打印详细信息')
    args = parser.parse_args()

    root = Path(args.root)
    dirs = [d.strip() for d in args.dirs.split(',') if d.strip()]
    suffix = args.suffix
    dry_run = args.dry_run
    log_file = Path(args.log)

    if not root.exists():
        print(f"[ERROR] root 目录不存在: {root}")
        sys.exit(1)

    print(f"Root: {root.resolve()}")
    print(f"处理目录: {dirs}")
    print(f"要删除后缀: '{suffix}'")
    print(f"{'DRY-RUN: 不会执行实际重命名' if dry_run else '实际执行重命名'}")
    print("------")

    log_rows = []
    total_files = 0
    total_changed = 0
    for d in dirs:
        dirp = root / d
        t, c = process_dir(dirp, suffix, dry_run, log_rows, args.verbose)
        total_files += t
        total_changed += c

    # 写日志（追加模式），包含时间戳、src、dst、可选错误字段
    if log_rows and not dry_run:
        write_header = not log_file.exists()
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp','src','dst','error'])
            if write_header:
                writer.writeheader()
            for row in log_rows:
                # ensure all keys present
                if 'error' not in row:
                    row['error'] = ''
                writer.writerow(row)
        print(f"已将重命名记录追加到 {log_file}")

    print("------")
    print(f"目录总文件计数（被遍历）: {total_files}")
    print(f"找到并处理的文件数: {total_changed}")
    if dry_run:
        print("注意：当前为 dry-run 模式（没有实际重命名）。移除 --dry-run 后再执行实际重命名。")

if __name__ == '__main__':
    main()
