"""
图片序号重排: 把目录内的图片按当前顺序重命名为 frame_0001, frame_0002, ... (连续无空洞).

典型用途: 删除部分图片后 (如删掉 data_diff 中不含块状盲元的图), 把剩余文件重排连续编号.
支持多目录"同名联动": blur 里 frame_0005 改名 frame_0002 时, sharp (和 mask) 里的同名文件
一起改 -> 配对关系不断.

用法 (建议先干跑预览):
  python paixv.py --dry-run --dirs D:\\qtt\\Restormer\\data_diff\\train_blur D:\\qtt\\Restormer\\data_diff\\train_sharp
  python paixv.py --dirs D:\\qtt\\Restormer\\data_diff\\train_blur D:\\qtt\\Restormer\\data_diff\\train_sharp

选项:
  --dirs     参与联动重命名的目录列表 (第一个目录决定顺序; 各目录内同名文件集合必须一致)
  --prefix   新文件名前缀 (默认 frame_)
  --start    起始编号 (默认 1)
  --ext      处理的扩展名 (默认 .png)
  --dry-run  只预览, 不改名

说明:
  - 按每个子目录 (递归) 分别重排; 排序依据 = 文件名中最后一段数字 (frame_0007 -> 7)
  - 已连续命名的目录重跑本脚本无副作用 (需改名 0 个)
  - 采用两阶段改名 (先临时名再终名), 不会因编号冲突出错; 若中途异常, 目录里会残留
    __paixv_tmp_* 临时文件, 重跑前请先手工清理
"""
import argparse
import os
import re


def nat_key(name):
    nums = re.findall(r'\d+', os.path.splitext(name)[0])
    return (int(nums[-1]) if nums else 10**12, name)


def main():
    ap = argparse.ArgumentParser(description='图片序号重排 (frame_0001 顺序)')
    ap.add_argument('--dirs', nargs='+', required=True, help='联动重命名的目录 (>=2)')
    ap.add_argument('--prefix', default='frame_')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--ext', default='.png')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if len(args.dirs) < 2:
        raise SystemExit('至少给 2 个目录 (如 blur + sharp)')

    roots = [os.path.abspath(d) for d in args.dirs]
    for r in roots:
        if not os.path.isdir(r):
            raise SystemExit(f'目录不存在: {r}')

    driver = roots[0]
    subdirs = []
    for cur, _, files in os.walk(driver):
        if any(f.lower().endswith(args.ext) for f in files):
            subdirs.append(os.path.relpath(cur, driver))
    subdirs.sort()
    if not subdirs:
        raise SystemExit('驱动目录下没有找到图片')

    # 残留临时文件检查
    for cur, _, files in os.walk(driver):
        left = [f for f in files if f.startswith('__paixv_tmp_')]
        if left:
            raise SystemExit(f'发现上次未完成的临时文件: {cur} ({left[:3]}...), 请先手工清理再重跑')

    total = 0
    for rel in subdirs:
        lists, ok = [], True
        for r in roots:
            d = os.path.join(r, rel)
            if not os.path.isdir(d):
                print(f'!! 跳过 {rel}: {r} 下没有该子目录')
                ok = False
                break
            lists.append(sorted([f for f in os.listdir(d) if f.lower().endswith(args.ext)], key=nat_key))
        if not ok:
            continue
        base = lists[0]
        for i, fs in enumerate(lists[1:], 1):
            if set(fs) != set(base):
                diff = sorted(set(base) ^ set(fs))[:3]
                print(f'!! 跳过 {rel}: 目录间文件不一致 ({roots[0]} {len(base)} 个 vs {roots[i]} {len(fs)} 个; '
                      f'差异示例 {diff})')
                ok = False
                break
        if not ok:
            continue

        plan = [(old, f'{args.prefix}{args.start + i:04d}{args.ext}')
                for i, old in enumerate(base)]
        changed = sum(1 for a, b in plan if a != b)
        print(f'[{rel}] {len(base)} 个文件, 需改名 {changed} 个')
        if args.dry_run:
            for a, b in plan[:3]:
                print(f'    {a} -> {b}')
            if len(plan) > 3:
                print('    ...')
            continue

        # 两阶段改名: 先全部改为临时名, 再改为目标名 (避免编号互相冲突)
        for j, (old, _) in enumerate(plan):
            tmp = f'__paixv_tmp_{j:06d}{args.ext}'
            for r in roots:
                os.rename(os.path.join(r, rel, old), os.path.join(r, rel, tmp))
        for j, (_, new) in enumerate(plan):
            tmp = f'__paixv_tmp_{j:06d}{args.ext}'
            for r in roots:
                os.rename(os.path.join(r, rel, tmp), os.path.join(r, rel, new))
        total += len(base)

    if args.dry_run:
        print('\n(干跑模式, 未执行任何改名; 去掉 --dry-run 正式执行)')
    else:
        print(f'\n完成: 共重排 {total} 个文件 x {len(roots)} 个目录')


if __name__ == '__main__':
    main()
