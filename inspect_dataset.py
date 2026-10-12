"""
数据集体检: 结构 / 配对 / 图片规格 / 块状盲元统计.

用法 (项目根目录):
  python inspect_dataset.py --root data_diff
  python inspect_dataset.py --root data_diff --limit 400 --thr 30
"""
import argparse
import glob
import os
from collections import defaultdict

import cv2
import numpy as np

BLUR_TOKENS = ('blur', 'lq', 'input')
SHARP_TOKENS = ('sharp', 'gt')
BUCKETS = [(6, 16), (16, 64), (64, 256), (256, 10**9)]


def norm_key(rel):
    """把相对路径里的 blur/sharp/lq/gt 目录 token 归一成 @@, 用于配对识别."""
    d, name = os.path.split(rel)
    low = d.lower()
    for t in ('blur', 'sharp', 'lq', 'gt'):
        idx = low.find(t)
        if idx >= 0:
            return (d[:idx] + '@@' + d[idx + len(t):]).replace('\\', '/') + '/' + name
    return None


def side_of(rel):
    d = os.path.dirname(rel).lower()
    if any(t in d for t in BLUR_TOKENS):
        return 'blur'
    if any(t in d for t in SHARP_TOKENS):
        return 'sharp'
    return None


def load8(path):
    """读图 -> uint8 灰度 (uint16 缩放)."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.dtype == np.uint16:
        im = (im / 256.0).astype(np.uint8)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def main():
    ap = argparse.ArgumentParser(description='数据集体检')
    ap.add_argument('--root', required=True, help='数据集根目录')
    ap.add_argument('--limit', type=int, default=0, help='块状统计最多扫描多少对 (0=全部)')
    ap.add_argument('--thr', type=int, default=30, help='盲元判定阈值 |blur-gt| (uint8 尺度)')
    ap.add_argument('--min_block', type=int, default=6, help='连通域最小像素数')
    args = ap.parse_args()
    root = args.root

    # ---------- [1] 结构 ----------
    print(f'[1] 目录结构: {root}')
    total_png = 0
    for cur, dirs, files in os.walk(root):
        dirs.sort()
        png = sum(1 for f in files if f.lower().endswith('.png'))
        total_png += png
        depth = os.path.relpath(cur, root).count(os.sep) if cur != root else 0
        if depth <= 3:
            rel = os.path.relpath(cur, root)
            print(f'{"  " * (depth + 1)}{"." if rel == "." else rel}/  ({png} png, {len(dirs)} subdirs)')
    print(f'    共 {total_png} 个 png')

    # ---------- [2] 配对 ----------
    files = sorted(glob.glob(os.path.join(root, '**', '*.png'), recursive=True))
    rels = [os.path.relpath(f, root) for f in files]
    groups = defaultdict(lambda: {'blur': [], 'sharp': []})
    n_unrecog = 0
    for rel in rels:
        s = side_of(rel)
        k = norm_key(rel)
        if s is None or k is None:
            n_unrecog += 1
            continue
        groups[k][s].append(rel)
    n_pairs = sum(1 for g in groups.values() if g['blur'] and g['sharp'])
    n_blur_only = sum(len(g['blur']) for g in groups.values() if g['blur'] and not g['sharp'])
    n_sharp_only = sum(len(g['sharp']) for g in groups.values() if g['sharp'] and not g['blur'])
    n_multi = sum(1 for g in groups.values() if len(g['blur']) > 1 or len(g['sharp']) > 1)
    print(f'\n[2] 配对情况 (blur/lq/input 与 sharp/gt 同路径配对)')
    print(f'    成功配对: {n_pairs} 对 | blur 落单: {n_blur_only} | sharp 落单: {n_sharp_only}'
          f' | 未识别: {n_unrecog} | 重复key: {n_multi}')
    if n_pairs:
        sample = next(g for g in groups.values() if g['blur'] and g['sharp'])
        print(f'    示例: {sample["blur"][0]}  <->  {sample["sharp"][0]}')

    # ---------- [3] 图片规格 ----------
    pair_list = sorted((g['blur'][0], g['sharp'][0])
                       for g in groups.values() if g['blur'] and g['sharp'])
    if args.limit:
        scan = pair_list[:args.limit]
    else:
        scan = pair_list
    show = min(5, len(pair_list))
    print(f'\n[3] 图片规格 (抽查 {show} 对)')
    for b, s in pair_list[:show]:
        raw_b = cv2.imread(os.path.join(root, b), cv2.IMREAD_UNCHANGED)
        raw_s = cv2.imread(os.path.join(root, s), cv2.IMREAD_UNCHANGED)
        fb = f'{raw_b.shape[1]}x{raw_b.shape[0]} {raw_b.dtype}' if raw_b is not None else '读失败'
        fs = f'{raw_s.shape[1]}x{raw_s.shape[0]} {raw_s.dtype}' if raw_s is not None else '读失败'
        print(f'    {b}: {fb}  |  sharp: {fs}')

    # ---------- [4] 块状盲元统计 ----------
    print(f'\n[4] 块状盲元统计 (|blur-gt|>{args.thr}, 块>={args.min_block}px)  扫描 {len(scan)} 对')
    bucket = {b: [0, 0] for b in BUCKETS}
    n_img_scan = n_img_block = bad = 0
    max_block = 0
    n_mask_px = vals_dark = vals_bright = 0
    val_sum = 0.0
    min_side = 10**9
    for b, s in scan:
        ib = load8(os.path.join(root, b))
        ish = load8(os.path.join(root, s))
        if ib is None or ish is None or ib.shape != ish.shape:
            bad += 1
            continue
        n_img_scan += 1
        min_side = min(min_side, min(ib.shape))
        mask = (cv2.absdiff(ib, ish) > args.thr).astype(np.uint8)
        n = int(mask.sum())
        if n == 0:
            continue
        m = mask.astype(bool)
        vals = ib[m].astype(np.float64)
        val_sum += float(vals.sum())
        n_mask_px += n
        vals_dark += int((vals < 10).sum())
        vals_bright += int((vals > 245).sum())
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        has_block = False
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < args.min_block:
                continue
            has_block = True
            max_block = max(max_block, area)
            for lo, hi in BUCKETS:
                if lo <= area < hi:
                    bucket[(lo, hi)][0] += 1
                    bucket[(lo, hi)][1] += area
                    break
        if has_block:
            n_img_block += 1
    print(f'    含块图像: {n_img_block} / {n_img_scan} '
          f'({100.0 * n_img_block / max(1, n_img_scan):.0f}%)'
          + (f' | 读失败/尺寸不符: {bad}' if bad else ''))
    if n_img_scan:
        warn = '  <-- 小于 384, 训练会走 padding 分支' if min_side < 384 else ''
        print(f'    图像最小边: {min_side}px{warn}')
    print(f'    {"块大小(px)":<12}{"块数":>8}{"平均面积":>10}')
    for (lo, hi), (cnt, area) in bucket.items():
        if cnt:
            label = f'{lo}-{hi - 1}' if hi < 10**9 else f'>={lo}'
            print(f'    {label:<12}{cnt:>8}{area / cnt:>10.1f}')
    if max_block:
        print(f'    最大单块: {max_block} px')
    if n_mask_px:
        print(f'    盲元填充值(blur侧): 均值 {val_sum / n_mask_px:.0f} | <10(黑) {100.0 * vals_dark / n_mask_px:.0f}%'
              f' | >245(白) {100.0 * vals_bright / n_mask_px:.0f}% | 盲元像素共 {n_mask_px}')


if __name__ == '__main__':
    main()
