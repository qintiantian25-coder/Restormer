"""
并排对比图生成: 原图 | 各模型修复 | GT, 直接目视检查用.

用法 (项目根目录):
  # 随机抽 20 张: 原位 | 旧模型(te_A) | 新模型(te_B) | GT 四联
  python montage_compare.py --blur_dir data_merged/test_blur --gt_dir data_merged/test_sharp \
      --pred results/te_A results/te_B --out_dir results/montage_AB --limit 20

  # 单模型: 原图 | 修复 | GT
  python montage_compare.py --blur_dir ... --gt_dir ... --pred results/te_B --out_dir ... --limit 30

  # 指定子序列全部: --subdir 001   |  指定文件: --names frame_0001.png,frame_0002.png
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='并排对比图')
    ap.add_argument('--blur_dir', required=True)
    ap.add_argument('--gt_dir', default=None)
    ap.add_argument('--pred', nargs='+', required=True, help='修复图目录 (可给多个模型, 按顺序并排)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--subdir', default=None, help='只选某个子目录 (如 001)')
    ap.add_argument('--names', default=None, help='逗号分隔的文件名/相对路径列表')
    ap.add_argument('--limit', type=int, default=20, help='随机抽样数量 (0=全部)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.blur_dir, '**', '*.png'), recursive=True))
    rels = [os.path.relpath(f, args.blur_dir) for f in files]
    if args.subdir:
        rels = [r for r in rels if os.path.basename(os.path.dirname(r)) == args.subdir]
    if args.names:
        wanted = {n.strip() for n in args.names.split(',') if n.strip()}
        rels = [r for r in rels if r in wanted or os.path.basename(r) in wanted]
    if args.limit and len(rels) > args.limit:
        rels = sorted(random.Random(args.seed).sample(rels, args.limit))

    os.makedirs(args.out_dir, exist_ok=True)
    saved = skipped = 0
    for rel in rels:
        b = cv2.imread(os.path.join(args.blur_dir, rel), cv2.IMREAD_GRAYSCALE)
        if b is None:
            skipped += 1
            continue
        panels = [(b, 'input')]
        ok = True
        for pdir in args.pred:
            p = cv2.imread(os.path.join(pdir, rel), cv2.IMREAD_GRAYSCALE)
            if p is None:
                ok = False
                break
            panels.append((p, os.path.basename(os.path.normpath(pdir))))
        if not ok:
            skipped += 1
            continue
        if args.gt_dir:
            g = cv2.imread(os.path.join(args.gt_dir, rel), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                panels.append((g, 'gt'))

        sep = np.full((b.shape[0], 3, 3), 255, np.uint8)
        row = []
        for i, (img, tag) in enumerate(panels):
            if i:
                row.append(sep)
            t = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
            cv2.putText(t, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255), 1, cv2.LINE_AA)
            row.append(t)
        name = rel.replace(os.sep, '_').replace('/', '_')
        cv2.imwrite(os.path.join(args.out_dir, name), np.hstack(row))
        saved += 1
    print(f'生成 {saved} 张对比图 -> {args.out_dir}' + (f' (跳过 {skipped})' if skipped else ''))


if __name__ == '__main__':
    main()
