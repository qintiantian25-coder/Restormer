"""
从 blur/sharp 配对生成盲元 mask PNG (|blur-sharp| > thr -> 255, 其余 0).

输出目录结构与 blur_dir 一致 (子目录/文件名镜像), 供训练 yml 的 dataroot_mask 使用.

用法 (项目根目录):
  python make_masks.py --blur_dir data_diff/train_blur --gt_dir data_diff/train_sharp \
      --out_dir data_diff/train_mask --thr 30
"""
import argparse
import glob
import os

import cv2
import numpy as np


def load8(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.dtype == np.uint16:
        im = (im / 256.0).astype(np.uint8)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def main():
    ap = argparse.ArgumentParser(description='生成盲元 mask PNG')
    ap.add_argument('--blur_dir', required=True, help='含盲元图目录')
    ap.add_argument('--gt_dir', required=True, help='干净图目录 (同结构)')
    ap.add_argument('--out_dir', required=True, help='mask 输出目录')
    ap.add_argument('--thr', type=int, default=30, help='盲元判定阈值 |blur-gt|')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.blur_dir, '**', '*.png'), recursive=True))
    n_ok = n_skip = tot_px = 0
    for f in files:
        rel = os.path.relpath(f, args.blur_dir)
        gt_path = os.path.join(args.gt_dir, rel)
        if not os.path.exists(gt_path):
            n_skip += 1
            continue
        b = load8(f)
        g = load8(gt_path)
        if b is None or g is None or b.shape != g.shape:
            n_skip += 1
            continue
        mask = ((cv2.absdiff(b, g) > args.thr) * 255).astype(np.uint8)
        out_path = os.path.join(args.out_dir, rel)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cv2.imwrite(out_path, mask)
        tot_px += int((mask > 0).sum())
        n_ok += 1
    print(f'完成: {n_ok} 张 mask -> {args.out_dir} (跳过 {n_skip}), 盲元像素共 {tot_px}')


if __name__ == '__main__':
    main()
