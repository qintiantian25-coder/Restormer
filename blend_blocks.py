"""
盲元块"无缝融合"后处理: 消除网络填充残留的接缝 / 色差 (保持平滑, 不引入纹理).

原理 (纯经典算法, 不训练):
  1. mask = |blur-gt| > thr (离线用; 部署时换 --mask_dir 传盲元位置表) -> 微膨胀
  2. base = cv2.inpaint(blur, mask) = 由周围扩散出的平滑估计
  3. corr = 低通(base - pred): 只修正"低频色差/接缝"部分, 不碰填充的细节
  4. 距离羽化权重回加: 块边缘权重->0 (不引入新接缝), 块内部权重->1

用法 (在项目根目录):
  python blend_blocks.py --blur_dir data_diff/val_blur --gt_dir data_diff/val_sharp \
      --pred_dir results/dx_after3 --out_dir results/dx_after3_blend
  # 部署: --mask_dir <盲元表目录(结构镜像)> 代替 --gt_dir
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='盲元块无缝融合后处理')
    ap.add_argument('--blur_dir', required=True, help='含盲元原图目录 (结构镜像)')
    ap.add_argument('--gt_dir', default=None, help='离线: 干净图, 用 |blur-gt|>thr 生成 mask')
    ap.add_argument('--mask_dir', default=None, help='部署: 盲元表 mask 目录 (255=盲元), 优先')
    ap.add_argument('--pred_dir', required=True, help='网络修复图目录')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--thr', type=int, default=30)
    ap.add_argument('--dilate', type=int, default=5, help='mask 膨胀核 (覆盖过渡带)')
    ap.add_argument('--feather', type=int, default=4, help='边缘羽化宽度 px')
    ap.add_argument('--lp', type=int, default=9, help='低频滤波核 (奇数)')
    args = ap.parse_args()
    if not args.gt_dir and not args.mask_dir:
        raise SystemExit('需要 --gt_dir (离线) 或 --mask_dir (部署)')

    files = sorted(glob.glob(os.path.join(args.pred_dir, '**', '*.png'), recursive=True))
    import basicsr.metrics as metrics
    n = 0
    err_before = err_after = 0.0
    cnt = 0
    ps_b, ps_a = [], []
    for f in files:
        rel = os.path.relpath(f, args.pred_dir)
        blur = cv2.imread(os.path.join(args.blur_dir, rel), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if blur is None or pred is None:
            continue
        gt = cv2.imread(os.path.join(args.gt_dir, rel), cv2.IMREAD_GRAYSCALE) if args.gt_dir else None

        if args.mask_dir:
            m = cv2.imread(os.path.join(args.mask_dir, rel), cv2.IMREAD_GRAYSCALE)
            mask = (m > 127).astype(np.uint8) if m is not None else np.zeros_like(blur)
        else:
            mask = (cv2.absdiff(blur, gt) > args.thr).astype(np.uint8)

        if mask.sum() == 0:
            out = pred.copy()
        else:
            md = cv2.dilate(mask, np.ones((args.dilate, args.dilate), np.uint8))
            base = cv2.inpaint(blur, md, 5, cv2.INPAINT_TELEA).astype(np.float32)
            corr = cv2.GaussianBlur(base - pred.astype(np.float32), (args.lp, args.lp), 0)
            dist = cv2.distanceTransform(md, cv2.DIST_L2, 3)
            w = np.clip((dist - 1.0) / max(args.feather, 1), 0, 1).astype(np.float32) * md
            out = np.clip(pred.astype(np.float32) + corr * w, 0, 255).round().astype(np.uint8)

        outp = os.path.join(args.out_dir, rel)
        os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
        cv2.imwrite(outp, out)
        n += 1

        if gt is not None:
            m = mask.astype(bool)
            if m.any():
                err_before += float(np.abs(pred.astype(np.float32) - gt)[m].mean())
                err_after += float(np.abs(out.astype(np.float32) - gt)[m].mean())
                cnt += 1
            ps_b.append(metrics.calculate_psnr(pred, gt, crop_border=0, test_y_channel=False))
            ps_a.append(metrics.calculate_psnr(out, gt, crop_border=0, test_y_channel=False))

    print(f'完成 {n} 张 -> {args.out_dir}')
    if cnt:
        print(f'盲元区平均误差: 融合前 {err_before / cnt:.2f} -> 融合后 {err_after / cnt:.2f} (灰度级)')
    if ps_b:
        print(f'全图 PSNR: 融合前 {np.mean(ps_b):.4f} -> 融合后 {np.mean(ps_a):.4f}')


if __name__ == '__main__':
    main()
