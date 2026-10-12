"""
块状盲元诊断: 统计修复图在"盲元块"区域的修复程度, 按块大小分桶.

盲元判定: |blur - gt| > thr 的像素; 连通域 >= min_block 像素视为"块".
输出: 各尺寸桶的 块数 / 修复前误差 / 修复后误差 / 模型改动量, 以及最差样例拼图.

用法 (项目根目录, 不需要 GPU):
  python diag_block.py --blur_dir data_merged/test_blur --gt_dir data_merged/test_sharp \
      --pred_dir results/test_blur_eval --save_dir results/block_diag
"""
import argparse
import glob
import os

import cv2
import numpy as np

BUCKETS = [(6, 16), (16, 64), (64, 256), (256, 10**9)]


def main():
    ap = argparse.ArgumentParser(description='块状盲元修复诊断')
    ap.add_argument('--blur_dir', required=True, help='含盲元图目录')
    ap.add_argument('--gt_dir', required=True, help='干净图目录')
    ap.add_argument('--pred_dir', required=True, help='修复图目录 (infer_png 输出)')
    ap.add_argument('--limit', type=int, default=0, help='只扫描前 N 张 (0=全部)')
    ap.add_argument('--thr', type=int, default=30, help='盲元判定阈值 |blur-gt|, 默认 30')
    ap.add_argument('--min_block', type=int, default=6, help='连通域最小像素数, 更小的忽略')
    ap.add_argument('--top', type=int, default=6, help='保存最差 N 个块的对比拼图')
    ap.add_argument('--save_dir', default=None, help='拼图保存目录 (不给则不保存)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.blur_dir, '**', '*.png'), recursive=True))
    if args.limit:
        files = files[:args.limit]

    bucket = {b: {'n': 0, 'area': 0, 'e_in': 0.0, 'e_out': 0.0, 'moved': 0.0} for b in BUCKETS}
    img_rows = []      # (rel, 最大块, 图级修复前误差, 图级修复后误差)
    worst = []         # (e_out, rel, (x,y,w,h), area, blur, gt, pred)
    n_img = n_skip = 0

    for f in files:
        rel = os.path.relpath(f, args.blur_dir)
        gt_p = os.path.join(args.gt_dir, rel)
        pr_p = os.path.join(args.pred_dir, rel)
        if not (os.path.exists(gt_p) and os.path.exists(pr_p)):
            n_skip += 1
            continue
        blur = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pr_p, cv2.IMREAD_GRAYSCALE)
        if blur is None or gt is None or pred is None:
            n_skip += 1
            continue
        n_img += 1

        mask = (cv2.absdiff(blur, gt) > args.thr).astype(np.uint8)
        if mask.sum() == 0:
            continue
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        eb = np.abs(blur.astype(np.float32) - gt)   # 修复前误差
        eo = np.abs(pred.astype(np.float32) - gt)   # 修复后误差
        mv = np.abs(pred.astype(np.float32) - blur)  # 模型改动量

        img_big = 0
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < args.min_block:
                continue
            sel = labels == i
            e_in, e_out, moved = float(eb[sel].mean()), float(eo[sel].mean()), float(mv[sel].mean())
            for lo, hi in BUCKETS:
                if lo <= area < hi:
                    b = bucket[(lo, hi)]
                    b['n'] += 1
                    b['area'] += area
                    b['e_in'] += e_in
                    b['e_out'] += e_out
                    b['moved'] += moved
                    break
            img_big = max(img_big, area)
            if args.save_dir:
                bbox = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
                worst.append((e_out, rel, bbox, area, blur, gt, pred))

        m = mask.astype(bool)
        img_rows.append((rel, img_big, float(eb[m].mean()), float(eo[m].mean())))

    print(f'扫描 {n_img} 张 (跳过 {n_skip}); 盲元判定 |blur-gt| > {args.thr}, 块 >= {args.min_block} px\n')
    print(f'{"块大小(px)":<12}{"块数":>7}{"平均面积":>10}{"修复前误差":>11}{"修复后误差":>11}{"模型改动量":>11}')
    for (lo, hi), b in bucket.items():
        if not b['n']:
            continue
        label = f'{lo}-{hi - 1}' if hi < 10**9 else f'>={lo}'
        print(f'{label:<12}{b["n"]:>7}{b["area"] / b["n"]:>10.1f}'
              f'{b["e_in"] / b["n"]:>11.1f}{b["e_out"] / b["n"]:>11.1f}{b["moved"] / b["n"]:>11.1f}')
    print('\n误差单位 = 灰度级(0-255)。判读: "修复后误差"越接近 0 越好; "模型改动量"接近 0 = 模型对块几乎没动。'
          '块越大 修复后误差 越高 = 大块短板。')

    if img_rows:
        img_rows.sort(key=lambda r: -r[2])   # 按图级修复前误差(块区)排
        print(f'\n图级块区误差 最差的 {min(10, len(img_rows))} 张:')
        print(f'  {"文件":<44}{"最大块px":>9}{"修复前":>8}{"修复后":>8}')
        for rel, big, e_in, e_out in img_rows[:10]:
            print(f'  {rel:<44}{big:>9}{e_in:>8.1f}{e_out:>8.1f}')
        all_in = np.mean([r[2] for r in img_rows])
        all_out = np.mean([r[3] for r in img_rows])
        print(f'\n全部 {len(img_rows)} 张的块区平均: 修复前 {all_in:.1f} -> 修复后 {all_out:.1f}')

    if args.save_dir and worst:
        os.makedirs(args.save_dir, exist_ok=True)
        worst.sort(key=lambda w: -w[0])
        margin = 24
        for rank, (e_out, rel, (x, y, w, h), area, blur, gt, pred) in enumerate(worst[:args.top], 1):
            H, W = blur.shape
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
            crops = [c[y0:y1, x0:x1] for c in (blur, gt, pred)]
            min_side = min(min(c.shape[:2]) for c in crops)
            scale = max(1, int(160 / max(1, min_side)))
            crops = [cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                     for c in crops]
            sep = np.full((crops[0].shape[0], 4), 255, np.uint8)
            panel = np.hstack([crops[0], sep, crops[1], sep, crops[2]])
            for j, tag in enumerate(('blur', 'gt', 'pred')):
                cv2.putText(panel, tag, (j * (crops[0].shape[1] + 4) + 4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255 if crops[0].max() < 128 else 0, 1, cv2.LINE_AA)
            name = f'rank{rank:02d}_area{area}_err{e_out:.0f}_' + rel.replace('/', '_').replace('\\', '_')
            cv2.imwrite(os.path.join(args.save_dir, name), panel)
        print(f'\n最差 {min(args.top, len(worst))} 块拼图 (blur | gt | pred) 已存到 {args.save_dir}')


if __name__ == '__main__':
    main()
