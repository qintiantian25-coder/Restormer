"""
盲元修复质量评估 (输出 vs 输入).

指标:
  Residual_X  — 像素与自身邻域中值的偏差 > X 的占比, 越小越好
  LocalStd    — 局部标准差均值 (5×5), 越小越好
  Roughness   — 归一化 Laplacian 高通能量, 越小越好
  NU          — 非均匀性 (局部均值变异系数), 越小越好
  EstSNR      — 估计信噪比 (Immerkaer), 越大越好

不依赖 RGB 预训练模型, 纯数学计算, 天然适配灰度红外图像.

用法:
  python evaluate_nr.py
  python evaluate_nr.py --output <修复图> --input <原图> --save <结果> --thresholds 5 10 20 30 50
"""

import os, re, csv, argparse
import cv2, numpy as np


DEFAULTS = {
    'output': r"/root/Qtt/FGAF-Net/results/FGAF-Net_BlindPixel_stage2_real/test",
    'input':  r"/root/Qtt/FGAF-Net/real_image/test_blur",
    'save':   r"/root/Qtt/FGAF-Net/results/FGAF-Net_BlindPixel_stage2_real/nr_eval",
}
DEFAULT_THRESHOLDS = [5, 10, 20, 30, 50]


# =====================================================================
# 指标
# =====================================================================

def compute_residual(img, thresh, kernel_size=5):
    """像素与自身邻域中值的偏差 > thresh 的占比 (%), 越小越好."""
    f = img.astype(np.float64)
    med = cv2.medianBlur(img, kernel_size).astype(np.float64)
    return float(100.0 * (np.abs(f - med) > thresh).sum() / img.size)


def compute_local_std(img, kernel_size=5):
    """局部标准差均值, 越小越好."""
    f = img.astype(np.float64)
    m = cv2.blur(f, (kernel_size, kernel_size))
    ms = cv2.blur(f**2, (kernel_size, kernel_size))
    return float(np.mean(np.sqrt(np.maximum(ms - m**2, 0))))


def compute_roughness(img):
    """归一化 Laplacian 高通能量, 越小越好."""
    f = img.astype(np.float64)
    lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
    hf = np.abs(cv2.filter2D(f, -1, lap))
    return float(np.mean(hf) / (np.mean(f) + 1e-10))


def compute_nu(img, block_size=32):
    """非均匀性 (局部均值变异系数), 越小越好."""
    f = img.astype(np.float64)
    h, w = f.shape
    means = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = f[y:min(y+block_size, h), x:min(x+block_size, w)]
            if block.size > block_size:
                means.append(np.mean(block))
    means = np.array(means)
    return float(100.0 * np.std(means) / (np.mean(means) + 1e-10))


def compute_est_snr(img):
    """Immerkaer 拉普拉斯噪声估计 SNR, 越大越好."""
    f = img.astype(np.float64)
    lap = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float64)
    laplacian = cv2.filter2D(f, -1, lap)
    noise_var = np.var(laplacian) / 72.0
    signal_var = max(np.var(f) - noise_var, 1e-10)
    return float(10.0 * np.log10(signal_var / noise_var))


# =====================================================================
# 工具
# =====================================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def _resolve_path(base_dir, seq_name, rel_path, img_name):
    p = os.path.join(base_dir, rel_path)
    if os.path.exists(p): return p
    p = os.path.join(base_dir, seq_name, img_name)
    if os.path.exists(p): return p
    return None


# =====================================================================
# 主逻辑
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        parser.add_argument(f'--{k}', default=v)
    parser.add_argument('--thresholds', nargs='+', type=int, default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    thresholds = args.thresholds
    OUTPUT_DIR = args.output
    INPUT_DIR  = args.input
    SAVE_DIR   = args.save
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 扫描输出目录
    out_records = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if not f.endswith('.png'): continue
            op = os.path.join(root, f)
            rel = os.path.relpath(op, OUTPUT_DIR).replace('\\', '/')
            out_records.append({
                'out_path': op, 'img_name': f, 'rel_path': rel,
                'seq': rel.split('/')[0] if '/' in rel else 'root',
            })
    out_records.sort(key=lambda r: natural_sort_key(r['rel_path']))
    print(f"找到 {len(out_records)} 张输出图片")

    seq_records = {}
    for r in out_records:
        seq_records.setdefault(r['seq'], []).append(r)

    METRICS = ([f'residual_{t}' for t in thresholds] +
               ['localstd', 'roughness', 'nu', 'estsnr'])
    DIRECTION = {f'residual_{t}': '↓' for t in thresholds}
    for m in ['localstd', 'roughness', 'nu']:
        DIRECTION[m] = '↓'
    DIRECTION['estsnr'] = '↑'

    keys = ['image', 'seq']
    for m in METRICS:
        keys += [f'{m}_out', f'{m}_in']

    per_image_rows, seq_stats = [], {}
    global_vals = {f'{m}_out': [] for m in METRICS}
    for m in METRICS:
        global_vals[f'{m}_in'] = []

    print("===> 开始评估...")

    for seq_name in sorted(seq_records, key=natural_sort_key):
        seq_recs = sorted(seq_records[seq_name], key=lambda r: natural_sort_key(r['rel_path']))
        sm = {f'{m}_out': [] for m in METRICS}
        for m in METRICS:
            sm[f'{m}_in'] = []

        for idx, rec in enumerate(seq_recs):
            out_img = cv2.imread(rec['out_path'], cv2.IMREAD_GRAYSCALE)
            if out_img is None: continue
            in_path = _resolve_path(INPUT_DIR, rec['seq'], rec['rel_path'], rec['img_name'])

            out_vals = {}
            for t in thresholds:
                out_vals[f'residual_{t}'] = compute_residual(out_img, t)
            out_vals['localstd'] = compute_local_std(out_img)
            out_vals['roughness'] = compute_roughness(out_img)
            out_vals['nu'] = compute_nu(out_img)
            out_vals['estsnr'] = compute_est_snr(out_img)

            in_vals = {}
            if in_path and os.path.exists(in_path):
                in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                if in_img is not None:
                    for t in thresholds:
                        in_vals[f'residual_{t}'] = compute_residual(in_img, t)
                    in_vals['localstd'] = compute_local_std(in_img)
                    in_vals['roughness'] = compute_roughness(in_img)
                    in_vals['nu'] = compute_nu(in_img)
                    in_vals['estsnr'] = compute_est_snr(in_img)

            row = {'image': rec['rel_path'], 'seq': rec['seq']}
            for m in METRICS:
                row[f'{m}_out'] = round(out_vals.get(m), 6) if out_vals.get(m) is not None else None
                row[f'{m}_in'] = round(in_vals.get(m), 6) if in_vals.get(m) is not None else None
            per_image_rows.append(row)

            for m in METRICS:
                for v, lst in [(out_vals, f'{m}_out'), (in_vals, f'{m}_in')]:
                    if v.get(m) is not None:
                        sm[lst].append(v[m])
                        global_vals[lst].append(v[m])

            if (idx+1) % 10 == 0 or idx == len(seq_recs)-1:
                print(f"  [{rec['seq']}] {idx+1}/{len(seq_recs)}  "
                      f"Res10={out_vals.get('residual_10'):.1f}% vs {in_vals.get('residual_10')}%  "
                      f"Std={out_vals.get('localstd'):.1f}  SNR={out_vals.get('estsnr'):.1f}")

        if sm['localstd_out']:
            st = {'count': len(sm['localstd_out'])}
            for m in METRICS:
                st[f'{m}_out'] = float(np.mean(sm[f'{m}_out'])) if sm[f'{m}_out'] else None
                st[f'{m}_in'] = float(np.mean(sm[f'{m}_in'])) if sm[f'{m}_in'] else None
            seq_stats[seq_name] = st

    # CSV
    csv_path = os.path.join(SAVE_DIR, 'nr_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_image_rows: w.writerow(row)
        for sn in sorted(seq_stats, key=natural_sort_key):
            st = seq_stats[sn]
            r = {'image': f'AVERAGE ({sn})', 'seq': sn}
            for m in METRICS:
                r[f'{m}_out'] = round(st[f'{m}_out'], 6) if st[f'{m}_out'] is not None else None
                r[f'{m}_in'] = round(st[f'{m}_in'], 6) if st[f'{m}_in'] is not None else None
            w.writerow(r)
        def _a(lst): return float(np.mean(lst)) if lst else None
        r = {'image': 'AVERAGE', 'seq': ''}
        for m in METRICS:
            r[f'{m}_out'] = round(_a(global_vals[f'{m}_out']), 6) if global_vals[f'{m}_out'] else None
            r[f'{m}_in'] = round(_a(global_vals[f'{m}_in']), 6) if global_vals[f'{m}_in'] else None
        w.writerow(r)

    def _a(lst): return float(np.mean(lst)) if lst else None
    print(f"\nCSV: {csv_path}")
    print(f"{'='*70}")
    print(f"总体平均 ({len(global_vals['localstd_out'])} 张):")
    print(f"{'':>14s} {'方向':>4s} {'输出':>14s}  {'输入':>14s}")
    for m in METRICS:
        so = f"{_a(global_vals[f'{m}_out']):.4f}"
        si = f"{_a(global_vals[f'{m}_in']):.4f}" if global_vals[f'{m}_in'] else "N/A"
        print(f"  {m:14s} {DIRECTION[m]:>4s} {so:>14s}  {si:>14s}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
