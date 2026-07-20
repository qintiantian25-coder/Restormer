"""
无参考图像质量评估 (输出 vs 输入).

指标 (均为无参考, 越小越好):
  NIQE  — Naturalness Image Quality Evaluator, NSS 特征 + MVG 距离
  DBCNN — Deep Blind CNN, 合成+真实失真联合训练

依赖: pip install pyiqa

用法:
  python evaluate_nr.py
"""

import os
import re
import csv
import cv2
import numpy as np
import torch
import pyiqa


# =====================================================================
# 配置
# =====================================================================

OUTPUT_DIR = r"/root/Qtt/FGAF-Net/results/FGAF-Net_BlindPixel_stage2_real/test"
GT_DIR     = r"/root/Qtt/FGAF-Net/real_image/test_sharp"
INPUT_DIR  = r"/root/Qtt/FGAF-Net/real_image/test_blur"
SAVE_DIR   = r"/root/Qtt/FGAF-Net/results/FGAF-Net_BlindPixel_stage2_real/nr_eval"

NR_METRICS = ['niqe', 'dbcnn']


# =====================================================================
# MRD 实现 (参考 mrd.m)
# =====================================================================

def compute_mrd(img_clear: np.ndarray, img_restored: np.ndarray) -> float:
    """
    Mean Relative Deviation.
    |GT - restored| / max(restored, 1) 逐像素均值, 越小越好.
    等价于 mrd.m: abs(clear - noise) / noise, noise==0 时设为 1.
    """
    clear = img_clear.astype(np.float64)
    restored = img_restored.astype(np.float64)
    denom = np.maximum(restored, 1.0)
    return float(np.mean(np.abs(clear - restored) / denom))


# =====================================================================
# IQA 计算
# =====================================================================

def _to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    return t


class IQAComputer:
    def __init__(self):
        self._metrics = {}

    def _get_metric(self, name):
        if name not in self._metrics:
            self._metrics[name] = pyiqa.create_metric(
                name,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            )
        return self._metrics[name]

    def compute(self, metric_name: str, img: np.ndarray) -> float:
        t = _to_tensor(img)
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)  # 灰度 → RGB
        with torch.no_grad():
            return float(self._get_metric(metric_name)(t).item())


# =====================================================================
# 工具
# =====================================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def _resolve_path(base_dir, seq_name, rel_path, img_name):
    p = os.path.join(base_dir, rel_path)
    if os.path.exists(p):
        return p
    p = os.path.join(base_dir, seq_name, img_name)
    if os.path.exists(p):
        return p
    return None


# =====================================================================
# 主逻辑
# =====================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    iqa = IQAComputer()

    # 扫描输出目录
    out_records = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if not f.endswith('.png'):
                continue
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

    keys = ['image', 'seq', 'mrd_out', 'mrd_in']
    for m in NR_METRICS:
        keys += [f'{m}_out', f'{m}_in']

    per_image_rows = []
    seq_stats = {}
    global_vals = {'mrd_out': [], 'mrd_in': []}
    for m in NR_METRICS:
        global_vals[f'{m}_out'] = []
        global_vals[f'{m}_in'] = []

    print("===> 开始评估...")

    for seq_name in sorted(seq_records, key=natural_sort_key):
        seq_recs = sorted(seq_records[seq_name], key=lambda r: natural_sort_key(r['rel_path']))
        sm = {'mrd_out': [], 'mrd_in': []}
        for m in NR_METRICS:
            sm[f'{m}_out'] = []
            sm[f'{m}_in'] = []

        for idx, rec in enumerate(seq_recs):
            out_path = rec['out_path']
            rel_path = rec['rel_path']

            in_path = _resolve_path(INPUT_DIR, seq_name, rel_path, rec['img_name'])

            out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
            if out_img is None:
                print(f"  警告: 无法读取 {out_path}, 跳过")
                continue

            gt_path = _resolve_path(GT_DIR, seq_name, rel_path, rec['img_name'])

            # MRD
            mrd_out, mrd_in = None, None
            if gt_path and os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is not None:
                    gh, gw = gt_img.shape[:2]
                    out_r = cv2.resize(out_img, (gw, gh)) if (h, w) != (gh, gw) else out_img
                    mrd_out = compute_mrd(gt_img, out_r)

            # 无参考指标
            nr_out = {m: iqa.compute(m, out_img) for m in NR_METRICS}

            nr_in = {}
            if in_path and os.path.exists(in_path):
                in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                if in_img is not None:
                    if mrd_out is not None and gt_path and os.path.exists(gt_path):
                        gt_img2 = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        if gt_img2 is not None:
                            gh, gw = gt_img2.shape[:2]
                            in_r = cv2.resize(in_img, (gw, gh)) if in_img.shape != (gh, gw) else in_img
                            mrd_in = compute_mrd(gt_img2, in_r)
                    nr_in = {m: iqa.compute(m, in_img) for m in NR_METRICS}

            row = {'image': rel_path, 'seq': seq_name,
                   'mrd_out': round(mrd_out, 6) if mrd_out is not None else None,
                   'mrd_in': round(mrd_in, 6) if mrd_in is not None else None}
            for m in NR_METRICS:
                row[f'{m}_out'] = round(nr_out[m], 6)
                row[f'{m}_in'] = round(nr_in[m], 6) if m in nr_in else None
            per_image_rows.append(row)

            def _append(lst, v):
                if v is not None:
                    lst.append(v)
            _append(sm['mrd_out'], mrd_out)
            _append(sm['mrd_in'], mrd_in)
            _append(global_vals['mrd_out'], mrd_out)
            _append(global_vals['mrd_in'], mrd_in)
            for m in NR_METRICS:
                sm[f'{m}_out'].append(nr_out[m])
                global_vals[f'{m}_out'].append(nr_out[m])
                if m in nr_in:
                    sm[f'{m}_in'].append(nr_in[m])
                    global_vals[f'{m}_in'].append(nr_in[m])

            if (idx + 1) % 10 == 0 or idx == len(seq_recs) - 1:
                parts = f"MRD={mrd_out}(out) vs {mrd_in}(in)  "
                parts += "  ".join(f"{m.upper()}={nr_out[m]:.3f}(out) vs {nr_in.get(m)}(in)" for m in NR_METRICS)
                print(f"  [{seq_name}] {idx + 1}/{len(seq_recs)}  {parts}")

        if sm['niqe_out']:
            st = {'count': len(sm['niqe_out']),
                  'mrd_out': float(np.mean(sm['mrd_out'])) if sm['mrd_out'] else None,
                  'mrd_in': float(np.mean(sm['mrd_in'])) if sm['mrd_in'] else None}
            for m in NR_METRICS:
                st[f'{m}_out'] = float(np.mean(sm[f'{m}_out']))
                st[f'{m}_in'] = float(np.mean(sm[f'{m}_in'])) if sm[f'{m}_in'] else None
            seq_stats[seq_name] = st

    # ---- CSV ----
    csv_path = os.path.join(SAVE_DIR, 'nr_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow(row)

        for sn in sorted(seq_stats, key=natural_sort_key):
            st = seq_stats[sn]
            r = {'image': f'AVERAGE ({sn})', 'seq': sn,
                 'mrd_out': round(st['mrd_out'], 6) if st['mrd_out'] is not None else None,
                 'mrd_in': round(st['mrd_in'], 6) if st['mrd_in'] is not None else None}
            for m in NR_METRICS:
                r[f'{m}_out'] = round(st[f'{m}_out'], 6)
                r[f'{m}_in'] = round(st[f'{m}_in'], 6) if st[f'{m}_in'] is not None else None
            writer.writerow(r)

        def _a(lst): return float(np.mean(lst)) if lst else None

        r = {'image': 'AVERAGE', 'seq': '',
             'mrd_out': round(_a(global_vals['mrd_out']), 6) if global_vals['mrd_out'] else None,
             'mrd_in': round(_a(global_vals['mrd_in']), 6) if global_vals['mrd_in'] else None}
        for m in NR_METRICS:
            r[f'{m}_out'] = round(_a(global_vals[f'{m}_out']), 6)
            r[f'{m}_in'] = round(_a(global_vals[f'{m}_in']), 6) if global_vals[f'{m}_in'] else None
        writer.writerow(r)

    def _a(lst): return float(np.mean(lst)) if lst else None

    print(f"\nCSV: {csv_path}")
    print(f"{'='*60}")
    print(f"总体平均 ({len(global_vals['niqe_out'])} 张):")
    print(f"{'':>12s} {'输出(修复后)':>14s}  {'输入(模糊)':>14s}")
    for label in ['mrd'] + NR_METRICS:
        so = f"{_a(global_vals[f'{label}_out']):.4f}"
        si = f"{_a(global_vals[f'{label}_in']):.4f}" if global_vals[f'{label}_in'] else "N/A"
        print(f"  {label.upper():10s} {so:>14s}  {si:>14s}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
