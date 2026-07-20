"""
Final evaluation table for IR blind-pixel restoration.

Three-tier evaluation:
  1. Blind pixel residual (objective detection)
  2. Local Std / SNR (statistical cleanness)
  3. DBCNN (IEEE TIP 2018, VGG branch covers point defects)

Usage:
    python eval_final.py
"""

import cv2, numpy as np
import torch, pyiqa


# =====  Tier 1: blind pixel residual detection  =====

def blind_residual(img, kernel=5, thresholds=(10, 20, 30, 50)):
    """像素与自身邻域中值偏差 > th 的占比 (%), 越小越好."""
    f = img.astype(np.float64)
    med = cv2.medianBlur(img, kernel).astype(np.float64)
    diff = np.abs(f - med)
    return {f'residual_{th}': float(100.0 * (diff > th).sum() / img.size)
            for th in thresholds}


# =====  Tier 2: statistical cleanness  =====

def local_std(img, kernel_size=5):
    """局部标准差均值, 越小越好."""
    f = img.astype(np.float64)
    mean = cv2.blur(f, (kernel_size, kernel_size))
    mean_sq = cv2.blur(f ** 2, (kernel_size, kernel_size))
    local = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    return float(np.mean(local))


def est_snr(img):
    """Immerkaer 拉普拉斯噪声估计 SNR, 越大越好."""
    f = img.astype(np.float64)
    lap = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    laplacian = cv2.filter2D(f, -1, lap)
    noise_var = np.var(laplacian) / 72.0
    signal_var = max(np.var(f) - noise_var, 1e-10)
    return float(10.0 * np.log10(signal_var / noise_var))


# =====  Tier 3: DBCNN  =====

def eval_dbcnn(metric, img_gray):
    img = cv2.resize(img_gray, (512, 512))
    t = torch.from_numpy(img.astype(np.float32)/255.).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).cuda()
    with torch.no_grad():
        return float(metric(t).item())


# =====  Main  =====

def main():
    images = {
        '盲元原图':       'image1.png',
        '传统方法':       'image2.png',
        'Restormer (合并)': 'image3.png',
    }

    device = torch.device('cuda')
    dbcnn = pyiqa.create_metric('dbcnn', device=device)

    results = {}
    for label, path in images.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        r = {}
        r.update(blind_residual(img))
        r['Local Std ↓']  = local_std(img)
        r['Est. SNR ↑']   = est_snr(img)
        r['DBCNN ↓']      = eval_dbcnn(dbcnn, img)
        results[label] = r

    # ---- Print Table ----
    print()
    print('## 红外盲元修复最终评价\n')

    print('### 第一级：盲元残差检测 (客观修复率)\n')
    print('| 图像 | residual_10 (%) | residual_20 (%) | residual_30 (%) | residual_50 (%) |')
    print('|------|----------------|----------------|----------------|----------------|')
    for label, r in results.items():
        print(f'| {label} | {r["residual_10"]:>14.6f} | {r["residual_20"]:>14.6f} | {r["residual_30"]:>14.6f} | {r["residual_50"]:>14.6f} |')
    print(f'\n*核大小 5×5, 像素总数 {img.size:,d}*\n')

    print('### 第二级：统计清洁度 + 第三级：感知质量\n')
    print('| 图像 | Local Std ↓ | Est. SNR ↑ | DBCNN ↓ |')
    print('|------|-------------|------------|---------|')
    for label, r in results.items():
        print(f'| {label} | {r["Local Std ↓"]:.4f} | {r["Est. SNR ↑"]:.2f} | {r["DBCNN ↓"]:.4f} |')

    print(f'\n*DBCNN: Ma et al., IEEE TIP 2018; VGG分支用于合成失真, 覆盖点状缺陷*')

    # ---- Improvement ratios ----
    print('\n### 修复率\n')
    b = results['盲元原图']
    t = results['传统方法']
    rs = results['Restormer (合并)']
    for th in [10, 20, 30, 50]:
        k = f'residual_{th}'
        t_pct = (1 - t[k]/max(b[k], 1e-10)) * 100
        r_pct = (1 - rs[k]/max(b[k], 1e-10)) * 100
        print(f'- {k}: 传统 {t_pct:.1f}%, Restormer {r_pct:.1f}%')


if __name__ == '__main__':
    main()
