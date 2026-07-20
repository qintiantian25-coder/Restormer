"""
No-reference clarity / cleanness metrics for grayscale IR blind-pixel images.

Metrics (non-learning, grayscale-native):
  - Local Std ↓      : lower = smoother flat regions (fewer blind pixels / noise)
  - SRIS ↑           : Signal-to-Residual-Intensity Spread (estimated SNR)
  - Edge Preservation : true edges vs noise ratio

Usage:
    python eval_sharpness.py
"""

import cv2, numpy as np
from scipy.ndimage import sobel, uniform_filter


def local_std(img, win=5):
    """Mean local standard deviation — lower = cleaner (fewer outliers)."""
    sq = uniform_filter(img.astype(np.float64)**2, win)
    mu = uniform_filter(img.astype(np.float64), win)
    std_map = np.sqrt(np.maximum(sq - mu**2, 0))
    return float(std_map.mean())


def estimated_snr(img, win=7):
    """
    Signal-to-Residual-Intensity Spread.
    SNR ≈ mean(local_mean) / mean(local_std).  Higher = cleaner.
    """
    mu = uniform_filter(img.astype(np.float64), win)
    std_map = np.sqrt(np.maximum(uniform_filter(img.astype(np.float64)**2, win) - mu**2, 0))
    eps = 1e-6
    return float((mu / (std_map + eps)).mean())


def edge_cleanliness(img, win=5):
    """
    Ratio of strong edges (true structure) to weak edges (noise/blind pixels).
    Higher = more true edges, less noise.
    """
    gx = sobel(img.astype(np.float64), axis=0)
    gy = sobel(img.astype(np.float64), axis=1)
    grad = np.sqrt(gx**2 + gy**2)
    strong = (grad > grad.mean() * 2).sum()
    weak = (grad < grad.mean() * 0.3).sum()
    return float(strong / max(weak, 1))


def main():
    images = {
        'image1 (盲元-blur)':      'image1.png',
        'image2 (传统方法)':        'image2.png',
        'image3 (Restormer合并)': 'image3.png',
    }

    results = {}
    for label, path in images.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        results[label] = {
            'Local Std ↓':   local_std(img),
            'Est. SNR ↑':    estimated_snr(img),
            'Edge Clean ↑':  edge_cleanliness(img),
        }

    metrics = list(next(iter(results.values())).keys())

    print('### 红外盲元图像无参考质量评价 (灰度图原生)\n')
    header = '| 图像 | ' + ' | '.join(metrics) + ' |'
    sep = '|------|' + '|'.join(['--------'] * len(metrics)) + '|'
    print(header)
    print(sep)
    for label, r in results.items():
        vals = ' | '.join(f'{r[m]:.4f}' for m in metrics)
        print(f'| {label} | {vals} |')

    print()
    print('| 指标 | 方向 | 原理 |')
    print('|------|------|------|')
    print('| Local Std | ↓ | 局部标准差均值, 越低越干净 |')
    print('| Est. SNR | ↑ | 局部信噪比估计, 越高噪声越少 |')
    print('| Edge Clean | ↑ | 强边缘/弱边缘比, 越高结构越清晰 |')


if __name__ == '__main__':
    main()
