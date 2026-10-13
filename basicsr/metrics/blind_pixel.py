import csv
import os

import numpy as np
try:
    import torch
except ImportError:
    torch = None


_COORD_CACHE = {}


def _to_numpy_hwc(img):
    if torch is not None and isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img.squeeze(0)
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
    return np.asarray(img)


def _to_gray_float64(img):
    arr = _to_numpy_hwc(img).astype(np.float64)
    if arr.ndim == 3:
        # Keep evaluation consistent for RGB/gray images.
        arr = arr[..., 0]
    return arr


def _load_blind_coords(csv_path, x_col='x', y_col='y'):
    cache_key = (csv_path, x_col, y_col)
    if cache_key in _COORD_CACHE:
        return _COORD_CACHE[cache_key]

    if not csv_path or not os.path.exists(csv_path):
        _COORD_CACHE[cache_key] = None
        return None

    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or x_col not in reader.fieldnames or y_col not in reader.fieldnames:
            _COORD_CACHE[cache_key] = None
            return None
        for row in reader:
            try:
                coords.append((int(float(row[x_col])), int(float(row[y_col]))))
            except Exception:
                continue

    if not coords:
        _COORD_CACHE[cache_key] = None
        return None

    arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
    _COORD_CACHE[cache_key] = arr
    return arr


def _blind_errors(img1, img2, csv_path, x_col='x', y_col='y'):
    pred = _to_gray_float64(img1)
    gt = _to_gray_float64(img2)
    if pred.shape != gt.shape:
        raise ValueError(f'Image shapes are different: {pred.shape}, {gt.shape}.')

    coords = _load_blind_coords(csv_path, x_col=x_col, y_col=y_col)
    if coords is None:
        return np.array([], dtype=np.float64)

    h, w = gt.shape[:2]
    x = coords[:, 0]
    y = coords[:, 1]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(valid):
        return np.array([], dtype=np.float64)

    x = x[valid]
    y = y[valid]
    return pred[y, x] - gt[y, x]


def calculate_blind_mae(img1, img2, csv_path, x_col='x', y_col='y'):
    err = _blind_errors(img1, img2, csv_path, x_col=x_col, y_col=y_col)
    if err.size == 0:
        return 0.0
    return float(np.abs(err).mean())


def calculate_blind_rmse(img1, img2, csv_path, x_col='x', y_col='y'):
    err = _blind_errors(img1, img2, csv_path, x_col=x_col, y_col=y_col)
    if err.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(err ** 2)))


def calculate_blind_psnr(img1, img2, csv_path, x_col='x', y_col='y', peak=255.0):
    err = _blind_errors(img1, img2, csv_path, x_col=x_col, y_col=y_col)
    if err.size == 0:
        return 0.0
    mse = float(np.mean(err ** 2))
    return float(10.0 * np.log10((peak * peak) / max(mse, 1e-12)))


def calculate_blind_count(img1, img2, csv_path, x_col='x', y_col='y'):
    err = _blind_errors(img1, img2, csv_path, x_col=x_col, y_col=y_col)
    return float(err.size)


# ============================================================
#  现场版: 不需要 CSV 坐标表, mask 由 |lq - gt| > thr 直接算出
#  (训练验证用; 无盲元的图返回 NaN, 由调用方跳过)
# ============================================================

def _mask_from_lq(gt, lq, thr=30):
    gt_a = _to_gray_float64(gt)
    lq_a = _to_gray_float64(lq)
    if gt_a.shape != lq_a.shape:
        raise ValueError(f'Image shapes are different: {gt_a.shape}, {lq_a.shape}.')
    return np.abs(lq_a - gt_a) > thr


def calculate_blind_mae_mask(img1, img2, lq, thr=30):
    """盲元区 MAE (灰度级). mask = |lq - gt| > thr; 无盲元返回 NaN."""
    mask = _mask_from_lq(img2, lq, thr)
    if not np.any(mask):
        return float('nan')
    pred = _to_gray_float64(img1)
    gt = _to_gray_float64(img2)
    return float(np.abs(pred[mask] - gt[mask]).mean())


def calculate_blind_psnr_mask(img1, img2, lq, thr=30, peak=255.0):
    """盲元区 PSNR. mask = |lq - gt| > thr; 无盲元返回 NaN."""
    mask = _mask_from_lq(img2, lq, thr)
    if not np.any(mask):
        return float('nan')
    pred = _to_gray_float64(img1)
    gt = _to_gray_float64(img2)
    err = pred[mask] - gt[mask]
    mse = float(np.mean(err ** 2))
    return float(10.0 * np.log10((peak * peak) / max(mse, 1e-12)))

