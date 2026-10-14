"""
推理时 mask 合成: 只修 mask 区, mask 外保持原图.

模型本身没有 mask 输入通道 (1 通道输入/输出); 这里在输出端做合成:
    out = 模型输出 x alpha + 原图 x (1 - alpha)
alpha 由 mask 生成, 可膨胀 (覆盖盲元边缘过渡带) 和羽化 (软化边界).
羽化 = 0 时 mask (膨胀后) 外的像素与输入逐位相同.

mask 约定: 与输入图同尺寸.
  - 支持任意图片格式/位深 (png/tif/...): >127 视为修复区;
  - 或 .mat (MATLAB): 取变量 loc (0/1 判定图, 如 YD_SW_A_D_12000_4hz_xs.mat),
    0/1 mask 自动按 >0.5 判定; v7 用 scipy 读, v7.3(HDF5) 用 h5py 读并转置
    (与 process_raw.load_calib 同口径).
"""
import os

import cv2
import numpy as np

MASK_EXTS = ('.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg', '.mat')


def imread_u(path):
    """Unicode 路径安全读图 (Windows 中文路径可用)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def find_mask(mask_dir, rel_stem):
    """在 mask_dir 下按相对路径主干名 (可含子目录) 找任意扩展名的 mask; 找不到返回 None."""
    for ext in MASK_EXTS:
        p = os.path.join(mask_dir, rel_stem + ext)
        if os.path.isfile(p):
            return p
    return None


def load_mask_array(path):
    """读 mask 文件 -> 2D 数组 (任意位深, 或 .mat 里的 0/1 loc 图)."""
    if path.lower().endswith('.mat'):
        with open(path, 'rb') as fh:
            head = fh.read(128)
        if head.startswith(b'MATLAB 5.0 MAT-file'):     # v5/v7: scipy
            from scipy.io import loadmat
            d = loadmat(path)
            if 'loc' not in d:
                raise ValueError(f'.mat 里没有 loc 变量: {path}')
            return np.asarray(d['loc'])
        # v7.3 (HDF5, 含 MATLAB 标准头 / h5py 生成): h5py 读并转置
        # (与 process_raw.load_calib 同口径: h5py 看到的是转置, 需 .T 还原)
        import h5py
        with h5py.File(path, 'r') as f:
            if 'loc' not in f:
                raise ValueError(f'.mat 里没有 loc 变量: {path}')
            arr = np.array(f['loc'], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f'.mat 的 loc 不是二维 ({arr.shape}): {path}')
        return arr.T
    m = imread_u(path)
    if m is None:
        raise IOError(f'读 mask 失败: {path}')
    return m


def load_mask(path, shape, dilate=0, feather=0):
    """读 mask -> (alpha, binary).

    alpha: 0~1 float32 合成权重 (膨胀 + 羽化后);
    binary: 羽化/膨胀前的 bool 判定区 (用于算 mask 区指标, 与训练 blind_psnr 口径一致).
    """
    m = load_mask_array(path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape != tuple(shape):
        raise ValueError(f'mask 尺寸 {m.shape} 与图像 {tuple(shape)} 不一致: {path}')
    if m.size == 0 or float(np.max(m)) <= 0:
        raise ValueError(f'mask 全黑 (没有修复区): {path}')
    thr = 0.5 if float(np.max(m)) <= 1.0 else 127.0   # 0/1 判定图(如 loc) 或 0~255/0~65535
    binary = m > thr
    md = binary.astype(np.uint8) * 255
    if dilate > 0:
        k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        md = cv2.dilate(md, k)
    alpha = md.astype(np.float32) / 255.
    if feather > 0:
        alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), feather), 0, 1)
    return alpha, binary


def apply_mask(orig, pred, alpha):
    """原图与修复图按 alpha 合成 (均 uint8 同尺寸, 2D 灰度; 返回 uint8)."""
    a = alpha if orig.ndim == 2 else alpha[..., None]
    out = orig.astype(np.float32) * (1 - a) + pred.astype(np.float32) * a
    return np.round(out).clip(0, 255).astype(np.uint8)


def region_psnr_mae(a, b, region):
    """region 内 PSNR / MAE (灰度级; 与训练 blind_psnr/blind_mae 同口径)."""
    d = a.astype(np.float32) - b.astype(np.float32)
    n = int(region.sum())
    if n == 0:
        return float('nan'), float('nan')
    mse = float(np.mean(d[region] ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-10)), float(np.mean(np.abs(d[region])))
