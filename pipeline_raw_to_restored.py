"""
End-to-end pipeline: raw files → NUC → stripe suppression → contrast → Restormer.

Replicates ceshi.m preprocessing in Python, then runs tiled Restormer inference.

Usage:
    python pipeline_raw_to_restored.py \
        --raw_dir "E:\\DD数据\\YD背景100帧" \
        --calib "E:\\DD数据\\非均匀校正系数\\YD_SW_A_D_72000_4hz_xs.mat" \
        --output_dir "E:\\DD数据\\results" \
        --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth
"""

import argparse
import os
import sys
import time
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from basicsr.models.archs.restormer_arch import Restormer


# ---------------------------------------------------------------------------
#  Preprocessing  (ceshi.m → Python)
# ---------------------------------------------------------------------------

def load_calib(mat_path):
    """Load NUC coefficients from .mat file.  Supports v5 and v7.3."""
    try:
        calib = loadmat(mat_path)
        K = calib['kk'].astype(np.float64)
        B = calib['bb'].astype(np.float64)
    except NotImplementedError:
        # MATLAB v7.3 stores transposed in HDF5 → must transpose back
        import h5py
        with h5py.File(mat_path, 'r') as f:
            K = np.array(f['kk'], dtype=np.float64).T
            B = np.array(f['bb'], dtype=np.float64).T
    if K.ndim == 3:
        K = K.squeeze()
    if B.ndim == 3:
        B = B.squeeze()
    K[K < 0.5] = 0.1
    return K, B


def read_raw(path, rows=6000, cols=6000):
    """Read a raw uint16 LE frame with MATLAB-compatible column-major order."""
    with open(path, 'rb') as f:
        data = f.read()
    n_pixels = len(data) // 2
    if n_pixels != rows * cols:
        side = int(np.sqrt(n_pixels))
        rows = cols = side
    # MATLAB fread([rows,cols],'uint16') → column-major (Fortran order)
    arr = np.frombuffer(data, dtype='<u2').reshape(cols, rows).T
    return arr.astype(np.float64)


def preprocess(frame, K, B, stripe_degree=5, stripe_mode='poly'):
    """Apply NUC → stripe suppression → contrast enhancement → uint8.

    stripe_mode:
        'poly'  - polynomial fit (original ceshi.m)
        'median' - median-based column/row equalisation (stronger)
        'both'   - polynomial first, then median residual
    """
    # [1] NUC
    data = (frame - B) / K

    rows, cols = data.shape

    # [2] Stripe suppression
    if stripe_mode in ('poly', 'both'):
        col_means = data.mean(axis=0)
        x = np.arange(cols, dtype=np.float64)
        p = np.polyfit(x, col_means, stripe_degree)
        trend_col = np.polyval(p, x)
        data -= (col_means - trend_col)

        row_means = data.mean(axis=1)
        y = np.arange(rows, dtype=np.float64)
        p_row = np.polyfit(y, row_means, stripe_degree)
        trend_row = np.polyval(p_row, y)
        data -= (row_means - trend_row).reshape(-1, 1)

    if stripe_mode in ('median', 'both'):
        # Median-based: subtracts median of each col/row, then smooths
        col_med = np.median(data, axis=0)
        col_med_smooth = np.polyval(np.polyfit(np.arange(cols), col_med, 3), np.arange(cols))
        data -= (col_med - col_med_smooth)

        row_med = np.median(data, axis=1)
        row_med_smooth = np.polyval(np.polyfit(np.arange(rows), row_med, 3), np.arange(rows))
        data -= (row_med - row_med_smooth).reshape(-1, 1)

    # [3] Contrast enhancement (np.quantile = linear interp, matches MATLAB quantile)
    lo = np.quantile(data, 0.001)
    hi = np.quantile(data, 0.999)
    if hi - lo > 1e-10:
        data = (data - lo) / (hi - lo)
    data = np.clip(data, 0, 1)
    data = data ** 0.6  # gamma

    return (data * 255.0).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Restormer inference  (tiled)
# ---------------------------------------------------------------------------

def _tiled_forward(model, x, tile_h, tile_w, overlap, device, use_fp16):
    b, c, h, w = x.shape
    th = min(tile_h, h)
    tw = min(tile_w, w)
    stride_h = th - overlap
    stride_w = tw - overlap

    def _starts(dim, t, stride):
        s = list(range(0, dim - t + 1, stride))
        if s and s[-1] + t < dim:
            s.append(dim - t)
        if not s:
            s = [0]
        return s

    h_starts = _starts(h, th, stride_h)
    w_starts = _starts(w, tw, stride_w)
    total = len(h_starts) * len(w_starts)

    accum = torch.zeros(b, c, h, w, device=device)
    count = torch.zeros(b, c, h, w, device=device)

    idx = 0
    t_start = time.time()

    for y0 in h_starts:
        for x0 in w_starts:
            patch = x[..., y0:y0 + th, x0:x0 + tw]
            with torch.no_grad():
                if use_fp16:
                    with torch.amp.autocast('cuda'):
                        out = model(patch)
                else:
                    out = model(patch)
            out = torch.clamp(out, 0, 1)
            accum[..., y0:y0 + th, x0:x0 + tw] += out
            count[..., y0:y0 + th, x0:x0 + tw] += 1.0
            torch.cuda.empty_cache()

            idx += 1
            if idx % 4 == 0 or idx == total:
                pct = idx / total * 100
                elapsed = time.time() - t_start
                eta = elapsed / idx * (total - idx) if idx > 0 else 0
                print(f'    tiles [{idx}/{total}] {pct:.0f}%  '
                      f'elapsed={elapsed:.0f}s  eta={eta:.0f}s', flush=True)

    return accum / count.clamp_min(1.0)


def run_restormer(model, img, device, tile_h=640, tile_w=512, overlap=128):
    """Tiled Restormer inference on a single 8-bit grayscale image."""
    h_in, w_in = img.shape
    t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    H = ((h_in + 8) // 8) * 8
    W = ((w_in + 8) // 8) * 8
    t = F.pad(t, (0, W - w_in, 0, H - h_in), 'reflect')

    restored = _tiled_forward(model, t, tile_h, tile_w, overlap, device, use_fp16=True)
    restored = restored[:, :, :h_in, :w_in]
    out = (restored.squeeze().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Raw → Restormer pipeline')
    parser.add_argument('--raw_dir', required=True, help='Directory of .raw files')
    parser.add_argument('--calib', required=True, help='NUC calibration .mat file')
    parser.add_argument('--output_dir', required=True, help='Directory for restored PNGs')
    parser.add_argument('--weights', required=True, help='Trained model .pth')
    parser.add_argument('--rows', type=int, default=6000)
    parser.add_argument('--cols', type=int, default=6000)
    parser.add_argument('--stripe_degree', type=int, default=3,
                        help='Polynomial degree for stripe suppression')
    parser.add_argument('--stripe_mode', default='poly',
                        choices=['poly', 'median', 'both'],
                        help='Stripe suppression method.')
    parser.add_argument('--tile_h', type=int, default=640)
    parser.add_argument('--tile_w', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=128)
    parser.add_argument('--save_pre', action='store_true',
                        help='Also save preprocessed (pre-model) image')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Process at most N frames (0 = all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        print(f'GPU: {p.name} ({p.total_memory / 1024**3:.0f} GB)')

    # --- load calibration ---
    print(f'Loading calibration: {args.calib}')
    K, B = load_calib(args.calib)
    print(f'  K shape={K.shape}  range=[{K.min():.3f}, {K.max():.3f}]')
    print(f'  B shape={B.shape}  range=[{B.min():.1f}, {B.max():.1f}]')

    # --- load model ---
    print(f'Loading model: {args.weights}')
    model = Restormer(
        inp_channels=1, out_channels=1, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
    ).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['params'])
    model.eval()

    # --- find raw files ---
    raw_files = sorted(glob(os.path.join(args.raw_dir, '*.raw')))
    if not raw_files:
        sys.exit(f'No .raw files found in {args.raw_dir}')
    if args.max_frames > 0:
        raw_files = raw_files[:args.max_frames]
    print(f'\nFound {len(raw_files)} raw files (max_frames={args.max_frames})')

    # --- process ---
    n_total = len(raw_files)
    t_total = time.time()
    for i, raw_path in enumerate(raw_files):
        fname = os.path.splitext(os.path.basename(raw_path))[0]
        out_path = os.path.join(args.output_dir, f'{fname}.png')

        elapsed_total = time.time() - t_total
        eta_total = elapsed_total / (i + 1) * (n_total - i - 1) if i > 0 else 0
        print(f'\n{"="*60}')
        print(f'[{i+1}/{n_total}] ({100*(i+1)/n_total:.1f}%)  '
              f'elapsed={elapsed_total/60:.1f}min  eta={eta_total/60:.1f}min')
        print(f'  {fname}')
        t_frame = time.time()

        # 1) read raw
        print(f'  [1/4] Reading raw...', end=' ', flush=True)
        raw = read_raw(raw_path, args.rows, args.cols)
        print(f'{raw.shape[1]}x{raw.shape[0]}  range=[{raw.min():.0f},{raw.max():.0f}]')
        # 2) preprocess
        print(f'  [2/4] Preprocessing...', end=' ', flush=True)
        pre = preprocess(raw, K, B, args.stripe_degree, args.stripe_mode)
        print(f'done  range=[{pre.min():.0f},{pre.max():.0f}]')
        if args.save_pre:
            pre_path = out_path.replace('.png', '_pre.png')
            cv2.imwrite(pre_path, pre)
            print(f'  [pre] Saved: {pre_path}')
        # 3) Restormer
        print(f'  [3/4] Restormer inference:')
        restored = run_restormer(model, pre, device,
                                 args.tile_h, args.tile_w, args.tile_overlap)
        # 4) save
        print(f'  [4/4] Saving → {out_path}', end=' ', flush=True)
        cv2.imwrite(out_path, restored)

        dt = time.time() - t_frame
        print(f'({dt:.0f}s)')

    elapsed = (time.time() - t_total) / 60
    print(f'\n{"="*60}')
    print(f'Done. {n_total} files → {args.output_dir}  ({elapsed:.0f}min)')


if __name__ == '__main__':
    main()
