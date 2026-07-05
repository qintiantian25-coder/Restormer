"""
Inference script for large blind-pixel image restoration with tiled blending.

Usage:
    python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
        --weights experiments/.../best_model.pth --tile 3072 --tile_overlap 128
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer


def _per_tile_weight(tile_h, tile_w, overlap, y0, x0, img_h, img_w):
    """Compute a 2D weight map for a tile at position (y0, x0).

    Weight ramps from 0→1 over *overlap* pixels at interior edges (where
    two tiles meet) and stays at 1 near the image border (no ramp needed).
    """
    wy = np.ones(tile_h, dtype=np.float32)
    wx = np.ones(tile_w, dtype=np.float32)

    # top edge: ramp only if not at image top
    if y0 > 0:
        ramp = np.linspace(0, 1, min(overlap, tile_h), dtype=np.float32)
        wy[:len(ramp)] = np.minimum(wy[:len(ramp)], ramp)
    # bottom edge: ramp only if not at image bottom
    if y0 + tile_h < img_h:
        ramp = np.linspace(1, 0, min(overlap, tile_h), dtype=np.float32)
        wy[-len(ramp):] = np.minimum(wy[-len(ramp):], ramp)

    # left edge
    if x0 > 0:
        ramp = np.linspace(0, 1, min(overlap, tile_w), dtype=np.float32)
        wx[:len(ramp)] = np.minimum(wx[:len(ramp)], ramp)
    # right edge
    if x0 + tile_w < img_w:
        ramp = np.linspace(1, 0, min(overlap, tile_w), dtype=np.float32)
        wx[-len(ramp):] = np.minimum(wx[-len(ramp):], ramp)

    return wy[:, None] * wx[None, :]  # [tile_h, tile_w]


def _tiled_forward(model, x, tile_size, overlap, device):
    """Tiled forward pass with correct per-tile blending weights."""
    b, c, h, w = x.shape
    tile = min(tile_size, h, w)
    stride = tile - overlap

    # --- compute tile grid ---
    def _starts(dim):
        s = list(range(0, dim - tile, stride))
        if not s or s[-1] < dim - tile:
            s.append(dim - tile)
        if not s:
            s = [0]
        return s

    h_starts = _starts(h)
    w_starts = _starts(w)
    total = len(h_starts) * len(w_starts)
    print(f'Tiles: {len(h_starts)} rows x {len(w_starts)} cols = {total} tiles')
    print(f'Tile size: {tile}, overlap: {overlap}, stride: {stride}')
    print(f'Image: {w}x{h}')

    accum = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    idx = 0
    t_start = time.time()

    for y0 in h_starts:
        for x0 in w_starts:
            # --- extract patch ---
            patch = x[..., y0:y0 + tile, x0:x0 + tile]

            with torch.no_grad():
                out = model(patch)
            out = torch.clamp(out, 0, 1)
            out_np = out.squeeze().cpu().numpy().astype(np.float64)

            # --- compute per-tile weight ---
            weight = _per_tile_weight(tile, tile, overlap, y0, x0, h, w)

            accum[y0:y0 + tile, x0:x0 + tile] += out_np * weight
            weight_sum[y0:y0 + tile, x0:x0 + tile] += weight

            idx += 1
            pct = idx / total * 100
            elapsed = time.time() - t_start
            eta = elapsed / idx * (total - idx) if idx > 0 else 0
            if device.type == 'cuda':
                mem = torch.cuda.max_memory_allocated(device) / 1024**3
            else:
                mem = 0
            print(f'  [{idx:3d}/{total}] ({pct:5.1f}%) '
                  f'pos=({y0:4d},{x0:4d})  '
                  f'elapsed={elapsed:.0f}s  eta={eta:.0f}s  '
                  f'GPU_peak={mem:.1f}G', flush=True)

    restored = accum / np.maximum(weight_sum, 1e-8)
    restored = torch.from_numpy(restored).unsqueeze(0).unsqueeze(0).float().to(device)
    print(f'Done. Total: {time.time() - t_start:.1f}s')
    return restored


def _check_image_range(img, path):
    """Warn if image does not look like 8-bit."""
    vmin, vmax = img.min(), img.max()
    print(f'Image range: [{vmin}, {vmax}], dtype={img.dtype}')
    if vmax <= 1.0 and img.dtype in (np.float32, np.float64):
        print('  → Image appears to be [0,1] float, using as-is')
        return img.astype(np.float32), True
    if vmax <= 255:
        print(f'  → 8-bit, normalising by 1/255')
        return img.astype(np.float32) / 255.0, True
    if vmax <= 65535:
        print(f'  → 16-bit, normalising by 1/65535')
        return img.astype(np.float32) / 65535.0, True
    return img.astype(np.float32) / vmax, False


def main():
    parser = argparse.ArgumentParser(description='Restormer blind-pixel inference')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Path for restored output')
    parser.add_argument('--weights', required=True, help='Path to trained model .pth')
    parser.add_argument('--tile', type=int, default=1024,
                        help='Tile size (multiple of 8). Default 1024.')
    parser.add_argument('--tile_overlap', type=int, default=128,
                        help='Tile overlap in pixels. Default 128.')
    args = parser.parse_args()

    assert args.tile % 8 == 0, 'tile must be a multiple of 8'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f'GPU: {props.name} ({props.total_memory / 1024**3:.0f} GB)')

    # --- load model ---
    print('Loading model...')
    model = Restormer(
        inp_channels=1, out_channels=1, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
    ).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['params'])
    model.eval()
    print(f'Loaded: {args.weights}')

    # --- load image ---
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f'Cannot read: {args.input}')
    h_in, w_in = img.shape
    print(f'Input: {w_in}x{h_in} ({w_in*h_in/1e6:.1f} MP), dtype={img.dtype}')

    # auto-detect bit depth & normalise
    img, ok = _check_image_range(img, args.input)
    if not ok:
        print('WARNING: unexpected value range, result may be wrong')

    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # --- pad to multiple of 8 ---
    H = ((h_in + 8) // 8) * 8
    W = ((w_in + 8) // 8) * 8
    pad_h, pad_w = H - h_in, W - w_in
    print(f'Pad: +{pad_h} rows, +{pad_w} cols  →  {W}x{H}')
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), 'reflect')

    # --- tiled inference ---
    restored = _tiled_forward(model, img_t, args.tile, args.tile_overlap, device)

    # --- unpad, save ---
    restored = restored[:, :, :h_in, :w_in]
    out = restored.squeeze().cpu().numpy()
    out = (out * 255.0).round().clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f'Saved: {args.output}')
    print(f'Output range: [{out.min()}, {out.max()}]')


if __name__ == '__main__':
    main()
