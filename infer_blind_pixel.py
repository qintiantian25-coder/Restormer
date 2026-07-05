"""
Inference script for large blind-pixel image restoration with tiled blending.

Usage:
    python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
        --weights experiments/.../best_model.pth

    python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
        --weights experiments/.../best_model.pth --tile 1024 --tile_overlap 128
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


def _make_blend_window(tile, overlap):
    """1D raised-cosine window: ramp 0->1 over *overlap* px at each end."""
    if overlap <= 0:
        return torch.ones(tile)
    ramp = 0.5 * (1.0 - torch.cos(math.pi * torch.arange(overlap) / overlap))
    window = torch.ones(tile)
    window[:overlap] = ramp
    window[-overlap:] = ramp.flip(0)
    return window


def _tiled_forward(model, x, tile, overlap, device):
    """Tiled forward pass with feathered blending.

    Each output tile is multiplied by a 2D weight window that ramps to zero
    at the edges inside the overlap region.  The final image is the sum of
    weighted tiles divided by the sum of weights — producing seamless blending.
    """
    b, c, h, w = x.shape
    tile = min(tile, h, w)
    stride = tile - overlap

    h_starts = list(range(0, h - tile, stride))
    if h_starts and h_starts[-1] < h - tile:
        h_starts.append(h - tile)
    if not h_starts:
        h_starts = [0]

    w_starts = list(range(0, w - tile, stride))
    if w_starts and w_starts[-1] < w - tile:
        w_starts.append(w - tile)
    if not w_starts:
        w_starts = [0]

    total_tiles = len(h_starts) * len(w_starts)
    print(f'Tiles: {len(h_starts)} rows x {len(w_starts)} cols = {total_tiles} tiles')
    print(f'Tile size: {tile}, overlap: {overlap}, stride: {stride}')
    print(f'Image: {w}x{h} (padded to {x.shape[3]}x{x.shape[2]})')

    # 2D blend weights
    w1d = _make_blend_window(tile, overlap).to(device)
    w2d = w1d.unsqueeze(0) * w1d.unsqueeze(1)  # [tile, tile]
    w2d = w2d.unsqueeze(0).unsqueeze(0)          # [1, 1, tile, tile]

    accum = torch.zeros(b, c, h, w, device=device)
    weight_sum = torch.zeros(b, c, h, w, device=device)

    tile_idx = 0
    t_start = time.time()

    for y0 in h_starts:
        for x0 in w_starts:
            patch = x[..., y0:y0 + tile, x0:x0 + tile]

            with torch.no_grad():
                out_patch = model(patch)

            out_patch = torch.clamp(out_patch, 0, 1)
            weighted = out_patch * w2d

            accum[..., y0:y0 + tile, x0:x0 + tile] += weighted
            weight_sum[..., y0:y0 + tile, x0:x0 + tile] += w2d

            tile_idx += 1
            pct = tile_idx / total_tiles * 100
            elapsed = time.time() - t_start
            eta = elapsed / tile_idx * (total_tiles - tile_idx) if tile_idx > 0 else 0
            mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0
            print(f'  [{tile_idx:3d}/{total_tiles}] ({pct:5.1f}%) '
                  f'pos=({y0:4d},{x0:4d})  '
                  f'elapsed={elapsed:.0f}s  eta={eta:.0f}s  '
                  f'GPU_peak={mem:.1f}G')

    restored = accum / weight_sum.clamp_min(1e-8)
    print(f'Done. Total time: {time.time() - t_start:.1f}s')
    return restored


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
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f'Cannot read: {args.input}')
    h_in, w_in = img.shape
    print(f'Input size: {w_in}x{h_in} ({w_in*h_in/1e6:.1f} MP)')

    img_t = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

    # --- pad to multiple of 8 ---
    H = ((h_in + 8) // 8) * 8
    W = ((w_in + 8) // 8) * 8
    pad_h = H - h_in
    pad_w = W - w_in
    print(f'Pad: +{pad_h} rows, +{pad_w} cols -> {W}x{H}')

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


if __name__ == '__main__':
    main()
