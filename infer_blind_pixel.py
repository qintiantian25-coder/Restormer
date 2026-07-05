"""
Inference script for large blind-pixel image restoration.

Usage:
    python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
        --weights experiments/.../best_model.pth
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer


def _tiled_forward(model, x, tile, overlap, device, use_fp16):
    """Tiled forward with uniform-weight blending (no feathering needed
    when overlap is large enough and model is consistent)."""
    b, c, h, w = x.shape
    tile = min(tile, h, w)
    stride = tile - overlap

    # Force consistent stride: all tiles share the same offset pattern
    h_starts = list(range(0, h - tile + 1, stride))
    w_starts = list(range(0, w - tile + 1, stride))
    # Cover trailing pixels
    if h_starts and h_starts[-1] + tile < h:
        h_starts.append(h - tile)
    if not h_starts:
        h_starts = [0]
    if w_starts and w_starts[-1] + tile < w:
        w_starts.append(w - tile)
    if not w_starts:
        w_starts = [0]

    total = len(h_starts) * len(w_starts)
    print(f'Tiles: {len(h_starts)} rows x {len(w_starts)} cols = {total}')
    print(f'Tile: {tile}  overlap: {overlap}  stride: {stride}')
    print(f'Padded image: {w}x{h}  fp16: {use_fp16}')

    accum = torch.zeros(b, c, h, w, device=device)
    count = torch.zeros(b, c, h, w, device=device)

    idx = 0
    t_start = time.time()

    for y0 in h_starts:
        for x0 in w_starts:
            patch = x[..., y0:y0 + tile, x0:x0 + tile]

            with torch.no_grad():
                if use_fp16:
                    with torch.amp.autocast('cuda'):
                        out = model(patch)
                else:
                    out = model(patch)

            out = torch.clamp(out, 0, 1)
            accum[..., y0:y0 + tile, x0:x0 + tile] += out
            count[..., y0:y0 + tile, x0:x0 + tile] += 1.0

            torch.cuda.empty_cache()

            idx += 1
            pct = idx / total * 100
            elapsed = time.time() - t_start
            eta = elapsed / idx * (total - idx) if idx > 0 else 0
            mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0
            print(f'  [{idx:3d}/{total}] ({pct:5.1f}%) '
                  f'pos=({y0:4d},{x0:4d})  '
                  f'elapsed={elapsed:.0f}s  eta={eta:.0f}s  '
                  f'peak_mem={mem:.1f}G', flush=True)

    restored = accum / count.clamp_min(1.0)
    print(f'Done. Total: {time.time() - t_start:.1f}s')
    return restored


def main():
    parser = argparse.ArgumentParser(description='Restormer blind-pixel inference')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--tile', type=int, default=2048)
    parser.add_argument('--tile_overlap', type=int, default=128)
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use fp16 autocast (default: on)')
    parser.add_argument('--fp32', action='store_false', dest='fp16',
                        help='Force fp32 inference')
    args = parser.parse_args()

    assert args.tile % 8 == 0, 'tile must be a multiple of 8'
    assert args.tile_overlap >= 32, 'overlap too small'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        print(f'GPU: {p.name} ({p.total_memory/1024**3:.0f} GB)')

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
    print(f'Input: {w_in}x{h_in}  dtype={img.dtype}  range=[{img.min()},{img.max()}]')

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.max() <= 1.0:
        img = img.astype(np.float32)
    else:
        img = img.astype(np.float32) / img.max()
    print(f'Normalized: range=[{img.min():.4f}, {img.max():.4f}]')

    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # --- pad to multiple of 8 ---
    H = ((h_in + 8) // 8) * 8
    W = ((w_in + 8) // 8) * 8
    pad_h, pad_w = H - h_in, W - w_in
    print(f'Pad: +{pad_h}h +{pad_w}w  →  {W}x{H}')
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), 'reflect')

    # --- tiled inference ---
    restored = _tiled_forward(model, img_t, args.tile, args.tile_overlap,
                              device, args.fp16)

    # --- unpad, save ---
    restored = restored[:, :, :h_in, :w_in]
    out = restored.squeeze().cpu().numpy()
    out = (out * 255.0).round().clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f'Saved: {args.output}  [{out.min()},{out.max()}]')


if __name__ == '__main__':
    main()
