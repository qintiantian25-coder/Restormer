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


def _tiled_forward(model, x, tile_h, tile_w, overlap, device, use_fp16):
    """Tiled forward with uniform-weight blending."""
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
    print(f'Tiles: {len(h_starts)} rows x {len(w_starts)} cols = {total}')
    print(f'Tile: {th}x{tw}  overlap: {overlap}  stride: {stride_h}x{stride_w}')
    print(f'Padded image: {w}x{h}  fp16: {use_fp16}')

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
    parser.add_argument('--tile', type=int, default=0,
                        help='Square tile size (e.g. 2048).  Overrides --tile_w/--tile_h')
    parser.add_argument('--tile_w', type=int, default=640,
                        help='Tile width (default 640, multiple of 8)')
    parser.add_argument('--tile_h', type=int, default=512,
                        help='Tile height (default 512, multiple of 8)')
    parser.add_argument('--tile_overlap', type=int, default=128)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--fp32', action='store_false', dest='fp16')
    args = parser.parse_args()

    if args.tile > 0:
        args.tile_w = args.tile_h = args.tile
    assert args.tile_w % 8 == 0 and args.tile_h % 8 == 0, 'tile must be multiple of 8'
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
    restored = _tiled_forward(model, img_t, args.tile_h, args.tile_w,
                              args.tile_overlap, device, args.fp16)

    # --- unpad, save ---
    restored = restored[:, :, :h_in, :w_in]
    out = restored.squeeze().cpu().numpy()
    out = (out * 255.0).round().clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f'Saved: {args.output}  [{out.min()},{out.max()}]')


if __name__ == '__main__':
    main()
