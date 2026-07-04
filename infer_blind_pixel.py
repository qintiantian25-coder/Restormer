"""
Inference script for 6000x6000 blind-pixel image restoration.

Usage:
    # After training, find the best model checkpoint:
    python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
        --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth

    # Or use a specific checkpoint:
    python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
        --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/net_g_100000.pth

    # Tiled inference (if full-image OOM):
    python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
        --weights path/to/model.pth --tile 2048 --tile_overlap 64
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to 6000x6000 input image')
    parser.add_argument('--output', required=True, help='Path for restored output')
    parser.add_argument('--weights', required=True, help='Path to trained model .pth')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size for tiled inference (e.g. 2048). None = full image')
    parser.add_argument('--tile_overlap', type=int, default=64,
                        help='Tile overlap in pixels')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- load model ---
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
    print(f'Input size: {w_in}x{h_in}')

    img_t = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

    # --- pad to multiple of 8 ---
    H = ((h_in + 8) // 8) * 8
    W = ((w_in + 8) // 8) * 8
    pad_h = H - h_in
    pad_w = W - w_in
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), 'reflect')

    # --- inference ---
    with torch.no_grad():
        if args.tile is None:
            restored = model(img_t)
        else:
            restored = _tiled_forward(model, img_t, args.tile, args.tile_overlap, device)

    # --- unpad, clamp, save ---
    restored = restored[:, :, :h_in, :w_in]
    restored = torch.clamp(restored, 0, 1)
    out = restored.squeeze().cpu().numpy()
    out = (out * 255.0).round().astype(np.uint8)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f'Saved: {args.output}')


def _tiled_forward(model, x, tile_size, overlap, device):
    """Tiled inference with overlap, blending at boundaries."""
    b, c, h, w = x.shape
    tile = min(tile_size, h, w)
    stride = tile - overlap

    h_starts = list(range(0, h - tile, stride)) + [h - tile]
    w_starts = list(range(0, w - tile, stride)) + [w - tile]

    accum = torch.zeros(b, c, h, w, device=device)
    count = torch.zeros(b, c, h, w, device=device)

    for y0 in h_starts:
        for x0 in w_starts:
            patch = x[..., y0:y0 + tile, x0:x0 + tile]
            out_patch = model(patch)
            accum[..., y0:y0 + tile, x0:x0 + tile] += out_patch
            count[..., y0:y0 + tile, x0:x0 + tile] += 1.0

    return accum / count


if __name__ == '__main__':
    main()
