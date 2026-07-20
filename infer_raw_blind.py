"""
Standalone raw → Restormer inference (no NUC, no stripe suppression).

Only contrast enhancement + blind-pixel restoration via Restormer.

Usage:
    python infer_raw_blind.py \
        --raw_dir "背景数据_灯管" \
        --output_dir "背景数据_灯管/results" \
        --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth
"""

import argparse, os, sys, time
from glob import glob

import cv2, numpy as np, torch, torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer


def read_raw(path, rows=6000, cols=6000):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 2
    if n != rows * cols:
        side = int(np.sqrt(n)); rows = cols = side
    return np.frombuffer(data, dtype='<u2').reshape(cols, rows).T.astype(np.float64)


def contrast_only(frame):
    """Quantile stretch + gamma → uint8."""
    lo = np.quantile(frame, 0.001)
    hi = np.quantile(frame, 0.999)
    out = (frame - lo) / max(hi - lo, 1e-10)
    out = np.clip(out, 0, 1) ** 0.6
    return (out * 255).round().clip(0, 255).astype(np.uint8)


def tiled_forward(model, x, th, tw, overlap, device, batch_size=8):
    b, c, h, w = x.shape
    th, tw = min(th, h), min(tw, w)
    sh, sw = th - overlap, tw - overlap

    def starts(dim, t, s):
        r = list(range(0, dim - t + 1, s))
        if r and r[-1] + t < dim: r.append(dim - t)
        return r or [0]

    hs, ws = starts(h, th, sh), starts(w, tw, sw)
    total = len(hs) * len(ws)
    accum = torch.zeros(b, c, h, w, device=device)
    count = torch.zeros(b, c, h, w, device=device)

    # Collect all tile positions
    positions = [(y0, x0) for y0 in hs for x0 in ws]

    idx = 0; t0 = time.time()
    bs = batch_size
    peak_mem = 0

    for i in range(0, len(positions), bs):
        batch_pos = positions[i:i+bs]
        patches = []
        for y0, x0 in batch_pos:
            patches.append(x[..., y0:y0+th, x0:x0+tw])
        batch = torch.cat(patches, dim=0)  # [B, C, H, W]

        with torch.no_grad(), torch.amp.autocast('cuda'):
            outs = model(batch)
        outs = torch.clamp(outs, 0, 1)

        for j, (y0, x0) in enumerate(batch_pos):
            accum[..., y0:y0+th, x0:x0+tw] += outs[j:j+1]
            count[..., y0:y0+th, x0:x0+tw] += 1.0

        idx += len(batch_pos)
        mem = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_mem = max(peak_mem, mem)
        if idx % (bs * 4) <= bs or idx == total:
            e = time.time() - t0
            print(f'    tiles [{idx}/{total}] {idx/total*100:.0f}%  elapsed={e:.0f}s  eta={e/idx*(total-idx):.0f}s  GPU_mem={mem:.1f}G', flush=True)

    print(f'  Tiled forward done: {time.time()-t0:.0f}s  peak_GPU={peak_mem:.1f}G')
    return accum / count.clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--tile_h', type=int, default=640)
    parser.add_argument('--tile_w', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=128)
    parser.add_argument('--batch', type=int, default=8,
                        help='Tile batch size for parallel inference')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = Restormer(inp_channels=1, out_channels=1, dim=48,
        num_blocks=[4,6,6,8], num_refinement_blocks=4,
        heads=[1,2,4,8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['params'])
    model.eval()
    print(f'Loaded: {args.weights}')

    files = sorted(glob(os.path.join(args.raw_dir, '*.raw')))
    if not files: sys.exit(f'No .raw files in {args.raw_dir}')
    print(f'Found {len(files)} files')

    t_total = time.time()
    for i, p in enumerate(files):
        name = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.output_dir, f'{name}.png')
        print(f'\n[{i+1}/{len(files)}] {name}')
        t0 = time.time()

        raw = read_raw(p)
        pre = contrast_only(raw)
        t = torch.from_numpy(pre.astype(np.float32)/255.).unsqueeze(0).unsqueeze(0).to(device)
        h_in, w_in = t.shape[2], t.shape[3]
        H = ((h_in+8)//8)*8; W = ((w_in+8)//8)*8
        t = F.pad(t, (0, W-w_in, 0, H-h_in), 'reflect')
        restored = tiled_forward(model, t, args.tile_h, args.tile_w, args.tile_overlap, device, args.batch)
        restored = restored[:,:,:h_in,:w_in]
        out_img = (restored.squeeze().cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
        # Rotate each 90° CCW, then left-right
        pre_rot = np.rot90(pre, k=1)
        out_rot = np.rot90(out_img, k=1)
        compare = np.hstack([pre_rot, out_rot])
        cv2.imwrite(out, compare)
        dt = time.time()-t0
        print(f'  → {os.path.basename(out)}  frame_time={dt:.0f}s', flush=True)

    dt_total = time.time()-t_total
    print(f'\nDone. {len(files)} files → {args.output_dir}  total={dt_total:.0f}s  avg={dt_total/len(files):.0f}s/frame', flush=True)


if __name__ == '__main__':
    main()
