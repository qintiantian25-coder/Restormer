"""
统一 raw → 修复后 PNG 处理管线.

两种模式:
  contrast (默认) — raw → 对比度增强 → Restormer → PNG
  nuc             — raw → NUC → 条纹抑制 → 对比度增强 → Restormer → PNG

用法:
  python process_raw.py \
      --raw_dir <输入文件夹> \
      --output_dir <输出文件夹> \
      --weights experiments/RealDenosing_BlindPixel_Merged/models/best_model.pth

  # NUC 模式
  python process_raw.py --mode nuc --calib <标定.mat> --raw_dir ... --output_dir ... --weights ...
"""

import argparse, os, sys, time
from glob import glob

import cv2, numpy as np, torch, torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer


# =====================================================================
# 预处理
# =====================================================================

def read_raw(path, rows=6000, cols=6000):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 2
    if n != rows * cols:
        side = int(np.sqrt(n)); rows = cols = side
    return np.frombuffer(data, dtype='<u2').reshape(cols, rows).T.astype(np.float64)


def preprocess_contrast(frame):
    lo, hi = np.quantile(frame, 0.001), np.quantile(frame, 0.999)
    out = (frame - lo) / max(hi - lo, 1e-10)
    out = np.clip(out, 0, 1) ** 0.6
    return (out * 255).round().clip(0, 255).astype(np.uint8)


def preprocess_nuc(frame, K, B, stripe_degree=3):
    data = (frame - B) / K

    col_means = data.mean(axis=0)
    x = np.arange(data.shape[1], dtype=np.float64)
    p = np.polyfit(x, col_means, stripe_degree)
    data -= (col_means - np.polyval(p, x))

    row_means = data.mean(axis=1)
    y = np.arange(data.shape[0], dtype=np.float64)
    p = np.polyfit(y, row_means, stripe_degree)
    data -= (row_means - np.polyval(p, y)).reshape(-1, 1)

    lo, hi = np.quantile(data, 0.001), np.quantile(data, 0.999)
    out = (data - lo) / max(hi - lo, 1e-10)
    out = np.clip(out, 0, 1) ** 0.6
    return (out * 255).round().clip(0, 255).astype(np.uint8)


def load_calib(mat_path):
    try:
        from scipy.io import loadmat
        calib = loadmat(mat_path)
        K = calib['kk'].astype(np.float64)
        B = calib['bb'].astype(np.float64)
    except NotImplementedError:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            K = np.array(f['kk'], dtype=np.float64).T
            B = np.array(f['bb'], dtype=np.float64).T
    if K.ndim == 3: K = K.squeeze()
    if B.ndim == 3: B = B.squeeze()
    K[K < 0.5] = 0.1
    return K, B


# =====================================================================
# Restormer 推理 (分块 batch)
# =====================================================================

def tiled_forward(model, x, th, tw, overlap, device, batch_size):
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
    positions = [(y, x) for y in hs for x in ws]

    idx = 0; t0 = time.time()
    for i in range(0, len(positions), batch_size):
        batch_pos = positions[i:i+batch_size]
        patches = torch.cat([x[..., y0:y0+th, x0:x0+tw] for y0, x0 in batch_pos], dim=0)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            outs = model(patches)
        outs = torch.clamp(outs, 0, 1)
        for j, (y0, x0) in enumerate(batch_pos):
            accum[..., y0:y0+th, x0:x0+tw] += outs[j:j+1]
            count[..., y0:y0+th, x0:x0+tw] += 1.0
        idx += len(batch_pos)
        if idx % (batch_size*4) <= batch_size or idx == total:
            e = time.time() - t0
            print(f'    tiles [{idx}/{total}] {idx/total*100:.0f}%  '
                  f'elapsed={e:.0f}s  eta={e/idx*(total-idx):.0f}s', flush=True)
    return accum / count.clamp_min(1.0)


# =====================================================================
# 主逻辑
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Raw → Restormer pipeline')
    parser.add_argument('--config', default='process_config.yml', help='YAML config file')
    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--mode', default=None, choices=['contrast', 'nuc'])
    parser.add_argument('--calib', default=None)
    parser.add_argument('--stripe_degree', type=int, default=None)
    parser.add_argument('--tile_w', type=int, default=None)
    parser.add_argument('--tile_h', type=int, default=None)
    parser.add_argument('--tile_overlap', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    cli = parser.parse_args()

    # Load config file for unspecified args
    import yaml
    with open(cli.config, 'r') as f:
        cfg = yaml.safe_load(f)

    class Args:
        pass
    args = Args()
    for key in ['raw_dir', 'output_dir', 'weights', 'mode', 'calib',
                'stripe_degree', 'tile_w', 'tile_h', 'tile_overlap', 'batch']:
        cli_val = getattr(cli, key, None)
        cfg_val = cfg.get(key)
        setattr(args, key, cli_val if cli_val is not None else cfg_val)

    if not args.raw_dir or not args.output_dir or not args.weights:
        sys.exit('Missing raw_dir/output_dir/weights (set in config or CLI)')

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        print(f'GPU: {p.name} ({p.total_memory/1024**3:.0f} GB)')
    print(f'Mode: {args.mode}  Tile: {args.tile_w}x{args.tile_h}  Batch: {args.batch}')

    # --- 模型 ---
    print(f'Loading model...')
    model = Restormer(inp_channels=1, out_channels=1, dim=48,
        num_blocks=[4,6,6,8], num_refinement_blocks=4,
        heads=[1,2,4,8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['params']); model.eval()

    # --- 标定 (NUC) ---
    K = B = None
    if args.mode == 'nuc':
        if not args.calib:
            sys.exit('--calib required for nuc mode')
        print(f'Loading calibration: {args.calib}')
        K, B = load_calib(args.calib)

    # --- 处理 ---
    files = sorted(glob(os.path.join(args.raw_dir, '*.raw')))
    if not files: sys.exit(f'No .raw files in {args.raw_dir}')
    print(f'\nFound {len(files)} raw files\n')

    t_total = time.time()
    for i, p in enumerate(files):
        name = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.output_dir, f'{name}.png')
        print(f'[{i+1}/{len(files)}] {name}')
        t0 = time.time()

        raw = read_raw(p)
        if args.mode == 'nuc':
            pre = preprocess_nuc(raw, K, B, args.stripe_degree)
        else:
            pre = preprocess_contrast(raw)
        t1 = time.time()
        print(f'  Preprocess: {t1-t0:.1f}s')

        t = torch.from_numpy(pre.astype(np.float32)/255.).unsqueeze(0).unsqueeze(0).to(device)
        hi, wi = t.shape[2], t.shape[3]
        H = ((hi+8)//8)*8; W = ((wi+8)//8)*8
        t = F.pad(t, (0, W-wi, 0, H-hi), 'reflect')

        restored = tiled_forward(model, t, args.tile_h, args.tile_w,
                                 args.tile_overlap, device, args.batch)
        restored = restored[:,:,:hi,:wi]
        out = (restored.squeeze().cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
        cv2.imwrite(out_path, out)
        t2 = time.time()
        print(f'  Inference: {t2-t1:.1f}s  Save: {time.time()-t2:.1f}s  Total: {t2-t0:.1f}s')
        print(f'  → {out_path}')

    elapsed = (time.time()-t_total)/60
    print(f'\nDone. {len(files)} files → {args.output_dir}  ({elapsed:.0f}min)')


if __name__ == '__main__':
    main()
