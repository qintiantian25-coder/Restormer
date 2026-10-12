"""
单张 PNG 的修复推理 (测试/验收用).

与 process_raw.py 的区别: 直接读 PNG (不做 raw/NUC/条纹/对比度预处理),
适合测试已经预处理好的模拟图 (如 data_merged/*_sim6000_blind.png)。

用法 (在项目根目录运行):
  python infer_png.py --input data_merged/658_4_sim6000_blind.png \
      --output results/658_4_repaired.png \
      --weights experiments/RealDenosing_BlindPixel_Merged/models/best_model.pth \
      --batch 8

  # 可选: 若有干净参考图, 顺手加 --gt 会打印 输入/修复后 的 PSNR 对比
  python infer_png.py --input ... --output ... --gt data_merged/658_4_sim6000.png ...
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer
from process_raw import tiled_forward


def main():
    parser = argparse.ArgumentParser(description='PNG -> Restormer -> PNG')
    parser.add_argument('--input', required=True, help='输入 PNG 路径')
    parser.add_argument('--output', required=True, help='输出 PNG 路径')
    parser.add_argument('--weights', default=None,
                        help='best_model.pth; 省略则随机初始化(仅用于管道自检)')
    parser.add_argument('--gt', default=None,
                        help='可选: 干净参考图, 提供后会计算 输入/修复后 的 PSNR/SSIM')
    parser.add_argument('--tile_w', type=int, default=640)
    parser.add_argument('--tile_h', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        print(f'GPU: {p.name} ({p.total_memory/1024**3:.0f} GB)')

    # --- 读图 ---
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f'读不到图片: {args.input}')
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535. * 255).round().astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hi, wi = img.shape
    print(f'Input: {args.input}  尺寸 {wi}x{hi} (宽x高)  dtype={img.dtype}')

    # --- 模型 ---
    model = Restormer(inp_channels=1, out_channels=1, dim=48,
                      num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                      heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type='BiasFree',
                      dual_pixel_task=False).to(device)
    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(ckpt['params'])
        print(f'Weights: {args.weights}')
    else:
        print('WARNING: 未提供 --weights, 使用随机初始化 (仅用于管道自检)')
    model.eval()

    # --- 分块推理 (与 process_raw 同一套: reflect pad 到 16 倍数 -> 640x512 滑窗 -> 平均融合) ---
    t = torch.from_numpy(img.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0).to(device)
    H = ((hi + 16) // 16) * 16
    W = ((wi + 16) // 16) * 16
    t = F.pad(t, (0, W - wi, 0, H - hi), 'reflect')

    t0 = time.time()
    out = tiled_forward(model, t, args.tile_h, args.tile_w, args.overlap,
                        device, args.batch)
    out = out[:, :, :hi, :wi]
    dt = time.time() - t0

    out_u8 = (out.squeeze().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.output, out_u8)
    mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0
    print(f'Output: {args.output}  用时 {dt:.1f}s  峰值显存 {mem:.1f}G')

    # --- 可选: 有 GT 就算指标 ---
    if args.gt:
        gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise SystemExit(f'GT 读不到: {args.gt}')
        if gt.shape != out_u8.shape:
            raise SystemExit(f'GT 尺寸 {gt.shape} 与输出 {out_u8.shape} 不一致')
        import basicsr.metrics as metrics
        psnr_in = metrics.calculate_psnr(img, gt, crop_border=0, test_y_channel=False)
        psnr_out = metrics.calculate_psnr(out_u8, gt, crop_border=0, test_y_channel=False)
        ssim_out = metrics.calculate_ssim(out_u8, gt, crop_border=0, test_y_channel=False)
        print(f'PSNR: 输入 {psnr_in:.4f} dB -> 修复后 {psnr_out:.4f} dB '
              f'(提升 {psnr_out - psnr_in:+.4f} dB)')
        print(f'SSIM: 修复后 {ssim_out:.4f}')


if __name__ == '__main__':
    main()
