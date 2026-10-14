"""
PNG 修复推理 (测试/验收用): 单张 或 整个目录.

与 process_raw.py 的区别: 直接读 PNG (不做 raw/NUC/条纹/对比度预处理),
适合测试已经预处理好的模拟图 / 用配对数据做离线评估。

单张模式 (在项目根目录运行):
  python infer_png.py --input data_merged/658_4_sim6000_blind.png \
      --output results/658_4_repaired.png \
      --weights experiments/RealDenosing_BlindPixel_Merged/models/best_model.pth \
      --batch 8

目录模式 (递归处理所有 *.png, 保持子目录结构输出; --gt 传配对目录则算平均指标):
  python infer_png.py --input data_merged/test_blur --output results/test_blur_eval \
      --gt data_merged/test_sharp \
      --weights experiments/RealDenosing_BlindPixel_Merged/models/best_model.pth \
      --batch 8

  # 只处理前 20 张快速预览:
  ... --limit 20

Mask 模式 (只修 mask 区, mask 外保持原图; 单张用 --mask, 目录/批量用 --mask_dir 或 --mask_mat):
  python infer_png.py --input xxx_blind.png --output results/xxx_maskfix.png \
      --mask xxx_mask.png --gt xxx_gt.png --mask_dilate 4 --mask_feather 3

  # 目录 + GT 评估: 一张 .mat mask 对所有图生效, GT 按"输入名去掉后缀"查找
  python infer_png.py --input test_in --output results/_eval --gt D:\\...\\GT \
      --gt_suffix _blind --mask_mat D:\\...\\xxx.mat --batch 8
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer
from mask_blend import apply_mask, find_mask, load_mask, region_psnr_mae
from process_raw import tiled_forward


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f'读不到图片: {path}')
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535. * 255).round().astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def restore(model, img, device, tile_h, tile_w, overlap, batch_size):
    hi, wi = img.shape
    t = torch.from_numpy(img.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0).to(device)
    # pad 到 16 的倍数 (已经是倍数则不 pad)
    H = hi if hi % 16 == 0 else ((hi + 15) // 16) * 16
    W = wi if wi % 16 == 0 else ((wi + 15) // 16) * 16
    t = F.pad(t, (0, W - wi, 0, H - hi), 'reflect')
    out = tiled_forward(model, t, tile_h, tile_w, overlap, device, batch_size)
    out = out[:, :, :hi, :wi]
    return (out.squeeze().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='PNG -> Restormer -> PNG (单张或目录)')
    parser.add_argument('--input', required=True, help='输入 PNG 或 目录')
    parser.add_argument('--output', required=True, help='输出 PNG (单张) 或 目录')
    parser.add_argument('--weights', default=None,
                        help='best_model.pth; 省略则随机初始化(仅用于管道自检)')
    parser.add_argument('--gt', default=None, help='可选: 干净参考(文件或目录), 提供后计算 PSNR/SSIM')
    parser.add_argument('--tile_w', type=int, default=640)
    parser.add_argument('--tile_h', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='目录模式: 只处理前 N 张 (0=全部)')
    parser.add_argument('--mask', default=None,
                        help='单张模式: mask 文件 (图片 或 .mat 里的 loc), 只修 mask 区 (mask 外保持原图)')
    parser.add_argument('--mask_dir', default=None, help='目录模式: mask 目录 (按相对路径同名查找)')
    parser.add_argument('--mask_mat', default=None, help='所有图都用这一张 mask (图片 或 .mat; 单张/目录模式都可用)')
    parser.add_argument('--mask_dilate', type=int, default=4, help='mask 膨胀像素 (4=与训练 mask 口径一致)')
    parser.add_argument('--mask_feather', type=int, default=3, help='边界羽化像素 (0=mask(膨胀后)外逐位不动)')
    parser.add_argument('--gt_suffix', default=None,
                        help='目录模式: 找 GT 时先去掉输入文件名的该后缀再查 (如 --gt_suffix _blind)')
    args = parser.parse_args()
    if args.mask_mat and not os.path.isfile(args.mask_mat):
        raise SystemExit(f'--mask_mat 不存在: {args.mask_mat}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        print(f'GPU: {p.name} ({p.total_memory/1024**3:.0f} GB)')

    # --- 模型 (先建, 两种模式共用) ---
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

    is_dir = os.path.isdir(args.input)

    if not is_dir:
        # ---------- 单张模式 ----------
        img = load_gray(args.input)
        hi, wi = img.shape
        print(f'Input: {args.input}  尺寸 {wi}x{hi} (宽x高)')
        alpha, binary = None, None
        mask_file = args.mask_mat or args.mask
        if mask_file:
            alpha, binary = load_mask(mask_file, img.shape, args.mask_dilate, args.mask_feather)
            print(f'Mask: {mask_file}  修复区 {int(binary.sum())} px '
                  f'(膨胀 {args.mask_dilate}px, 羽化 {args.mask_feather}px)')
        t0 = time.time()
        out_u8 = restore(model, img, device, args.tile_h, args.tile_w, args.overlap, args.batch)
        if mask_file:
            out_u8 = apply_mask(img, out_u8, alpha)
        dt = time.time() - t0
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        cv2.imwrite(args.output, out_u8)
        mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0
        print(f'Output: {args.output}  用时 {dt:.1f}s  峰值显存 {mem:.1f}G')
        if args.gt:
            gt = load_gray(args.gt)
            if gt.shape != out_u8.shape:
                raise SystemExit(f'GT 尺寸 {gt.shape} 与输出 {out_u8.shape} 不一致')
            import basicsr.metrics as metrics
            psnr_in = metrics.calculate_psnr(img, gt, crop_border=0, test_y_channel=False)
            psnr_out = metrics.calculate_psnr(out_u8, gt, crop_border=0, test_y_channel=False)
            ssim_out = metrics.calculate_ssim(out_u8, gt, crop_border=0, test_y_channel=False)
            print(f'PSNR: 输入 {psnr_in:.4f} dB -> 修复后 {psnr_out:.4f} dB '
                  f'(提升 {psnr_out - psnr_in:+.4f} dB)')
            print(f'SSIM: 修复后 {ssim_out:.4f}')
            if binary is not None:
                bp_in, bm_in = region_psnr_mae(img, gt, binary)
                bp_out, bm_out = region_psnr_mae(out_u8, gt, binary)
                print(f'盲元区 (mask 内 {int(binary.sum())} px): PSNR {bp_in:.2f} -> {bp_out:.2f} dB   '
                      f'MAE {bm_in:.2f} -> {bm_out:.2f} 灰度级')
        return

    # ---------- 目录模式 ----------
    files = sorted(glob.glob(os.path.join(args.input, '**', '*.png'), recursive=True))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'目录里没有 *.png: {args.input}')
    print(f'共 {len(files)} 张 -> {args.output}'
          + (f'  (GT: {args.gt})' if args.gt else ''))

    import basicsr.metrics as metrics
    t_all = time.time()
    psnrs_in, psnrs_out, ssims_out, skipped_gt = [], [], [], 0
    bp_ins, bp_outs, bm_outs = [], [], []
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, args.input)
        out_path = os.path.join(args.output, rel)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        img = load_gray(f)
        out_u8 = restore(model, img, device, args.tile_h, args.tile_w, args.overlap, args.batch)
        binary = None
        line = f'[{i}/{len(files)}] {rel}'
        if args.mask_mat or args.mask_dir:
            mp = args.mask_mat or find_mask(args.mask_dir, os.path.splitext(rel)[0])
            if mp:
                alpha, binary = load_mask(mp, img.shape, args.mask_dilate, args.mask_feather)
                out_u8 = apply_mask(img, out_u8, alpha)
                line += '  [mask]'
            else:
                line += '  [无对应 mask, 整图输出]'
        cv2.imwrite(out_path, out_u8)

        if args.gt:
            gt_path = os.path.join(args.gt, rel)
            if not os.path.exists(gt_path) and args.gt_suffix:
                stem, ext = os.path.splitext(rel)
                if stem.endswith(args.gt_suffix):
                    gt_path = os.path.join(args.gt, stem[:-len(args.gt_suffix)] + ext)
            if os.path.exists(gt_path):
                gt = load_gray(gt_path)
                if gt.shape == out_u8.shape:
                    p_in = metrics.calculate_psnr(img, gt, crop_border=0, test_y_channel=False)
                    p_out = metrics.calculate_psnr(out_u8, gt, crop_border=0, test_y_channel=False)
                    s_out = metrics.calculate_ssim(out_u8, gt, crop_border=0, test_y_channel=False)
                    psnrs_in.append(p_in); psnrs_out.append(p_out); ssims_out.append(s_out)
                    line += f'  PSNR {p_in:.2f} -> {p_out:.2f}  SSIM {s_out:.4f}'
                    if binary is not None:
                        bp_in, bm_in = region_psnr_mae(img, gt, binary)
                        bp_out, bm_out = region_psnr_mae(out_u8, gt, binary)
                        bp_ins.append(bp_in); bp_outs.append(bp_out); bm_outs.append(bm_out)
                        line += f'  盲元区 {bp_in:.2f}->{bp_out:.2f}dB(MAE {bm_out:.2f})'
                else:
                    skipped_gt += 1
                    line += '  [GT 尺寸不一致, 跳过]'
            else:
                skipped_gt += 1
                line += '  [无对应 GT, 跳过指标]'
        print(line, flush=True)

    dt = time.time() - t_all
    print(f'\n完成: {len(files)} 张, 总用时 {dt/60:.1f} 分钟'
          + (f', 峰值显存 {torch.cuda.max_memory_allocated(device)/1024**3:.1f}G'
             if device.type == 'cuda' else ''))
    if psnrs_out:
        print(f'===== 平均指标 ({len(psnrs_out)} 张) =====')
        print(f'PSNR: 输入 {np.mean(psnrs_in):.4f} dB -> 修复后 {np.mean(psnrs_out):.4f} dB '
              f'(平均提升 {np.mean(psnrs_out) - np.mean(psnrs_in):+.4f} dB)')
        print(f'SSIM: {np.mean(ssims_out):.4f}')
    if bp_outs:
        print(f'===== 盲元区平均 (mask 内, {len(bp_outs)} 张) =====')
        print(f'PSNR: {np.mean(bp_ins):.4f} dB -> {np.mean(bp_outs):.4f} dB   '
              f'MAE: {np.mean(bm_outs):.4f} 灰度级')
    if skipped_gt:
        print(f'注意: {skipped_gt} 张因缺 GT/尺寸不符未计入平均指标')


if __name__ == '__main__':
    main()
