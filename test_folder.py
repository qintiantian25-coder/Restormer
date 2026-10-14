"""
固定测试入口: test_in/ 里丢什么就测什么, 结果同名输出到 results/.

用法 (先改下面 CONFIG 区, 然后一条命令):
  python test_folder.py

要点:
  - 支持格式: png / tif / tiff / bmp / jpg / jpeg (原样读取, 16bit 自动转 8bit, 彩色转灰度);
              raw (uint16, 6000x6000 或边长相等方可自动推断)
  - raw 若在 CONFIG 里填了 RAW_CALIB (.mat), 走 NUC -> 条纹抑制 -> 对比度;
    不填则仅条纹抑制 -> 对比度 (与 process_raw 的 nuc 模式一致)
  - 其余格式不做预处理 (适合已处理好的测试图, 如 *_sim6000_blind.png)
  - mask 开关 (CONFIG: USE_MASK + MASK_DIR 或 MASK_MAT): 开着时 "有 mask 就用 mask" —
    有对应 mask 的图只修 mask 区 (mask 外保持原图; 详见 mask_blend.py),
    没有 mask 的图照常整图修复;
    关掉 (USE_MASK=False, 默认) 则全部整图, 与之前完全一样.
    MASK_DIR: 按同名查找 (test_in/a.raw -> MASK_DIR/a.mat 或 a.png), 尺寸须一致;
    MASK_MAT: 直接指定一个 mask 文件 (.mat 的 loc / 图片) 的路径, 所有测试图都用它 (优先于 MASK_DIR);
    图片 mask >127 为修复区, .mat 取 loc (0/1 判定图)
  - 输出: results/<同名>.png (重名会覆盖旧结果); 同一文件夹里同名不同扩展名 (a.png + a.tif)
    只处理排序靠前的那个, 另一个打印警告并跳过
  - 分块推理参数与 process_raw 一致 (tile 640x512 / overlap 128), 大图自动滑窗
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer
from mask_blend import apply_mask, find_mask, imread_u, load_mask
from process_raw import (load_calib, preprocess_nuc, preprocess_nuc_no_calib,
                         read_raw, tiled_forward)

# ======================================================================
#  ↓↓↓ 直接在这里改路径/参数 ↓↓↓
TEST_DIR = r'D:\qtt\Restormer\test_in'        # 待测文件放这里 (自动创建)
OUT_DIR = r'D:\qtt\Restormer\results'         # 结果输出这里 (同名 .png)
WEIGHTS = r'D:\qtt\Restormer\experiments\RealDenosing_BlindPixel_Merged\models\best_model.pth'
RAW_CALIB = None          # raw 的 NUC 标定 .mat 路径; None = 仅条纹抑制+对比度
STRIPE_DEGREE = 3
BATCH = 8                 # 24G 卡用 8; 8G 卡用 2
USE_MASK = False          # ← mask 开关: True = 有 mask 的图只修 mask 区, 没有 mask 的照常整图修复; False = 全部整图
MASK_DIR = None           # mask 目录 (按同名找 mask: test_in/a.png -> MASK_DIR/a.png / a.mat)
MASK_MAT = None           # 或直接指定一个 mask 文件 (.mat 的 loc / 图片) 的完整路径: 所有测试图都用它 (优先于 MASK_DIR)
MASK_DILATE = 4           # mask 膨胀像素 (4 = 与训练 mask 口径一致, 覆盖盲元边缘过渡带)
MASK_FEATHER = 3          # 边界羽化 (0 = mask 外严格不动; >0 = 边缘软过渡, 约 2~3 倍宽度内轻微混合)
TILE_W, TILE_H, TILE_OVERLAP = 640, 512, 128
#  ↑↑↑ 直接在这里改路径/参数 ↑↑↑
# ======================================================================

EXTS = ('.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg', '.raw')


def imwrite_u(path, img):
    """Unicode 路径安全写图."""
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f'编码失败: {path}')
    buf.tofile(path)


def load_input(path, kk, bb):
    """读任意格式 -> uint8 灰度 (推理输入)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.raw':
        frame = read_raw(path)
        if kk is not None:
            return preprocess_nuc(frame, kk, bb, STRIPE_DEGREE)
        return preprocess_nuc_no_calib(frame, STRIPE_DEGREE)
    img = imread_u(path)
    if img is None:
        raise IOError(f'读图失败: {path}')
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535. * 255).round().astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def main():
    ap = argparse.ArgumentParser(description='固定测试入口: test_in -> results')
    ap.add_argument('--dir', default=None, help='覆盖 CONFIG 的 TEST_DIR')
    ap.add_argument('--out', default=None, help='覆盖 CONFIG 的 OUT_DIR')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--calib', default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--mask_dir', default=None, help='mask 目录 (传了就开启 mask 模式)')
    ap.add_argument('--mask_mat', default=None, help='直接指定 mask 文件 (.mat/图片) 路径 (传了就开启 mask 模式)')
    ap.add_argument('--use_mask', action='store_true', help='强制开启 mask 模式 (用 CONFIG 的 MASK_DIR/MASK_MAT)')
    ap.add_argument('--no_mask', action='store_true', help='强制关闭 mask 模式 (覆盖 CONFIG 的 USE_MASK)')
    ap.add_argument('--mask_dilate', type=int, default=None)
    ap.add_argument('--mask_feather', type=int, default=None)
    args = ap.parse_args()
    test_dir = args.dir or TEST_DIR
    out_dir = args.out or OUT_DIR
    weights = args.weights or WEIGHTS
    calib = args.calib if args.calib is not None else RAW_CALIB
    batch = args.batch or BATCH
    mask_dir = args.mask_dir if args.mask_dir is not None else MASK_DIR
    mask_mat = args.mask_mat if args.mask_mat is not None else MASK_MAT
    mask_dilate = args.mask_dilate if args.mask_dilate is not None else MASK_DILATE
    mask_feather = args.mask_feather if args.mask_feather is not None else MASK_FEATHER
    use_mask = (USE_MASK or args.use_mask or (args.mask_dir is not None)
                or (args.mask_mat is not None))
    if args.no_mask:
        use_mask = False
    if use_mask and not mask_dir and not mask_mat:
        raise SystemExit('mask 开关开着但 MASK_DIR / MASK_MAT 都是空的: 填一个, 或加 --no_mask 关闭')
    if use_mask:
        if mask_mat and not os.path.isfile(mask_mat):
            print(f'!! MASK_MAT 不存在: {mask_mat}')
            mask_mat = None
        if not mask_mat and mask_dir and not os.path.isdir(mask_dir):
            print(f'!! MASK_DIR 不存在: {mask_dir}')
            mask_dir = None
        if not mask_mat and not mask_dir:
            print('!! mask 源都不可用 (MASK_MAT / MASK_DIR), mask 模式自动关闭, 全部整图输出')
            use_mask = False

    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(test_dir)
                   if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(test_dir, f)))
    if not files:
        print(f'没有待测文件。把图片放进: {test_dir}')
        print(f'支持格式: {", ".join(EXTS)}')
        return
    if not os.path.isfile(weights):
        raise SystemExit(f'权重不存在: {weights}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        print(f'GPU: {p.name} ({p.total_memory/1024**3:.0f} GB)')

    print('Loading model...')
    model = Restormer(inp_channels=1, out_channels=1, dim=48,
                      num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                      heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type='BiasFree',
                      dual_pixel_task=False).to(device)
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['params'])
    model.eval()

    kk = bb = None
    if calib:
        if os.path.isfile(calib):
            kk, bb = load_calib(calib)
            print(f'Calibration: {calib}')
        else:
            print(f'!! 标定文件不存在, raw 将仅做条纹抑制: {calib}')
    if use_mask:
        if mask_mat:
            print(f'Mask 模式: 所有图都用指定 mask (膨胀 {mask_dilate}px, 羽化 {mask_feather}px)  <- {mask_mat}')
        else:
            print(f'Mask 模式: 有 mask 的图只修 mask 区 (膨胀 {mask_dilate}px, 羽化 {mask_feather}px), '
                  f'无 mask 的整图修复  <- {mask_dir}')

    print(f'\n共 {len(files)} 个待测文件 -> {out_dir}\n')
    used_stems = {}
    done = 0
    failed = 0
    n_masked = 0
    t_all = time.time()
    for i, name in enumerate(files, 1):
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(out_dir, stem + '.png')
        if stem in used_stems:
            print(f'[{i}/{len(files)}] {name}  !! 和 {used_stems[stem]} 输出同名 ({stem}.png), 跳过')
            continue
        used_stems[stem] = name
        existed = os.path.exists(out_path)
        t0 = time.time()
        tag = ''
        try:
            img = load_input(os.path.join(test_dir, name), kk, bb)
            h, w = img.shape
            t = torch.from_numpy(img.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0).to(device)
            H = h if h % 16 == 0 else ((h + 15) // 16) * 16
            W = w if w % 16 == 0 else ((w + 15) // 16) * 16
            t = F.pad(t, (0, W - w, 0, H - h), 'reflect')
            out = tiled_forward(model, t, TILE_H, TILE_W, TILE_OVERLAP, device, batch)
            out = out[:, :, :h, :w].squeeze().cpu().numpy()
            out_u8 = (out * 255).round().clip(0, 255).astype(np.uint8)
            if use_mask:
                mp = mask_mat if mask_mat else find_mask(mask_dir, stem)
                if mp:
                    alpha, _ = load_mask(mp, (h, w), mask_dilate, mask_feather)
                    out_u8 = apply_mask(img, out_u8, alpha)
                    tag = '  [mask]'
                    n_masked += 1
                else:
                    tag = '  [无mask, 整图]'
            imwrite_u(out_path, out_u8)
        except Exception as e:
            print(f'[{i}/{len(files)}] {name}  !! 失败: {e}')
            failed += 1
            continue
        note = '  (覆盖旧结果)' if existed else ''
        print(f'[{i}/{len(files)}] {name}  ({w}x{h})  {time.time()-t0:.1f}s  -> {stem}.png{note}{tag}')
        done += 1
    tail = f', 失败 {failed} 张' if failed else ''
    tail2 = f' (mask 修复 {n_masked} 张, 整图 {done - n_masked} 张)' if use_mask else ''
    print(f'\n完成: {done}/{len(files)} 张{tail}{tail2}, 总用时 {(time.time()-t_all)/60:.1f} 分钟 -> {out_dir}')


if __name__ == '__main__':
    main()
