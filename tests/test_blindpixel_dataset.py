"""回归测试: Dataset_PairedImage_BlindPixel 的随机裁剪/增强对齐 与 padding 路径.

背景 (P1): cv2.copyMakeBorder 会把单通道 (H,W,1) 数组压成 (H,W), 导致小于
gt_size 的图像在 padding 分支后返回 (1,gt_size) 而不是 (1,gt_size,gt_size)
张量。修复见 paired_image_dataset.py 的 padding 分支 (补回通道维)。

覆盖:
  1. 正常 640x512 数据路径: 随机裁 gt_size, lq/gt/mask 形状正确
  2. 小尺寸切片的 padding 路径 (单轴 / 双轴超界): 形状正确, 不退化不崩
  3. 对齐关系: 每一像素上 (mask>0) 与 (lq!=gt) 完全一致 (含随机几何增强)
  4. 验证相位 (phase=val): 不做裁剪, 返回整图尺寸

运行: python tests/test_blindpixel_dataset.py   (也可被 pytest 收集)
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basicsr.data.paired_image_dataset import Dataset_PairedImage_BlindPixel

GT_SIZE = 384
SAMPLES_PER_CASE = 30


def _make_pair(root, sub, h, w, n_blind, seed):
    """构造一对合成图: gt 全非零随机, lq 在盲元坐标处置 0, mask 置 255."""
    rng = np.random.default_rng(seed)
    gt = rng.integers(1, 256, size=(h, w), dtype=np.uint8)
    lq = gt.copy()
    mask = np.zeros((h, w), np.uint8)
    ys = rng.integers(0, h, n_blind)
    xs = rng.integers(0, w, n_blind)
    lq[ys, xs] = 0
    mask[ys, xs] = 255
    for name, img in (('gt', gt), ('lq', lq), ('mask', mask)):
        d = os.path.join(root, name, sub)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, 'f0001.png'), img)


def _make_opt(root, phase):
    return {
        'name': 'RegTest',
        'type': 'Dataset_PairedImage_BlindPixel',
        'phase': phase,
        'scale': 1,
        'dataroot_gt': os.path.join(root, 'gt'),
        'dataroot_lq': os.path.join(root, 'lq'),
        'dataroot_mask': os.path.join(root, 'mask'),
        'in_ch': 1,
        'gt_size': GT_SIZE,
        'geometric_augs': True,
        'filename_tmpl': '{}',
        'io_backend': {'type': 'disk'},
        'blind_weight': 0.0,
    }


def _check_train_case(ds, idx, case):
    """形状 + 对齐关系断言; 返回累计盲元像素数."""
    total_blind = 0
    for _ in range(SAMPLES_PER_CASE):
        sample = ds[idx]
        lq, gt, mask = sample['lq'], sample['gt'], sample['mask']
        for name, t in (('lq', lq), ('gt', gt), ('mask', mask)):
            assert tuple(t.shape) == (1, GT_SIZE, GT_SIZE), (
                f'{case}: {name} 形状 {tuple(t.shape)} != (1,{GT_SIZE},{GT_SIZE}) '
                f'(padding 路径通道维退化回归)')
        mask_bool = mask[0] > 0.5
        diff = lq[0] != gt[0]
        assert torch.equal(mask_bool, diff), (
            f'{case}: mask>0 与 lq!=gt 不逐像素一致 -> 裁剪/增强错位')
        total_blind += int(diff.sum())
    assert total_blind > 0, f'{case}: {SAMPLES_PER_CASE} 次采样均无盲元像素, 测试无效'
    return total_blind


def test_train_normal_slice():
    """640x512 切片 (H=512, W=640): 正常裁剪路径."""
    with tempfile.TemporaryDirectory() as root:
        _make_pair(root, '001', 512, 640, 300, seed=1)
        ds = Dataset_PairedImage_BlindPixel(_make_opt(root, 'train'))
        total = _check_train_case(ds, 0, '640x512 正常路径')
        print(f'  [PASS] 640x512 正常路径: {SAMPLES_PER_CASE} 次采样形状/对齐 OK, '
              f'平均盲元 {total / SAMPLES_PER_CASE:.1f} px/crop')


def test_train_small_slice_padding():
    """小于 gt_size 的切片: 单轴 (300x400) 与双轴 (200x300) padding 路径."""
    with tempfile.TemporaryDirectory() as root:
        _make_pair(root, '002', 300, 400, 120, seed=2)  # 仅 H 轴需要 padding
        _make_pair(root, '003', 200, 300, 80, seed=3)   # H/W 双轴都需要 padding
        ds = Dataset_PairedImage_BlindPixel(_make_opt(root, 'train'))
        assert len(ds.paths) == 2, f'期望 2 对样本, 实际 {len(ds.paths)}'
        for idx, case in ((0, '300x400 单轴 padding'), (1, '200x300 双轴 padding')):
            total = _check_train_case(ds, idx, case)
            print(f'  [PASS] {case}: {SAMPLES_PER_CASE} 次采样形状/对齐 OK, '
                  f'平均盲元 {total / SAMPLES_PER_CASE:.1f} px/crop')


def test_val_phase_whole_image():
    """验证相位: 不裁剪, 返回整图 (1,512,640)."""
    with tempfile.TemporaryDirectory() as root:
        _make_pair(root, '001', 512, 640, 300, seed=1)
        ds = Dataset_PairedImage_BlindPixel(_make_opt(root, 'val'))
        sample = ds[0]
        for name in ('lq', 'gt', 'mask'):
            assert tuple(sample[name].shape) == (1, 512, 640), (
                f'val: {name} 形状 {tuple(sample[name].shape)} != (1,512,640)')
        print('  [PASS] val 相位整图: (1,512,640), 未裁剪')


def main():
    print('Dataset_PairedImage_BlindPixel 回归测试')
    test_train_normal_slice()
    test_train_small_slice_padding()
    test_val_phase_whole_image()
    print('全部通过')


if __name__ == '__main__':
    main()
