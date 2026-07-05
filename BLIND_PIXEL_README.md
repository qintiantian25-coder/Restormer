# Restormer 盲元修复 — 训练 & 推理指南

## 数据

```
ceshi_full.png              # 6000×6000 原始图像 (含盲元，未修复)
GT.png                      # 2048×2048 无噪声 GT → resize 到 GT_6000.png

data5/
├── train_blur/   001..006/ frame_0001~0054.png   (6序列, 48-54帧)
├── train_sharp/  001..006/ (一一对应, 无噪声 GT)
├── test_blur/    001..004/ frame_0001~0052.png   (4序列, 51-52帧)
├── test_sharp/   001..004/
├── val_blur/     001..002/ frame_0001~0052.png   (2序列, 50-52帧)
└── val_sharp/    001..002/
```

| 属性 | 值 |
|------|-----|
| Patch 尺寸 | 640×512 (W×H) |
| 通道 | 1 (灰度) |
| 划分 | train 6 / test 4 / val 2 |
| blur 来源 | ceshi_full.png 上直接裁剪（盲元可见） |
| sharp 来源 | GT.png 同一位置裁剪（无噪声真值） |

## 训练

### 重新训练（clean GT，从头开始）

```bash
# 1. 删除旧实验目录（干净 GT 不需要旧权重）
rm -rf experiments/RealDenosing_BlindPixel_Gray_NoMask

# 2. 开始训练
./train.sh Denoising/Options/RealDenosing_BlindPixel_Gray_NoMask.yml
```

### 续训

```bash
./train.sh Denoising/Options/RealDenosing_BlindPixel_Gray_NoMask.yml
```

```
640×512 patch
    ↓ random crop 384×384
    ↓ geometric augment (翻转/旋转, 8 种)
    ↓ Restormer (dim=48, grayscale)
    ↓ L1 loss (无盲元加权)
```

| 参数 | 设定 |
|------|------|
| 模型 | Restormer, inp/out_channels=1 |
| gt_size | 384 |
| Batch size | 4 / GPU |
| 优化器 | AdamW, lr=3e-4, weight_decay=1e-3 |
| 学习率 | CosineAnnealingRestartCyclicLR, 三周期各 50k |
| 总迭代 | 150000 |
| 精度 | fp16 (autocast + GradScaler) |
| 增强 | geometric_augs |
| 保存 | 仅 best_model.pth，PSNR 提升时更新 |

### 第二阶段：加权 loss（如果盲元残影明显）

首先生成盲元 mask（基于局部中值检测，不依赖 GT 质量）：

```bash
# 从 blur 图像自身检测盲元
python generate_masks.py --root data5 --split train --threshold 30
python generate_masks.py --root data5 --split val --threshold 30
```

然后 warm-start 训练：

```bash
./train.sh Denoising/Options/RealDenosing_BlindPixel_Gray_Masked.yml
```

mask 位置 loss 权重 10×，从 NoMask best_model 续训。阈值可用 `--visualize` 检查调整。

### 监控

```bash
tensorboard --logdir experiments/RealDenosing_BlindPixel_Gray_NoMask/tb_logger
cat experiments/RealDenosing_BlindPixel_Gray_NoMask/train_log.txt
cat experiments/RealDenosing_BlindPixel_Gray_NoMask/val_log.txt
```

## 推理

### 单张全图

默认 640×512 分块 (与训练一致)，fp16，均匀加权融合：

```bash
python infer_blind_pixel.py \
    --input ceshi_full.png \
    --output restored.png \
    --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth

# 方形大块
python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
    --weights .../best_model.pth --tile 2048 --tile_overlap 128
```

| 参数 | 说明 |
|------|------|
| `--input` | 原始含盲元图像 |
| `--output` | 修复后保存路径 |
| `--weights` | best_model.pth 路径 |
| `--tile_w / --tile_h` | 分块尺寸 (8 的倍数)，默认 640×512 |
| `--tile` | 方形分块快捷方式 |
| `--tile_overlap` | 块间重叠，默认 128 |
| `--fp32` | 强制 fp32 (默认 fp16) |

### 测试集批量评估

```bash
# model 修复版
python eval_test.py

# 保存修复图片
python eval_test.py --save results/test_restored

# baseline: blur 原始 vs sharp
python eval_test.py --no_model
```

输出每个序列和总体的 PSNR/SSIM。

## 盲元分析

```bash
# 生成 restored.png 的盲元残影检测图 (红色标记)
python gen_residual_mask.py --input restored.png

# 从 blur 图像批量生成 mask
python generate_masks.py --root data5 --split train --threshold 30
python generate_masks.py --root data5 --split train --threshold 30 --visualize
```

`generate_masks.py` 基于局部中值滤波，不依赖 GT。`--threshold` 越小越敏感。

## 文件说明

| 文件 | 用途 |
|------|------|
| `basicsr/data/paired_image_dataset.py` | `Dataset_PairedImage_BlindPixel`，子目录扫描 + CSV/PNG mask |
| `basicsr/models/image_restoration_model.py` | fp16 AMP + best_model 保存 + 加权 loss + val_log |
| `basicsr/models/base_model.py` | `save_training_state` 自定义文件名 |
| `basicsr/train.py` | train_log + best_model.state 优先续训 |
| `Denoising/Options/RealDenosing_BlindPixel_Gray_NoMask.yml` | 无 mask 训练 |
| `Denoising/Options/RealDenosing_BlindPixel_Gray_Masked.yml` | 加权 loss 训练 (warm-start) |
| `infer_blind_pixel.py` | 全图分块推理 |
| `eval_test.py` | 测试集批量评估 (PSNR/SSIM) |
| `generate_masks.py` | 盲元 mask 自动检测 |
| `resize_gt.py` | GT 图像缩放 |
| `train.sh` | 单卡非分布式启动 |
