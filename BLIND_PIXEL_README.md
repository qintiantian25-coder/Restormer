# Restormer 盲元修复 — 训练 & 推理指南

## 数据

```
ceshi_full.png              # 6000×6000 原始图像 (含盲元，未修复)
GT.png                      # 6000×6000 干净图像 (全管线修复)

data5/
├── train_blur/   001..006/ frame_0001~0054.png   (6序列, 48-54帧)
├── train_sharp/  001..006/ (一一对应)
├── test_blur/    001..004/ frame_0001~0053.png   (4序列, 47-53帧)
├── test_sharp/   001..004/
├── val_blur/     001..002/ frame_0001~0051.png   (2序列, 48-51帧)
└── val_sharp/    001..002/
```

| 属性 | 值 |
|------|-----|
| Patch 尺寸 | 640×512 (W×H) |
| 通道 | 1 (灰度) |
| 划分 | train 6 / test 4 / val 2 |
| blur 来源 | ceshi_full.png 上直接裁剪（盲元可见） |
| sharp 来源 | GT.png 同一位置裁剪（盲元已修复） |

## 训练

```bash
./train.sh Denoising/Options/RealDenosing_BlindPixel_Gray_NoMask.yml
```

训练策略：

```
640×512 patch
    ↓ random crop 384×384
    ↓ geometric augment (翻转/旋转, 8 种)
    ↓ Restormer (dim=48, grayscale)
    ↓ L1 loss
```

| 参数 | 设定 |
|------|------|
| 模型 | Restormer, inp/out_channels=1 |
| gt_size | 384 (从 640×512 中随机裁) |
| Batch size | 4 / GPU |
| 优化器 | AdamW, lr=3e-4, weight_decay=1e-3 |
| 学习率 | CosineAnnealingRestartCyclicLR, 两周期各 50k |
| 总迭代 | 100000 |
| 精度 | fp16 (AMP autocast + GradScaler) |
| 增强 | geometric_augs (8 种翻转/旋转) |
| 保存 | 仅 best_model.pth，PSNR 提升时更新 |

验证指标变化：

| iter | PSNR | SSIM |
|------|------|------|
| 4k | 34.38 | 0.8807 |
| 20k | 39.72 | 0.9526 |
| 40k | 41.52 | 0.9662 |
| 48k | 41.72 | 0.9676 | ← 第一周期最佳 |
| 52k | 40.78 | 0.9629 | ← lr 重启 (余弦退火复位) |
| 72k | 41.82 | 0.9682 | |
| 100k | **42.76** | **0.9741** | ← 最终 |

如果盲元修复效果不理想，可以在 YAML 中加 `dataroot_mask` 和 `blind_weight: 10.0` 启用加权 loss（无需改代码）。

### 续训

训练中断或想继续训练，直接重新运行训练命令即可，自动从 `best_model.state` 恢复：

```bash
./train.sh Denoising/Options/RealDenosing_BlindPixel_Gray_NoMask.yml
```

### 监控

```bash
tensorboard --logdir experiments/RealDenosing_BlindPixel_Gray_NoMask/tb_logger

# 或直接查看日志
cat experiments/RealDenosing_BlindPixel_Gray_NoMask/train_log.txt
cat experiments/RealDenosing_BlindPixel_Gray_NoMask/val_log.txt
```

## 推理

分块推理 + 羽化融合，拼缝无痕。每块输出进度、耗时、显存。

```bash
# RTX 6000 Pro (48G) — 推荐
python infer_blind_pixel.py \
    --input ceshi_full.png \
    --output restored.png \
    --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth \
    --tile 3072 --tile_overlap 128

# 其他显卡参考
python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
    --weights .../best_model.pth --tile 1024 --tile_overlap 128   # <24G 显存
python infer_blind_pixel.py --input ceshi_full.png --output restored.png \
    --weights .../best_model.pth --tile 2048 --tile_overlap 128   # 24-40G 显存
```

| 参数 | 说明 |
|------|------|
| `--input` | 原始含盲元图像 |
| `--output` | 修复后保存路径 |
| `--weights` | best_model.pth 路径 |
| `--tile` | 分块大小 (8 的倍数)，默认 1024 |
| `--tile_overlap` | 块间重叠像素，默认 128 |

## 为什么分块训练 + 分块推理可行

Restormer 是全卷积网络，所有操作都与输入尺寸无关：

- **MDTA 注意力** — 计算在通道维度 (48×48 矩阵)，不随空间尺寸缩放
- **LayerNorm** — 逐通道统计量，与 H、W 无关
- **没有位置编码** — 没有 learnable positional embedding
- 3 次 PixelUnshuffle(2) 下采样 → 输入必须是 8 的倍数

推理使用 **raised-cosine 羽化融合**，每个 tile 边缘权重渐变到 0，消除拼缝。

## 文件说明

| 文件 | 用途 |
|------|------|
| `basicsr/data/data_util.py` | 新增 `paired_paths_from_folder_recursive` |
| `basicsr/data/paired_image_dataset.py` | 新增 `Dataset_PairedImage_BlindPixel`，支持子目录扫描 |
| `basicsr/models/image_restoration_model.py` | fp16 AMP + best_model 保存 + 加权 loss + val_log |
| `basicsr/models/base_model.py` | `save_training_state` 支持自定义文件名 |
| `basicsr/train.py` | train_log + best_model.state 优先续训 |
| `Denoising/Options/RealDenosing_BlindPixel_Gray_NoMask.yml` | 训练配置 |
| `infer_blind_pixel.py` | 推理脚本 (分块+羽化融合+进度输出) |
| `train.sh` | 单卡非分布式启动 |
