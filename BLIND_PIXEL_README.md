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
| 精度 | fp16 |
| 增强 | geometric_augs (8 种翻转/旋转) |
| 保存频率 | 每 5000 iter 保存 checkpoint + best_model.pth |

如果盲元修复效果不理想，可以在 YAML 中加 `dataroot_mask` 和 `blind_weight: 10.0` 启用加权 loss（无需改代码）。

## 推理

```bash
# 全图一次推理 (94G 显存推荐)
python infer_blind_pixel.py \
    --input ceshi_full.png \
    --output restored.png \
    --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth

# 分块推理 (备选，显存不够时使用)
python infer_blind_pixel.py \
    --input ceshi_full.png \
    --output restored.png \
    --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth \
    --tile 2048 --tile_overlap 64
```

| 参数 | 说明 |
|------|------|
| `--input` | 6000×6000 含盲元图像 |
| `--output` | 修复后的输出路径 |
| `--weights` | 训练产出的 .pth 文件 |
| `--tile` | 分块大小 (8 的倍数)，不指定则全图推理 |
| `--tile_overlap` | 块间重叠像素，默认 64 |

## 为什么分块训练 + 全图推理可行

Restormer 是全卷积网络，所有操作都与输入尺寸无关：

- **MDTA 注意力** — 计算在通道维度 (48×48 矩阵)，不随空间尺寸缩放
- **LayerNorm** — 逐通道统计量，与 H、W 无关
- **没有位置编码** — 没有 learnable positional embedding
- 3 次 PixelUnshuffle(2) 下采样 → 输入必须是 8 的倍数

## 显存估算

| 阶段 | 尺寸 | 最大特征图 |
|------|------|-----------|
| 训练 (fp16) | 384×384 | ~100 MB |
| 推理 全图 | 6000×6000 | ~7 GB (fp16) / ~14 GB (fp32) |

94G 显存下 6000×6000 全图推理绰绰有余，不需要分块。

## 监控

```bash
tensorboard --logdir experiments/RealDenosing_BlindPixel_Gray_NoMask/tb_logger
```

关注：
- `l_pix` 持续下降 → 正常
- `metrics/psnr` 验证集上升 → 正常
- `metrics/psnr` 不升反降 + l_pix 还在降 → 过拟合，需要更强正则化
