# Restormer 红外盲元修复

## 推理

1. 修改 `process_config.yml` 中的路径
2. 执行:

```bash
python process_raw.py
```

### 配置文件 `process_config.yml`

```yaml
raw_dir: "背景数据_灯管"           # 输入 raw 目录
output_dir: "results"              # 输出 PNG 目录
weights: "experiments/.../best_model.pth"
mode: "contrast"                   # contrast 或 nuc (NUC+条纹抑制)
# calib: "xxx.mat"                 # nuc 模式必填
tile_w: 640  #2048
tile_h: 512  #2048
tile_overlap: 128
batch: 8      #16                  # 并行 batch (显存大可调大)
```

### 评估

```bash
python evaluate_nr.py --output <修复图> --input <原图> --save <结果> --thresholds 5 10 20 30 50
```

| 指标 | 方向 | 含义 |
|------|------|------|
| residual_X (%) | ↓ | 像素与 5×5 邻域中值偏差 > X 的占比 |
| localstd | ↓ | 局部标准差均值 |
| roughness | ↓ | 归一化 Laplacian 高通能量 |
| estsnr | ↑ | Immerkaer 估计信噪比 |

## 训练

### 数据准备

```
data_new/
├── train_blur/   001/ frame_0001.png ...   (含盲元)
├── train_sharp/  001/ (一一对应, GT)
├── val_blur/     001/ ...
└── val_sharp/    001/ ...
```

- 640×512 灰度 PNG，文件名一一对应，子目录自动递归扫描

### 修改配置

编辑 `Denoising/Options/RealDenosing_BlindPixel_Merged.yml`：

```yaml
datasets:
  train:
    dataroot_gt: data_new/train_sharp     # ← 改这里
    dataroot_lq: data_new/train_blur      # ← 改这里
    iters: [180000]                       # ← 总迭代数
  val:
    dataroot_gt: data_new/val_sharp       # ← 改这里
    dataroot_lq: data_new/val_blur        # ← 改这里

train:
  total_iter: 180000                      # ← 和 iters 一致
  scheduler:
    periods: [60000, 60000, 60000]        # ← 和=total_iter, 每段长度
    restart_weights: [1, 1, 1]            # ← 和 periods 等长
    eta_mins: [0.000001, 0.000001, 0.000001]
```

### 训练

```bash
# 从头训练
rm -rf experiments/RealDenosing_BlindPixel_Merged
./train.sh Denoising/Options/RealDenosing_BlindPixel_Merged.yml
```

### 续训 / 加轮数

中断后直接重新运行，自动从 `latest.pth` 完整恢复（模型+优化器+scheduler）：

```bash
./train.sh Denoising/Options/RealDenosing_BlindPixel_Merged.yml
```

想加 200 轮：把 `iters` 和 `total_iter` 改大，`periods`/`restart_weights`/`eta_mins` 补上对应段数，然后 `./train.sh`。

### 保存文件

仅两个文件，PSNR 提升时同步更新：

| 文件 | 内容 |
|------|------|
| `best_model.pth` | 最佳模型权重 |
| `latest.pth` | 权重 + 优化器 + scheduler + iter (完整续训状态) |

## 文件

| 文件 | 用途 |
|------|------|
| `process_raw.py` | raw → 修复 PNG |
| `process_config.yml` | 推理参数 |
| `evaluate_nr.py` | 无参考质量评估 |
| `train.sh` | 训练启动 |
| `Denoising/Options/RealDenosing_BlindPixel_Merged.yml` | 训练配置 |
| `basicsr/` | 模型引擎 |
| `experiments/` | 模型产出 |

## 依赖

```bash
pip install torch opencv-python numpy scipy h5py pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
```
