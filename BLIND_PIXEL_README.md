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
mode: "contrast"                   # contrast（仅对比度）或 nuc（NUC+条纹抑制+对比度）
calib: ""                          # nuc 模式标定 .mat，不填则仅条纹抑制
stripe_degree: 3                   # 条纹抑制多项式阶数
tile_w: 640                        # 分块宽（训练 patch 为 384×384，640×512 分块实测无拼接痕迹）
tile_h: 512                        # 分块高
tile_overlap: 128                  # 相邻分块重叠像素（消除拼接痕迹）
batch: 24                          # 并行 batch（显存大可调大）
```

需约 35.3G 显存，640×512 分块（overlap 128）、batch=24 并行推理，单帧 6000×6000 从 raw 到 PNG 约 12 秒。

### 处理模式

| mode | calib | 管线 |
|------|-------|------|
| `contrast` | — | 对比度增强 → Restormer |
| `nuc` | 有 | NUC → 条纹抑制 → 对比度增强 → Restormer |
| `nuc` | 无 | 条纹抑制 → 对比度增强 → Restormer |

### 评估

```bash
python evaluate_nr.py --output <修复图> --input <原图> --save <结果> --thresholds 5 10 20 30 50
```

| 指标 | 方向 | 含义 |
|------|------|------|
| residual_X (%) | ↓ | 像素与 5×5 邻域中值偏差 > X 的占比 |
| localstd | ↓ | 局部标准差均值 |
| roughness | ↓ | 归一化 Laplacian 高通能量 |
| nu | ↓ | 非均匀性（局部均值变异系数） |
| estsnr | ↑ | Immerkaer 估计信噪比 |

注意：必须显式传 `--output` / `--input` / `--save`（脚本内置默认路径来自其他项目，直接运行会报错）。

## 训练

### 数据准备

```
data_new/
├── train_blur/   001/ frame_0001.png ...   (含盲元)
├── train_sharp/  001/ (一一对应, GT)
├── val_blur/     001/ ...
└── val_sharp/    001/ ...
```

- 640×512 灰度 PNG（实测 uint8，如 data6/train_blur 共 594 张），文件名一一对应，子目录自动递归扫描
- 当前仓库实际数据目录是 `data6/`（含 train/val/test 各 blur/sharp）；训练配置里提交的 `data_merged/` 路径不存在，**训练前必须改成实际目录**

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

中断后直接重新运行，自动续训：

```bash
./train.sh Denoising/Options/RealDenosing_BlindPixel_Merged.yml
```

续训的自动检测逻辑（`basicsr/train.py`）：

| 存在 `models/latest.pth`? | 行为 |
|------|------|
| 有 | 完整恢复（权重 + 优化器 + scheduler + iter 续接） |
| 无 | 回退用 `best_model.pth` 权重 warm-start，**从 iter 0 重新训练** |

注意：`latest.pth` 仅在验证 PSNR 创新高时生成。旧版训练产物（如当前 `experiments/` 下的目录）没有 `latest.pth`，直接重跑会退化为 warm-start 而非真正续训。

想加 200 轮：把 `iters` 和 `total_iter` 改大，`periods`/`restart_weights`/`eta_mins` 补上对应段数，然后 `./train.sh`。

### 保存文件

训练中不做周期保存（`save_checkpoint_freq: 999999`），仅在验证 PSNR 创新高时**成对更新**：

| 文件 | 内容 |
|------|------|
| `models/best_model.pth` | 最佳模型权重（仅 params） |
| `models/latest.pth` | 完整续训状态（权重 + 优化器 + scheduler + iter） |

目录中可能残留的 `net_g_latest.pth`、`training_states/-1.state` 是旧版代码的产物，当前版本不再生成、续训也不会使用，可删除。

## 文件

| 文件 | 用途 |
|------|------|
| `requirements.txt` | 依赖清单（一键安装入口） |
| `install_deps.sh` | 依赖一键安装脚本（清华镜像源） |
| `packages/` | 打包的离线依赖 wheel 文件夹 |
| `process_raw.py` | raw → 修复 PNG |
| `process_config.yml` | 推理参数 |
| `evaluate_nr.py` | 无参考质量评估 |
| `train.sh` | 训练启动 |
| `Denoising/Options/RealDenosing_BlindPixel_Merged.yml` | 训练配置 |
| `basicsr/` | 模型引擎 |
| `experiments/` | 模型产出 |

## 依赖

所有依赖已汇总到 `requirements.txt`，一键安装（清华镜像源）：

```bash
# 方式一: 直接联网安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式二: 把所有依赖的库打包到一个文件夹 packages/（离线备份 / 内网机器用）
pip download -r requirements.txt -d packages -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式三: 从已打包的 packages/ 文件夹离线安装
pip install --no-index --find-links=packages -r requirements.txt
```

或用脚本一键完成（等价封装上面三条命令）：

```bash
./install_deps.sh          # 联网安装
./install_deps.sh bundle   # 打包全部依赖到 packages/ 文件夹
./install_deps.sh offline  # 从 packages/ 文件夹离线安装
```

安装完成后，可选执行 `python setup.py develop --no_cuda_ext` 将本仓库注册为可导入包。

依赖清单（详见 `requirements.txt`）:

| 库 | 用途 |
|----|------|
| torch / torchvision | 深度学习框架（GPU 版建议用官方渠道安装） |
| numpy / opencv-python / scipy / scikit-image | 图像处理 |
| h5py | 读取 NUC 标定 .mat |
| PyYAML | 读取 .yml 配置 |
| einops | Restormer 的 rearrange |
| tensorboard | 训练日志 (tb_logger) |
| lpips | （可选）Defocus_Deblurring 上游测试 |
