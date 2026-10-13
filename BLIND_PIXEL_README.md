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
batch: 2                           # 并行 batch，按显存选（见下表）；只影响速度，不影响修复结果
```

### 显存与耗时（实测）

推理显存只由 **「分块面积 × batch」** 决定，**与整图尺寸无关**——6000×6000 的输入/累加缓冲恒定 ~0.4G，峰值只取决于单个分块。所以 8G 卡照样能推 6000×6000，只是慢一点。

| 分块 | batch | 实测 peak allocated | 适用显存 |
|------|-------|--------------------|---------|
| 640×512 | 1 | 2.1 G | 8G |
| 640×512 | **2** | **3.4 G**（真实 raw 端到端实测 3.5 G，单帧 30 s） | **8G（默认配置）** |
| 640×512 | 3 | 4.9 G | 8G（卡上无其他进程时） |
| 640×512 | 8 | 12.2 G | 24G |
| 640×512 | 16 | 23.7 G | 40G |
| 640×512 | 24 | 35.3 G | 96G（脚本原先的默认值） |
| 384×384 | 4 | 3.2 G | 8G（更省，但块数 529 > 192，更慢） |

- 经验公式：`peak ≈ 0.7G + 1.42G × batch`（640×512 分块，fp16 autocast）
- 上表是脚本打印的 `GPU_mem`（torch peak allocated）；`nvidia-smi` 上看到的一般再高 10~25%（allocator reserved）
- **OOM 就调小 `batch`，不要缩分块**——分块尺寸影响拼接痕迹，batch 只影响速度

耗时：空闲卡上 batch 24 单帧 6000×6000 约 12 秒；batch 2 在 8G 消费级卡上约 1~3 分钟（随算力浮动，非实测）。

### 处理模式

| mode | calib | 管线 |
|------|-------|------|
| `contrast` | — | 对比度增强 → Restormer |
| `nuc` | 有 | NUC → 条纹抑制 → 对比度增强 → Restormer |
| `nuc` | 无 | 条纹抑制 → 对比度增强 → Restormer |

### PNG 推理（`infer_png.py`）— 单张 / 目录

测试已经预处理好的 PNG，不做 raw/NUC/条纹/对比度，直接 PNG → Restormer → PNG。支持单张和整个目录（递归扫描、保持子目录结构）：

```bash
# 单张
python infer_png.py --input data_merged/658_4_sim6000_blind.png \
    --output results/658_4_repaired.png \
    --weights experiments/RealDenosing_BlindPixel_Merged/models/best_model.pth \
    --batch 8

# 整个目录 (如 test_blur 723 张); --gt 传配对目录则打印平均 PSNR/SSIM
python infer_png.py --input data_merged/test_blur --output results/test_blur_eval \
    --gt data_merged/test_sharp \
    --weights experiments/RealDenosing_BlindPixel_Merged/models/best_model.pth \
    --batch 8
```

- 单张模式可选 `--gt <干净参考图>`；目录模式加 `--limit N` 可只跑前 N 张快速预览
- 分块参数与 `process_raw.py` 一致（640×512 / overlap 128）；`--batch` 按显存调（24G 卡用 8）
- 640×512 等本身就是 16 倍数的图不会多余 pad，单张 = 一次前向（与训练验证口径一致）

### 块状盲元诊断（`diag_block.py`，不用 GPU）

```bash
python diag_block.py --blur_dir data_merged/test_blur --gt_dir data_merged/test_sharp \
    --pred_dir results/test_blur_eval --save_dir results/block_diag
```

- 盲元判定：`|blur-gt| > thr`（默认 30）的连通域，块 ≥ 6 px
- 输出：按块大小分桶的 块数 / 修复前误差 / 修复后误差 / 模型改动量；`--save_dir` 下保存最差块的 (blur|gt|pred) 对比拼图
- 判读：某桶"修复后误差"高 + "模型改动量"≈0 → 模型对该尺寸的块基本没动（能力短板）

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
data_merged/
├── train_blur/   001/ frame_0001.png ...   (含盲元)
├── train_sharp/  001/ (一一对应, GT)
├── val_blur/     001/ ...
├── val_sharp/    001/ ...
└── test_blur/  test_sharp/ ...
```

- 640×512 灰度 PNG（uint8），文件名一一对应，子目录自动递归扫描
- 当前配置指向 `data_merged/`：train 1090 / val 418 / test 455 对。它等于 `data5/`(2094 张) + `data6/`(1832 张) 的合并副本，`data5/`、`data6/` 是原始来源（保留，不再被配置引用）
- `val:` 段现指向 **`data_merged/val_*`**（686 对，2026-10-12 从 `test_*` 切换而来）；`test_*`（723 对）留作纯测试集，不再参与 best 门控。⚠️ 切换验证集后 PSNR 与历史记录不可直接对比
- 训练时网络看到的是 `gt_size`(384×384) 的**随机方形裁剪**，640×512 只是数据切片的尺寸（见 `Dataset_PairedImage_BlindPixel.__getitem__`）
- 推理分块的边长必须 **≥ 训练 patch**（384）才不出边界痕迹；640×512 分块已实测无拼接痕迹
- 小于 `gt_size` 的切片会先 reflect padding 再裁剪（`paired_image_dataset.py` 的 padding 分支）。该分支曾因 `cv2.copyMakeBorder` 把单通道 `(H,W,1)` 压成 `(H,W)` 而返回 `(1,gt_size)` 的错误形状，已修复（`img_lq/img_gt` 补回通道维），并有回归测试覆盖

### 回归测试

```bash
python tests/test_blindpixel_dataset.py     # 也可被 pytest 收集
```

覆盖：640×512 正常裁剪路径、小尺寸切片 padding 路径（单轴 300×400 / 双轴 200×300）、blur/GT/mask 逐像素对齐（含随机几何增强）、val 相位整图不裁剪。改动数据集代码后应重跑。

### 修改配置

编辑 `Denoising/Options/RealDenosing_BlindPixel_Merged.yml`：

```yaml
datasets:
  train:
    dataroot_gt: data_merged/train_sharp  # ← 训练 GT（当前: 已验证分布 + 加权损失）
    dataroot_lq: data_merged/train_blur   # ← 训练 blur
    dataroot_mask: data_merged/train_mask # ← 盲元 mask（blind_weight 加权损失用, make_masks.py 生成）
    blind_weight: 10.0                    # ← 盲元像素损失权重
    batch_size_per_gpu: 1                 # ← 数据加载 batch（和 mini_batch_sizes 保持一致）
    mini_batch_sizes: [1]                 # ← 实际训练 batch（按显存选，见下表）
    gt_size: 384                          # ← 训练 patch，别超过推理分块边长
    gt_sizes: [384]
    iters: [30000]                        # ← 总迭代数
  val:
    dataroot_gt: data_merged/val_sharp    # ← 验证集 (2026-10-12 起用 val_*; test_* 留给纯测试)
    dataroot_lq: data_merged/val_blur

train:
  total_iter: 30000                       # ← 和 iters 一致
  blind_weight: 10.0                      # ← 盲元加权（生效值; 与数据集 mask 配套）
  optim_g:
    lr: !!float 2e-5                      # ← 微调口径（从零训练用 1e-4）
  scheduler:
    periods: [30000]                      # ← 和=total_iter（单周期余弦; 多周期重启会把 lr 打回满值, 不适合微调）
    restart_weights: [1]                  # ← 和 periods 等长
    eta_mins: [0.000001]
```

### 训练显存（实测）

训练步显存 ≈ **90 KB × patch 面积(像素) × batch**（fp16 autocast，含梯度与 AdamW 状态），与 patch 是否方形无关：

| 训练 patch | batch | 实测 peak allocated | 适用显存 |
|-----------|-------|--------------------|---------|
| 384×384 | **1** | **13.0 G**（真实训练全程含验证实测 12.96 G / reserved 16.6 G） | **24G（默认配置）** |
| 384×384 | 2 | 25.5 G | 32G |
| 384×384 | 4 | 50.9 G | 64G+ |
| 320×320 | 2 | 17.9 G | 24G |
| 256×256 | 4 | 22.8 G | 24G（余量小） |
| 512×512 | 1 | 22.6 G | 24G（余量小） |
| 512×640 | 1 | 28.1 G | ❌ 24G 装不下 |

- 24G 卡的可用预算按 ~22G 算：**patch 面积 × batch ≤ 约 250k 像素**
- ⚠️ **`basicsr/train.py` 里的 `torch.backends.cudnn.benchmark` 已从 `True` 改成 `False`，不要改回去**：开着它时 cuDNN 会枚举卷积实现并为候选方案申请 workspace，实测在验证阶段（第一次遇到 640×512 卷积形状）申请了 **54.9 G**，24G 卡直接 OOM。关掉后整条链路（训练 + 验证 + 存 checkpoint）峰值才 12.96 G
- batch 从 4 降到 1 时把 `lr` 一起降（1e-4），否则等效学习率变大
- 本仓库**不支持梯度累积**（没有 accum 实现），batch 只能真降
- 想训 512×640 整片需要 32G+ 的卡，或自己给 `TransformerBlock` 加 `torch.utils.checkpoint`
- val 是整张 640×512 无梯度前向（~2G），与训练步不叠加

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

续训的自动检测逻辑（`basicsr/train.py`，三种情况）：

| `experiments/<name>/models/` 里有什么 | 行为 |
|------|------|
| `latest.pth` | **完整续训**：权重 + 优化器 + scheduler + iter 续接；实验目录不动（不归档） |
| 只有 `best_model.pth` | **warm-start**：best 先复制到 `experiments/<name>_warmstart.pth`，旧实验目录归档为 `<name>_archived_<时间戳>`，然后在**新**目录下从 iter 0 重新训练（全新优化器/scheduler） |
| 都没有（或 yml 里 pretrain 路径失效） | 从头训练（打印提示，不再报 FileNotFoundError） |

注意：

- `latest.pth` 与 `best_model.pth` 在验证 PSNR 创新高时**成对写入**；若只有 `best_model.pth`（旧版产物），重跑就是 warm-start 而不是迭代续接
- `best_psnr` 随 `latest.pth` 保存并在续训时恢复 → 续训后不会被更差的验证结果覆盖 `best_model.pth`；warm-start 是全新训练，best 会被第一次验证刷新（**开训前建议先备份旧 `best_model.pth`**）
- **扩数据**：直接往 `data_merged/train_*` 的对应子目录放图即可（文件名与 sharp 一一对应），启动时自动重扫，无需改配置
- **加轮数**：把 `datasets.train.iters` 与 `train.total_iter` 一起改大，`periods`/`restart_weights`/`eta_mins` 补上对应段数（`sum(periods) == total_iter`）。真续训时 YAML 里新的 `periods` 会覆盖存档里的旧值并立即生效
- 更换验证集（如 `test_*` ↔ `val_*`）后 PSNR 与历史记录不可直接对比；续训时检测到验证集更换会自动重置 best 门槛（`latest.pth` 会记录 `val_gt`）

### 验证指标与 best 门控

- `val.metrics` 现在包含全图 `psnr`/`ssim` **和盲元区 `blind_psnr`/`blind_mae`**（mask 由 `|val_blur−val_sharp|>30` 现场生成，无盲元的验证图自动跳过；`blind_mae` 单位=灰度级，越小越好）
- **best 门控优先用 `blind_psnr`**（yml 里存在该项时）——`best_model.pth`/`latest.pth` 按"盲元区 PSNR 最高"保存；`val_log.txt` 每次验证记录全部指标
- 想回退到全图 `psnr` 门控：删掉 yml 里的 `blind_psnr` 指标项即可

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
| `packages/` | 打包的离线依赖 wheel 文件夹（离线安装用，当前不存在） |
| `process_raw.py` | raw → 修复 PNG |
| `infer_png.py` | 单张/目录 PNG 测试推理（已预处理图，免 raw 链路） |
| `diag_block.py` | 块状盲元诊断（按块大小统计修复前后误差 + **过渡带残影统计** + 最差样例拼图，不用 GPU） |
| `inspect_dataset.py` | 数据集体检：结构/配对/规格/配对质量/块状盲元统计（`--root <目录>`） |
| `make_masks.py` | 生成盲元 mask PNG（\|blur−gt\|>阈值 → 255）：`--blur_dir X --gt_dir Y --out_dir Z [--dilate N]`（膨胀 N 像素把盲元边缘过渡带一并加权） |
| `montage_compare.py` | 并排对比图（原图 \| 各模型修复 \| GT），目视检查用 |
| `blend_blocks.py` | 盲元块无缝融合后处理（低频色差/接缝校正，保持平滑）：`--blur_dir X --gt_dir Y --pred_dir Z --out_dir W` |
| `paixv.py` | 删图后序号重排（frame_0001 顺序；多目录联动同名改；**路径直接改文件顶部 DIRS，DRY_RUN 先预览**） |
| `process_config.yml` | 推理参数（`batch` 按显存选，见「显存与耗时」） |
| `evaluate_nr.py` | 无参考质量评估 |
| `train.sh` | 训练启动 |
| `tests/test_blindpixel_dataset.py` | 数据集回归测试（裁剪/增强对齐 + padding 路径） |
| `Denoising/Options/RealDenosing_BlindPixel_Merged.yml` | 训练配置（24G 卡口径：batch 1 / lr 1e-4） |
| `basicsr/` | 模型引擎 |
| `experiments/RealDenosing_BlindPixel_Merged/` | 当前模型产出（`best_model.pth` / `latest.pth`） |
| `data_merged/` | 训练数据（= `data5/` + `data6/` 的合并副本） |
| `data5/`、`data6/` | 原始数据来源（保留，配置里不再引用） |
| `背景数据_灯管/` | 推理用的 raw 帧 + NUC 标定 `.mat` |
| `results/` | 推理输出 PNG |

> 表中显存数字的实测环境：torch 2.7.1+cu128，fp16 autocast。显存占用只取决于分块面积与 batch，与显卡型号和整图尺寸无关。

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
