# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Restormer (CVPR 2022 Oral) is a Vision Transformer for high-resolution image restoration — deraining, motion deblurring, defocus deblurring, and Gaussian/real denoising. Built on the BasicSR toolbox. The core engine lives in `basicsr/`; each task has its own directory with configs, data download scripts, and evaluation code.

**This fork's primary purpose is infrared blind-pixel (dead-pixel) repair** (`红外盲元修复`): a single-GPU, grayscale Restormer trained on paired 640×512 grayscale images (blur = frames with blind pixels, sharp = clean GT), plus a raw-format inference pipeline and no-reference evaluation. Target hardware: **24 GB for training, 8 GB for inference** — both configs are sized for that (see the blind-pixel workflow section). The authoritative workflow doc is `BLIND_PIXEL_README.md` (Chinese). The upstream task directories (Deraining, Motion_Deblurring, Defocus_Deblurring, Denoising) remain but are not the active work.

## Commands

```bash
# Install (dev mode, no CUDA extensions)
python setup.py develop --no_cuda_ext
# deps: pip install torch opencv-python numpy scipy h5py pyyaml

# Train blind-pixel model (single GPU; auto-resumes from latest checkpoint)
./train.sh Denoising/Options/RealDenosing_BlindPixel_Merged.yml   # runs: python basicsr/train.py -opt <yml> --launcher none

# Raw → repaired PNG inference (6000×6000 uint16 .raw)
python process_raw.py --raw_dir <in> --output_dir <out> --weights experiments/RealDenosing_BlindPixel_Merged/models/best_model.pth
python process_raw.py --mode nuc --calib <标定.mat> ...   # NUC + stripe suppression + contrast before the network
# Or edit paths in process_config.yml and run `python process_raw.py`

# No-reference quality evaluation (residual_X, localstd, roughness, NU, estsnr)
python evaluate_nr.py --output <修复图> --input <原图> --save <结果> --thresholds 5 10 20 30 50

# Upstream demo inference
python demo.py --task Single_Image_Defocus_Deblurring --input_dir ./demo/degraded/ --result_dir ./demo/restored/
```

Task values for `--task`: `Motion_Deblurring`, `Single_Image_Defocus_Deblurring`, `Deraining`, `Real_Denoising`, `Gaussian_Gray_Denoising`, `Gaussian_Color_Denoising`.

There are no unit tests. Testing is done by running task-specific test scripts (e.g., `cd Denoising && python test_real_denoising_sidd.py`).

## Architecture

### Model registration

Both architectures and models use **dynamic discovery** rather than explicit registries. `basicsr/models/archs/__init__.py` scans for `*_arch.py` files and imports all classes; `basicsr/models/__init__.py` does the same for `*_model.py` files. To wire up a new architecture, just drop a file in the right directory — no manual registration needed.

### Restormer model (`basicsr/models/archs/restormer_arch.py`)

A UNet-style encoder-decoder Transformer with skip connections:

- **Patch embed**: 3×3 Conv (no stride) → `dim=48`
- **Encoder**: 4 levels (4, 6, 6, 8 TransformerBlocks), downsampling between levels via Conv3×3 + PixelUnshuffle(2)
- **Bottleneck**: 8 TransformerBlocks at dim=384
- **Decoder**: Mirrors encoder (6, 6, 4 blocks), upsampling via Conv3×3 + PixelShuffle(2), concat skip connections
- **Refinement**: 4 TransformerBlocks + Conv3×3 output with residual connection to input

Core building blocks:
- **MDTA** (Multi-DConv Head Transposed Attention): depthwise-conv-augmented QKV projection, cross-channel attention (channels as sequence dimension instead of spatial)
- **GDFN** (Gated-Dconv Feed-Forward): depthwise conv + gated linear unit (GELU(x1) * x2)
- **TransformerBlock**: pre-norm MDTA + pre-norm GDFN
- **LayerNorm**: operates on channel dimension; two variants (`BiasFree` and `WithBias`)

### Training loop (`basicsr/train.py`)

Uses **progressive training**: starts with small patches (128×128) and large batch sizes, then over 300k iterations increases patch size to 384×384 while reducing batch size. Transition points are defined in the YAML config under `datasets.train` as `mini_batch_sizes`, `iters`, and `gt_sizes`.

The training loop dynamically rebuilds the dataloader at each transition boundary to pick up the new `gt_size` and `mini_batch_size`.

The blind-pixel config uses a single stage (`mini_batch_sizes: [4]`, `gt_sizes: [384]`, `iters: [180000]`) rather than progressive growth.

### Data pipeline

- **Dataset_PairedImage** (`basicsr/data/paired_image_dataset.py`): main paired LQ/GT dataset class. Supports `disk`, `lmdb`, and `meta_info_file` backends.
- **Dataset_PairedImage_BlindPixel** (same file, line ~137): the fork's custom dataset for the blind-pixel task — grayscale (`in_ch: 1`), 640×512 PNG pairs with matching filenames, recursive subdirectory scanning. Registered by name in the YAML as `type: Dataset_PairedImage_BlindPixel`.
- **Dataset_GaussianDenoising**: adds Gaussian noise on-the-fly during training; `sigma_range` controls noise level distribution.
- **Dataset_DefocusDeblur_DualPixel_16bit**: loads left+right views concatenated into 6-channel input.
- **Prefetch**: `CPUPrefetcher` and `CUDAPrefetcher` overlap data loading with GPU computation.

### YAML config structure

Every YAML has these sections:
- `network_g`: Restormer hyperparams (dim, num_blocks, heads, ffn_expansion_factor, LayerNorm_type, inp_channels/out_channels)
- `datasets.train/val`: dataset type, paths, geometric_augs, progressive training schedule
- `train`: total_iter (300k), scheduler (CosineAnnealingRestartCyclicLR), optimizer (AdamW, lr=3e-4, weight_decay=1e-4), MixUp augment toggle, L1Loss
- `val`: val_freq (4e3), metrics (PSNR, sometimes SSIM)
- `path`: pretrained weights and resume checkpoint locations

### Model class hierarchy

`BaseModel` (`basicsr/models/base_model.py`) → `ImageCleanModel` (`basicsr/models/image_restoration_model.py`). `ImageCleanModel` handles the full train/val/test loop: `optimize_parameters` runs forward pass + loss + backward + optimizer step; `test` runs tiled inference with reflect-padding to multiples of 8.

## Blind-pixel repair workflow

This is the active work in this fork.

### Training (`Denoising/Options/RealDenosing_BlindPixel_Merged.yml`)

- Grayscale (`inp_channels: 1`), single GPU, fp16, 180k iterations, L1 loss, val every 4k iters with PSNR/SSIM (computed on grayscale directly, `test_y_channel: false`). The config is tuned for a **24 GB training GPU**: `mini_batch_sizes: [1]` + `batch_size_per_gpu: 1` + `lr: 1e-4` (measured 13 GB/step at patch 384). On a 64 GB+ card set both batch fields to 4 and `lr` back to 3e-4 (measured 51 GB/step).
- **Training VRAM**: ≈ 90 KB × patch area(px) × batch, independent of patch shape (measured at 384²/320²/256²/512²/512×640). Budget rule for 24 GB: patch area × batch ≤ ~250k px. End-to-end verified on the real trainer (train + val + checkpoint save): 12.96 GB allocated / 16.6 GB reserved.
- **Do not re-enable `torch.backends.cudnn.benchmark`** in `basicsr/train.py` (it is deliberately `False`): with it on, cuDNN's autotuner asked for a **54.9 GB workspace** the first time it saw the 640×512 validation conv shape, which OOMs a 24 GB card. With it off the same run peaks at 12.96 GB.
- **Data layout**: `data_merged/` is what the config points at (train 1090 / val 418 / test 455 pairs); it is a merge copy of `data5/` (2094) + `data6/` (1832). Each split has `train_blur/`, `train_sharp/`, ... with numbered subdirectories (`001/`, ...) holding 640×512 grayscale PNGs; blur/sharp names correspond one-to-one. Note the `val:` section currently points at `data_merged/test_*` (test set used as validation).
- Training patches are **random square `gt_size` crops** (384×384) taken from the 640×512 slices — the model never sees a 640×512 or full-frame input during training.
- To point at new data, edit `dataroot_gt`/`dataroot_lq` for train and val.
- **Resume**: rerunning `./train.sh` auto-resumes from the latest state — model, optimizer, and scheduler. To extend training, raise `iters` and `train.total_iter` together and append matching entries to `scheduler.periods`/`restart_weights`/`eta_mins` (their sum must equal `total_iter`). Changing `lr` in the YAML does take effect on resume (the scheduler recomputes lr from the YAML value every iteration).
- **Checkpoints** in `experiments/<name>/`: `models/best_model.pth` (best weights, PSNR-gated) and `models/latest.pth` (full resume state). `net_g_latest.pth` / `training_states/-1.state` are legacy artifacts from older code — no longer written and unused on resume.

### Inference (`process_raw.py` + `process_config.yml`)

Raw 6000×6000 uint16 `.raw` frames → repaired PNG. Pipeline modes (set in `process_config.yml`):
- `contrast`: quantile contrast stretch (0.001/0.999, ^0.6 gamma) → Restormer
- `nuc` with calib `.mat`: NUC (two-point, K/B) → stripe suppression (polynomial fit, `stripe_degree`) → contrast → Restormer
- `nuc` without calib: stripe suppression only, then contrast

Tiled inference: `tile_w`/`tile_h` (default 640×512) with `tile_overlap` 128, batched. **Inference VRAM depends only on tile area × batch — not on the frame size** (the full-image input/accumulate buffers are a constant ~0.4 GB), so an 8 GB GPU can still do 6000×6000. Measured (fp16 autocast): `peak ≈ 0.7 GB + 1.42 GB × batch` at 640×512 tiles → batch 2 = 3.4 GB (8 GB card, current default), batch 8 = 12.2 GB (24 GB), batch 24 = 35.3 GB (96 GB). Verified end-to-end on a real 6000×6000 raw frame at batch 2: 3.5 GB, ~30 s/frame (contended GPU). `batch` changes speed only; **tile side must be ≥ the 384 training patch**, and 640×512 is verified seam-free. ~12 s per 6000×6000 frame at batch 24 on an idle card.

### Evaluation (`evaluate_nr.py`)

No-reference metrics for grayscale infrared images (no GT needed): `residual_X` (% pixels deviating > X from 5×5 median), `localstd` (5×5 local std mean), `roughness` (normalized Laplacian high-pass energy), `NU` (non-uniformity), `estsnr` (Immerkaer estimated SNR). Lower is better except estsnr. A related metric module lives at `basicsr/metrics/blind_pixel.py`.

## Key dependencies

PyTorch 1.8.1, einops (rearrange), OpenCV, scikit-image, lpips, tb-nightly (TensorBoard), h5py (for .mat calib files). The `setup.py` defines optional CUDA extensions (deform_conv, fused_act, upfirdn2d) that are skipped with `--no_cuda_ext`.
