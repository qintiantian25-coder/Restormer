# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Restormer (CVPR 2022 Oral) is a Vision Transformer for high-resolution image restoration — deraining, motion deblurring, defocus deblurring, and Gaussian/real denoising. Built on the BasicSR toolbox. The core engine lives in `basicsr/`; each task has its own directory with configs, data download scripts, and evaluation code.

## Commands

```bash
# Install (dev mode, no CUDA extensions)
python setup.py develop --no_cuda_ext

# Train (8 GPUs, distributed)
./train.sh <path_to_yml>    # e.g. ./train.sh Denoising/Options/RealDenoising_Restormer.yml

# Demo inference
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

### Data pipeline

- **Dataset_PairedImage** (`basicsr/data/paired_image_dataset.py`): main paired LQ/GT dataset class. Supports `disk`, `lmdb`, and `meta_info_file` backends.
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

### Key dependencies

PyTorch 1.8.1, einops (rearrange), OpenCV, scikit-image, lpips, tb-nightly (TensorBoard). The `setup.py` defines optional CUDA extensions (deform_conv, fused_act, upfirdn2d) that are skipped with `--no_cuda_ext`.
