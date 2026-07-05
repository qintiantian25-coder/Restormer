"""
Evaluate model on test set: PSNR / SSIM per sequence, plus blind pixel stats.

Usage:
    python eval_test.py \
        --weights experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth \
        --blur data5/test_blur --sharp data5/test_sharp

    # also check original blur vs sharp (baseline):
    python eval_test.py --weights none --blur data5/test_blur --sharp data5/test_sharp
"""

import argparse, os, sys
import cv2, numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='experiments/RealDenosing_BlindPixel_Gray_NoMask/models/best_model.pth')
    parser.add_argument('--blur', default='data5/test_blur')
    parser.add_argument('--sharp', default='data5/test_sharp')
    parser.add_argument('--no_model', action='store_true', help='Evaluate blur vs sharp (no model)')
    parser.add_argument('--save', default=None, help='Save restored images to dir')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = None
    if not args.no_model:
        from basicsr.models.archs.restormer_arch import Restormer
        model = Restormer(
            inp_channels=1, out_channels=1, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
        ).to(device)
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['params'])
        model.eval()
        print(f'Loaded: {args.weights}')

    seqs = sorted(os.listdir(args.blur))
    total_psnr, total_ssim, total_frames = 0.0, 0.0, 0

    for seq in seqs:
        blur_dir = os.path.join(args.blur, seq)
        sharp_dir = os.path.join(args.sharp, seq)
        if not os.path.isdir(blur_dir) or not os.path.isdir(sharp_dir):
            continue

        seq_psnr, seq_ssim, n = 0.0, 0.0, 0
        frames = sorted(os.listdir(blur_dir))

        for fname in tqdm(frames, desc=f'  {seq}'):
            if not fname.endswith('.png'):
                continue
            blur = cv2.imread(os.path.join(blur_dir, fname), cv2.IMREAD_GRAYSCALE)
            sharp = cv2.imread(os.path.join(sharp_dir, fname), cv2.IMREAD_GRAYSCALE)
            if blur is None or sharp is None:
                continue

            if model is not None:
                t = torch.from_numpy(blur.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
                h, w = t.shape[2], t.shape[3]
                H = ((h+8)//8)*8; W = ((w+8)//8)*8
                t = F.pad(t, (0, W-w, 0, H-h), 'reflect')
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        out = model(t)
                out = out[:,:,:h,:w].clamp(0,1)
                pred = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
            else:
                pred = blur

            if args.save:
                save_dir = os.path.join(args.save, seq)
                os.makedirs(save_dir, exist_ok=True)
                if model is not None:
                    compare = np.hstack([blur, pred, sharp])  # blur | restored | GT
                else:
                    compare = np.hstack([blur, blur, sharp])  # blur | blur | GT
                cv2.imwrite(os.path.join(save_dir, fname), compare)

            psnr = peak_signal_noise_ratio(sharp, pred, data_range=255)
            ssim = structural_similarity(sharp, pred, data_range=255)
            seq_psnr += psnr; seq_ssim += ssim; n += 1

        avg_p = seq_psnr / n; avg_s = seq_ssim / n
        print(f'  {seq}: PSNR={avg_p:.2f}  SSIM={avg_s:.4f}  ({n} frames)')
        total_psnr += seq_psnr; total_ssim += seq_ssim; total_frames += n

    print(f'\nOverall: PSNR={total_psnr/total_frames:.2f}  SSIM={total_ssim/total_frames:.4f}  ({total_frames} frames)')


if __name__ == '__main__':
    main()
