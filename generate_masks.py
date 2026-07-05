"""
Generate blind-pixel masks from blur images alone (no GT needed).

A blind pixel is detected when its value deviates significantly from the
median of its local neighbourhood.  This does NOT rely on GT quality.

Usage:
    python generate_masks.py --root data5 --split train --threshold 30
"""

import argparse
import os

import cv2
import numpy as np
from scipy.ndimage import median_filter


def detect_blind_pixels(img, threshold, kernel=5):
    """Return binary mask of blind-pixel candidates.

    A pixel is flagged when |pixel - median(local_kxk)| > threshold.
    """
    img = img.astype(np.float32)
    local_median = median_filter(img, size=kernel)
    diff = np.abs(img - local_median)
    return (diff > threshold).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data5')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--threshold', type=int, default=30,
                        help='|pixel - local_median| > threshold → blind pixel')
    parser.add_argument('--kernel', type=int, default=5,
                        help='Neighbourhood size for median filter (odd, >=3)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save side-by-side visual checks for first 5 frames')
    args = parser.parse_args()

    blur_root = os.path.join(args.root, f'{args.split}_blur')
    mask_root = os.path.join(args.root, f'{args.split}_mask')

    total_pixels = 0
    total_blind = 0
    viz_count = 0

    for sub in sorted(os.listdir(blur_root)):
        blur_dir = os.path.join(blur_root, sub)
        if not os.path.isdir(blur_dir):
            continue
        mask_dir = os.path.join(mask_root, sub)
        os.makedirs(mask_dir, exist_ok=True)

        for fname in sorted(os.listdir(blur_dir)):
            if not fname.endswith('.png'):
                continue
            blur_path = os.path.join(blur_dir, fname)
            blur = cv2.imread(blur_path, cv2.IMREAD_GRAYSCALE)
            if blur is None:
                continue

            mask = detect_blind_pixels(blur, args.threshold, args.kernel)

            n_blind = mask.sum() // 255
            total_pixels += mask.size
            total_blind += n_blind

            mask_path = os.path.join(mask_dir, fname)
            cv2.imwrite(mask_path, mask)

            if args.visualize and viz_count < 5:
                viz_count += 1
                # Side-by-side: blur | mask | diff_map
                diff_map = np.abs(blur.astype(np.float32) -
                                  median_filter(blur.astype(np.float32), size=args.kernel))
                diff_map = (np.clip(diff_map / args.threshold, 0, 3) * 85).astype(np.uint8)
                viz = np.hstack([blur, mask, diff_map])
                viz_path = os.path.join(mask_dir, f'__CHECK_{viz_count}_{fname}')
                cv2.imwrite(viz_path, viz)
                print(f'  Visual check: {viz_path}')

    pct = total_blind / max(total_pixels, 1) * 100
    print(f'{args.split}: {total_blind} blind px / {total_pixels} = {pct:.3f}%')
    print(f'Masks saved to {mask_root}/')


if __name__ == '__main__':
    main()
