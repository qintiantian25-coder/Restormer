"""
Upsample GT image to target resolution.

Usage:
    python resize_gt.py --input GT.png --output GT_6000.png --width 6000 --height 6000
"""

import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--width', type=int, default=6000)
parser.add_argument('--height', type=int, default=6000)
args = parser.parse_args()

img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
if img is None:
    raise SystemExit(f'Cannot read {args.input}')

h, w = img.shape[:2]
print(f'Input:  {w}x{h}')

out = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_LANCZOS4)

print(f'Output: {args.width}x{args.height}')
cv2.imwrite(args.output, out)
print(f'Saved: {args.output}')
