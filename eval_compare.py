"""
Compare 3 images with no-reference metrics: NIQE, DBCNN, MRD.

Usage:
    python eval_compare.py
"""

import cv2, numpy as np, torch, pyiqa


def compute_mrd(img_clear, img_restored):
    """|clear - restored| / max(restored, 1), lower is better."""
    c = img_clear.astype(np.float64)
    r = img_restored.astype(np.float64)
    return float(np.mean(np.abs(c - r) / np.maximum(r, 1.0)))


def to_tensor(img):
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    return t


EVAL_SIZE = 512  # resize for NR metrics (DBCNN OOM on 6000x6000)

def eval_nr(metric, img_gray):
    """Run no-reference metric on resized image."""
    small = cv2.resize(img_gray, (EVAL_SIZE, EVAL_SIZE))
    t = to_tensor(small).repeat(1, 3, 1, 1).cuda()
    with torch.no_grad():
        return float(metric(t).item())


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  (eval size: {EVAL_SIZE}x{EVAL_SIZE})\n')

    niqe_metric = pyiqa.create_metric('niqe', device=device)
    dbcnn_metric = pyiqa.create_metric('dbcnn', device=device)

    images = {
        'image1 (盲元-blur)':      'image1.png',
        'image2 (传统方法)':        'image2.png',
        'image3 (Restormer合并)': 'image3.png',
    }

    nr_metrics = {
        'NIQE ↓':    pyiqa.create_metric('niqe', device=device),
        'DBCNN ↓':   pyiqa.create_metric('dbcnn', device=device),
        'BRISQUE ↓': pyiqa.create_metric('brisque', device=device),
        'MUSIQ ↑':   pyiqa.create_metric('musiq', device=device),
    }

    results = {}
    for label, path in images.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        r = {}
        for name, metric in nr_metrics.items():
            r[name] = eval_nr(metric, img)
        results[label] = r

    # Print markdown table
    cols = list(nr_metrics.keys())
    header = '| 图像 | ' + ' | '.join(cols) + ' |'
    sep = '|------|' + '|'.join(['--------'] * len(cols)) + '|'
    print(header)
    print(sep)
    for label, r in results.items():
        vals = ' | '.join(f'{r[c]:.4f}' for c in cols)
        print(f'| {label} | {vals} |')

    print()
    print('| 指标 | 类型 | 说明 |')
    print('|------|------|------|')
    print('| NIQE | 无参考 | 基于NSS特征的自然度 |')
    print('| DBCNN | 无参考 | 深度盲卷积网络 |')
    print('| BRISQUE | 无参考 | 空间域自然场景统计 |')
    print('| MUSIQ | 无参考 | 多尺度Transformer |')


if __name__ == '__main__':
    main()
