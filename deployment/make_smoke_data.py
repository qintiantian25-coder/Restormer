r"""生成离线冒烟测试数据 (不涉密, 纯合成图).

输出 (写到本脚本所在目录的 test_data\ 下):
  smoke_test.png  — 640x512, 平滑背景 + 几个亮/暗点块 (供 test_folder 冒烟)
  smoke_test.raw  — 512x512 uint16 合成 raw (供 raw 链路冒烟)

用法 (任意 cwd, 建议用部署环境的 python 运行):
  DEPLOY_ROOT\env311\python.exe DEPLOY_ROOT\Restormer\deployment\make_smoke_data.py
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / 'test_data'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    h, w = 512, 640
    base = np.linspace(30, 200, w)[None, :].repeat(h, 0)
    img = np.clip(base + rng.normal(0, 3.0, (h, w)), 0, 255).astype(np.uint8)
    img[120:150, 220:280] = 250      # 亮块 (模拟块状盲元)
    img[300:305, 100:106] = 10       # 暗点
    img[400:410, 500:512] = 255      # 亮点
    ok, buf = cv2.imencode('.png', img)      # tofile 写盘, 兼容中文路径
    if not ok:
        raise SystemExit('png 编码失败')
    buf.tofile(str(OUT / 'smoke_test.png'))

    raw = np.clip(np.linspace(5000, 45000, 512)[None, :].repeat(512, 0)
                  + rng.normal(0, 200, (512, 512)), 0, 65535).astype('<u2')
    raw[200:260, 300:360] = 60000
    raw.tofile(str(OUT / 'smoke_test.raw'))

    print('written:', OUT / 'smoke_test.png')
    print('written:', OUT / 'smoke_test.raw')


if __name__ == '__main__':
    main()
