#!/usr/bin/env bash
# ============================================================
#  Restormer 红外盲元修复 — 依赖一键安装脚本 (清华镜像源)
#
#  用法:
#    ./install_deps.sh            # 直接联网安装 (清华镜像源)
#    ./install_deps.sh bundle     # 把所有依赖的 wheel 打包到 packages/ 文件夹
#    ./install_deps.sh offline    # 从已打包的 packages/ 文件夹离线安装
# ============================================================

set -e
cd "$(dirname "$0")"

MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
REQ="requirements.txt"
PKG_DIR="packages"

case "${1:-install}" in
    bundle)
        echo "==> 下载全部依赖到 $PKG_DIR/ 文件夹 ..."
        mkdir -p "$PKG_DIR"
        pip download -r "$REQ" -d "$PKG_DIR" -i "$MIRROR"
        echo "==> 完成: $PKG_DIR/ 已包含全部依赖, 可复制到离线机器"
        ;;
    offline)
        echo "==> 从 $PKG_DIR/ 文件夹离线安装 ..."
        pip install --no-index --find-links="$PKG_DIR" -r "$REQ"
        echo "==> 完成"
        ;;
    *)
        echo "==> 使用清华镜像源联网安装 ..."
        pip install -r "$REQ" -i "$MIRROR"
        echo "==> 完成"
        ;;
esac

echo ""
echo "下一步(可选): python setup.py develop --no_cuda_ext"
