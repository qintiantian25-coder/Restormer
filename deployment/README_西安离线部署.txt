============================================================
 Restormer 红外盲元修复 —— 西安离线部署说明 (env311)
============================================================

【1. 最终目录结构】

把下面几项放到同一个部署根目录 DEPLOY_ROOT（盘符、路径任意，
路径里有空格也可以）:

DEPLOY_ROOT\
  ├─ env311\                    ← 解压 env311_win_x64.tar.gz 得到
  ├─ Restormer\                 ← 整个项目文件夹原样放进来
  ├─ check_env.bat              ← 从 Restormer\deployment\ 复制过来
  ├─ run_test_folder.bat        ← 从 Restormer\deployment\ 复制过来
  ├─ run_process_raw.bat        ← 从 Restormer\deployment\ 复制过来
  └─ README_西安离线部署.txt     ← 从 Restormer\deployment\ 复制过来

注意:
  - 三个 .bat 必须放在 DEPLOY_ROOT（与 env311、Restormer 同级）。
    它们用"自己所在目录"定位 env311 和 Restormer，不依赖 PATH、
    不依赖 Conda，整个目录搬到别的盘也能直接用。
  - mask 模式需要 Restormer\test_in_mask\ 下放好 mask 文件（见 §3）；
    没有这个文件时程序会打印警告并自动关闭 mask 模式，仍可正常出图。

【2. 环境解压】

- 解压 env311_win_x64.tar.gz 到 DEPLOY_ROOT\env311
  （解压后该目录下应直接看到 python.exe）。
  Windows 10 自带 tar，在 DEPLOY_ROOT 下执行:
      tar -xzf env311_win_x64.tar.gz
  或用 7-Zip 连续解压两层（tar.gz -> tar -> 文件夹）。
- 第一次使用前执行一次（双击即可，窗口一闪而过属正常）:
      DEPLOY_ROOT\env311\Scripts\conda-unpack.exe
- 不需要安装 Conda，不需要安装 Python，正常情况下不需要安装
  CUDA Toolkit（torch 自带 CUDA 运行库）。
- 唯一前提: 目标机的 NVIDIA 驱动正常（nvidia-smi 能列出显卡）。

【3. SHA256 校验】

文件   : env311_win_x64.tar.gz
期望值 : 65DFBB80A951BE82EFD503C1D016D315CC6495F0E90712E30915AD86B7FF1FB1

校验方法（PowerShell，在文件所在目录右键 -> 在终端中打开，然后）:
      Get-FileHash .\env311_win_x64.tar.gz -Algorithm SHA256

输出的 Hash 必须与上面完全一致（不区分大小写）。
不一致就不要解压，重新拷贝。

【4. 第一次运行顺序】

  1) 运行 nvidia-smi，确认驱动正常、能看到显卡
  2) 解压 env311（见 §2）
  3) 运行一次 conda-unpack.exe
  4) 双击 check_env.bat，看到 "ALL CHECKS PASSED" 才可以继续
  5) 冒烟测试（不涉密）:
       把 Restormer\deployment\test_data\smoke_test.png
       复制到   Restormer\test_in\
       双击 run_test_folder.bat
     应无报错，Restormer\results\ 下出现同名 png。
     （若 deployment\test_data 缺失: 用 env311\python.exe 运行
       Restormer\deployment\make_smoke_data.py 现场重新生成即可）
  6) 冒烟通过后再跑真实 RAW（用法见 §7 B）。

【5. 已验证环境（本环境打包前实测）】

  Python 3.11.17 / torch 2.5.1+cu121 / torchvision 0.20.1+cu121
  RTX 3060 Laptop 6GB 上实际推理 5/5 成功，无 OOM、无 CUDA 错误。
  参数: 分块 640x512, overlap 128, batch 2, 显存约 3.4GB。
  env311 是 conda-pack 打包，并已用"解压到全新目录 + conda-unpack +
  不激活原 conda + 直接用 python.exe"方式验证过。

【6. 西安目标机】

  Windows 10 / RTX 3070 8GB / NVIDIA Driver 531.29
  环境是按该驱动（CUDA 12.1）的兼容性选择的；但首次到目标机仍必须
  运行 check_env.bat 做最终确认。
  需要随行携带的文件清单见 DEPLOYMENT_CHECKLIST.txt（除代码外至少
  还有: best_model.pth、mask 文件、NUC 标定 .mat(如需 NUC)、测试输入）。

【7. 运行用法】

  A) 测试图修复（png/tif/tiff/bmp/jpg/raw 均可）
     1. 把图放进 Restormer\test_in\（不存在会自动创建）
     2. 双击 run_test_folder.bat
     3. 结果在 Restormer\results\，与输入同名 .png
     * 固定参数: tile 640x512 / overlap 128 / batch 2（8G 卡口径）
     * 改了参数想恢复？不用改，bat 每次都按上面三个值传参。

  B) 真实 RAW 正式修复（6000x6000 uint16）
     run_process_raw.bat RAW_DIR OUTPUT_DIR [标定文件.mat]
     例:
       run_process_raw.bat D:\data\raw D:\data\out
       run_process_raw.bat D:\data\raw D:\data\out D:\data\标定.mat
     * 带第 3 个参数 = NUC + 条纹抑制 + 对比度 + 修复
     * 不带第 3 个参数 = 按 Restormer\process_config.yml 的配置走
       （其配置里若找不到标定文件，则仅条纹抑制 + 对比度）

  C) 调 batch（显存不够时）
     用记事本打开 run_test_folder.bat 或 run_process_raw.bat，
     把里面的 "--batch 2" 改成 "--batch 1"，保存后重跑。
     不要改分块尺寸（会影响拼接痕迹）。

  D) 如果 .bat 被执行策略禁止（组策略限制），手动等价命令:
     cd /d DEPLOY_ROOT\Restormer
     ..\env311\python.exe test_folder.py --batch 2
     ..\env311\python.exe process_raw.py --raw_dir <输入> --output_dir <输出> --weights experiments\RealDenosing_BlindPixel_Merged\models\best_model.pth --tile_w 640 --tile_h 512 --tile_overlap 128 --batch 2

【8. 常见错误与处理】

  1) [ERROR] python.exe not found
     → env311 没解压到 DEPLOY_ROOT\env311；注意解压后不要多套一层
       （不能是 env311\env311\python.exe）；三个 bat 必须与 env311、
       Restormer 放在同一级目录。
  2) import torch 失败
     → 环境解压不完整，或没执行 conda-unpack.exe；
       重跑一次 conda-unpack，仍不行就重新解压再 conda-unpack。
  3) torch.cuda.is_available() = False
     → 先看 nvidia-smi：命令失败 = 驱动问题（见第 4 条）；
       命令正常但这里为 False = 显卡被占用或异常，重启机器后再试。
  4) nvidia-smi 失败
     → NVIDIA 驱动没装好，用离线驱动安装包重装 531.29 驱动。
  5) No .raw files in ...
     → RAW_DIR 里没有 .raw 文件，检查路径和文件扩展名。
  6) weights not found / 权重不存在
     → 确认 Restormer\experiments\RealDenosing_BlindPixel_Merged\
       models\best_model.pth 存在。
  7) calibration .mat 不存在 / No calibration file
     → 程序会退化为"仅条纹抑制"继续跑（不中断）；
       如需 NUC，检查 --calib 路径或 process_config.yml 里的 calib。
  8) CUDA out of memory
     → 按 §7 C 把 bat 里的 --batch 2 改成 --batch 1；
       不要缩小分块。

============================================================
