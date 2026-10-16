@echo off
setlocal EnableExtensions
rem ============================================================
rem  env311 offline deployment self-check  (check_env.bat)
rem  Purpose : first step on the target machine. Checks python
rem            environment, dependencies and CUDA availability.
rem  Place   : this file must sit in DEPLOY_ROOT, next to
rem            "env311" and "Restormer":
rem              DEPLOY_ROOT\env311\python.exe
rem              DEPLOY_ROOT\Restormer\test_folder.py
rem  Notes   : no Conda required, no conda activate, no PATH use.
rem ============================================================

set "ROOT=%~dp0"
set "PY=%ROOT%env311\python.exe"
set "FAIL=0"

echo ========================================
echo  env311 deployment check
echo  root: %ROOT%
echo ========================================
echo.

if not exist "%PY%" (
    echo [ERROR] python.exe not found:
    echo         "%PY%"
    echo         Unpack env311 to "%ROOT%env311" first.
    goto :fail
)

echo [OK] python.exe found
echo.
echo ---- python version ----
"%PY%" --version
if errorlevel 1 (
    echo [ERROR] python.exe cannot run
    goto :fail
)
echo.

if exist "%ROOT%Restormer\test_folder.py" (
    echo [OK] Restormer\test_folder.py found
) else (
    echo [WARN] Restormer\test_folder.py not found - check folder layout
)
echo.

echo ---- dependency imports ----
for %%M in (torch torchvision cv2 yaml einops scipy h5py skimage tqdm) do (
    "%PY%" -c "import %%M" >nul 2>"%TEMP%\env311_chk_err.txt"
    if errorlevel 1 (
        echo [FAIL] import %%M
        type "%TEMP%\env311_chk_err.txt"
        set "FAIL=1"
    ) else (
        echo [OK]   import %%M
    )
)
echo.

echo ---- torch / CUDA ----
"%PY%" -c "import torch; print('torch              :', torch.__version__); print('torch.version.cuda :', torch.version.cuda); print('cuda available     :', torch.cuda.is_available()); print('device count       :', torch.cuda.device_count()); print('device 0           :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if errorlevel 1 (
    echo [FAIL] torch / CUDA check failed
    set "FAIL=1"
) else (
    "%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if errorlevel 1 (
        echo [ERROR] torch.cuda.is_available() = False
        echo         - run  nvidia-smi  to check the NVIDIA driver
        echo         - check that the GPU is visible to Windows
        set "FAIL=1"
    )
)
echo.

if "%FAIL%"=="1" goto :fail

echo ========================================
echo  ALL CHECKS PASSED
echo ========================================
echo.
pause
exit /b 0

:fail
echo ========================================
echo  *** SOME CHECKS FAILED ***
echo ========================================
echo.
pause
exit /b 1
