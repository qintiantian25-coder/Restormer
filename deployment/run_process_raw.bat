@echo off
setlocal EnableExtensions
rem ============================================================
rem  One-click run of process_raw.py  (raw frames -> repaired
rem  PNG, offline, no Conda needed)
rem  Place : this file must sit in DEPLOY_ROOT, next to
rem          "env311" and "Restormer".
rem
rem  Usage : run_process_raw.bat RAW_DIR OUTPUT_DIR [CALIB_FILE]
rem  Examples:
rem    run_process_raw.bat D:\data\raw D:\data\out
rem    run_process_raw.bat D:\data\raw D:\data\out D:\data\calib.mat
rem
rem  With CALIB_FILE    -> nuc mode: NUC + stripe + contrast
rem  Without CALIB_FILE -> follows process_config.yml settings
rem  Fixed deployment baseline: tile 640x512 / overlap 128 /
rem                            batch 2  (8GB-GPU pin)
rem ============================================================

set "ROOT=%~dp0"
set "PY=%ROOT%env311\python.exe"
set "PROJ=%ROOT%Restormer"
set "WEIGHTS=%PROJ%\experiments\RealDenosing_BlindPixel_Merged\models\best_model.pth"

echo ========================================
echo  Restormer process_raw.py
echo  root: %ROOT%
echo ========================================
echo.

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage
set "RAW_DIR=%~1"
set "OUT_DIR=%~2"
set "CALIB=%~3"

if not exist "%PY%" (
    echo [ERROR] python.exe not found: "%PY%"
    pause
    exit /b 2
)
if not exist "%PROJ%\process_raw.py" (
    echo [ERROR] process_raw.py not found: "%PROJ%\process_raw.py"
    pause
    exit /b 3
)
if not exist "%WEIGHTS%" (
    echo [ERROR] weights not found:
    echo         "%WEIGHTS%"
    pause
    exit /b 4
)
if not exist "%RAW_DIR%" (
    echo [ERROR] RAW_DIR not found: "%RAW_DIR%"
    pause
    exit /b 5
)

echo RAW_DIR : "%RAW_DIR%"
echo OUT_DIR : "%OUT_DIR%"
if not "%CALIB%"=="" (
    if not exist "%CALIB%" (
        echo [ERROR] calibration .mat not found: "%CALIB%"
        pause
        exit /b 6
    )
    echo CALIB   : "%CALIB%"
) else (
    echo CALIB   : (none) - follows process_config.yml
)
echo.

pushd "%PROJ%"
if not "%CALIB%"=="" (
    "%PY%" process_raw.py --raw_dir "%RAW_DIR%" --output_dir "%OUT_DIR%" --weights "%WEIGHTS%" --mode nuc --calib "%CALIB%" --tile_w 640 --tile_h 512 --tile_overlap 128 --batch 2
) else (
    "%PY%" process_raw.py --raw_dir "%RAW_DIR%" --output_dir "%OUT_DIR%" --weights "%WEIGHTS%" --tile_w 640 --tile_h 512 --tile_overlap 128 --batch 2
)
set "RC=%ERRORLEVEL%"
popd
echo.
echo errorlevel = %RC%
if not "%RC%"=="0" (
    echo [ERROR] process_raw.py failed - see the log above.
) else (
    echo [OK] done. PNG files are in: "%OUT_DIR%"
)
echo.
pause
exit /b %RC%

:usage
echo Usage: %~nx0 RAW_DIR OUTPUT_DIR [CALIB_FILE]
echo.
echo   RAW_DIR     : folder with .raw files (uint16, e.g. 6000x6000)
echo   OUTPUT_DIR  : folder for repaired PNG (created automatically)
echo   CALIB_FILE  : optional NUC calibration .mat (enables nuc mode)
echo.
pause
exit /b 1
