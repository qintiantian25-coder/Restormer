@echo off
setlocal EnableExtensions
rem ============================================================
rem  One-click run of test_folder.py  (offline, no Conda needed)
rem  Place   : this file must sit in DEPLOY_ROOT, next to
rem            "env311" and "Restormer".
rem  Input   : drop files into   Restormer\test_in\   (auto-created)
rem  Output  : Restormer\results\<same name>.png
rem  Fixed deployment baseline : tile 640x512 / overlap 128 /
rem            batch 2  (validated on 6GB RTX3060; safe for 8GB)
rem ============================================================

set "ROOT=%~dp0"
set "PY=%ROOT%env311\python.exe"
set "PROJ=%ROOT%Restormer"

echo ========================================
echo  Restormer test_folder.py
echo  root: %ROOT%
echo ========================================
echo.

if not exist "%PY%" (
    echo [ERROR] python.exe not found:
    echo         "%PY%"
    echo         Unpack env311 to "%ROOT%env311" first.
    pause
    exit /b 2
)
if not exist "%PROJ%\test_folder.py" (
    echo [ERROR] test_folder.py not found:
    echo         "%PROJ%\test_folder.py"
    echo         Check layout: DEPLOY_ROOT\Restormer\test_folder.py
    pause
    exit /b 3
)

pushd "%PROJ%"
"%PY%" test_folder.py --batch 2
set "RC=%ERRORLEVEL%"
popd
echo.
echo errorlevel = %RC%
if not "%RC%"=="0" (
    echo [ERROR] test_folder.py failed - see the log above.
) else (
    echo [OK] test_folder.py finished.
    echo      results are in: "%PROJ%\results"
)
echo.
pause
exit /b %RC%
