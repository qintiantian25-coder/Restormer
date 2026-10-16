#Requires -Version 5.1
<#
  build_deploy_project.ps1 —— 构建"西安正式部署版 Restormer"目录
  ============================================================
  作用: 只复制两个正式入口 (test_folder.py / process_raw.py) 真正需要的内容,
        生成一个可随 env311 一起带走的精简 Restormer 目录。
  源  : 本脚本所在 deployment 目录的上一级 (即 Restormer 项目根), 只读不改。
  目标: -Target 参数 (默认 D:\qtt\xian_deploy_test\Restormer)

  复制清单 (已按 import 链路 + 实际运行路径审计, 并用"缩小副本实际跑通
  test_folder.py 与 process_raw.py"验证过):
    test_folder.py / process_raw.py / mask_blend.py
    basicsr\ 整包 (它是动态扫描 *_arch.py 的包, 整包复制最稳; 约 3.6MB)
    process_config.yml
    experiments\RealDenosing_BlindPixel_Merged\models\best_model.pth (仅此一份权重)
    test_in_mask\<test_folder.py 的 MASK_MAT 实际引用的 mask .mat>
    <process_config.yml 的 calib 实际引用的 NUC calibration .mat> (kk/bb, 按原相对目录复制)
        * mask .mat 与 calibration .mat 是两份完全不同的文件, 不要混用;
        * calib 配置了但源文件不存在时, 本脚本默认报错退出 (见 -AllowMissingCalib)。
    test_in\ (空目录, 运行时可放测试图)
    deployment\test_data\ 冒烟数据 (不涉密) + make_smoke_data.py (若存在)

  不复制: .git / __pycache__ / 训练数据 / 历史 results / 日志 / latest.pth /
          其他 checkpoint / warmstart / 论文等无关内容。

  用法 (普通 CMD 或 PowerShell 均可):
    powershell -ExecutionPolicy Bypass -File build_deploy_project.ps1
    powershell -ExecutionPolicy Bypass -File build_deploy_project.ps1 -Force
    powershell -ExecutionPolicy Bypass -File build_deploy_project.ps1 -PythonExe D:\qtt\deploy_test\env311\python.exe

  可选开关:
    -Force              目标目录已存在且非空时, 先清空目标目录再重建 (只动目标目录)
    -SkipVerify         跳过"用 python 导入两个入口"的验证步骤
    -PythonExe          指定验证用 python.exe (默认自动探测常见位置)
    -WithRootScripts    额外把 deployment\ 的 3 个 bat + 2 个说明复制到目标上一级
                        (即 DEPLOY_ROOT, 与 env311 同级; 默认不复制)
    -AllowMissingCalib  人工明确选择"无 calib 部署": calib 配置了但源文件不存在时,
                        不报错退出, 而是警告并跳过 (默认不加此开关时构建失败)
#>
[CmdletBinding()]
param(
    [string]$Target = 'D:\qtt\xian_deploy_test\Restormer',
    [string]$PythonExe = '',
    [switch]$Force,
    [switch]$SkipVerify,
    [switch]$WithRootScripts,
    [switch]$AllowMissingCalib
)

$ErrorActionPreference = 'Stop'

function Info([string]$m) { Write-Host "[INFO] $m" }
function Ok([string]$m)   { Write-Host "[ OK ] $m" -ForegroundColor Green }
function Warn([string]$m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Die([string]$m)  { Write-Host "[FAIL] $m" -ForegroundColor Red; exit 1 }

# ---------- 定位源项目 (相对脚本位置, 不依赖 cwd) ----------
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
foreach ($must in 'test_folder.py', 'process_raw.py', 'basicsr', 'process_config.yml') {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $must))) {
        Die "源项目不完整: 找不到 $RepoRoot\$must"
    }
}
Info "源项目: $RepoRoot"

# ---------- 目标目录安全检查 (禁止指向源项目本身/内部/父级) ----------
$TargetFull = [System.IO.Path]::GetFullPath($Target)
$RepoFull   = [System.IO.Path]::GetFullPath($RepoRoot)
if ($TargetFull -eq $RepoFull -or
    $TargetFull.StartsWith($RepoFull + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
    $RepoFull.StartsWith($TargetFull + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    Die "目标目录不能是源项目本身/内部/父级: $TargetFull"
}
if (Test-Path -LiteralPath $TargetFull) {
    $existing = @(Get-ChildItem -LiteralPath $TargetFull -Recurse -File -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0) {
        if (-not $Force) {
            Die "目标目录已存在且非空: $TargetFull`n       确认重建请加 -Force (只会清空该目标目录, 不动源项目)"
        }
        Info "清空旧目标目录: $TargetFull"
        Remove-Item -LiteralPath $TargetFull -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $TargetFull | Out-Null
Ok "目标目录: $TargetFull"

# ---------- 解析配置引用的资源 (不猜文件名, 从源配置解析) ----------

# 1) mask: test_folder.py 的 MASK_MAT
$maskRel = $null
$tfPath = Join-Path $RepoRoot 'test_folder.py'
$hitM = Select-String -LiteralPath $tfPath -Pattern "^MASK_MAT\s*=\s*'([^']+)'" -Encoding UTF8 | Select-Object -First 1
if ($hitM) { $maskRel = $hitM.Matches[0].Groups[1].Value -replace '/', '\' }
if (-not $maskRel) {
    $maskRel = 'test_in_mask\YD_SW_A_D_12000_4hz_xs.mat'
    Warn "未能从 test_folder.py 解析 MASK_MAT, 按默认名处理: $maskRel"
}
$maskRooted = $false
if ($maskRel) { $maskRooted = [System.IO.Path]::IsPathRooted($maskRel) }
if ($maskRooted) {
    Warn "MASK_MAT 配置为绝对路径, 部署包中无法保证解析, 跳过复制: $maskRel"
}
Info "mask (test_folder.py 的 MASK_MAT 引用): $maskRel"

# 2) NUC calibration: process_config.yml 的 calib (kk/bb), 只在 nuc 流程用到
$calibRel = $null
$cfgPath = Join-Path $RepoRoot 'process_config.yml'
$hitC = Select-String -LiteralPath $cfgPath -Pattern '^\s*calib\s*:\s*(.+?)\s*$' -Encoding UTF8 | Select-Object -First 1
if ($hitC) {
    $v = $hitC.Matches[0].Groups[1].Value.Trim()
    if ($v -match '^"(.*)"$' -or $v -match "^'(.*)'$") { $v = $Matches[1].Trim() }
    elseif ($v.Contains('#')) { $v = ($v -split '#', 2)[0].Trim() }
    if ($v -ne '' -and $v -ne '~' -and $v -ine 'null') { $calibRel = $v -replace '/', '\' }
}
$calibConfigured = [bool]$calibRel
$calibRooted     = $false
if ($calibConfigured) { $calibRooted = [System.IO.Path]::IsPathRooted($calibRel) }
$calibSrc        = $null
$calibSrcExists  = $false
$calibCopied     = $false
$calibSize       = 0
if ($calibConfigured) {
    if ($calibRooted) { $calibSrc = $calibRel } else { $calibSrc = Join-Path $RepoRoot $calibRel }
    $calibSrcExists = Test-Path -LiteralPath $calibSrc -PathType Leaf
    Info "NUC calibration (process_config.yml 的 calib 引用): $calibRel"

    # ---- 缺 calib 的失败判定 (放在复制前, 快速失败) ----
    if ($calibRooted) {
        if (-not $AllowMissingCalib) {
            Die "process_config.yml 的 calib 是绝对路径, 无法随部署包迁移:`n       $calibRel`n       请改为项目根相对路径后重跑; 或显式加 -AllowMissingCalib 表示人工确认不要 NUC"
        }
        Warn "calib 为绝对路径, 已按 -AllowMissingCalib 跳过复制 (部署后该路径需另行保证)"
    } elseif (-not $calibSrcExists) {
        if (-not $AllowMissingCalib) {
            Die "process_config.yml 引用的 NUC 标定文件不存在:`n       $calibSrc`n       正式构建默认失败 (避免静默生成缺 calibration 的部署包)。`n       如确实要做'无 calib 部署', 请显式加 -AllowMissingCalib。"
        }
        Warn "calib 已配置但源文件不存在, 已按 -AllowMissingCalib 跳过: $calibSrc"
    }
} else {
    Info 'process_config.yml 未配置 calib (configured: NO), 部署包不含 NUC 标定'
}

# ---------- 复制函数 ----------
function Copy-One([string]$Rel) {
    $src = Join-Path $RepoRoot $Rel
    if (-not (Test-Path -LiteralPath $src)) { Warn "源文件不存在, 跳过: $Rel"; return }
    $dst = Join-Path $TargetFull $Rel
    $parent = Split-Path -Parent $dst
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath $src -Destination $dst -Force
}
function Copy-Tree([string]$Rel) {
    $src = Join-Path $RepoRoot $Rel
    if (-not (Test-Path -LiteralPath $src)) { Warn "源目录不存在, 跳过: $Rel"; return }
    $dst = Join-Path $TargetFull $Rel
    robocopy "$src" "$dst" /E /XJ /XD __pycache__ .git /XF *.pyc *.pyo | Out-Null
    if ($LASTEXITCODE -ge 8) { Die "robocopy 失败 (exit=$LASTEXITCODE): $src -> $dst" }
    $global:LASTEXITCODE = 0
}

# ---------- 按清单复制 ----------
Info '复制入口与本地模块 (test_folder.py / process_raw.py / mask_blend.py)'
foreach ($f in 'test_folder.py', 'process_raw.py', 'mask_blend.py') { Copy-One $f }
Info '复制 basicsr 整包 (排除 __pycache__)'
Copy-Tree 'basicsr'
Info '复制 process_config.yml'
Copy-One 'process_config.yml'

$weightsRel = 'experiments\RealDenosing_BlindPixel_Merged\models\best_model.pth'
Info "复制权重 (仅 best_model.pth, 不含 latest/warmstart): $weightsRel"
Copy-One $weightsRel

if (-not $maskRooted) {
    Info "复制 mask .mat (区域修复用): $maskRel"
    Copy-One $maskRel
}

if ($calibConfigured -and -not $calibRooted -and $calibSrcExists) {
    Info "复制 NUC calibration .mat (kk/bb, 按原相对目录): $calibRel"
    Copy-One $calibRel
    $calibCopied = $true
    $calibSize = (Get-Item -LiteralPath $calibSrc).Length
    Ok ("NUC calibration 已复制: {0}  ({1:N1} MB)" -f $calibRel, ($calibSize / 1MB))
}

Info '创建空 test_in (运行时可放测试图)'
New-Item -ItemType Directory -Force -Path (Join-Path $TargetFull 'test_in') | Out-Null

if (Test-Path -LiteralPath (Join-Path $RepoRoot 'deployment\test_data')) {
    Info '复制 deployment\test_data 冒烟数据 (不涉密)'
    Copy-Tree 'deployment\test_data'
} else {
    Warn 'deployment\test_data 不存在 (可在现场用 make_smoke_data.py 生成)'
}
Copy-One 'deployment\make_smoke_data.py'

if ($WithRootScripts) {
    Info '复制部署脚本到目标上一级 (DEPLOY_ROOT, 与 env311 同级)'
    $rootDir = Split-Path -Parent $TargetFull
    foreach ($f in 'check_env.bat', 'run_test_folder.bat', 'run_process_raw.bat',
                   'README_西安离线部署.txt', 'DEPLOYMENT_CHECKLIST.txt') {
        $src = Join-Path $RepoRoot (Join-Path 'deployment' $f)
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src -Destination (Join-Path $rootDir $f) -Force
        } else {
            Warn "缺少 deployment\$f"
        }
    }
}

# ---------- 导入验证 (证明缩小副本可运行) ----------
if (-not $SkipVerify) {
    $py = $PythonExe
    if (-not $py) {
        $cands = @(
            'D:\qtt\deploy_test\env311\python.exe',
            'D:\qtt\deploy\env311\python.exe',
            (Join-Path (Split-Path -Parent $RepoFull) 'env311\python.exe')
        )
        foreach ($c in $cands) { if (Test-Path -LiteralPath $c) { $py = $c; break } }
    }
    if (-not $py) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $py = $cmd.Source }
    }
    if ($py -and (Test-Path -LiteralPath $py)) {
        Info "导入验证 (python: $py)"
        $code = "import basicsr, test_folder, process_raw, mask_blend; print('IMPORT_OK'); print('basicsr=' + (getattr(basicsr, '__file__', None) or str(list(basicsr.__path__)[0])))"
        $old = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        Push-Location $TargetFull
        $out = (& $py -c $code 2>&1 | Out-String)
        $rc = $LASTEXITCODE
        Pop-Location
        $ErrorActionPreference = $old
        if ($rc -eq 0 -and $out -match 'IMPORT_OK') {
            Ok '导入验证通过: test_folder / process_raw / mask_blend / basicsr 均可导入'
            $bl = ($out -split "`n" | Where-Object { $_ -match '^basicsr=' }) -join ''
            $bl = $bl -replace '^basicsr=', ''
            if ($bl -and -not $bl.Trim().StartsWith($TargetFull, [System.StringComparison]::OrdinalIgnoreCase)) {
                Warn "注意: basicsr 从 [ $($bl.Trim()) ] 解析, 不是副本目录 (env311 里可能装过 basicsr), 建议核查"
            }
        } else {
            Warn "导入验证失败 (exit=$rc):`n$out"
        }
    } else {
        Warn '未找到可用于验证的 python.exe (可加 -PythonExe 指定, 或手动验证):'
        Warn "    cd /d `"$TargetFull`"  然后  <env311>\python.exe -c `"import test_folder, process_raw`""
    }
}

# ---------- 构建报告 ----------
$files = @(Get-ChildItem -LiteralPath $TargetFull -Recurse -File -ErrorAction SilentlyContinue)
$count = $files.Count
$bytes = ($files | Measure-Object -Property Length -Sum).Sum
if (-not $bytes) { $bytes = 0 }
$weightsOk = Test-Path -LiteralPath (Join-Path $TargetFull $weightsRel) -PathType Leaf
$maskOk    = (-not $maskRooted) -and (Test-Path -LiteralPath (Join-Path $TargetFull $maskRel) -PathType Leaf)
$cfgOk     = Test-Path -LiteralPath (Join-Path $TargetFull 'process_config.yml') -PathType Leaf
$calibOk   = $calibConfigured -and (-not $calibRooted) -and (Test-Path -LiteralPath (Join-Path $TargetFull $calibRel) -PathType Leaf)

Info '扫描目标目录中的开发机绝对路径...'
$scanFiles = @(Get-ChildItem -LiteralPath $TargetFull -Recurse -File -Include *.py, *.yml, *.yaml, *.txt, *.bat, *.ps1, *.md -ErrorAction SilentlyContinue)
$hits = @()
if ($scanFiles.Count -gt 0) {
    foreach ($pat in @('D:\qtt', 'D:/qtt', 'C:\Users\lfh98', 'C:\Users', '/root/Qtt', '/root/')) {
        $found = $scanFiles | Select-String -SimpleMatch -Pattern $pat -Encoding UTF8 -ErrorAction SilentlyContinue
        foreach ($h in $found) {
            $hits += ("{0} : line {1} : {2}" -f $h.Path.Replace($TargetFull + '\', ''), $h.LineNumber, $pat)
        }
    }
}

Write-Host ''
Write-Host '========================================'
Write-Host '  西安部署版 Restormer 构建报告'
Write-Host '========================================'
Write-Host ("  目标目录       : {0}" -f $TargetFull)
Write-Host ("  文件总数       : {0}" -f $count)
Write-Host ("  总大小         : {0:N1} MB ({1:N0} 字节)" -f ($bytes / 1MB), $bytes)
if ($calibOk) { $calibText = '存在' }
elseif ($calibConfigured) { $calibText = '缺失 !!' }
else { $calibText = '未配置' }
$calibPathText = '-'
if ($calibConfigured) { $calibPathText = $calibRel }
$calSrcText = '-'; $calCopiedText = '-'; $calSizeText = '-'
if ($calibConfigured) {
    if ($calibSrcExists) { $calSrcText = 'YES' } else { $calSrcText = 'NO' }
    if ($calibCopied)    { $calCopiedText = 'YES' } else { $calCopiedText = 'NO' }
    if ($calibCopied)    { $calSizeText = ('{0:N1} MB' -f ($calibSize / 1MB)) }
}

Write-Host ("  best_model.pth : {0}" -f $(if ($weightsOk) { '存在' } else { '缺失 !!' }))
Write-Host ("  mask           : {0}  ({1})" -f $(if ($maskOk) { '存在' } else { '缺失 !!' }), $maskRel)
Write-Host ("  process_config : {0}" -f $(if ($cfgOk) { '存在' } else { '缺失 !!' }))
Write-Host ("  calibration    : {0}  ({1})" -f $calibText, $calibPathText)
Write-Host ''
Write-Host '  NUC calibration:'
Write-Host ("    configured    : {0}" -f $(if ($calibConfigured) { 'YES' } else { 'NO' }))
Write-Host ("    source exists : {0}" -f $calSrcText)
Write-Host ("    copied        : {0}" -f $calCopiedText)
Write-Host ("    relative path : {0}" -f $calibPathText)
Write-Host ("    file size     : {0}" -f $calSizeText)
if ($hits.Count -eq 0) {
    Write-Host '  开发机路径残留 : 无'
} else {
    Write-Host ("  开发机路径残留 : {0} 处" -f $hits.Count)
    $hits | Select-Object -First 20 | ForEach-Object { Write-Host ("      " + $_) }
}
Write-Host '========================================'
Write-Host ''
if ($calibConfigured -and -not $calibOk) {
    Warn '构建完成, 但 NUC calibration 缺失 (无 calib 部署; process_raw 将只能走条纹抑制+对比度)'
} else {
    Ok '构建完成 (未修改任何源项目文件; 未执行任何 git 操作)'
}
exit 0
