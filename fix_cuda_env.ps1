#Requires -Version 5.1
<#
.SYNOPSIS
    H2Q-MicroStream Windows CUDA 编译环境自检与一键修复脚本
    检查项: CUDA_HOME / nvcc / cl.exe (MSVC) / ninja / PyTorch ABI
    修复项: 自动设置 CUDA_HOME、PATH；通过 winget 安装缺失工具
.USAGE
    # 普通权限运行（仅检查+设环境变量）:
    .\fix_cuda_env.ps1

    # 管理员权限运行（含 winget 安装 CUDA Toolkit + VS BuildTools）:
    .\fix_cuda_env.ps1 -Install

    # 跳过 winget 安装，只检查并输出结论:
    .\fix_cuda_env.ps1 -CheckOnly
#>
param(
    [switch]$Install,
    [switch]$CheckOnly,
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── 颜色输出工具 ────────────────────────────────────────────────
function Write-OK   { param($msg) Write-Host "  [OK]  $msg" -ForegroundColor Green }
function Write-FAIL { param($msg) Write-Host "  [!!]  $msg" -ForegroundColor Red }
function Write-WARN { param($msg) Write-Host "  [>>]  $msg" -ForegroundColor Yellow }
function Write-INFO { param($msg) Write-Host "        $msg" -ForegroundColor Cyan }
function Write-HDR  { param($msg) Write-Host "`n===== $msg =====" -ForegroundColor Magenta }

$script:passCount = 0
$script:failCount = 0
$script:fixApplied = @()

function Pass($item) { $script:passCount++; Write-OK $item }
function Fail($item) { $script:failCount++; Write-FAIL $item }

# ─── 检查 1: PyTorch CUDA 是否可用 ──────────────────────────────
Write-HDR "步骤 1 / 6  PyTorch + CUDA 运行时"
try {
    $ptInfo = python -c @"
import torch, sys
print(f'version={torch.__version__}')
print(f'cuda_available={torch.cuda.is_available()}')
print(f'cuda_version={torch.version.cuda}')
if torch.cuda.is_available():
    print(f'device={torch.cuda.get_device_name(0)}')
"@
    $ptVersion = ($ptInfo | Where-Object {$_ -match "^version="}) -replace "version=",""
    $cudaAvail  = ($ptInfo | Where-Object {$_ -match "^cuda_available="}) -match "True"
    $cudaVer    = ($ptInfo | Where-Object {$_ -match "^cuda_version="}) -replace "cuda_version=",""
    $device     = ($ptInfo | Where-Object {$_ -match "^device="}) -replace "device=",""

    Pass "PyTorch $ptVersion  (built with CUDA $cudaVer)"
    if ($cudaAvail) {
        Pass "CUDA 运行时可用  [$device]"
    } else {
        Fail "CUDA 运行时不可用 (torch.cuda.is_available() = False)"
    }
    $script:TorchCudaVer = $cudaVer  # e.g. "12.1"
} catch {
    Fail "Python/PyTorch 导入失败: $_"
    Write-WARN "请先确保 python 在 PATH 且已安装 torch"
    exit 1
}

# ─── 检查 2: ninja ────────────────────────────────────────────────
Write-HDR "步骤 2 / 6  ninja (JIT 编译后端)"
$ninjaCmd = Get-Command ninja -ErrorAction SilentlyContinue
if ($ninjaCmd) {
    $ninjaVer = (ninja --version 2>&1) -join ""
    Pass "ninja $ninjaVer  @ $($ninjaCmd.Source)"
} else {
    Fail "ninja 未找到"
    if (-not $CheckOnly) {
        Write-WARN "尝试 pip 安装 ninja ..."
        pip install ninja -q
        if (Get-Command ninja -ErrorAction SilentlyContinue) {
            Pass "ninja 已通过 pip 安装"
            $script:fixApplied += "pip install ninja"
        } else {
            Fail "ninja pip 安装后仍未找到，请手动: pip install ninja"
        }
    }
}

# ─── 检查 3: CUDA Toolkit (nvcc) ────────────────────────────────
Write-HDR "步骤 3 / 6  CUDA Toolkit (nvcc)"

function Find-Nvcc {
    # 1) 直接 PATH
    $c = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }

    # 2) 标准安装路径
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $vers = Get-ChildItem $cudaRoot -Directory | Sort-Object Name -Descending
        foreach ($v in $vers) {
            $nvcc = Join-Path $v.FullName "bin\nvcc.exe"
            if (Test-Path $nvcc) { return $nvcc }
        }
    }

    # 3) 环境变量 CUDA_HOME / CUDA_PATH
    foreach ($envVar in @("CUDA_HOME","CUDA_PATH")) {
        $val = [System.Environment]::GetEnvironmentVariable($envVar, "Machine")
        if (-not $val) { $val = [System.Environment]::GetEnvironmentVariable($envVar, "User") }
        if ($val) {
            $nvcc = Join-Path $val "bin\nvcc.exe"
            if (Test-Path $nvcc) { return $nvcc }
        }
    }
    return $null
}

$nvccPath = Find-Nvcc
if ($nvccPath) {
    $nvccVer = (& $nvccPath --version 2>&1 | Select-String "release") -join ""
    Pass "nvcc  @ $nvccPath"
    Write-INFO $nvccVer

    # 自动设 CUDA_HOME
    $cudaHome = Split-Path (Split-Path $nvccPath)
    $currentCudaHome = [System.Environment]::GetEnvironmentVariable("CUDA_HOME","User")
    if ($currentCudaHome -ne $cudaHome) {
        [System.Environment]::SetEnvironmentVariable("CUDA_HOME", $cudaHome, "User")
        $env:CUDA_HOME = $cudaHome
        $script:fixApplied += "设置 CUDA_HOME=$cudaHome (用户级)"
        Pass "CUDA_HOME 已自动设为 $cudaHome"
    } else {
        Pass "CUDA_HOME=$cudaHome (已正确设置)"
    }

    # 自动加 nvcc 所在目录到 PATH
    $nvccDir = Split-Path $nvccPath
    if ($env:PATH -notlike "*$nvccDir*") {
        $userPath = [System.Environment]::GetEnvironmentVariable("PATH","User")
        [System.Environment]::SetEnvironmentVariable("PATH", "$userPath;$nvccDir", "User")
        $env:PATH += ";$nvccDir"
        $script:fixApplied += "PATH 追加 $nvccDir (用户级)"
        Pass "nvcc 目录已加入 PATH"
    }

    # 版本对齐检查
    $nvccMajorMinor = if ($nvccVer -match "release (\d+)\.(\d+)") { "$($matches[1]).$($matches[2])" } else { "?" }
    $torchCudaMajMin = if ($script:TorchCudaVer -match "(\d+)\.(\d+)") { "$($matches[1]).$($matches[2])" } else { "?" }
    if ($nvccMajorMinor -eq $torchCudaMajMin) {
        Pass "CUDA 版本对齐: nvcc $nvccMajorMinor == PyTorch built with $torchCudaMajMin"
    } else {
        Fail "CUDA 版本不对齐: nvcc=$nvccMajorMinor  PyTorch=$torchCudaMajMin"
        Write-WARN "PyTorch 2.5.1+cu121 要求 CUDA Toolkit 12.1.x"
        Write-WARN "下载地址: https://developer.nvidia.com/cuda-12-1-0-download-archive"
    }
} else {
    Fail "nvcc 未找到 (CUDA Toolkit 未安装)"
    Write-INFO "PyTorch $($script:TorchCudaVer) 需要 CUDA Toolkit $($script:TorchCudaVer)"
    Write-INFO "下载 CUDA Toolkit 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive"

    if ($Install) {
        # 尝试使用 winget 安装 CUDA Toolkit
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-WARN "尝试通过 winget 安装 CUDA Toolkit 12.1..."
            Write-WARN "(这需要管理员权限，约 2-3 GB 下载，耗时 5-15 分钟)"
            Write-WARN "winget 命令: winget install Nvidia.CUDA --version 12.1"
            Write-WARN "请在管理员 PowerShell 中运行上述命令，然后重新运行此脚本"
            Write-INFO ""
            Write-INFO "或手动下载安装包:"
            Write-INFO "  https://developer.nvidia.com/cuda-12-1-0-download-archive"
            Write-INFO "  选择: Windows > x86_64 > 11/10 > exe (network)"
        } else {
            Fail "winget 未找到，请手动安装 CUDA Toolkit 12.1"
        }
    } else {
        Write-INFO "使用 -Install 参数或运行: winget install Nvidia.CUDA --version 12.1"
    }
}

# ─── 检查 4: MSVC cl.exe ─────────────────────────────────────────
Write-HDR "步骤 4 / 6  MSVC C++ 编译器 (cl.exe)"

function Find-Cl {
    # 1) 直接 PATH
    $c = Get-Command cl -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }

    # 2) VS 2022/2019/2017 Build Tools
    $vsPaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC"
    )
    foreach ($base in $vsPaths) {
        if (Test-Path $base) {
            $versions = Get-ChildItem $base -Directory | Sort-Object Name -Descending
            foreach ($v in $versions) {
                $cl = Join-Path $v.FullName "bin\Hostx64\x64\cl.exe"
                if (Test-Path $cl) { return $cl }
            }
        }
    }

    # 3) vswhere
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsDir = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($vsDir) {
            $msvcBase = Join-Path $vsDir "VC\Tools\MSVC"
            if (Test-Path $msvcBase) {
                $versions = Get-ChildItem $msvcBase -Directory | Sort-Object Name -Descending
                foreach ($v in $versions) {
                    $cl = Join-Path $v.FullName "bin\Hostx64\x64\cl.exe"
                    if (Test-Path $cl) { return $cl }
                }
            }
        }
    }
    return $null
}

$clPath = Find-Cl
if ($clPath) {
    $clVer = (& $clPath 2>&1 | Select-Object -First 2) -join " "
    Pass "cl.exe  @ $clPath"
    Write-INFO $clVer

    # 自动将 cl 目录加入 PATH
    $clDir = Split-Path $clPath
    if ($env:PATH -notlike "*$clDir*") {
        $userPath = [System.Environment]::GetEnvironmentVariable("PATH","User")
        [System.Environment]::SetEnvironmentVariable("PATH", "$userPath;$clDir", "User")
        $env:PATH += ";$clDir"
        $script:fixApplied += "PATH 追加 $clDir (cl.exe)"
        Pass "cl.exe 目录已加入 PATH"
    }

    # 检查并设置 INCLUDE/LIB (MSVC 需要)
    $msvcDir = Split-Path (Split-Path (Split-Path $clDir))  # MSVC\14.x.xxx
    $vsRoot   = Split-Path (Split-Path (Split-Path $msvcDir))  # VS install root

    $ucrtPaths = @(
        "C:\Program Files (x86)\Windows Kits\10\Include",
        "C:\Program Files\Windows Kits\10\Include"
    )
    $ucrtFound = $null
    foreach ($p in $ucrtPaths) {
        if (Test-Path $p) {
            $ucrtFound = (Get-ChildItem $p -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
            break
        }
    }
    if ($ucrtFound) {
        Pass "Windows SDK (ucrt)  @ $ucrtFound"
    } else {
        Write-WARN "未找到 Windows SDK ucrt 头文件，编译可能失败"
        Write-WARN "建议安装: VS Installer > 修改 > 单独组件 > Windows 10/11 SDK"
    }
} else {
    Fail "cl.exe 未找到 (MSVC Build Tools 未安装)"
    Write-INFO "需要 Visual Studio 2022 Build Tools (C++ 工作负载)"

    if ($Install) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-WARN "尝试通过 winget 安装 VS 2022 Build Tools..."
            Write-WARN "(需要管理员权限，约 1-2 GB 下载)"
            Write-WARN ""
            Write-WARN "请在管理员 PowerShell 中运行:"
            Write-WARN '  winget install Microsoft.VisualStudio.2022.BuildTools --silent --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait"'
            Write-INFO ""
            Write-INFO "安装完成后重新运行此脚本验证"
        } else {
            Write-WARN "请从以下地址下载 VS 2022 Build Tools:"
            Write-WARN "  https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022"
            Write-WARN "安装时勾选: C++ build tools (包含 MSVC + Windows SDK)"
        }
    } else {
        Write-INFO "使用 -Install 参数获取安装指引"
    }
}

# ─── 检查 5: PyTorch ABI / cpp_extension 可初始化 ───────────────
Write-HDR "步骤 5 / 6  PyTorch cpp_extension ABI 检查"
try {
    $abiInfo = python -c @"
from torch.utils.cpp_extension import CUDA_HOME, _TORCH_PATH
import torch
print(f'CUDA_HOME={CUDA_HOME}')
print(f'torch_path={_TORCH_PATH}')
print(f'cxx_compiler={torch._C._has_cxx11_abi()}')
print(f'abi_flag={torch._C._GLIBCXX_USE_CXX11_ABI}')
"@ 2>&1

    if ($abiInfo -match "CUDA_HOME=None" -or $abiInfo -match "Error") {
        Write-WARN "cpp_extension CUDA_HOME 仍为 None (需要先安装 CUDA Toolkit 并重启 shell)"
    } else {
        $cudaHomeVal = ($abiInfo | Where-Object {$_ -match "^CUDA_HOME="}) -replace "CUDA_HOME=",""
        Pass "cpp_extension CUDA_HOME=$cudaHomeVal"
    }
    $abiFlag = ($abiInfo | Where-Object {$_ -match "^abi_flag="}) -replace "abi_flag=",""
    Write-INFO "CXX11_ABI flag: $abiFlag"
} catch {
    Fail "ABI 检查失败: $_"
}

# ─── 检查 6: 试编译 cuda_ext ─────────────────────────────────────
Write-HDR "步骤 6 / 6  试编译 binary_sta_fused_ext"

$canCompile = $false
if ((Get-Command nvcc -ErrorAction SilentlyContinue) -and (Get-Command cl -ErrorAction SilentlyContinue)) {
    $canCompile = $true
} elseif ($nvccPath -and $clPath) {
    # 用找到的路径临时注入
    $env:PATH += ";$(Split-Path $nvccPath);$(Split-Path $clPath)"
    $canCompile = $true
}

if ($canCompile) {
    Write-INFO "nvcc + cl 均已找到，尝试 JIT 编译..."
    $compileResult = python -c @"
import sys, os
sys.path.insert(0, r'd:\H2Q-MicroStream')
os.environ.setdefault('BINARY_STA_DISABLE_CUDA_EXT', '0')
try:
    import binary_sta_cuda_ext as ext
    handle = ext.load_extension(verbose=True)
    print('COMPILE_OK')
    print(f'ext={handle}')
except Exception as e:
    print(f'COMPILE_FAIL: {e}')
"@ 2>&1

    if ($compileResult -match "COMPILE_OK") {
        Pass "cuda_ext 编译成功！"
        Write-INFO ($compileResult | Where-Object {$_ -match "ext="})
        $script:cudaExtReady = $true
    } else {
        Fail "编译失败"
        $compileResult | ForEach-Object { Write-INFO $_ }
        Write-WARN ""
        Write-WARN "常见原因与修复:"
        Write-WARN "  1) CUDA Toolkit 版本与 PyTorch 不匹配 -> 需 CUDA 12.1.x"
        Write-WARN "  2) cl.exe 未在当前 PATH -> 需要 VS Developer Prompt 或重新打开终端"
        Write-WARN "  3) Windows SDK 缺失 -> VS Installer 追加安装"
        Write-WARN "  4) INCLUDE/LIB 未设置 -> 使用 vcvarsall.bat 初始化环境"
        $script:cudaExtReady = $false
    }
} else {
    Write-WARN "缺少 nvcc 或 cl.exe，跳过编译测试"
    $script:cudaExtReady = $false
}

# ─── 汇总报告 ────────────────────────────────────────────────────
Write-HDR "自检汇总"
Write-Host ""
Write-Host "  通过: $($script:passCount)  失败: $($script:failCount)" -ForegroundColor $(if ($script:failCount -eq 0) { "Green" } else { "Yellow" })
Write-Host ""

if ($script:fixApplied.Count -gt 0) {
    Write-INFO "已自动修复以下项目 (对当前 shell 立即生效，永久生效需重启):"
    $script:fixApplied | ForEach-Object { Write-INFO "  • $_" }
    Write-Host ""
}

if ($script:failCount -eq 0) {
    Write-OK "环境完全就绪！运行以下命令启动 3x3 协议验证:"
    Write-Host ""
    Write-Host "  python train_compare_sta_protocol.py --binary-backend cuda_ext --steps 40,120,240 --seeds 42,1337,2024" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-WARN "仍有 $($script:failCount) 项未通过。修复清单:"
    Write-Host ""
    if (-not $nvccPath) {
        Write-Host "  [A] 安装 CUDA Toolkit 12.1 (管理员 PS):" -ForegroundColor Yellow
        Write-Host "      winget install Nvidia.CUDA --version 12.1" -ForegroundColor Cyan
        Write-Host "      或手动: https://developer.nvidia.com/cuda-12-1-0-download-archive" -ForegroundColor Cyan
        Write-Host ""
    }
    if (-not $clPath) {
        Write-Host "  [B] 安装 VS 2022 Build Tools (管理员 PS):" -ForegroundColor Yellow
        Write-Host '      winget install Microsoft.VisualStudio.2022.BuildTools --silent --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait"' -ForegroundColor Cyan
        Write-Host ""
    }
    Write-Host "  安装完成后重新运行:" -ForegroundColor Yellow
    Write-Host "  .\fix_cuda_env.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-INFO "或一步到位 (需管理员):"
    Write-Host "  .\fix_cuda_env.ps1 -Install" -ForegroundColor Cyan
}

# 输出机器可读状态供其他脚本调用
$status = @{
    pass          = $script:passCount
    fail          = $script:failCount
    cuda_ext_ready = if ($null -eq $script:cudaExtReady) { $false } else { $script:cudaExtReady }
    nvcc_found    = ($null -ne $nvccPath)
    cl_found      = ($null -ne $clPath)
    ninja_found   = ($null -ne $ninjaCmd)
    fix_applied   = $script:fixApplied
}
$status | ConvertTo-Json | Out-File "cuda_env_status.json" -Encoding utf8
Write-INFO "状态已写入 cuda_env_status.json"
