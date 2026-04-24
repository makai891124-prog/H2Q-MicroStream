"""
fix_cuda_env.py  —  H2Q-MicroStream Windows CUDA compilation environment
                    self-check and fix guide.

Usage:
    python fix_cuda_env.py           # Full check + auto PATH setup guide
    python fix_cuda_env.py --json    # Machine-readable JSON output
    python fix_cuda_env.py --fix     # Write a helper .ps1 that sets env vars
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent

# Expected CUDA version (must match torch cu build tag)
EXPECTED_CUDA_MAJOR_MINOR = None  # auto-detected from torch


# ── helpers ──────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 5) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return -1, str(e)


def _find_nvcc() -> str | None:
    # 1. PATH
    p = shutil.which("nvcc")
    if p:
        return p
    # 2. Standard Windows install path
    cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if cuda_root.exists():
        for ver_dir in sorted(cuda_root.iterdir(), reverse=True):
            nvcc = ver_dir / "bin" / "nvcc.exe"
            if nvcc.exists():
                return str(nvcc)
    # 3. Env vars
    for var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(var) or ""
        if val:
            nvcc = Path(val) / "bin" / "nvcc.exe"
            if nvcc.exists():
                return str(nvcc)
    return None


def _find_cl() -> str | None:
    # 1. PATH
    p = shutil.which("cl")
    if p:
        return p
    # 2. VS 2022/2019/2017 Build Tools
    vs_bases = [
        Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC"),
    ]
    for base in vs_bases:
        if base.exists():
            for ver_dir in sorted(base.iterdir(), reverse=True):
                cl = ver_dir / "bin" / "Hostx64" / "x64" / "cl.exe"
                if cl.exists():
                    return str(cl)
    # 3. vswhere
    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    if vswhere.exists():
        rc, out = _run([str(vswhere), "-latest", "-products", "*",
                        "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property", "installationPath"])
        if rc == 0 and out:
            msvc_base = Path(out.strip()) / "VC" / "Tools" / "MSVC"
            if msvc_base.exists():
                for ver_dir in sorted(msvc_base.iterdir(), reverse=True):
                    cl = ver_dir / "bin" / "Hostx64" / "x64" / "cl.exe"
                    if cl.exists():
                        return str(cl)
    return None


# ── main check ───────────────────────────────────────────────────

def check_env() -> dict:
    result: dict = {"platform": platform.version(), "python": sys.version.split()[0]}

    # PyTorch
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["torch_cuda_version"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["device"] = torch.cuda.get_device_name(0) if result["cuda_available"] else None
        global EXPECTED_CUDA_MAJOR_MINOR
        EXPECTED_CUDA_MAJOR_MINOR = torch.version.cuda  # e.g. "12.1"
    except ImportError:
        result["torch_error"] = "torch not installed"
        return result

    # ninja
    ninja = shutil.which("ninja")
    result["ninja"] = ninja
    if ninja:
        _, v = _run(["ninja", "--version"])
        result["ninja_version"] = v

    # CUDA_HOME (as seen by torch.utils.cpp_extension)
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        result["cpp_ext_CUDA_HOME"] = CUDA_HOME
    except Exception:
        result["cpp_ext_CUDA_HOME"] = None

    # nvcc
    nvcc = _find_nvcc()
    result["nvcc"] = nvcc
    if nvcc:
        rc, ver = _run([nvcc, "--version"])
        result["nvcc_version"] = ver.splitlines()[-1] if ver else "?"
        # Extract version for comparison
        import re
        m = re.search(r"release (\d+\.\d+)", ver)
        result["nvcc_release"] = m.group(1) if m else None
    else:
        result["nvcc_release"] = None

    # cl.exe
    cl = _find_cl()
    result["cl"] = cl
    if cl:
        rc, ver = _run([cl])  # cl prints version to stderr with no args
        result["cl_version"] = ver.splitlines()[0] if ver else "?"

    # CUDA_HOME env
    result["CUDA_HOME_env"] = os.environ.get("CUDA_HOME")
    result["CUDA_PATH_env"] = os.environ.get("CUDA_PATH")

    # Source files exist
    result["sources_exist"] = all(
        (WORKDIR / "cuda_ext" / f).exists()
        for f in ("binary_sta_fused.cpp", "binary_sta_fused_kernel.cu")
    )

    # Missing list
    missing = []
    if not result["cuda_available"]:
        missing.append("CUDA_runtime: torch.cuda.is_available() is False")
    if not nvcc:
        missing.append(
            "nvcc/CUDA_Toolkit_12.1: "
            "winget install Nvidia.CUDA --version 12.1  "
            "(admin PS, ~2 GB)"
        )
    elif result.get("nvcc_release") and EXPECTED_CUDA_MAJOR_MINOR:
        # Compare major.minor
        nvcc_mm = result["nvcc_release"]
        torch_mm = EXPECTED_CUDA_MAJOR_MINOR
        import re
        def mm(s):
            m = re.match(r"(\d+\.\d+)", s)
            return m.group(1) if m else s
        if mm(nvcc_mm) != mm(torch_mm):
            missing.append(
                f"CUDA_version_mismatch: nvcc={nvcc_mm} vs torch={torch_mm}. "
                f"Need CUDA Toolkit {torch_mm}: "
                "https://developer.nvidia.com/cuda-toolkit-archive"
            )
    if not cl:
        missing.append(
            "cl.exe/MSVC_BuildTools: "
            'winget install Microsoft.VisualStudio.2022.BuildTools --silent --override '
            '"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait"  '
            "(admin PS, ~2 GB)"
        )
    if not ninja:
        missing.append("ninja: pip install ninja")

    # CUDA_HOME fix needed?
    if nvcc and not result["cpp_ext_CUDA_HOME"]:
        cuda_home = str(Path(nvcc).parent.parent)
        missing.append(
            f"CUDA_HOME_unset: nvcc found at {nvcc} but CUDA_HOME is not set. "
            f"Run in this shell: $env:CUDA_HOME='{cuda_home}'"
        )

    result["missing"] = missing
    result["can_compile"] = len(missing) == 0
    return result


def print_status(d: dict) -> None:
    OK = "[OK]"
    FAIL = "[!!]"
    WARN = "[>>]"

    print()
    print("=" * 58)
    print("  H2Q-MicroStream  CUDA Compilation Environment Check")
    print("=" * 58)
    torch_ver = d.get("torch_version", "N/A")
    cuda_ver  = d.get("torch_cuda_version", "N/A")
    device    = d.get("device", "N/A")
    cuda_ok   = d.get("cuda_available", False)
    print(f"  {OK if cuda_ok else FAIL}  PyTorch {torch_ver}  (built with CUDA {cuda_ver})")
    print(f"       Device: {device}")
    print()

    ninja = d.get("ninja")
    print(f"  {OK if ninja else FAIL}  ninja: {ninja or '(not found)'}")

    nvcc = d.get("nvcc")
    nvcc_ver = d.get("nvcc_version", "")
    print(f"  {OK if nvcc else FAIL}  nvcc:  {nvcc or '(not found)'}")
    if nvcc_ver:
        print(f"         {nvcc_ver}")

    cl = d.get("cl")
    cl_ver = d.get("cl_version", "")
    print(f"  {OK if cl else FAIL}  cl:    {cl or '(not found)'}")
    if cl_ver:
        print(f"         {cl_ver}")

    cuda_home = d.get("cpp_ext_CUDA_HOME")
    print(f"  {OK if cuda_home else WARN}  CUDA_HOME (cpp_ext): {cuda_home or '(None — must set env var)'}")

    sources_ok = d.get("sources_exist", False)
    print(f"  {OK if sources_ok else FAIL}  CUDA source files: {sources_ok}")
    print()

    missing = d.get("missing", [])
    if not missing:
        print(f"  {OK} Environment complete. Ready to compile cuda_ext!")
        print()
        print("  Run the protocol:")
        print("    python cuda_ext_protocol.py")
    else:
        print(f"  {FAIL} {len(missing)} issue(s) found. Fix checklist:")
        print()
        for i, m in enumerate(missing, 1):
            tag, _, desc = m.partition(":")
            print(f"  [{i}] {tag.strip()}")
            if desc.strip():
                # Wrap long lines
                words = desc.strip().split()
                line = "      "
                for w in words:
                    if len(line) + len(w) > 72:
                        print(line)
                        line = "      " + w + " "
                    else:
                        line += w + " "
                if line.strip():
                    print(line)
            print()

        print()
        print("  After installing, reopen terminal and re-run:")
        print("    python fix_cuda_env.py")
        print()
        print("  Or let the protocol script handle it:")
        print("    python cuda_ext_protocol.py --diag")
    print("=" * 58)
    print()


def write_fix_ps1(d: dict) -> None:
    """Write a tiny .ps1 that sets CUDA_HOME and PATH for the current session."""
    nvcc = d.get("nvcc")
    cl   = d.get("cl")
    lines = [
        "# Auto-generated by fix_cuda_env.py",
        "# Source this in your PowerShell session to set CUDA env vars:",
        "#   . .\\apply_cuda_env.ps1",
        "",
    ]

    if nvcc:
        cuda_home = str(Path(nvcc).parent.parent).replace("\\", "/")
        nvcc_dir  = str(Path(nvcc).parent).replace("\\", "\\\\")
        lines += [
            f'$env:CUDA_HOME = "{cuda_home}"',
            f'$env:CUDA_PATH = "{cuda_home}"',
            f'if ($env:PATH -notlike "*{nvcc_dir}*") {{',
            f'    $env:PATH = "$env:PATH;{nvcc_dir}"',
            "}",
            f'Write-Host "CUDA_HOME set to: {cuda_home}"',
            "",
        ]
    else:
        lines += [
            '# nvcc not found — install CUDA Toolkit 12.1 first:',
            '# winget install Nvidia.CUDA --version 12.1',
            "",
        ]

    if cl:
        cl_dir = str(Path(cl).parent).replace("\\", "\\\\")
        lines += [
            f'if ($env:PATH -notlike "*{cl_dir}*") {{',
            f'    $env:PATH = "$env:PATH;{cl_dir}"',
            "}",
            f'Write-Host "cl.exe dir added: {cl_dir}"',
            "",
        ]
    else:
        lines += [
            '# cl.exe not found — install VS 2022 Build Tools:',
            '# winget install Microsoft.VisualStudio.2022.BuildTools ...',
            "",
        ]

    lines.append('Write-Host "Done. Now run: python cuda_ext_protocol.py"')

    ps1_path = WORKDIR / "apply_cuda_env.ps1"
    ps1_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {ps1_path}")
    print("  Usage (in PowerShell): . .\\apply_cuda_env.ps1")


def try_set_cuda_home_in_process(d: dict) -> bool:
    """Set CUDA_HOME in the current Python process environment if nvcc is found."""
    nvcc = d.get("nvcc")
    if not nvcc:
        return False
    if d.get("cpp_ext_CUDA_HOME"):
        return True  # already set
    cuda_home = str(Path(nvcc).parent.parent)
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CUDA_PATH"] = cuda_home
    nvcc_dir = str(Path(nvcc).parent)
    if nvcc_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + nvcc_dir
    # Reload cpp_extension to pick up new env
    try:
        import importlib
        import torch.utils.cpp_extension as cppext
        importlib.reload(cppext)
        return True
    except Exception:
        return False


def try_compile_ext(verbose: bool = False) -> bool:
    try:
        import binary_sta_cuda_ext as ext_mod
        ext_mod.load_extension(verbose=verbose)
        return True
    except Exception as e:
        print(f"  Compile error: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA env self-check for H2Q-MicroStream")
    parser.add_argument("--json",    action="store_true", help="Output JSON diagnostics")
    parser.add_argument("--fix",     action="store_true", help="Write apply_cuda_env.ps1 helper")
    parser.add_argument("--compile", action="store_true", help="After checks, attempt JIT compile")
    args = parser.parse_args()

    d = check_env()

    # Try to set CUDA_HOME in this process if nvcc found but env unset
    if d.get("nvcc") and not d.get("cpp_ext_CUDA_HOME"):
        if try_set_cuda_home_in_process(d):
            # Re-check with updated env
            d = check_env()

    if args.json:
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return

    print_status(d)

    if args.fix:
        write_fix_ps1(d)

    if args.compile:
        if d.get("can_compile"):
            print("  Attempting JIT compile of binary_sta_fused_ext ...")
            ok = try_compile_ext(verbose=True)
            if ok:
                print("  SUCCESS: cuda_ext compiled and loaded.")
                print("  Run: python cuda_ext_protocol.py")
            else:
                print("  FAILED: see error above.")
        else:
            print("  Skipping compile: environment incomplete.")
            print("  Fix issues above first, then run: python fix_cuda_env.py --compile")

    # Write status JSON for other scripts to consume
    status_path = WORKDIR / "cuda_env_status.json"
    status_path.write_text(
        json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Status written to: {status_path}")


if __name__ == "__main__":
    main()
