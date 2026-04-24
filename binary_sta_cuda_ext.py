"""Runtime loader for Binary STA CUDA fused extension.

Compile prerequisites (Windows):
  - CUDA Toolkit 12.1  (must match torch.version.cuda)
    winget install Nvidia.CUDA --version 12.1
    → CUDA_HOME env var must point to the toolkit root
  - MSVC Build Tools 2022 (C++ workload, includes cl.exe + Windows SDK)
    winget install Microsoft.VisualStudio.2022.BuildTools ...
  - ninja:  pip install ninja
  - Run fix_cuda_env.ps1 to auto-detect/set env, then reopen terminal.

Diagnostic: import binary_sta_cuda_ext; print(binary_sta_cuda_ext.diagnose())
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.cpp_extension import load, CUDA_HOME

_EXT = None
_LOAD_ERROR: str | None = None


def _sources() -> list[str]:
    root = Path(__file__).resolve().parent / "cuda_ext"
    return [
        str(root / "binary_sta_fused.cpp"),
        str(root / "binary_sta_fused_kernel.cu"),
    ]


def diagnose() -> Dict[str, Any]:
    """Return a dict with full environment diagnostics for debugging."""
    nvcc = shutil.which("nvcc")
    cl   = shutil.which("cl")
    ninja = shutil.which("ninja")
    nvcc_ver: str | None = None
    if nvcc:
        try:
            r = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=5)
            nvcc_ver = r.stdout.strip().splitlines()[-1] if r.returncode == 0 else r.stderr
        except Exception as e:
            nvcc_ver = f"error: {e}"

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_HOME": CUDA_HOME,
        "CUDA_HOME_env": os.environ.get("CUDA_HOME"),
        "CUDA_PATH_env": os.environ.get("CUDA_PATH"),
        "nvcc_path": nvcc,
        "nvcc_version": nvcc_ver,
        "cl_path": cl,
        "ninja_path": ninja,
        "sources_exist": all(Path(s).exists() for s in _sources()),
        "disable_env": os.environ.get("BINARY_STA_DISABLE_CUDA_EXT", "0"),
        "load_error": _LOAD_ERROR,
        "ext_loaded": _EXT is not None,
        "missing": _missing_prereqs(),
    }


def _missing_prereqs() -> list[str]:
    missing = []
    if not torch.cuda.is_available():
        missing.append("CUDA runtime (torch.cuda.is_available() is False)")
    if CUDA_HOME is None and not shutil.which("nvcc"):
        missing.append(
            "nvcc / CUDA Toolkit — install: winget install Nvidia.CUDA --version 12.1  "
            "then set CUDA_HOME=<toolkit root>"
        )
    if not shutil.which("cl"):
        missing.append(
            "cl.exe / MSVC Build Tools — install: "
            'winget install Microsoft.VisualStudio.2022.BuildTools --silent --override '
            '"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait"'
        )
    if not shutil.which("ninja"):
        missing.append("ninja — install: pip install ninja")
    return missing


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if os.environ.get("BINARY_STA_DISABLE_CUDA_EXT", "0") == "1":
        return False
    return True


def load_extension(verbose: bool = False):
    global _EXT, _LOAD_ERROR
    if _EXT is not None:
        return _EXT
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    if not is_available():
        raise RuntimeError("CUDA extension unavailable: CUDA disabled or env blocked")

    missing = _missing_prereqs()
    if missing:
        msg = (
            "Cannot compile binary_sta_fused_ext — missing prerequisites:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\nRun: python fix_cuda_env.py"
        )
        _LOAD_ERROR = msg
        raise RuntimeError(msg)

    profile = os.environ.get("BINARY_STA_CUDA_EXT_PROFILE", "fast").strip().lower()
    if profile not in {"fast", "consistent"}:
        profile = "fast"

    extra_cuda_cflags = [
        "-O3",
        "-lineinfo",
        "-allow-unsupported-compiler",
        "-Xcompiler",
        "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
    ]
    if profile == "fast":
        extra_cuda_cflags.append("--use_fast_math")
    else:
        # Consistency profile: avoid fast-math transforms to reduce numeric drift.
        extra_cuda_cflags.extend(["--fmad=false"])

    try:
        # CUDA 12.1 on Windows can reject newer MSVC versions via host_config.h;
        # this flag keeps nvcc working with up-to-date Build Tools.
        _EXT = load(
            name="binary_sta_fused_ext",
            sources=_sources(),
            verbose=verbose,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=["-O3", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"],
        )
        return _EXT
    except Exception as exc:
        text = str(exc)
        _LOAD_ERROR = text
        raise RuntimeError(f"binary_sta_fused_ext compile failed: {text}") from exc


def fused_forward(
    packed_codes: torch.Tensor,
    values: torch.Tensor,
    num_planes: int,
    chunk_size: int,
    routing_mode: str,
    temperature: float,
) -> torch.Tensor:
    ext = load_extension(verbose=False)
    use_softmax = routing_mode == "softmax"
    return ext.binary_sta_fused_forward(
        packed_codes,
        values,
        int(num_planes),
        int(chunk_size),
        True,
        bool(use_softmax),
        float(temperature),
    )
