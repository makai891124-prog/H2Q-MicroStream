"""
run_cuda_strategy.py
====================
Execute two ready-to-run strategies for Binary STA backend behavior.

Usage examples:
  python run_cuda_strategy.py --mode conservative --run-protocol --steps 40 120 240
  python run_cuda_strategy.py --mode aggressive --run-protocol --steps 10 30 60

Strategy meanings:
  conservative:
    - Keep default backend as packbits.
    - Opportunistically use cuda_ext only in inference and long sequence.
  aggressive:
    - Keep threshold unchanged.
    - Use cuda_ext consistency profile (no fast-math) to reduce loss drift.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def _bootstrap_toolchain_env() -> None:
    try:
        import fix_cuda_env as envfix

        d = envfix.check_env()
        nvcc = d.get("nvcc")
        cl = d.get("cl")

        if nvcc:
            cuda_home = str(Path(nvcc).parent.parent)
            os.environ.setdefault("CUDA_HOME", cuda_home)
            os.environ.setdefault("CUDA_PATH", cuda_home)
            nvcc_dir = str(Path(nvcc).parent)
            if nvcc_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + nvcc_dir

        if cl:
            cl_dir = str(Path(cl).parent)
            if cl_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + cl_dir
    except Exception:
        # Keep strategy script best-effort: protocol diagnostics will print details.
        return


def _set_conservative_env(min_seq: int) -> None:
    # Keep default packbits, allow cuda_ext only for eval and long sequence.
    os.environ["BINARY_STA_PACKBITS_INFER_CUDA_EXT"] = "1"
    os.environ["BINARY_STA_CUDA_EXT_MODE"] = "infer_long"
    os.environ["BINARY_STA_CUDA_EXT_MIN_SEQ"] = str(max(1, min_seq))
    os.environ["BINARY_STA_CUDA_EXT_PROFILE"] = "fast"


def _set_aggressive_env() -> None:
    # Force best numeric consistency profile during cuda_ext compilation/runtime.
    os.environ["BINARY_STA_PACKBITS_INFER_CUDA_EXT"] = "0"
    os.environ["BINARY_STA_CUDA_EXT_MODE"] = "always"
    os.environ["BINARY_STA_CUDA_EXT_PROFILE"] = "consistent"


def _print_effective_env() -> None:
    print("Effective strategy env:")
    keys = [
        "BINARY_STA_PACKBITS_INFER_CUDA_EXT",
        "BINARY_STA_CUDA_EXT_MODE",
        "BINARY_STA_CUDA_EXT_MIN_SEQ",
        "BINARY_STA_CUDA_EXT_PROFILE",
    ]
    for k in keys:
        print(f"  {k}={os.environ.get(k, '')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["conservative", "aggressive"], required=True)
    parser.add_argument("--min-seq", type=int, default=256,
                        help="Long sequence threshold for conservative mode")
    parser.add_argument("--run-protocol", action="store_true",
                        help="Run cuda_ext_protocol.py after setting strategy")
    parser.add_argument("--steps", type=int, nargs="+", default=[40, 120, 240])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    args = parser.parse_args()

    _bootstrap_toolchain_env()

    if args.mode == "conservative":
        _set_conservative_env(args.min_seq)
        print("[Strategy] conservative")
        print("- default backend stays packbits")
        print("- cuda_ext is enabled only for eval + long sequence")
    else:
        _set_aggressive_env()
        print("[Strategy] aggressive")
        print("- threshold is unchanged")
        print("- cuda_ext uses consistency compile profile")

    _print_effective_env()

    if not args.run_protocol:
        print("\nDone. Add --run-protocol to execute the 3x3 validation immediately.")
        return

    cmd = [
        PYTHON,
        str(WORKDIR / "cuda_ext_protocol.py"),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--steps",
        *[str(s) for s in args.steps],
    ]
    print("\nRunning:")
    print("  " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(WORKDIR)).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
