"""
post_install_verify.py
======================
Run this AFTER CUDA Toolkit 12.1 + VS 2022 Build Tools are installed.
Opens a FRESH Python process (to pick up new PATH), verifies the
environment, compiles cuda_ext, and launches cuda_ext_protocol.py.

Usage:
    python post_install_verify.py          # verify + compile + protocol
    python post_install_verify.py --diag   # verify only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
PYTHON  = sys.executable


def _fresh_check() -> dict:
    """Run fix_cuda_env.py --json in a fresh subprocess to pick up new PATH."""
    r = subprocess.run(
        [PYTHON, str(WORKDIR / "fix_cuda_env.py"), "--json"],
        capture_output=True,
        text=True,
        cwd=str(WORKDIR),
        timeout=30,
    )
    if r.returncode != 0:
        return {"error": r.stderr.strip()}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": r.stdout[-500:]}


def _add_cuda_to_path(d: dict) -> bool:
    """If nvcc found but CUDA_HOME not set, configure env and return True."""
    nvcc = d.get("nvcc")
    if not nvcc:
        return False
    cuda_home = str(Path(nvcc).parent.parent)
    nvcc_dir  = str(Path(nvcc).parent)
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault("CUDA_PATH", cuda_home)
    if nvcc_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + nvcc_dir
    return True


def _add_cl_to_path(d: dict) -> bool:
    cl = d.get("cl")
    if not cl:
        return False
    cl_dir = str(Path(cl).parent)
    if cl_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + cl_dir
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", action="store_true",
                        help="Only run diagnostics; skip compile and protocol")
    parser.add_argument("--protocol-args", default="",
                        help="Extra args to pass to cuda_ext_protocol.py")
    args = parser.parse_args()

    print("=" * 58)
    print("  Post-Install Verification for H2Q cuda_ext")
    print("=" * 58)

    # 1. Fresh environment check (subprocess picks up new PATH after install)
    print("\nStep 1: Environment check (fresh subprocess) ...")
    d = _fresh_check()
    if "error" in d:
        print(f"  ERROR: {d['error']}")
        sys.exit(1)

    # 2. Print status
    ok  = "[OK]"
    nok = "[!!]"
    pr  = "[>>]"
    print(f"  {ok if d.get('cuda_available') else nok}  CUDA runtime: {d.get('device', 'N/A')}")
    print(f"  {ok if d.get('nvcc') else nok}  nvcc: {d.get('nvcc') or 'NOT FOUND'}")
    print(f"  {ok if d.get('cl') else nok}  cl:   {d.get('cl') or 'NOT FOUND'}")
    print(f"  {ok if d.get('ninja') else nok}  ninja: {d.get('ninja') or 'NOT FOUND'}")
    print(f"  {ok if d.get('cpp_ext_CUDA_HOME') else pr}  CUDA_HOME: {d.get('cpp_ext_CUDA_HOME') or '(not set)'}")
    print(f"  {ok if d.get('sources_exist') else nok}  Source files: {d.get('sources_exist')}")

    missing = d.get("missing", [])
    if missing:
        print(f"\n  {len(missing)} issue(s) remaining:")
        for m in missing:
            tag, _, desc = m.partition(":")
            print(f"    [{tag.strip()}]: {desc.strip()[:80]}")
        print()
        print("  Cannot compile until all issues are resolved.")
        print("  After fixing, run: python post_install_verify.py")
        sys.exit(1)

    if args.diag:
        print("\n  Diag complete. Run without --diag to compile and run protocol.")
        return

    # 3. Set env vars in current process
    _add_cuda_to_path(d)
    _add_cl_to_path(d)
    print(f"\n  CUDA_HOME set to: {os.environ.get('CUDA_HOME')}")

    # 4. Try compile
    print("\nStep 2: JIT compile binary_sta_fused_ext ...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "binary_sta_cuda_ext",
            str(WORKDIR / "binary_sta_cuda_ext.py")
        )
        ext_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext_mod)
        ext_mod.load_extension(verbose=True)
        print("  SUCCESS: cuda_ext compiled and loaded!\n")
    except Exception as e:
        print(f"  COMPILE FAILED: {e}\n")
        print("  Possible reasons:")
        print("    - CUDA Toolkit version mismatch (need exactly 12.1)")
        print("    - VS Build Tools not fully installed yet (wait and retry)")
        print("    - cl.exe not in PATH (run: python fix_cuda_env.py --fix)")
        sys.exit(1)

    # 5. Run protocol
    print("Step 3: Running 3x3 protocol (cuda_ext vs packbits vs sta_v2) ...")
    protocol_args = args.protocol_args.split() if args.protocol_args else []
    cmd = [PYTHON, str(WORKDIR / "cuda_ext_protocol.py")] + protocol_args
    print(f"  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(WORKDIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
