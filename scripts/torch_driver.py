import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Pre-import torch and numpy (if available) and inject into executed scripts' globals.
try:  # Lazy-friendly: don't fail if not present in this environment
    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover
    _torch = None  # type: ignore

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore


def run_file_with_exec(py_file: Path) -> int:
    """Execute a Python file using compile/exec in an isolated globals dict.

    Returns 0 on success, non-zero on exception. Never raises.
    """
    try:
        src = py_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[torch_driver] READ-FAIL {py_file}: {e}")
        return 1

    # Prepare an isolated module-like globals namespace
    g: Dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(py_file),
        "__package__": None,
        "__cached__": None,
        "__builtins__": __builtins__,
    }

    # Inject commonly used modules
    if _torch is not None:
        g["torch"] = _torch
    if _np is not None:
        g["np"] = _np
        g["numpy"] = _np

    # Temporarily adjust cwd and sys.path so relative imports/files work
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    try:
        os.chdir(str(py_file.parent))
        if str(py_file.parent) not in sys.path:
            sys.path.insert(0, str(py_file.parent))
        compiled_code = compile(src, str(py_file), "exec")
        exec(compiled_code, g)
        print(f"[torch_driver] OK {py_file}")
        return 0
    except SystemExit as se:
        # Treat SystemExit as non-fatal; record its code but keep going
        exit_code = se.code if isinstance(se.code, int) else 0
        print(f"[torch_driver] SYS-EXIT {py_file} rc={exit_code}")
        # Consider SystemExit as success to avoid failing the run loop
        return 0
    except Exception as e:
        print(f"[torch_driver] FAIL {py_file}: {e}")
        return 1
    finally:
        # Restore environment for next file
        try:
            os.chdir(str(old_cwd))
        except Exception:
            pass
        sys.path[:] = old_sys_path


def discover_py_files(inputs_dir: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return sorted(p for p in inputs_dir.rglob("*.py") if p.is_file())
    return sorted(p for p in inputs_dir.glob("*.py") if p.is_file())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Torch driver: exec() all Python files in a directory, never fail on one file")
    parser.add_argument("--inputs-dir", required=True, help="Directory containing Python files")
    parser.add_argument("--non-recursive", action="store_true", help="Only run files directly under inputs-dir (no recursion)")
    args = parser.parse_args(argv)

    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists() or not inputs_dir.is_dir():
        print(f"[torch_driver] Inputs dir not found: {inputs_dir}", file=sys.stderr)
        return 2

    files = discover_py_files(inputs_dir, recursive=not args.non_recursive)
    if not files:
        print("[torch_driver] No Python files found", file=sys.stderr)
        return 0

    total = len(files)
    n_ok = 0
    n_fail = 0
    for f in files:
        rc = run_file_with_exec(f)
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1

    print(f"[torch_driver] Summary: total={total} ok={n_ok} fail={n_fail}")
    # Always return 0 per requirement "not fail"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
