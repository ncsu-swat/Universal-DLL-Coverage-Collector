#!/usr/bin/env python3
"""
Prune Results/*/valid_by_api tree using api.txt.

Reads an API list (one dotted API per line, e.g., torch.nn.functional.relu) and
deletes directories under the valid_by_api folder that are not on any API path.

Rules:
- Only directories are removed; files like index.csv, api.txt are preserved.
- For each API, we keep valid_dir/<root> (e.g., torch) and all ancestors down to
  the leaf path (e.g., torch/nn/functional/relu), if they exist.
- Directories not in the keep set are removed recursively.

Default locations:
  valid_dir: /home/fqin2/Universal-DLL-Coverage-Collector/Results/torch/valid_by_api
  api_file:  <valid_dir>/api.txt

CAUTION: This is destructive. Consider backing up first if needed.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Set


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def api_to_dir(valid_dir: Path, api: str) -> Path:
    parts = api.split(".")
    return valid_dir.joinpath(*parts)


def collect_ancestors(path: Path, stop_at: Path) -> List[Path]:
    res: List[Path] = []
    cur = path
    stop_at = stop_at.resolve()
    while True:
        res.append(cur)
        if cur.resolve() == stop_at:
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    return res


def prune_tree_to_keep(root: Path, keep_dirs: Set[Path]) -> None:
    root = root.resolve()
    keep_dirs_resolved = {p.resolve() for p in keep_dirs}
    if root.resolve() not in keep_dirs_resolved:
        keep_dirs_resolved.add(root.resolve())

    for cur_root, dirnames, _filenames in os.walk(root, topdown=True):
        cur_path = Path(cur_root)
        # Filter dirnames in-place so os.walk does not descend into removed dirs
        to_iterate: List[str] = []
        for d in dirnames:
            child = (cur_path / d).resolve()
            if child in keep_dirs_resolved:
                to_iterate.append(d)
            else:
                try:
                    shutil.rmtree(child)
                    print(f"Removed: {child}")
                except Exception as e:
                    print(f"[WARN] Failed to remove {child}: {e}")
        dirnames[:] = to_iterate


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune valid_by_api tree using api.txt")
    ap.add_argument("--valid-dir", type=Path, default=Path("/home/fqin2/Universal-DLL-Coverage-Collector/Results/torch/valid_by_api"))
    ap.add_argument("--api-file", type=Path, default=None, help="Path to api.txt (default: <valid_dir>/api.txt)")
    args = ap.parse_args()

    valid_dir: Path = args.valid_dir
    api_file: Path = args.api_file or (valid_dir / "api.txt")

    if not valid_dir.exists() or not valid_dir.is_dir():
        raise FileNotFoundError(f"valid_dir not found or not a directory: {valid_dir}")
    if not api_file.exists():
        raise FileNotFoundError(f"API list file not found: {api_file}")

    apis = read_lines(api_file)
    if not apis:
        print("No APIs found in list; nothing to prune.")
        return 0

    # Compute keep set across all API roots present in the list
    keep: Set[Path] = set()
    for api in apis:
        p = api_to_dir(valid_dir, api)
        # Identify root under valid_dir (top-level namespace like torch, tf)
        parts = api.split('.')
        if not parts:
            continue
        root = valid_dir / parts[0]
        if not root.exists():
            # No such root; skip
            continue
        keep.add(root.resolve())
        # Only consider existing leaf directories; collect ancestors if present
        # If leaf doesn't exist, still keep ancestors that exist
        cur = p
        while True:
            if cur.exists() and cur.is_dir():
                for anc in collect_ancestors(cur, stop_at=root):
                    keep.add(anc.resolve())
                break
            if cur == root:
                break
            cur = cur.parent

    # Prune each top-level directory under valid_dir that is not kept
    # Only operate on directories; leave files alone
    for entry in valid_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.resolve() not in keep:
            try:
                shutil.rmtree(entry)
                print(f"Removed: {entry}")
            except Exception as e:
                print(f"[WARN] Failed to remove {entry}: {e}")
        else:
            # Recursively prune within this kept root
            prune_tree_to_keep(entry, keep)

    print("Pruning complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
