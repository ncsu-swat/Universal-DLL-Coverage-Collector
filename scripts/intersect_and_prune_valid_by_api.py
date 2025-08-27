#!/usr/bin/env python3
"""
Compute the intersection between api.txt (in valid_by_api) and a FlashFuzz API list,
write it to api_intersect.txt, and prune valid_by_api/torch to only keep intersect APIs.

By default, operates on:
  - valid_dir: /home/fqin2/Universal-DLL-Coverage-Collector/Results/torch/valid_by_api
  - flash_list: /home/fqin2/FlashFuzz/api_list/torch2.2-flashfuzz.txt

Safety notes:
  - This is destructive: it deletes non-intersect directories under valid_by_api/torch.
  - api.txt must already exist in valid_dir (one API per line).
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
    # Map dotted API to a nested directory under valid_dir
    # e.g., "torch.nn.functional.relu" -> valid_dir/torch/nn/functional/relu
    parts = api.split(".")
    return valid_dir.joinpath(*parts)


def collect_ancestors(path: Path, stop_at: Path) -> List[Path]:
    """Collect all ancestors including path itself down to stop_at (inclusive)."""
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
    """Delete subdirectories under root that are not in keep_dirs.

    Assumes keep_dirs contains root and all ancestor directories for kept leaves.
    """
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
                # Remove entire subtree
                try:
                    shutil.rmtree(child)
                    print(f"Removed: {child}")
                except Exception as e:
                    print(f"[WARN] Failed to remove {child}: {e}")
        dirnames[:] = to_iterate


def main() -> int:
    ap = argparse.ArgumentParser(description="Intersect API lists and prune valid_by_api/torch tree.")
    ap.add_argument("--valid-dir", type=Path, default=Path("/home/fqin2/Universal-DLL-Coverage-Collector/Results/torch/valid_by_api"))
    ap.add_argument("--flash-list", type=Path, default=Path("/home/fqin2/FlashFuzz/api_list/torch2.2-flashfuzz.txt"))
    args = ap.parse_args()

    valid_dir: Path = args.valid_dir
    flash_list_path: Path = args.flash_list
    api_txt = valid_dir / "api.txt"
    api_intersect = valid_dir / "api_intersect.txt"
    torch_root = valid_dir / "torch"

    if not api_txt.exists():
        raise FileNotFoundError(f"api.txt not found: {api_txt}")
    if not flash_list_path.exists():
        raise FileNotFoundError(f"FlashFuzz list not found: {flash_list_path}")
    if not torch_root.exists() or not torch_root.is_dir():
        raise FileNotFoundError(f"torch folder not found under valid_dir: {torch_root}")

    ours = set(read_lines(api_txt))
    theirs = set(read_lines(flash_list_path))
    inter = sorted(ours & theirs)

    write_lines(api_intersect, inter)
    print(f"Wrote {len(inter)} intersect APIs to {api_intersect}")

    # Build keep set of directories (include all ancestors up to torch_root)
    keep: Set[Path] = set()
    keep.add(torch_root.resolve())
    for api in inter:
        p = api_to_dir(valid_dir, api)
        # Only keep those under torch_root
        if not str(p).startswith(str(torch_root)):
            continue
        for anc in collect_ancestors(p, stop_at=torch_root):
            keep.add(anc.resolve())

    prune_tree_to_keep(torch_root, keep)
    print("Pruning complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
