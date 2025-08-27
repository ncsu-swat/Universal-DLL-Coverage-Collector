#!/usr/bin/env python3
"""
Classify Torch valid samples by API name based on filename pattern.

Input filenames look like:
  - torch.nn.functional.leaky_relu_1291.py
  - torch.Tensor.mul__480.py   # API ends with underscore -> double underscore before id
  - torch.optim.Adam_572.py

We extract the API name as the part before the trailing _<id>.py, then
create a nested directory structure under the output directory by
replacing dots with path separators. Each file is placed into its API
directory using symlinks by default (configurable).

Usage examples:
  - Dry-run (no changes):
      python scripts/classify_torch_valid_by_api.py --dry-run --max-files 20

  - Create symlinked classification under Results/torch/valid_by_api:
      python scripts/classify_torch_valid_by_api.py

  - Move files instead of symlinking (destructive):
      python scripts/classify_torch_valid_by_api.py --mode move

  - Build a full JSON index with file lists:
      python scripts/classify_torch_valid_by_api.py --full-index
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


FILENAME_RE = re.compile(r"^(?P<api>.+)_(?P<id>\d+)\.py$")


def extract_api_from_filename(filename: str) -> Optional[Tuple[str, int]]:
    """Extract (api, id) from a filename, or None if it doesn't match.

    We expect filenames like "torch.foo.bar_123.py" or "torch.Tensor.round__214.py".
    The regex takes everything up to the last "_\\d+.py" as API (so API may end with
    an underscore).
    """
    base = os.path.basename(filename)
    m = FILENAME_RE.match(base)
    if not m:
        return None
    api = m.group("api")
    try:
        sample_id = int(m.group("id"))
    except ValueError:
        return None
    return api, sample_id


def safe_relpath(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(str(path), start=str(start))
    except Exception:
        return str(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def place_file(
    src: Path,
    dest_dir: Path,
    mode: str = "symlink",
    allow_overwrite: bool = False,
) -> Path:
    """Place src inside dest_dir according to mode. Returns the destination path.

    mode: 'symlink' (default), 'hardlink', 'copy', or 'move'.
    When a file with the same name exists, we avoid clobbering by adding a suffix.
    """
    ensure_dir(dest_dir)
    dest = dest_dir / src.name

    def finalize_path(p: Path) -> Path:
        if allow_overwrite:
            return p
        if not p.exists():
            return p
        # Find a non-colliding name by adding -dupN before extension
        stem = p.stem
        suffix = p.suffix  # .py
        n = 1
        while True:
            candidate = p.with_name(f"{stem}-dup{n}{suffix}")
            if not candidate.exists():
                return candidate
            n += 1

    dest = finalize_path(dest)

    if mode == "symlink":
        # Use relative symlink for portability if possible
        try:
            rel_src = os.path.relpath(src, start=dest.parent)
            os.symlink(rel_src, dest)
        except FileExistsError:
            pass
        except FileNotFoundError:
            # Parent may not exist in rare race, ensure and retry
            ensure_dir(dest.parent)
            rel_src = os.path.relpath(src, start=dest.parent)
            os.symlink(rel_src, dest)
        except OSError:
            # Fallback to hardlink then copy
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
    elif mode == "hardlink":
        try:
            os.link(src, dest)
        except OSError:
            # Fallback to copy if hardlink not possible (e.g., cross-device)
            shutil.copy2(src, dest)
    elif mode == "copy":
        shutil.copy2(src, dest)
    elif mode == "move":
        shutil.move(src, dest)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return dest


def classify(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    dry_run: bool = False,
    max_files: Optional[int] = None,
    full_index: bool = False,
    allow_overwrite: bool = False,
) -> Dict[str, List[str]]:
    """Classify files by API name.

    Returns a mapping api -> list of destination file paths (as strings). If
    full_index is False, the lists will be empty to save memory.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    ensure_dir(output_dir)

    api_to_files: Dict[str, List[str]] = defaultdict(list)
    counts: Counter[str] = Counter()

    files_iter: Iterable[Path] = (p for p in input_dir.iterdir() if p.is_file())

    processed = 0
    skipped = 0

    for p in files_iter:
        if max_files is not None and processed >= max_files:
            break
        if p.suffix != ".py":
            skipped += 1
            continue
        extracted = extract_api_from_filename(p.name)
        if not extracted:
            skipped += 1
            continue
        api, _sid = extracted
        # Build destination directory: replace dots with path separators
        api_dir_rel = api.replace(".", os.sep)
        dest_dir = output_dir / api_dir_rel

        dest_path: Path
        if dry_run:
            dest_path = dest_dir / p.name
        else:
            dest_path = place_file(p, dest_dir, mode=mode, allow_overwrite=allow_overwrite)

        counts[api] += 1
        if full_index:
            api_to_files[api].append(str(dest_path))
        else:
            # Keep minimal memory footprint
            api_to_files.setdefault(api, [])
        processed += 1

    # Write indexes
    if not dry_run:
        # CSV with counts
        csv_path = output_dir / "index.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["api", "count"])
            for api, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                writer.writerow([api, cnt])

        # JSON index (counts + optional file lists)
        json_obj = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "mode": mode,
            "total_processed": processed,
            "total_skipped": skipped,
            "apis": {
                api: {
                    "count": counts[api],
                    "files": api_to_files[api] if full_index else None,
                }
                for api in sorted(api_to_files.keys())
            },
        }
        json_path = output_dir / "index.json"
        with json_path.open("w") as jf:
            json.dump(json_obj, jf, indent=2)

    return api_to_files


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify Torch valid samples by API name.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/fqin2/Universal-DLL-Coverage-Collector/Results/torch/valid"),
        help="Directory containing valid .py samples (default: Results/torch/valid)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/fqin2/Universal-DLL-Coverage-Collector/Results/torch/valid_by_api"),
        help="Output directory to create the classification tree (default: Results/torch/valid_by_api)",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy", "move"],
        default="symlink",
        help="How to place files into the classification tree (default: symlink)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not modify the filesystem; just show what would happen.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of files processed (useful for testing).",
    )
    parser.add_argument(
        "--full-index",
        action="store_true",
        help="Include file lists in the JSON index (can be large).",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing files at destination.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("Classifying Torch valid samples by API name:")
    print(f"  input_dir      = {args.input_dir}")
    print(f"  output_dir     = {args.out_dir}")
    print(f"  mode           = {args.mode}")
    print(f"  dry_run        = {args.dry_run}")
    print(f"  max_files      = {args.max_files}")
    print(f"  full_index     = {args.full_index}")
    print(f"  allow_overwrite= {args.allow_overwrite}")

    try:
        classify(
            input_dir=args.input_dir,
            output_dir=args.out_dir,
            mode=args.mode,
            dry_run=args.dry_run,
            max_files=args.max_files,
            full_index=args.full_index,
            allow_overwrite=args.allow_overwrite,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
