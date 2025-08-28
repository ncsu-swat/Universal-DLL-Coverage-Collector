import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Optional, List, Tuple

def _run_subdir(subdir: Path, driver_path: Path, prof: Path, timeout: int) -> Tuple[str, int, str, str]:
	"""Run driver (torch or tf) on a subdirectory with LLVM_PROFILE_FILE set.

	Returns: (subdir, returncode, stdout, stderr)
	"""
	prof.parent.mkdir(parents=True, exist_ok=True)
	env = os.environ.copy()
	env["LLVM_PROFILE_FILE"] = str(prof)
	# Constrain threading/concurrency to 1 for PyTorch / TensorFlow and math libs
	env.update({
		"OMP_NUM_THREADS": "1",
		"OMP_THREAD_LIMIT": "1",
		"MKL_NUM_THREADS": "1",
		"OPENBLAS_NUM_THREADS": "1",
		"NUMEXPR_NUM_THREADS": "1",
		"BLIS_NUM_THREADS": "1",
		"VECLIB_MAXIMUM_THREADS": "1",
		# TensorFlow specific knobs
		"TF_NUM_INTRAOP_THREADS": "1",
		"TF_NUM_INTEROP_THREADS": "1",
	})
	cmd = [sys.executable, str(driver_path), "--inputs-dir", str(subdir)]
	try:
		proc = subprocess.run(
			cmd,
			cwd=str(subdir),
			text=True,
			capture_output=True,
			env=env,
			timeout=timeout,
		)
		return (str(subdir), proc.returncode, proc.stdout, proc.stderr)
	except subprocess.TimeoutExpired as te:
		if isinstance(te.stdout, (bytes, bytearray)):
			stdout = te.stdout.decode(errors="ignore")
		else:
			stdout = str(te.stdout) if te.stdout is not None else ""
		if isinstance(te.stderr, (bytes, bytearray)):
			stderr = te.stderr.decode(errors="ignore")
		else:
			stderr = str(te.stderr) if te.stderr is not None else f"Timeout after {timeout}s"
		return (str(subdir), 124, stdout, stderr)
	except Exception as e:
		return (str(subdir), 1, "", f"Exception: {e}")


def _find_profdata_tool() -> Optional[str]:
	candidates: List[Optional[str]] = [
		os.environ.get("LLVM_PROFDATA"),
		"llvm-profdata",
		"llvm-profdata-18",
		"llvm-profdata-17",
	]
	for c in candidates:
		if not c:
			continue
		try:
			res = subprocess.run([c, "--version"], capture_output=True, text=True)
			if res.returncode == 0:
				return c
		except FileNotFoundError:
			continue
	return None


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="titanfuzz driver: run torch/tf driver per subdirectory and merge coverage")
	parser.add_argument("--inputs-dir", required=True, help="Top-level directory containing subdirectories of Python files")
	parser.add_argument("--profraw-root", required=True, help="Root directory for .profraw outputs")
	parser.add_argument("--profdata-out", required=True, help="Path to write merged .profdata output")
	parser.add_argument("--timeout-sec", type=int, default=180, help="Per-subdir timeout in seconds")
	parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1), help="Parallel worker processes")
	parser.add_argument("--dll", choices=["torch", "tf"], default=os.environ.get("DLL", "torch"), help="Which framework to preload in driver")
	args = parser.parse_args(argv)

	inputs_dir = Path(args.inputs_dir)
	profroot_path = Path(args.profraw_root)
	profdata_out = Path(args.profdata_out)
	timeout = args.timeout_sec
	jobs = max(1, int(args.jobs))

	if not inputs_dir.exists():
		print(f"[driver] Inputs dir not found: {inputs_dir}", file=sys.stderr)
		return 2
	profroot_path.mkdir(parents=True, exist_ok=True)
	profdata_out.parent.mkdir(parents=True, exist_ok=True)

	# Select driver script based on dll
	if args.dll == "tf":
		driver_path = Path(__file__).parent / "tf_driver.py"
		if not driver_path.exists():
			print(f"[driver] tf_driver.py not found at {driver_path}", file=sys.stderr)
			return 2
	else:
		driver_path = Path(__file__).parent / "torch_driver.py"
		if not driver_path.exists():
			print(f"[driver] torch_driver.py not found at {driver_path}", file=sys.stderr)
			return 2

	# Discover subdirectories directly under inputs_dir
	subdirs = sorted([p for p in inputs_dir.iterdir() if p.is_dir()])
	if not subdirs:
		print("[driver] No subdirectories found under inputs-dir", file=sys.stderr)
		return 2

	print(f"[driver] Running {driver_path.name} for {len(subdirs)} subdirs with {jobs} workers, timeout={timeout}s")

	# Dispatch work across processes: one .profraw per subdir
	futures: List[Future[Tuple[str, int, str, str]]] = []
	results: List[Tuple[str, int, str, str]] = []
	with ProcessPoolExecutor(max_workers=jobs) as exe:
		for sd in subdirs:
			rel = sd.relative_to(inputs_dir)
			out_prof = profroot_path / rel / "coverage.profraw"
			futures.append(exe.submit(_run_subdir, sd, driver_path, out_prof, timeout))
		for fut in as_completed(futures):
			results.append(fut.result())

	# Report
	failures = 0
	for (sd_str, rc, out, err) in results:
		rel_str = os.path.relpath(sd_str, str(inputs_dir))
		if rc == 0:
			print(f"[driver] OK subdir {rel_str}")
		elif rc == 124:
			print(f"[driver] TIMEOUT subdir {rel_str}", file=sys.stderr)
		else:
			print(f"[driver] FAIL subdir {rel_str} rc={rc}", file=sys.stderr)
			if out:
				print(out)
			if err:
				print(err, file=sys.stderr)
		if rc != 0:
			failures += 1

	# Merge profraws -> profdata
	all_profraws = [str(p) for p in profroot_path.rglob("*.profraw")]
	if not all_profraws:
		print("[driver] No .profraw files generated; aborting merge", file=sys.stderr)
		return 1
	tool = _find_profdata_tool()
	if not tool:
		print("[driver] llvm-profdata not found", file=sys.stderr)
		return 1
	merge_cmd = [tool, "merge", "--num-threads=0", "--failure-mode=all" ,"-sparse", "-o", str(profdata_out)] + all_profraws
	print(f"[driver] Merging {len(all_profraws)} profraw -> {profdata_out}")
	merge = subprocess.run(merge_cmd, capture_output=True, text=True)
	if merge.returncode != 0:
		print(merge.stdout)
		print(merge.stderr, file=sys.stderr)
		return merge.returncode


	# Exit non-zero if any test failed, but merge succeeded
	if failures:
		print(f"[driver] Completed with {failures} failures", file=sys.stderr)
		return 1
	print("[driver] Completed OK")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
