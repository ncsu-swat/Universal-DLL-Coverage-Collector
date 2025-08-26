import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

class DLLCovCollector:
    def __init__(self, ver: str, target: str, output: str, dll: str, itv: int, baseline: str,filter: Optional[str] = None):
        self.ver = ver
        self.target = target
        self.output = output
        self.dll = dll
        self.itv = itv
        self.filter = filter
        self.baseline = baseline
        self.driver = f"scripts/{baseline}_driver.py"
        # check driver file exist
        driver_path = Path(self.driver)
        if not driver_path.exists():
            raise FileNotFoundError(f"Driver script not found: {self.driver}")
        self.docker_image = f"ncsu-swat/torch-{self.ver}-instrumented"
        self.docker_name = f"torch_cov_{self.ver}-{baseline}" if baseline else f"torch_cov_{self.ver}"
        self.docker_id = ""
        self.result_dir = f"{output}/{baseline}"
        
        
    def check_image(self):
        # use docker image to check
        cmd = ["docker", "images", "-q", self.docker_image]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to find Docker image: {result.stderr.strip()}, please run build.sh.")
        ok = bool(result.stdout.strip())
        if not ok:
            raise RuntimeError(f"Docker image '{self.docker_image}' not found. Please build it via build.sh")
        return ok

    def start_docker(self):
        cmd = ["docker", "run", "-td", "--name", self.docker_name, self.docker_image]
        print(f"Creating Docker container {self.docker_name} ")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create Docker container: {result.stderr.strip()}")
        self.docker_id = result.stdout.strip()

    def stop_docker(self):
        cmd = ["docker", "stop", self.docker_id]
        print(f"Stopping Docker container {self.docker_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to stop Docker container: {result.stderr.strip()}")
        
    def rm_docker(self):
        cmd = ["docker", "rm", "-fv", self.docker_id]
        print(f"Removing Docker container {self.docker_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to remove Docker container: {result.stderr.strip()}")

    def classify_python_files_with_itv(self, regex_pattern: Optional[str] = None) -> Dict[str, int]:
        """
        Classify Python files into time buckets per top-level subdirectory.

        For each immediate subdirectory under self.target (treated as a group), compute that group's
        own minimum timestamp and bucket files relative to that min using intervals of length self.itv.
        All groups share the same bucket labels (e.g., 0-60, 60-120, ...), and files are copied into
        self.result_dir/<bucket>/<parent>/<...> preserving the path relative to that parent.

        If files exist directly under self.target (no parent folder), they will be bucketed relative
        to their own min (root group ""), and placed under self.result_dir/<bucket>/.

        Notes:
        - On many systems (Linux), true creation/birth time is unavailable; we use modification time
          (os.path.getmtime) as a proxy.
        - If multiple files share the same basename within a destination directory, a numeric suffix
          is appended to avoid collisions.

        Optional filtering:
        - regex_pattern: Optional regex string applied to the path relative to self.target.

        Returns:
            Dict[str, int]: Mapping of bucket label (e.g., "0-60") to the number of files copied (across all groups).
        """
        # Validate inputs
        if not self.target or not os.path.isdir(self.target):
            raise FileNotFoundError(f"Target directory does not exist: {self.target}")
        if self.itv <= 0:
            raise ValueError(f"Interval (itv) must be a positive integer (seconds), got: {self.itv}")
        os.makedirs(self.result_dir, exist_ok=True)

        try:
            compiled_regex = re.compile(regex_pattern) if regex_pattern else None
        except re.error as e:
            print(f"Invalid regex pattern '{regex_pattern}': {e}")
            compiled_regex = None

        def get_creation_time(path: str) -> float:
            """Return the chosen timestamp for bucketing.

            Per request, we use modification time as the canonical timestamp:
            os.path.getmtime(path) -> float seconds since epoch.
            """
            try:
                return float(os.path.getmtime(path))
            except Exception:
                # Fallback: stat and use st_mtime directly if getmtime fails
                st = os.stat(path)
                return float(st.st_mtime)

        # 1) Collect Python files and their times, grouping by top-level directory
        # group_name "" corresponds to files directly under target
        grouped: Dict[str, List[Tuple[str, float]]] = {}
        for root, _dirs, files in os.walk(self.target):
            for name in files:
                if not name.endswith(".py"):
                    continue
                if name == "__init__.pyc" or name.endswith(".pyc"):
                    continue
                full = os.path.join(root, name)
                # Apply regex filter if provided (match against path relative to target)
                rel_to_target = os.path.relpath(full, self.target)
                if compiled_regex and not compiled_regex.search(rel_to_target):
                    continue
                try:
                    ts = get_creation_time(full)
                except FileNotFoundError:
                    continue
                # Determine top-level group
                parts = rel_to_target.split(os.sep)
                group = parts[0] if len(parts) > 1 else ""
                grouped.setdefault(group, []).append((full, ts))

        if not grouped:
            print("No Python files found to classify.")
            return cast(Dict[str, int], {})

        counts: Dict[str, int] = {}
        total = 0
        # 2) For each group, bucket relative to that group's min timestamp
        for group, group_files in grouped.items():
            if not group_files:
                continue
            group_min = min(ts for _p, ts in group_files)
            group_max = max(ts for _p, ts in group_files)
            print(f"Group '{group or 'ROOT'}' min_ts: {group_min}, max_ts: {group_max}")

            def bucket_label_g(ts: float) -> str:
                offset = max(0, int(ts - group_min))
                start = (offset // self.itv) * self.itv
                end = start + self.itv
                return f"{start}-{end}"

            for full, ts in group_files:
                total += 1
                label = bucket_label_g(ts)
                counts[label] = counts.get(label, 0) + 1

                # Build destination dir: result_dir/label/<group>/path-within-group-parent
                if group:
                    group_root = os.path.join(self.target, group)
                    rel_within_group = os.path.relpath(os.path.dirname(full), group_root)
                    dest_dir = os.path.join(self.result_dir, label, group)
                    if rel_within_group and rel_within_group != ".":
                        dest_dir = os.path.join(dest_dir, rel_within_group)
                else:
                    # Files directly under target: preserve their subpath relative to target
                    rel_parent = os.path.relpath(os.path.dirname(full), self.target)
                    dest_dir = os.path.join(self.result_dir, label)
                    if rel_parent and rel_parent != ".":
                        dest_dir = os.path.join(dest_dir, rel_parent)

                os.makedirs(dest_dir, exist_ok=True)

                dest_path = os.path.join(dest_dir, os.path.basename(full))
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(os.path.basename(full))
                    i = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(dest_dir, f"{base}__{i}{ext}")
                        i += 1

                shutil.copy(full, dest_path)

        bucket_count = len(counts)
        print(
            f"Classified and copied {total} Python files into {bucket_count} buckets under '{self.result_dir}'."
        )
        return counts

    def copy_to_docker(self, from_path: str, to_path: str):

        src = os.path.abspath(from_path)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source path does not exist: {src}")

        container_ref = self.docker_id or self.docker_name
        if not container_ref:
            raise RuntimeError("Docker container is not initialized. Call start_docker() first.")


        # Compute destination directory and ensure it exists inside the container
        is_src_dir = os.path.isdir(src)
        if is_src_dir:
            dest_dir = to_path  # copy directory into this directory
        else:
            # If to_path ends with '/', treat it as a directory; else treat it as a full file path
            dest_dir = to_path if to_path.endswith("/") else (os.path.dirname(to_path) or "/")

        mk = subprocess.run(
            ["docker", "exec", container_ref, "mkdir", "-p", dest_dir],
            capture_output=True, text=True
        )
        if mk.returncode != 0:
            raise RuntimeError(f"Failed to create destination directory '{dest_dir}' in container: {mk.stderr.strip()}")

        # Determine final docker cp destination
        if is_src_dir:
            dest = f"{container_ref}:{dest_dir}"
        else:
            dest = f"{container_ref}:{to_path if not to_path.endswith('/') else dest_dir}"

        cp = subprocess.run(["docker", "cp", src, dest], capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"Failed to copy '{src}' to '{dest}': {cp.stderr.strip()}")
        print(f"Copied to container: {src} -> {dest}")

    def copy_from_docker(self, from_path: str, to_path: str):
        container_ref = self.docker_id or self.docker_name
        if not container_ref:
            raise RuntimeError("Docker container is not initialized. Call start_docker() first.")

        to_abs = os.path.abspath(to_path)
        os.makedirs(os.path.dirname(to_abs) or to_abs, exist_ok=True)
        src = f"{container_ref}:{from_path}"
        cp = subprocess.run(["docker", "cp", src, to_abs], capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"Failed to copy '{src}' to host '{to_abs}': {cp.stderr.strip()}")
        print(f"Copied from container: {from_path} -> {to_abs}")

    def exec_in_docker(self, args: List[str], workdir: Optional[str] = None) -> subprocess.CompletedProcess[str]:
        container_ref = self.docker_id or self.docker_name
        if not container_ref:
            raise RuntimeError("Docker container is not initialized. Call start_docker() first.")
        cmd = ["docker", "exec"]
        if workdir:
            cmd += ["-w", workdir]
        cmd.append(container_ref)
        cmd += args
        return subprocess.run(cmd, capture_output=True, text=True)

    def collect(self):
        try:
            # 1) Determine interval directories: if target already has interval buckets (\d+-\d+), use them;
            #    otherwise classify target Python files into buckets under result_dir.
            use_existing = False
            if os.path.isdir(self.target):
                try:
                    names = [d for d in os.listdir(self.target) if os.path.isdir(os.path.join(self.target, d))]
                    use_existing = any(re.match(r"^\d+-\d+$", n) for n in names)
                except Exception:
                    use_existing = False

            if use_existing:
                result_root = Path(self.target)
                baseline_name = Path(self.target).name
            else:
                counts = self.classify_python_files_with_itv(self.filter)
                if not counts:
                    print("No files to process; exiting.")
                    return
                result_root = Path(self.result_dir)
                baseline_name = result_root.name

            # 2) Ensure docker image/container
            self.check_image()
            self.start_docker()

            container_root = "/root"
            inputs_root = f"{container_root}/inputs"
            profraw_root = f"{container_root}/profraw"

            # Prepare container directories
            for d in (container_root, inputs_root, profraw_root):
                mk = self.exec_in_docker(["mkdir", "-p", d])
                if mk.returncode != 0:
                    raise RuntimeError(f"Failed to create {d} in container: {mk.stderr.strip()}")

            # Copy acetest_driver into container once
            driver_host = os.path.join(os.path.dirname(__file__), self.driver)
            if not os.path.exists(driver_host):
                raise FileNotFoundError(f"Driver not found: {driver_host}")
            driver_container = f"{container_root}/{self.driver}"
            self.copy_to_docker(driver_host, driver_container)

            # 3) For each interval bucket, copy inputs and execute all .py (single driver call per bucket)
            # bucket directories directly under self.result_dir
            interval_dirs = sorted([p for p in result_root.iterdir() if p.is_dir()])
            for bucket_dir in interval_dirs:
                self.start_docker()
                print(f"Processing bucket: {bucket_dir.name}")
                bucket_label = bucket_dir.name
                # Copy this interval into container
                container_bucket_dir = f"{inputs_root}/{bucket_label}"
                # Copy the bucket directory under inputs_root, resulting in inputs_root/bucket_label
                self.copy_to_docker(str(bucket_dir), inputs_root)

                # Ensure profraw bucket directory exists in container
                mkp = self.exec_in_docker(["mkdir", "-p", f"{profraw_root}/{bucket_label}"])
                if mkp.returncode != 0:
                    raise RuntimeError(f"Failed to mkdir for profraw: {mkp.stderr.strip()}")

                # Run driver once for the entire bucket
                run = self.exec_in_docker([
                    "python", driver_container,
                    "--inputs-dir", container_bucket_dir,
                    "--profraw-root", f"{profraw_root}/{bucket_label}",
                    "--profdata-out", f"{profraw_root}/{bucket_label}/merged.profdata",
                    "--jobs", "16",
                    "--timeout-sec", "5"
                ], workdir=container_root)
                if run.returncode != 0:
                    print(f"[WARN] Driver failed for bucket {bucket_label}: {run.stderr.strip()}\n{run.stdout}")

                # 4) Copy profraws for this bucket back to host
                host_prof_bucket = Path(self.output) / "profdata" / baseline_name / bucket_label
                os.makedirs(host_prof_bucket, exist_ok=True)
                try:
                    # Copy contents of the bucket (avoid duplicating the bucket folder name)
                    self.copy_from_docker(f"{profraw_root}/{bucket_label}/merged.profdata", str(host_prof_bucket))
                    # copy profile.log
                    self.copy_from_docker(f"{profraw_root}/{bucket_label}/profile.log", str(host_prof_bucket))
                except RuntimeError as e:
                    print(f"[WARN] Failed to copy profraws for {bucket_label}: {e}")
                self.stop_docker()
            self.start_docker()
            print("Success")
        except Exception as e:
            print("Fail:", e)
        finally:
            # stop and remove the docker container
            if self.docker_id:
                try:
                    self.stop_docker()
                finally:
                    self.rm_docker()



class TorchCovCollector(DLLCovCollector):
    pass
