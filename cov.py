import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple, cast

class DLLCovCollector:
    def __init__(self, ver: str, target: str, output: str, dll: str, itv: int, baseline: Optional[str] = None):
        self.ver = ver
        self.target = target
        self.output = output
        self.dll = dll
        self.itv = itv
        self.docker_image = f"ncsu-swat/torch-{self.ver}-instrumented"
        self.docker_name = f"torch_cov_{self.ver}-{baseline}" if baseline else f"torch_cov_{self.ver}"
        self.docker_id = ""
        self.result_dir = f"{output}/_tmp/{baseline}"
        
        
    def check_image(self):
        # use docker image to check
        cmd = ["docker", "images", "-q", self.docker_image]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to find Docker image: {result.stderr.strip()}, please run build.sh.")
        return bool(result.stdout.strip())

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
        

    def classify_python_files_with_itv(self, pattern: Optional[str] = None) -> Dict[str, int]:
        """
        Walk self.target for Python files, obtain their creation time from file metadata,
        bucket them using self.itv seconds starting at the minimum timestamp found, and
        move files into self.result_dir/<start-end>/... preserving their relative paths.

        Notes:
        - On Linux, true creation time (birth time) may be unavailable; we fall back to
          st_ctime (inode change time) or st_mtime when st_birthtime is not present.
        - If multiple files share the same name from different subdirectories, we
          preserve the relative directory structure under each bucket to avoid collisions.

        Optional filtering:
        - pattern: Optional regex string. If provided, only files whose path relative to
          self.target matches the regex will be included.

        Returns:
            Dict[str, int]: Mapping of bucket label (e.g., "0-60") to the number of files moved.
        """
        # Validate inputs
        if not self.target or not os.path.isdir(self.target):
            raise FileNotFoundError(f"Target directory does not exist: {self.target}")
        if self.itv <= 0:
            raise ValueError(f"Interval (itv) must be a positive integer (seconds), got: {self.itv}")

        os.makedirs(self.result_dir, exist_ok=True)

        regex = re.compile(pattern) if pattern else None

        def get_creation_time(path: str) -> float:
            st = os.stat(path)
            # Prefer st_birthtime if available (macOS, some filesystems), else fall back
            # to st_ctime (change time on Unix) and, as a last resort, st_mtime.
            birth = getattr(st, "st_birthtime", None)
            if birth is not None:
                return float(birth)
            # On Linux, getctime == st_ctime which is change time; still acceptable as a proxy
            try:
                return float(os.path.getctime(path))
            except Exception:
                return float(st.st_mtime)

        # 1) Collect all Python files and their times
        py_files: List[Tuple[str, float]] = []
        for root, _dirs, files in os.walk(self.target):
            for name in files:
                if not name.endswith(".py"):
                    continue
                # Skip typical cache files
                if name == "__init__.pyc" or name.endswith(".pyc"):
                    continue
                full = os.path.join(root, name)
                # Apply regex filter if provided (match against path relative to target)
                if regex:
                    rel_path = os.path.relpath(full, self.target)
                    if not regex.search(rel_path):
                        continue
                try:
                    ts = get_creation_time(full)
                except FileNotFoundError:
                    # File may have disappeared between walk and stat; skip it
                    continue
                py_files.append((full, ts))

        if not py_files:
            print("No Python files found to classify.")
            return cast(Dict[str, int], {})

        # 2) Determine bucket boundaries relative to the minimum timestamp
        min_ts = min(ts for _p, ts in py_files)
        max_ts = max(ts for _p, ts in py_files)
        # Max range in seconds from min
        max_span = int(max_ts - min_ts)
        # Precompute bucket label helper
        def bucket_label(ts: float) -> str:
            offset = max(0, int(ts - min_ts))
            start = (offset // self.itv) * self.itv
            end = start + self.itv
            return f"{start}-{end}"

        # 3) Move files into bucketed directories, preserving relative paths
        counts: Dict[str, int] = {}
        for full, ts in py_files:
            label = bucket_label(ts)
            counts[label] = counts.get(label, 0) + 1

            rel_parent = os.path.relpath(os.path.dirname(full), self.target)
            # If file is directly under target, rel_parent == '.'; handle that gracefully
            dest_dir = os.path.join(self.result_dir, label)
            if rel_parent and rel_parent != ".":
                dest_dir = os.path.join(dest_dir, rel_parent)
            os.makedirs(dest_dir, exist_ok=True)

            dest_path = os.path.join(dest_dir, os.path.basename(full))

            # If a file with the same name already exists, add a numeric suffix
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(os.path.basename(full))
                i = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{base}__{i}{ext}")
                    i += 1

            shutil.move(full, dest_path)

        # Optionally, report the overall range for visibility
        total = len(py_files)
        bucket_count = len(counts)
        print(
            f"Classified and moved {total} Python files into {bucket_count} buckets under '{self.result_dir}'.\n"
            f"Time span: 0-{((max_span // self.itv) + 1) * self.itv} seconds from minimum timestamp."
        )
        return counts
  

    def collect(self):
        try:
            self.classify_python_files_with_itv()
            self.check_image()
            self.start_docker()
            self.stop_docker()
            print("Sucess")
        except:
            print("Fail")
        finally:
            # rm the docker and purge the volumn
            if self.docker_id:
                self.rm_docker()
