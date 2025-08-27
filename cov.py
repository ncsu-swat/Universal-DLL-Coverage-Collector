import os
import re
import shutil
import subprocess
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, cast

# Optional dependency for parsing llvm-cov HTML output
try:
    from bs4 import BeautifulSoup, Tag  # type: ignore
except Exception:  # pragma: no cover - if bs4 missing, fallback parser will be used
    BeautifulSoup = None  # type: ignore
    Tag = None  # type: ignore

class DLLCovCollector:
    def __init__(self, ver: str, target: str, output: str, dll: str, itv: int, baseline: str, num_parallel: int, filter: Optional[str] = None):
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
        self.num_parallel = num_parallel

    def loop_until_control_c(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Received Ctrl+C, exiting...")

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

    def restart_docker(self):
        # check if the docker is running
        cmd = ["docker", "ps", "-q", "-f", f"name={self.docker_name}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to check Docker container status: {result.stderr.strip()}")
        container_id = result.stdout.strip()
        if container_id:
            print(f"Stopping running Docker container {self.docker_name}")
            self.stop_docker()
            print(f"Removing Docker container {self.docker_name}")
            self.rm_docker()
        print(f"Starting Docker container {self.docker_name}")
        self.start_docker()

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

    # ---- Coverage utilities ----
    def _find_tool(self, names: List[str]) -> Optional[str]:
        """Return the first available tool name inside the container by probing --version."""
        for n in names:
            res = self.exec_in_docker([n, "--version"])  # type: ignore[list-item]
            if res.returncode == 0:
                return n
        return None

    def _find_libtorch(self) -> Optional[str]:
        """Attempt to locate libtorch_cpu.so inside the container (preferred binary for llvm-cov)."""
        preferred = "/root/pytorch/build/lib/libtorch_cpu.so"
        chk = self.exec_in_docker(["bash", "-lc", f"test -f {preferred}"])
        if chk.returncode == 0:
            return preferred
        # Fallback common places
        candidates = [
            "/root/pytorch/torch/lib/libtorch_cpu.so",
        ]
        for c in candidates:
            chk2 = self.exec_in_docker(["bash", "-lc", f"test -f {c}"])
            if chk2.returncode == 0:
                return c
        # Fallback to searching (best-effort)
        find = self.exec_in_docker(["bash", "-lc", "set -o pipefail; find /root -type f -name libtorch_cpu.so 2>/dev/null | head -n1"])
        if find.returncode == 0 and find.stdout.strip():
            return find.stdout.strip().splitlines()[0]
        return None

    def _merge_profdata_in_container(self, out_path: str, inputs: List[str]) -> None:
        tool = self._find_tool(["llvm-profdata", "llvm-profdata-18", "llvm-profdata-17"])
        if not tool:
            raise RuntimeError("llvm-profdata not found in container")
        cmd = [tool, "merge", "--num-threads=0", "--failure-mode=all", "-sparse", "-o", out_path] + inputs
        res = self.exec_in_docker(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"llvm-profdata merge failed: {res.stderr}\n{res.stdout}")

    def _llvm_cov_show_html(self, binaries: List[str], instr_profile: str, html_dir: str) -> None:
        tool = self._find_tool(["llvm-cov", "llvm-cov-18", "llvm-cov-17"])
        if not tool:
            raise RuntimeError("llvm-cov not found in container")
        # ensure output dir exists and is empty
        self.exec_in_docker(["bash", "-lc", f"rm -rf {html_dir} && mkdir -p {html_dir}"])
        cmd = [tool, "show", *binaries, "--show-branches=count", f"--instr-profile={instr_profile}", "-format=html", f"-output-dir={html_dir}"]
        res = self.exec_in_docker(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"llvm-cov show failed: {res.stderr}\n{res.stdout}")

    @staticmethod
    def extract_coverage_data(html_content: str, required_substrings: List[str]) -> Tuple[List[Tuple[str, int, int]], int, int]:
        """
        Parse llvm-cov HTML index and collect covered lines for files whose path
        contains ALL required_substrings (logical AND). Returns:
            - list of (path, covered_lines, total_lines)
            - sum_covered
            - sum_total
        """
        # Fallback simple parser if BeautifulSoup is unavailable
        if BeautifulSoup is None or Tag is None:
            results: List[Tuple[str, int, int]] = []
            sum_cov = 0
            sum_tot = 0
            # Heuristic pairing
            # Find tuples like: some path text followed by (covered/total)
            link_texts = re.findall(r">([^<>]+)</a>", html_content)
            nums = re.findall(r"\((\d+)/(\d+)\)", html_content)
            for path, (cov, tot) in zip(link_texts, nums):
                if required_substrings and not all(sub in path for sub in required_substrings):
                    continue
                c = int(cov)
                t = int(tot)
                results.append((path, c, t))
                sum_cov += c
                sum_tot += t
            return results, sum_cov, sum_tot

        soup = BeautifulSoup(html_content, 'html.parser')
        rows = soup.find_all('tr', class_='light-row')

        results = []
        sum_cov = 0
        sum_tot = 0

        for row in rows:
            if not isinstance(row, Tag):
                continue
            tds = [td for td in row.find_all('td') if isinstance(td, Tag)]
            if len(tds) < 5:
                continue
            link = tds[0].find('a')
            if not isinstance(link, Tag):
                continue
            path = link.get_text(strip=True)
            if required_substrings and not all(sub in path for sub in required_substrings):
                continue
            pre_tag = tds[4].find('pre')
            if not pre_tag:
                continue
            coverage_text = pre_tag.get_text(strip=True)
            m = re.search(r"\((\d+)/(\d+)\)", coverage_text)
            if not m:
                continue
            covered = int(m.group(1))
            total = int(m.group(2))
            results.append((path, covered, total))
            sum_cov += covered
            sum_tot += total

        return results, sum_cov, sum_tot

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
            torch_driver_host = os.path.join(os.path.dirname(__file__), "scripts","torch_driver.py")
            torch_driver = f"{container_root}/scripts/torch_driver.py"
            self.copy_to_docker(torch_driver_host, torch_driver)

            # 3) For each interval bucket, copy inputs and execute all .py (single driver call per bucket)
            # bucket directories directly under self.result_dir
            interval_dirs = sorted([p for p in result_root.iterdir() if p.is_dir()])

            # Prepare cumulative profdata and coverage output paths inside container
            cumulative_prof = f"{profraw_root}/cumulative.profdata"
            coverage_summary: Dict[str, int] = {}

            # Locate libtorch once (binary for llvm-cov)
            libtorch = self._find_libtorch()
            if not libtorch:
                raise RuntimeError("libtorch_cpu.so not found in container")

            html_root = f"{container_root}/cov-html"
            mkh = self.exec_in_docker(["mkdir", "-p", html_root])
            if mkh.returncode != 0:
                raise RuntimeError(f"Failed to create html dir: {mkh.stderr.strip()}")

            for bucket_dir in interval_dirs:
                print(f"Processing bucket: {bucket_dir.name}")
                bucket_label = bucket_dir.name
                host_prof_bucket = Path(self.output) / "profdata" / baseline_name / bucket_label
                host_prof_bucket.mkdir(parents=True, exist_ok=True)
                host_prof_file = host_prof_bucket / "merged.profdata"

                # Ensure profraw bucket directory exists in container
                mkp = self.exec_in_docker(["mkdir", "-p", f"{profraw_root}/{bucket_label}"])
                if mkp.returncode != 0:
                    raise RuntimeError(f"Failed to mkdir for profraw: {mkp.stderr.strip()}")

                if host_prof_file.exists():
                    # Reuse existing result; copy into container
                    print(f"[resume] Found existing profdata for {bucket_label}, reusing")
                    self.copy_to_docker(str(host_prof_file), f"{profraw_root}/{bucket_label}/merged.profdata")
                else:
                    # Copy this interval into container and run driver
                    container_bucket_dir = f"{inputs_root}/{bucket_label}"
                    self.copy_to_docker(str(bucket_dir), inputs_root)
                    run = self.exec_in_docker([
                        "python", driver_container,
                        "--inputs-dir", container_bucket_dir,
                        "--profraw-root", f"{profraw_root}/{bucket_label}",
                        "--profdata-out", f"{profraw_root}/{bucket_label}/merged.profdata",
                        "--jobs", f"{self.num_parallel}",
                        "--timeout-sec", f"{self.itv*2}"
                    ], workdir=container_root)
                    if run.returncode != 0:
                        print(f"[WARN] Driver failed for bucket {bucket_label}: {run.stderr.strip()}\n{run.stdout}")
                    # Copy fresh results to host for future resume
                    try:
                        self.copy_from_docker(f"{profraw_root}/{bucket_label}/merged.profdata", str(host_prof_bucket))
                        self.copy_from_docker(f"{profraw_root}/{bucket_label}/profile.log", str(host_prof_bucket))
                    except RuntimeError as e:
                        print(f"[WARN] Failed to copy profraws for {bucket_label}: {e}")
                
                # Merge cumulatively: previous cumulative + current bucket -> new cumulative
                bucket_prof = f"{profraw_root}/{bucket_label}/merged.profdata"
                exists = self.exec_in_docker(["bash", "-lc", f"test -f {cumulative_prof}"])
                if exists.returncode == 0:
                    tmp_out = f"{profraw_root}/cumulative.tmp.profdata"
                    self._merge_profdata_in_container(tmp_out, [cumulative_prof, bucket_prof])
                    mv = self.exec_in_docker(["bash", "-lc", f"mv -f {tmp_out} {cumulative_prof}"])
                    if mv.returncode != 0:
                        raise RuntimeError(f"Failed to update cumulative profdata: {mv.stderr}")
                else:
                    cp = self.exec_in_docker(["bash", "-lc", f"cp -f {bucket_prof} {cumulative_prof}"])
                    if cp.returncode != 0:
                        raise RuntimeError(f"Failed to init cumulative profdata: {cp.stderr}")

                # Generate HTML coverage for cumulative profile and parse ATen coverage
                html_dir = f"{html_root}/{bucket_label}"
                try:
                    self._llvm_cov_show_html([libtorch], cumulative_prof, html_dir)
                    cat = self.exec_in_docker(["bash", "-lc", f"cat {html_dir}/index.html"])
                    if cat.returncode != 0:
                        raise RuntimeError(f"Failed to read index.html: {cat.stderr}")
                    html_content = cat.stdout
                    rows, sum_cov, sum_tot = self.extract_coverage_data(html_content, ["aten/src/ATen/native"])
                    coverage_summary[bucket_label] = sum_cov

                    # Generate a host-side text report per iteration
                    try:
                        report_lines: List[str] = []
                        report_lines.append(f"Bucket: {bucket_label}")
                        pct = (100.0 * sum_cov / sum_tot) if sum_tot > 0 else 0.0
                        report_lines.append(f"Total (aten/src/ATen/native): {sum_cov}/{sum_tot} ({pct:.2f}%)")
                        report_lines.append("")
                        report_lines.append("Files:")
                        # Sort files by covered desc, then total desc
                        for path, covered, total in sorted(rows, key=lambda x: (-x[1], -x[2], x[0])):
                            file_pct = (100.0 * covered / total) if total > 0 else 0.0
                            report_lines.append(f"- {path}: {covered}/{total} ({file_pct:.2f}%)")
                        report_content = "\n".join(report_lines) + "\n"
                        # Ensure host bucket dir exists and write file
                        host_prof_bucket.mkdir(parents=True, exist_ok=True)
                        with open(host_prof_bucket / "coverage.txt", "w", encoding="utf-8") as f:
                            f.write(report_content)
                    except Exception as werr:
                        print(f"[WARN] Failed to write coverage.txt for {bucket_label}: {werr}")
                except Exception as e:
                    print(f"[WARN] Coverage computation failed for {bucket_label}: {e}")
                
                # 4) Nothing else to copy if we resumed; fresh runs already copied above
            # Print final per-bucket cumulative covered-line counts
            if coverage_summary:
                def key_fn(s: str) -> Tuple[int, int]:
                    a, b = s.split('-')
                    return (int(a), int(b))
                for k in sorted(coverage_summary.keys(), key=key_fn):
                    print(f"{k}: {coverage_summary[k]}")
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

class TFCovCollector(DLLCovCollector):
    pass
