import subprocess
from typing import Optional

class DLLCovCollector:
    def __init__(self, ver: str, target: str, output: str, baseline: Optional[str] = None):
        self.ver = ver
        self.target = target
        self.output = output
        self.docker_image = f"ncsu-swat/torch-{self.ver}-instrumented"
        self.docker_name = f"torch_cov_{self.ver}-{baseline}" if baseline else f"torch_cov_{self.ver}"
        self.docker_id = ""
        
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
        

    def collect(self):
        try:
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
