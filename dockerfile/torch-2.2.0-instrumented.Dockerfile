# Base image: Ubuntu 24.04 LTS (x86_64)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=64
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    software-properties-common \
    wget \
    # Add deadsnakes PPA for newer Python versions
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    cmake \
    btop \
    # python3.10-distutils is not available on Ubuntu 24.04
    # Install pip via get-pip.py
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    # Install LLVM and Clang for coverage
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 18 \
    && rm llvm.sh \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python and python3, and lld the default linker, clang-18 as the Clang
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/ld ld /usr/bin/lld-18 1 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/clang-18 1 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-18 1 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 1

# Install Python build dependencies
RUN python3 -m pip install --no-cache-dir \
    astunparse \
    ninja \
    pyyaml \
    setuptools \
    wheel \
    cffi \
    typing_extensions \
    requests \
    numpy==1.23.5



# Clone PyTorch from source
WORKDIR /root
RUN git clone --branch v2.2.0 --depth 1 --recurse-submodules --shallow-submodules https://github.com/pytorch/pytorch.git


# Build and install PyTorch with coverage
WORKDIR /root/pytorch
ENV CC=clang-18 CXX=clang++-18 USE_CPP_CODE_COVERAGE=1 \
    CMAKE_C_FLAGS="-g -O0 -fprofile-instr-generate -fcoverage-mapping" \
    CMAKE_CXX_FLAGS="-g -O0 -fprofile-instr-generate -fcoverage-mapping -Wno-error"


# # Note: The build process can take a very long time and consume a lot of memory.
# # We are disabling many features to speed up the build.
RUN BUILD_TEST=0 USE_MKLDNN=0 USE_OPENMP=0 USE_CUDA=0 USE_NCCL=0 \
    python3 setup.py develop


WORKDIR /root

CMD ["bash"]
