# Nightcall training image (Linux x86_64, CPU PyTorch).
# Build from repo root:  docker build -f docker/Dockerfile -t nightcall-train .
# Run training:          docker run --rm -v nightcall_ckpts:/workspace/checkpoints nightcall-train \
#                          python python/training/train.py --total-steps 100000 --num-envs 4

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl zip unzip tar pkg-config \
    python3 python3-pip python3-venv python3-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh

WORKDIR /workspace
COPY . .

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET=x64-linux \
    -DNIGHTCALL_BUILD_PYTHON=ON \
    -DNIGHTCALL_BUILD_TESTS=OFF \
    -DPython_EXECUTABLE=/usr/bin/python3 \
    && cmake --build build --target nightcall_sim

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r python/requirements.txt

ENV PYTHONPATH=/workspace/build/python:/workspace/python/training
WORKDIR /workspace

CMD ["python3", "python/training/train.py", "--help"]
