# Nightcall training image (Linux x86_64, CPU PyTorch).
# Build:   docker build -f docker/Dockerfile -t nightcall-train:cpu .
# Compose: docker compose run --rm train
# Run:     docker run --rm -e NUM_ENVS=4 -v nightcall_ckpts:/workspace/checkpoints nightcall-train:cpu

FROM ubuntu:22.04 AS builder

COPY docker/entrypoint-train.sh /usr/local/bin/entrypoint-train.sh
RUN chmod +x /usr/local/bin/entrypoint-train.sh

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf autoconf-archive automake gettext libtool m4 ffmpeg \
    build-essential linux-libc-dev ninja-build git curl zip unzip tar pkg-config \
    python3 python3-pip python3-venv python3-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && command -v autoconf && command -v automake && command -v libtoolize && command -v m4

ARG CMAKE_VERSION=3.31.10
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
    | tar -xz -C /usr/local --strip-components=1

RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh

WORKDIR /workspace
COPY . .

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET=x64-linux \
    -DNIGHTCALL_BUILD_PYTHON=ON \
    -DNIGHTCALL_BUILD_TESTS=OFF \
    && cmake --build build --target nightcall_sim

ENV PATH="/workspace/build/vcpkg_installed/x64-linux/tools/python3:${PATH}"

RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r python/requirements.txt

ENV PYTHONPATH=/workspace/build/python:/workspace/python/training
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint-train.sh"]
CMD []
