# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
ARG BASE_IMAGE="ubuntu"
ARG BASE_IMAGE_TAG="24.04"

ARG BASE_IMAGE="nvcr.io/nvidia/cuda"
ARG BASE_IMAGE_TAG="12.8.1-runtime-ubuntu24.04"

ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
ARG BASE_IMAGE_TAG="25.03-cuda12.8-devel-ubuntu24.04"
ARG NIXL_COMMIT=78695c2900cd7fff506764377386592dfc98e87e
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq libffi-dev python3-dev python3-pip python3-venv git

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir /opt/dynamo && \
    uv venv /opt/dynamo/venv --python 3.12

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update -y && \
    apt-get install -y \
    # NIXL build dependencies
    cmake \
    meson \
    ninja-build \
    pybind11-dev \
    # Rust build dependencies
    libclang-dev \
    # Install utilities
    nvtop \
    tmux \
    vim

# Install NIXL Python module
# TODO: Move gds_path selection based on arch into NIXL build
RUN GIT_TERMINAL_PROMPT=0 git clone https://github.com/ai-dynamo/nixl.git /opt/nixl && cd /opt/nixl && git checkout $NIXL_COMMIT
RUN if [ "$ARCH" = "arm64" ]; then \
        cd /opt/nixl && uv pip install . --config-settings=setup-args="-Dgds_path=/usr/local/cuda/targets/sbsa-linux/"; \
    else \
        cd /opt/nixl && uv pip install . ; \
    fi


RUN uv pip install ai-dynamo[vllm]==0.2.0 genai-perf

RUN huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B
RUN COPY ./examples examples/
RUN COPY ./tutorials tutorials/

ENTRYPOINT [ "/usr/bin/bash" ]
CMD []
