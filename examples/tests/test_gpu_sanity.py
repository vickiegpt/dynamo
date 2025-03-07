# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess

import pytest

pytestmark = pytest.mark.gpu


def test_detect_gpu():
    try:
        find_cmd = [
            "find",
            "/proc/driver/nvidia/gpus",
            "-name",
            "information",
            "-type",
            "f",
            "-exec",
            "cat",
            "{}",
            ";",
        ]
        info_result = subprocess.run(find_cmd, capture_output=True, text=True)
        gpu_found = len(info_result.stdout.strip()) > 0
        if gpu_found:
            print(f"\nGPU information files content:\n{info_result.stdout}")
        else:
            print("No GPU information files found in /proc/driver/nvidia/gpus")
    except Exception as e:
        print(f"Error checking for GPU: {e}")
        gpu_found = False

    assert gpu_found, "No NVIDIA GPUs detected!"
