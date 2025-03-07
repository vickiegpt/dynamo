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

import pytest
import subprocess

pytestmark = pytest.mark.gpu


def test_detect_gpu():
    try:
        result = subprocess.run(['ls', '-la', '/proc/driver/nvidia/gpus'], 
                                capture_output=True, text=True, check=True)
        print(f"NVIDIA GPUs found in /proc/driver/nvidia/gpus:\n{result.stdout}")
        gpu_found = len(result.stdout.strip()) > 0
    except:
        gpu_found = False

    assert gpu_found, "No NVIDIA GPUs detected in /proc/driver/nvidia/gpus"