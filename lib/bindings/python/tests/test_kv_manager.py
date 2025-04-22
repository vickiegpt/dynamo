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


import asyncio

from dynamo.llm import KvManager

# pytestmark = pytest.mark.pre_merge


async def test_initialization():
    kv_manager = KvManager("CPU")
    kv_manager = KvManager("cpu")
    kv_manager = KvManager("CUDA")
    kv_manager = KvManager("cuda")
    kv_manager = KvManager("CUDA:0")
    kv_manager = KvManager("cuda:0")
    # kv_manager = KvManager("CUDA", pin_memory=True)
    print(kv_manager)


if __name__ == "__main__":
    asyncio.run(test_initialization())
