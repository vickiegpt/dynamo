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

import os
import logging

from pydantic import BaseModel

from .encode_worker_onnx import EncoderConfig
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)

config = ServiceConfig.get_instance()
encoder_config = EncoderConfig(**config.get("EncodeWorker", {}))

if encoder_config.encode_framework == "onnx":
    from .encode_worker_onnx import EncodeWorker
elif encoder_config.encode_framework == "pytorch":
    from .encode_worker import EncodeWorker
else:
    error_msg = (
        f"Unsupported encode_framework: '{encoder_config.encode_framework}'. "
        "Valid options are 'onnx' or 'pytorch'."
    )
    logger.error(error_msg)
    raise ImportError(error_msg)

__all__ = ["EncodeWorker"] 