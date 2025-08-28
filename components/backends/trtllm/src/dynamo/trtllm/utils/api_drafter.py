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

import logging
from typing import List

from tensorrt_llm._torch.speculative.external_api import APIDrafter

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


class DynamoAPIDrafter(APIDrafter):
    """
    Custom Dynamo drafter to support internal Dynamo endpoints instead of only HTTP endpoints.
    """

    def __init__(self, spec_config, draft_client):
        super().__init__(spec_config)
        if draft_client is None:
            raise ValueError(
                "next_client must be provided when using parallel speculative decoding"
            )
        self.client = draft_client
        self.max_draft_len = spec_config.max_draft_len

    async def get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        end_id: int,
        max_sequence_length: int,
    ) -> List[int]:
        request_data = {
            "token_ids": prefix,
            "sampling_options": {},
            "stop_conditions": {
                "max_tokens": self.max_draft_len,
            },
        }

        draft_tokens: List[int] = []
        try:
            response = await self.client.round_robin(request_data)
            async for chunk in response:
                chunk_data = chunk.data()
                if chunk_data.get("finish_reason"):
                    break
                draft_tokens.extend(chunk_data.get("token_ids", []))
                if len(draft_tokens) >= self.max_draft_len:
                    break
            return draft_tokens[: self.max_draft_len]
        except Exception as e:
            logging.error(
                f"Failed to get draft tokens for Dynamo endpoint {self.endpoint} with error: {e}"
            )
            raise e
