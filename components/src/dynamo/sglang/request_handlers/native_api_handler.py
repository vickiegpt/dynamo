# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SGLang Native APIs: https://docs.sglang.ai/basic_usage/native_api.html
# Code: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py

import asyncio
import logging
from typing import List, Optional, Tuple

import sglang as sgl
from sglang.srt.managers.io_struct import ProfileReqInput

from dynamo._core import Component


class NativeApiHandler:
    """Mixin to add sglang native API endpoints to worker handlers"""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        metrics_labels: Optional[List[Tuple[str, str]]] = None,
    ):
        self.component = component
        self.engine = engine
        self.metrics_labels = metrics_labels
        self.native_api_tasks = []

    async def init_native_apis(
        self,
    ) -> List[asyncio.Task]:
        """
        Initialize and register native API endpoints.
        Returns list of tasks to be gathered.
        """
        logging.info("Initializing native SGLang API endpoints")

        self.tm = self.engine.tokenizer_manager

        tasks = []

        model_info_ep = self.component.endpoint("get_model_info")
        start_profile_ep = self.component.endpoint("start_profile")
        stop_profile_ep = self.component.endpoint("stop_profile")
        tasks.extend(
            [
                model_info_ep.serve_endpoint(
                    self.get_model_info,
                    graceful_shutdown=True,
                    metrics_labels=self.metrics_labels,
                    http_endpoint_path="/get_model_info",
                ),
                start_profile_ep.serve_endpoint(
                    self.start_profile,
                    graceful_shutdown=True,
                    metrics_labels=self.metrics_labels,
                    http_endpoint_path="/start_profile",
                ),
                stop_profile_ep.serve_endpoint(
                    self.stop_profile,
                    graceful_shutdown=True,
                    metrics_labels=self.metrics_labels,
                    http_endpoint_path="/stop_profile",
                ),
            ]
        )

        self.native_api_tasks = tasks
        logging.info(f"Registered {len(tasks)} native API endpoints")
        return tasks

    async def get_model_info(self, request: dict):
        result = {
            "model_path": self.tm.server_args.model_path,
            "tokenizer_path": self.tm.server_args.tokenizer_path,
            "preferred_sampling_params": self.tm.server_args.preferred_sampling_params,
            "weight_version": self.tm.server_args.weight_version,
        }

        yield {"data": [result]}

    async def start_profile(self, request: dict):
        try:
            obj = ProfileReqInput.model_validate(request)
        except Exception:
            obj = None

        if obj is None:
            obj = ProfileReqInput()

        output_dir = obj.output_dir or f"profile_{self.tm.server_args.model_path}"

        await self.tm.start_profile(
            output_dir=output_dir,
            start_step=obj.start_step,
            num_steps=obj.num_steps,
            activities=obj.activities,
            with_stack=obj.with_stack,
            record_shapes=obj.record_shapes,
            profile_by_stage=obj.profile_by_stage,
        )

        yield {"data": [{"status": "started profile"}]}

    async def stop_profile(self, request: dict):
        asyncio.create_task(self.tm.stop_profile())
        yield {
            "data": [
                {
                    "status": (
                        "Stopped profile. This might take a long time to complete. "
                        f"Results should be available in the 'profile_{self.tm.server_args.model_path}' directory."
                    )
                }
            ]
        }
