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
import ctypes
import threading
from contextlib import asynccontextmanager
from ctypes import c_char_p, c_int64, c_uint32
from typing import Any, Optional

from common.parser import LLMAPIConfig
from common.processor import ChatProcessor, CompletionsProcessor
from tensorrt_llm._torch import LLM
from tensorrt_llm.logger import logger
from transformers import AutoTokenizer

from triton_distributed.llm import KvMetricsPublisher


class TritonResult:
    OK = 0
    ERR = 1


class ChatProcessorMixin:
    def __init__(self, engine_config: LLMAPIConfig):
        self._engine_config = engine_config
        logger.info(f"Using LLM API config: {self._engine_config}")
        # model name for chat processor
        self._model_name = self._engine_config.model_name
        logger.info(f"Set model name: {self._model_name}")

        # model for LLMAPI input
        self._model = self._model_name

        if self._engine_config.model_path:
            self._model = self._engine_config.model_path
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.model_path
            )
            logger.info(f"Using model from path: {self._engine_config.model_path}")
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.model_name
            )

        self._init_engine()

        if self._engine_config.extra_args.get("tokenizer", None):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.extra_args.get("tokenizer", None)
            )

        self.chat_processor = ChatProcessor(self._model_name, self._tokenizer)
        self.completions_processor = CompletionsProcessor(self._model_name)


class WorkerId:
    def __init__(self, completions_lease_id: str, chat_lease_id: str):
        self.completions_lease_id = completions_lease_id
        self.chat_lease_id = chat_lease_id

    def id(self):
        return "%s^%s" % (self.completions_lease_id, self.chat_lease_id)


class BaseTensorrtLLMEngine(ChatProcessorMixin, WorkerId):
    def __init__(
        self,
        engine_config: LLMAPIConfig,
        metrics_publisher: KvMetricsPublisher,
        worker_id: WorkerId,
    ):
        super().__init__(engine_config)
        logger.info(f"Using LLM API config: {self.engine_config}")
        self._init_engine()
        self.worker_id = worker_id
        self.metrics_publisher = metrics_publisher
        if metrics_publisher is not None:
            self.lib = ctypes.CDLL("/opt/triton/llm_binding/lib/libtriton_llm_capi.so")
            self.lib.triton_llm_init.argtypes = [c_char_p, c_char_p, c_int64]
            self.lib.triton_llm_init.restype = c_uint32
            result = self.lib.triton_llm_init(
                "triton-init".encode(), "tensorrt-llm-ctx".encode(), worker_id.id()
            )
            if result == TritonResult.OK:
                logger.info(
                    "KVCacheEventManager initialized successfully. Ready to publish KV Cache Events"
                )
            else:
                logger.error("KVCacheEventManager initialization failed!")

            self.lib.triton_kv_event_publish_stored.argtypes = [
                ctypes.c_uint64,  # event_id
                ctypes.POINTER(ctypes.c_uint32),  # token_ids
                ctypes.POINTER(ctypes.c_size_t),  # num_block_tokens
                ctypes.POINTER(ctypes.c_uint64),  # block_ids
                ctypes.c_size_t,  # num_blocks
                ctypes.POINTER(ctypes.c_uint64),  # parent_hash
                ctypes.c_uint64,  # lora_id
            ]
            self.lib.triton_kv_event_publish_stored.restype = (
                ctypes.c_uint32
            )  # triton_llm_result_t

            self.lib.triton_kv_event_publish_removed.argtypes = [
                ctypes.c_uint64,  # event_id
                ctypes.POINTER(ctypes.c_uint64),  # block_ids
                ctypes.c_size_t,  # num_blocks
            ]
            self.lib.triton_kv_event_publish_removed.restype = (
                ctypes.c_uint32
            )  # triton_llm_result_t

            # TODO: Tanmay Get these from the engine config
            self.request_active_slots = 0
            self.request_total_slots = 4
            self.kv_active_block = 0
            self.kv_total_blocks = 4
            # [NOTE] Now that the component must has proper metrics reported
            # to be properly selected by the router
            self.metrics_publisher.publish(
                self.request_active_slots,
                self.request_total_slots,
                self.kv_active_block,
                self.kv_total_blocks,
            )

    def _init_engine(self):
        logger.info("Initializing engine")
        # Run the engine in a separate thread running the AsyncIO event loop.
        self._llm_engine: Optional[Any] = None
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._event_thread = threading.Thread(
            target=asyncio.run, args=(self._run_llm_engine(),)
        )
        self._event_thread.start()
        with self._llm_engine_start_cv:
            while self._llm_engine is None:
                self._llm_engine_start_cv.wait()

        # The 'threading.Thread()' will not raise the exception here should the engine
        # failed to start, so the exception is passed back via the engine variable.
        if isinstance(self._llm_engine, Exception):
            e = self._llm_engine
            logger.error(f"Failed to start engine: {e}")
            if self._event_thread is not None:
                self._event_thread.join()
                self._event_thread = None
            raise e

    async def _run_llm_engine(self):
        # Counter to keep track of ongoing request counts.
        self._ongoing_request_count = 0

        @asynccontextmanager
        async def async_llm_wrapper():
            # Create LLM in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                llm = await loop.run_in_executor(
                    None,
                    lambda: LLM(model=self._model, **self._engine_config.to_dict()),
                )
                yield llm
            finally:
                if "llm" in locals():
                    # Run shutdown in a thread to avoid blocking
                    await loop.run_in_executor(None, llm.shutdown)

        try:
            async with async_llm_wrapper() as engine:
                # Capture the engine event loop and make it visible to other threads.
                self._event_loop = asyncio.get_running_loop()

                # Signal the engine is started and make it visible to other threads.
                with self._llm_engine_start_cv:
                    self._llm_engine = engine
                    self._llm_engine_start_cv.notify_all()

                logger.info("Engine loaded and ready to serve...")

                # Wait for the engine shutdown signal.
                await self._llm_engine_shutdown_event.wait()

                # Wait for the ongoing requests to complete.
                while self._ongoing_request_count > 0:
                    logger.info(
                        "Awaiting remaining {} requests".format(
                            self._ongoing_request_count
                        )
                    )
                    await asyncio.sleep(1)

                # Cancel all tasks in the event loop.
                for task in asyncio.all_tasks(loop=self._event_loop):
                    if task is not asyncio.current_task():
                        task.cancel()

        except Exception as e:
            # Signal and pass the exception back via the engine variable if the engine
            # failed to start. If the engine has started, re-raise the exception.
            with self._llm_engine_start_cv:
                if self._llm_engine is None:
                    self._llm_engine = e
                    self._llm_engine_start_cv.notify_all()
                    return
            raise e

        self._llm_engine = None
        logger.info("Shutdown complete")

    async def cooldown(self):
        while True:
            await asyncio.sleep(5)
            self.request_active_slots = max(0, self.request_active_slots - 1)
            self.kv_active_block = max(0, self.kv_active_block - 1)
            self.metrics_publisher.publish(
                self.request_active_slots,
                self.request_total_slots,
                self.kv_active_block,
                self.kv_total_blocks,
            )
