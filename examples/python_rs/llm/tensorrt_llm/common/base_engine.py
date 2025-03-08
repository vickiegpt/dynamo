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
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

from common.parser import LLMAPIConfig
from common.processor import ChatProcessor, CompletionsProcessor
from common.kv_cache_event_publisher import KVCacheEventPublisher
from triton_distributed.llm import KvMetricsPublisher
from tensorrt_llm._torch import LLM
from tensorrt_llm.logger import logger
from transformers import AutoTokenizer
from typing import Callable

from queue import Queue
import weakref
import traceback

class ChatProcessorMixin:
    def __init__(self, engine_config: LLMAPIConfig):
        self._engine_config = engine_config
        logger.info(f"Using LLM API config: {engine_config.to_dict()}")
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

        if self._engine_config.extra_args.get("tokenizer", None):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.extra_args.get("tokenizer", None)
            )

        self.chat_processor = ChatProcessor(self._model_name, self._tokenizer)
        self.completions_processor = CompletionsProcessor(self._model_name)


class ManagedThread(threading.Thread):
    """ A thread that will put exceptions into an external queue if the task fails.

    There are two approaches to stop the thread:
        1. Set stop_event to stop the loop
        2. Let `task` return False

    Args:
        task (Callable[..., bool]): The task to run repeatedly in the thread, should return False if break the loop.
        error_queue (Queue): The queue to put exceptions into if the task fails.
        name (str): The name of the thread.
        **kwargs: The arguments to pass to the task
    """

    def __init__(self,
                 task: Callable[..., bool],
                 error_queue: Queue,
                 name: Optional[str] = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None,
                 **kwargs):
        super().__init__(name=name)
        self.task = task
        self.error_queue = error_queue
        self.kwargs = kwargs
        self.loop = loop
        self.daemon = True

        self.stop_event = threading.Event()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def run(self):

        while not self.stop_event.is_set():
            task = self.task
            if isinstance(task, weakref.WeakMethod):
                task = task()
                if task is None:
                    # Normally, this should not happen.
                    logger.warning("WeakMethod is expired.")
                    break

            try:
                future = asyncio.run_coroutine_threadsafe(task(**self.kwargs), self.loop)
                _ = future.result()
            except Exception as e:
                logger.error(
                    f"Error in thread {self.name}: {e}\n{traceback.format_exc()}"
                )
                self.error_queue.put(e)

        logger.info(f"Thread {self.name} stopped.")

    def stop(self):
        self.stop_event.set()


class BaseTensorrtLLMEngine(ChatProcessorMixin):
    def __init__(self, engine_config: LLMAPIConfig, worker_id: str, publish_stats: bool= False, publish_kv_cache_events: bool= False):
        super().__init__(engine_config)
        logger.info(f"Using LLM API config: {self._engine_config}")
        self._worker_id = worker_id
        self._publish_stats = publish_stats
        self._publish_kv_cache_events = publish_kv_cache_events
        self._init_engine()

    def _init_engine(self):
        logger.info("Initializing engine")
        # Run the engine in a separate thread running the AsyncIO event loop.
        self._llm_engine: Optional[Any] = None
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._event_thread = threading.Thread(
            target=asyncio.run, args=(self._run_llm_engine(),)
        )

        self.publish_kv_cache_events_thread = None
        self.publish_stats_thread = None
        
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

        self._error_queue = Queue()

        try:
            if self._publish_stats:
                self._init_publish_metrics_thread()

            if self._publish_kv_cache_events:
                self._init_publish_kv_cache_events_thread()
        except Exception as e:
            logger.error(f"Failed to initialize publish metrics threads: {e}")
            raise e

    def _init_publish_metrics_thread(self):
        """
        self.kv_metrics_publisher = KvMetricsPublisher()
        request_active_slots = 0
        request_total_slots = 4
        kv_active_block = 0
        kv_total_blocks = 4
        # [NOTE] Publish initial metrics to the metrics publisher
        # so that the router can select this worker.
        self.kv_metrics_publisher.publish(
            request_active_slots,
            request_total_slots,
            kv_active_block,
            kv_total_blocks,
        )
        """

        # Prepare threads for publishing stats but don't start them yet. 
        # TRTLLM needs to start generating tokens first before stats
        # can be retrieved.
        self.publish_stats_thread = ManagedThread(
            self.publish_stats_task,
            error_queue=self._error_queue,
            name="publish_stats_thread",
            )
        
    def _init_publish_kv_cache_events_thread(self):
        # self.kv_cache_events_publisher = KVCacheEventPublisher()
        # Prepare threads for publishing kv cache events but don't start them yet. 
        # TRTLLM needs to start generating tokens first before kv cache events
        # can be retrieved.
        self.publish_kv_cache_events_thread = ManagedThread(
            self.publish_kv_cache_events_task,
            error_queue=self._error_queue,
            name="publish_kv_cache_events_thread",
            )

    async def publish_stats_task(self):
        '''
        Publish stats to the metrics publisher.
        '''
        logger.info(f"Tanmay:: Stats: Starting")
        stats = self._llm_engine.get_stats_async(timeout=5)
        async for stat in stats:
            logger.info(f"Tanmay:: Stats: {stat}")

        await asyncio.sleep(5)

        return True

    async def publish_kv_cache_events_task(self):   
        '''
        Publish kv cache events to the events publisher.
        '''
        logger.info(f"Tanmay:: Events: Starting")
        events = self._llm_engine.get_kv_cache_events_async(timeout=5)
        async for event in events:
            logger.info(f"Tanmay:: Events: {event}")

        await asyncio.sleep(5)

        return True

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

                # Stop the publishing threads
                if self.publish_stats_thread and self.publish_stats_thread.is_alive():
                    self.publish_stats_thread.stop()
                    self.publish_stats_thread.join()
                if self.publish_kv_cache_events_thread and self.publish_kv_cache_events_thread.is_alive():
                    self.publish_kv_cache_events_thread.stop()
                    self.publish_kv_cache_events_thread.join()


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
