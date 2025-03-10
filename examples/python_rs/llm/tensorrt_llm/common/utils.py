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
import traceback
import weakref
from queue import Queue
from typing import AsyncIterator, Callable, Optional, Union

from common.protocol import Tokens
from tensorrt_llm.logger import logger

from dynamo.llm import KvRouter
from dynamo.runtime import dynamo_endpoint

logger.set_level("debug")


class Scheduler:
    def __init__(self, kv_router: KvRouter):
        self.kv_router = kv_router

    @dynamo_endpoint(Tokens, str)
    async def generate(self, request) -> AsyncIterator[str]:
        lora_id = 0
        worker_id = None
        try:
            worker_id = await self.kv_router.schedule(request.tokens, lora_id)
        except Exception:
            logger.warning(f"Error during worker selection: {traceback.format_exc()}")
            worker_id = ""

        yield str(worker_id)


async def get_worker_id(scheduler: Scheduler, request, tokenizer) -> str:
    # NOTE: this will increase TTFT since we are encoding the prompt here
    # prompt is also encoded in the worker.
    # TODO: we need to implement our own request processing and protocols to send only token ids to llmapi worker.
    token_ids = tokenizer.encode(request.prompt)
    worker_id_generator: AsyncIterator = scheduler.generate(
        Tokens(tokens=token_ids).model_dump_json()
    )

    worker_id = await worker_id_generator.__anext__()  # only one worker id is returned

    logger.debug(f"Scheduling to worker_id: {worker_id}")
    return worker_id


class ManagedThread(threading.Thread):
    def __init__(
        self,
        task: Optional[Union[Callable[..., bool], weakref.WeakMethod]],
        error_queue: Optional[Queue] = None,
        name: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        **kwargs,
    ):
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
            task: Optional[Union[Callable[..., bool], weakref.WeakMethod]] = self.task
            if isinstance(task, weakref.WeakMethod):
                task = task()
                if task is None:
                    # Normally, this should not happen.
                    logger.warning("WeakMethod is expired.")
                    break

            if task is None:
                break

            try:
                if self.loop is None:
                    logger.error("[ManagedThread] Loop not initialized!")
                    break
                future = asyncio.run_coroutine_threadsafe(
                    task(**self.kwargs), self.loop
                )
                _ = future.result()
            except Exception as e:
                logger.error(
                    f"Error in thread {self.name}: {e}\n{traceback.format_exc()}"
                )
                if self.error_queue is not None:
                    self.error_queue.put(e)

        logger.info(f"Thread {self.name} stopped.")

    def stop(self):
        self.stop_event.set()


async def wait_for_workers(client, min_workers: int):
    wait_task = client.wait_for_endpoints()
    await asyncio.sleep(1)

    while not wait_task.done():
        logger.info("Waiting for workers to be ready...")
        await asyncio.sleep(5)

    while len(client.endpoint_ids()) < min_workers:
        logger.info(
            f"Waiting for more workers... Current: {len(client.endpoint_ids())}, Required: {min_workers}"
        )
        await asyncio.sleep(5)

    logger.info(
        f"Required number of workers ({min_workers}) are ready:\n"
        + "\n".join(f"id: {id}" for id in client.endpoint_ids())
    )
