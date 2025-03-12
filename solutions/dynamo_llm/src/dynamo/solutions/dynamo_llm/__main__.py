# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time

from ._parser import parse_known_args
from ._process_manager import Command, ProcessManager

processes = []

# llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B dynamo-init.process.chat/completions


def _configure_http(args, unknown_args):
    command = ["llmctl", "http", "remove", "chat-models", args.model]

    subprocess.call(command)

    command = [
        "llmctl",
        "http",
        "add",
        "chat-models",
        args.model,
    ]

    if args.router == "prefix":
        command.append("dynamo-init.process.chat/completions")
    else:
        command.append("dynamo-init.vllm.generate")

    subprocess.call(command)


def _http_commands(args, unknown_args):
    commands = []

    commands.append(
        Command(
            [
                "http",
                "--port",
                f"{args.http_port}",
            ],
            name="http",
        )
    )

    commands.append(
        Command(
            ["llmctl", "http", "remove", "chat-models", args.model],
            call=True,
            name="llmctl",
        )
    )

    llmctl_args = [
        "llmctl",
        "http",
        "add",
        "chat-models",
        args.model,
    ]

    if args.router == "prefix":
        llmctl_args.append("dynamo-init.process.chat/completions")
    else:
        llmctl_args.append("dynamo-init.vllm.generate")

    commands.append(Command(args=llmctl_args, call=True, name="llmctl"))
    return commands


def _vllm_worker_commands(args, unknown_args):
    command_args = []
    if not args.router:
        command_args.append("vllm-routerless-worker")
    else:
        command_args.append("vllm-worker")
        command_args.append("--enable-prefix-caching")

    command_args.extend(unknown_args)
    command_args.append("--model")
    command_args.append(args.model)
    command_args.append("--block-size")
    command_args.append(str(args.block_size))
    command_args.append("--max-model-len")
    command_args.append(str(args.max_model_len))
    return [Command(args=command_args, name="vllm worker")]


# RUST_LOG=info python3 router/processor.py \
#   --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
# --enable-prefix-caching \
# --block-size 64 \
# --max-model-len 16384


def _processor_commands(args, unknown_args):
    command_args = ["vllm-processor"]
    command_args.append("--model")
    command_args.append(args.model)
    command_args.append("--tokenizer")
    command_args.append(args.model)
    command_args.append("--enable-prefix-caching")
    command_args.append("--block-size")
    command_args.append(str(args.block_size))
    command_args.append("--max-model-len")
    command_args.append(str(args.max_model_len))

    return [Command(args=command_args, name="processor")]


# RUST_LOG=info python3 router/kv_router.py \
#     --routing-strategy prefix \
#     --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --custom-router \
#     --min-workers 1


def _router_commands(args, unknown_args):
    commands = []

    if args.router == "prefix":
        commands.extend(_processor_commands(args, unknown_args))

        command_args = ["vllm-kv_router.py"]
        command_args.append("--model-name")
        command_args.append(args.model)
        command_args.append("--custom-router")
        command_args.append("True")
        command_args.append("--min-workers")
        command_args.append("1")
        command_args.append("--block-size")
        command_args.append(str(args.block_size))

        commands.append(Command(args=command_args, name="kv router"))

    return commands


def main():
    known_args, unknown_args = parse_known_args()

    with ProcessManager(known_args, unknown_args) as process_manager:
        process_manager.add_commands(_http_commands(known_args, unknown_args))
        process_manager.add_commands(_router_commands(known_args, unknown_args))
        process_manager.add_commands(_vllm_worker_commands(known_args, unknown_args))
        process_manager.start()
        while True:
            time.sleep(5)

    # print("started", known_args, unknown_args)
    # _launch_http(known_args, unknown_args)
    # _launch_router(known_args, unknown_args)
    # _launch_vllm_worker(known_args, unknown_args)


#    while True:
# time.sleep(10)


if __name__ == "__main__":
    main()
