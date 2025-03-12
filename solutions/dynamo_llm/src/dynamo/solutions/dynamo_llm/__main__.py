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


def _launch_vllm_worker(args, unknown_args):
    global processes

    command = []
    if not args.router:
        command.append("vllm-routerless-worker")
    else:
        command.append("vllm-worker")
        command.append("--enable-prefix-caching")

    command.extend(unknown_args)
    command.append("--model")
    command.append(args.model)
    command.append("--block-size")
    command.append(str(args.block_size))
    command.append("--max-model-len")
    command.append(str(args.max_model_len))
    print(command)
    processes.append(
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
        )
    )


# RUST_LOG=info python3 router/processor.py \
#   --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
# --enable-prefix-caching \
# --block-size 64 \
# --max-model-len 16384


def _launch_processor(args, unknown_args):
    global processes

    command = ["python3", "/workspace/components/vllm/router/processor.py"]
    command.append("--model")
    command.append(args.model)
    command.append("--tokenizer")
    command.append(args.model)
    command.append("--enable-prefix-caching")
    command.append("--block-size")
    command.append(str(args.block_size))
    command.append("--max-model-len")
    command.append(str(args.max_model_len))

    processes.append(
        subprocess.Popen(
            command, stdin=subprocess.DEVNULL, cwd="/workspace/components/vllm"
        )
    )


# RUST_LOG=info python3 router/kv_router.py \
#     --routing-strategy prefix \
#     --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --custom-router \
#     --min-workers 1


def _launch_router(args, unknown_args):
    global processes

    if args.router == "prefix":
        _launch_processor(args, unknown_args)

        command = ["python3", "/workspace/components/vllm/router/kv_router.py"]
        command.append("--model-name")
        command.append(args.model)
        command.append("--custom-router")
        command.append("True")
        command.append("--min-workers")
        command.append("1")
        command.append("--block-size")
        command.append(str(args.block_size))

        processes.append(
            subprocess.Popen(
                command, stdin=subprocess.DEVNULL, cwd="/workspace/components/vllm"
            )
        )


def main():
    known_args, unknown_args = parse_known_args()

    with ProcessManager(known_args, unknown_args) as process_manager:
        process_manager.add_commands(_http_commands(known_args, unknown_args))
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
