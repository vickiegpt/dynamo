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

import argparse
import json
import logging
import math
import os
import random
import signal
import subprocess
import time
from typing import Literal
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import requests
import yaml

DECODE_NUM_REQUESTS_RANGE = [
    1,
    5,
    10,
    25,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_dynamo_serve_cmd(config_file_path):
    return [
        "dynamo",
        "serve",
        "graphs.agg:Frontend",
        "-f",
        config_file_path,
    ]


def get_prefill_genai_perf_cmd(
    isl,
    artifact_dir,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    osl=5,
    port=8000,
):
    return [
        "genai-perf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        model,
        "--service-kind",
        "openai",
        "--endpoint-type",
        "chat",
        "--endpoint",
        "/v1/chat/completions",
        "--streaming",
        "--url",
        f"http://localhost:{port}",
        "--synthetic-input-tokens-mean",
        str(isl),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        "5",
        "--output-tokens-stddev",
        "0",
        "--extra-inputs",
        "max_tokens:5",
        "--extra-inputs",
        "min_tokens:5",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        '{"nvext":{"ignore_eos":true}}',
        "--concurrency",
        "1",
        "--request-count",
        "1",
        "--warmup-request-count",
        "3",
        "--artifact-dir",
        artifact_dir,
        "--random-seed",
        str(seed),
    ]


def get_decode_genai_perf_cmd(
    isl,
    osl,
    artifact_dir,
    num_request,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    port=8000,
):
    return [
        "genai-perf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        model,
        "--service-kind",
        "openai",
        "--endpoint-type",
        "chat",
        "--endpoint",
        "/v1/chat/completions",
        "--streaming",
        "--url",
        f"http://localhost:{port}",
        "--synthetic-input-tokens-mean",
        str(isl),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(osl),
        "--output-tokens-stddev",
        "0",
        "--extra-inputs",
        f"max_tokens:{osl}",
        "--extra-inputs",
        f"min_tokens:{osl}",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        '{"nvext":{"ignore_eos":true}}',
        "--concurrency",
        str(num_request),
        "--num-dataset-entries",
        str(num_request),
        "--request-count",
        str(num_request),
        "--warmup-request-count",
        "3",
        "--artifact-dir",
        artifact_dir,
        "--random-seed",
        str(seed),
    ]


def convert_config(config: dict, target: Literal["prefill", "decode"]) -> dict:
    config = config.copy()

    # all profiles runs with a single prefill/decode worker, hence router doesn't matter
    if "Common" in config and "router" in config["Common"]:
        config["Common"]["router"] = "round-robin"
    else:
        config["Processor"]["router"] = "round-robin"

    # disable planner
    if "Planner" in config:
        config["Planner"]["no-operation"] = True

    if target == "prefill":
        if "PrefillWorker" in config:
            # make PrefillWorker into VllmWorker
            del config["VllmWorker"]
            config["VllmWorker"] = config["PrefillWorker"]
            del config["PrefillWorker"]

        # to profile prefill, we disable prefix caching
        config["VllmWorker"]["enable-prefix-caching"] = False
    elif target == "decode":
        if "PrefillWorker" in config:
            del config["PrefillWorker"]

        # to profile prefill, we enable prefix caching to pass the prefill stage
        config["VllmWorker"]["enable-prefix-caching"] = True

    # set num workers to 1
    config["VllmWorker"]["ServiceArgs"]["workers"] = 1

    # set PP to 1
    if (
        "pipeline-parallel-size" in config["VllmWorker"]
        and config["VllmWorker"]["pipeline-parallel-size"] > 1
    ):
        logger.warning("Currently we only support TP, setting PP to 1")
        config["VllmWorker"]["pipeline-parallel-size"] = 1

    # always local prefill
    config["VllmWorker"]["remote-prefill"] = False
    config["VllmWorker"]["conditional-disagg"] = False

    return config


def set_config_tp_size(config: dict, tp_size: int):
    config["VllmWorker"]["tensor-parallel-size"] = tp_size
    config["VllmWorker"]["ServiceArgs"]["resources"]["gpu"] = tp_size
    return config


def get_available_gpu_count():
    try:
        import pynvml

        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()

        if gpu_count > 0:
            logger.info(f"Detected {gpu_count} GPUs in the system:")
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_mb = memory.total / (1024 * 1024)
                free_memory_mb = memory.free / (1024 * 1024)
                logger.info(
                    f"  GPU {i}: {name}, Total Memory: {total_memory_mb:.2f} MB, Free Memory: {free_memory_mb:.2f} MB"
                )
        else:
            logger.warning("No GPUs detected with pynvml.")

        pynvml.nvmlShutdown()
        return gpu_count
    except ImportError:
        logger.error(
            "pynvml module not found. Please install it with 'pip install pynvml'"
        )
        return 0
    except pynvml.NVMLError as e:
        logger.error(f"NVML Error: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return 0


def get_model_name(config: dict) -> str:
    if "Common" in config and "served_model_name" in config["Common"]:
        return config["Common"]["served_model_name"]
    else:
        return config["Frontend"]["served_model_name"]


def get_port(config: dict) -> int:
    if "Common" in config and "port" in config["Common"]:
        return config["Common"]["port"]
    else:
        return config["Frontend"]["port"]


def wait_for_server_ready(model_name: str, port: int, timeout: int = 300):
    logger.info("Waiting for the server to be ready...")
    endpoint_url = f"http://localhost:{port}/v1/chat/completions"
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < timeout:
        try:
            # Send a simple request to check if the server is up
            response = requests.post(
                endpoint_url,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1,
                },
                timeout=5,
            )
            if response.status_code == 200:
                logger.info(
                    f"Server is ready after {time.time() - start_time:.2f} seconds"
                )
                server_ready = True
                break
            else:
                logger.info(
                    f"Server returned status code {response.status_code}, waiting..."
                )
        except (requests.RequestException, ConnectionError) as e:
            logger.info(f"Server not ready yet: {e}")

        time.sleep(5)

    return server_ready


def get_kv_cache_size_from_dynamo_log(dynamo_log_fn: str) -> int:
    try:
        with open(dynamo_log_fn, "r") as f:
            for line in f:
                if "Maximum concurrency for" in line:
                    line = line.strip().split("Maximum concurrency for ")[1]
                    token_count = int(line.split(" tokens per request: ")[0])
                    concurrency = float(line.split(" tokens per request: ")[1][:-1])

                    logger.info(
                        f"Found KV cache info: {token_count} x {concurrency} = {int(token_count * concurrency)}"
                    )
                    return int(token_count * concurrency)
    except Exception as e:
        logger.warning(f"Failed to parse KV cache size from line: {line}. Error: {e}")
    return 0


def get_gap_result(artifact_dir: str) -> dict:
    with open(f"{artifact_dir}/profile_export_genai_perf.json", "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the dynamo config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiling_results",
        help="Path to the output results directory",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get the number of available GPUs
    available_gpus = get_available_gpu_count()

    profile_tp_size = [2**i for i in range(int(math.log2(available_gpus)) + 1)]
    logger.info(f"Profiling TP sizes: {profile_tp_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name(config)
    port = get_port(config)

    # then profile decode
    
    logger.info("Profiling decode...")
    decode_config = convert_config(config, "decode")
    for tp_size in profile_tp_size:
        logger.info(f"Profiling decode with TP size {tp_size}...")
        logger.info(f"Dynamo config: {decode_config}")

        x_kv_usage = []
        y_context_length = []
        z_itl = []

        work_dir = f"{args.output_dir}/decode_tp{tp_size}"
        os.makedirs(work_dir, exist_ok=True)

        decode_config = set_config_tp_size(decode_config, tp_size)
        decode_config_fn = f"{work_dir}/config.yaml"
        dynamo_log_fn = f"{work_dir}/dynamo.log"
        with open(decode_config_fn, "w") as f:
            yaml.dump(decode_config, f)

        # Start the dynamo serve process
        logger.info(f"Starting dynamo serve with TP size {tp_size}...")
        dynamo_serve_cmd = get_dynamo_serve_cmd(decode_config_fn)
        with open(dynamo_log_fn, "w") as dynamo_log_f:
            dynamo_process = subprocess.Popen(
                dynamo_serve_cmd,
                stdout=dynamo_log_f,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,  # Use process group for clean termination
            )

        if not wait_for_server_ready(model_name, port):
            logger.error(f"Server did not become ready, skip profiling tp={tp_size}")
            break

        max_kv_tokens = get_kv_cache_size_from_dynamo_log(dynamo_log_fn)

        osl = 500
        for isl in range(1000, 11000, 1000):
            logger.info(f"Profiling decode with isl {isl}...")
            max_concurrency = max_kv_tokens // (isl + osl)
            sweep_num_request = range(1, max_concurrency, max_concurrency // 5)
            logger.info(f"Num request range: {list(sweep_num_request)}")

            for num_request in sweep_num_request:
                logger.info(f"Profiling decode with num_request {num_request}...")

                # first warm-up the engine by pre-computing all prefill tokens
                # we use the same random seed to make sure the prompt is the same
                seed = random.randint(0, 1000000)
                genai_perf_artifact_dir = f"{work_dir}/gap_warmup"
                genai_perf_cmd = get_decode_genai_perf_cmd(
                    isl,
                    osl,
                    genai_perf_artifact_dir,
                    num_request,
                    seed=seed,
                    model=model_name,
                    port=port,
                )
                gap_process = subprocess.Popen(
                    genai_perf_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                gap_process.communicate()
                # then send out the real requests, hopefully, this will skip all prefill computation
                genai_perf_artifact_dir = (
                    f"{work_dir}/gap"
                )
                genai_perf_cmd = get_decode_genai_perf_cmd(
                    isl,
                    osl,
                    genai_perf_artifact_dir,
                    num_request,
                    seed=seed,
                    model=model_name,
                    port=port,
                )
                gap_process = subprocess.Popen(
                    genai_perf_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = gap_process.communicate()
                if gap_process.returncode == 0:
                    logger.info("Genai-perf profiling completed successfully")
                    logger.info(stdout)
                    gap_result = get_gap_result(genai_perf_artifact_dir)
                    itl = gap_result["inter_token_latency"]["avg"]
                    x_kv_usage.append((isl + osl / 2) * num_request / max_kv_tokens)
                    y_context_length.append(isl + osl / 2)
                    z_itl.append(itl)
                else:
                    logger.error(
                        f"Genai-perf failed with error code: {gap_process.returncode}"
                    )
                    logger.error(f"stderr: {stderr}")

        print(f"x_kv_usage: {x_kv_usage}")
        print(f"y_context_length: {y_context_length}")
        print(f"z_itl: {z_itl}")
        xi =np.linspace(min(x_kv_usage), max(x_kv_usage), 100)
        yi = np.linspace(min(y_context_length), max(y_context_length), 100)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x_kv_usage, y_context_length), z_itl, (X, Y), method='cubic')

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create the surface plot with customizations
        surf = ax.plot_surface(
            X, Y, Z, 
            cmap=cm.coolwarm,  # Change colormap
            linewidth=0.2,     # Add grid lines
            antialiased=True,
            alpha=0.8
        )

        # Add a color bar with custom settings
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Z Value', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Add labels with custom font sizes
        ax.set_xlabel('Active KV Percentage', fontsize=12)
        ax.set_ylabel('Decode Context Length', fontsize=12)
        ax.set_zlabel('ITL', fontsize=12)

        # Set viewing angle
        ax.view_init(elev=30, azim=45)  # elevation and azimuth angles

        # Add a grid
        ax.grid(True)

        # Adjust tick label font size
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Save the figure if needed
        plt.savefig(f'{work_dir}/decode_tp{tp_size}.png', dpi=300, bbox_inches='tight')


        # Send SIGINT to the dynamo process to terminate it gracefully
        os.killpg(os.getpgid(dynamo_process.pid), signal.SIGINT)
        dynamo_process.communicate()
