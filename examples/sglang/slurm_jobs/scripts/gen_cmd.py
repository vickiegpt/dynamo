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

"""
Use this script to generate dynamo/sglang flags for h100 or gb200 disagg
"""

def get_prefill_command_args(config_flag: str, host_ip: str, port: int, total_nodes: int, rank: int, total_gpus: int) -> dict:
    """
    Get the command arguments for a specific config and worker type.
    
    Args:
        config_flag: One of "h100_dynamo", "h100_sglang", "gb200_dynamo", "gb200_sglang"
        worker_type: "prefill" or "decode"
        
    Returns:
        Dictionary with 'script' and 'args' keys
    """

    # TODO: validate if config_flag is valid and exists in support matrix

    if config_flag == "h100_dynamo":
        return {
            "script": "python3 components/worker.py",
            "args": [
                "--model-path /model/ "
                "--served-model-name deepseek-ai/DeepSeek-R1 "
                "--skip-tokenizer-init "
                "--disaggregation-mode prefill "
                "--disaggregation-transfer-backend nixl "
                "--disaggregation-bootstrap-port 30001 "
                f"--dist-init-addr {host_ip}:{port} "
                f"--nnodes {total_nodes} "
                f"--node-rank {rank} "
                f"--tp-size {total_gpus} "
                f"--dp-size {total_gpus} "
                "--enable-dp-attention "
                "--decode-log-interval 1 "
                "--enable-deepep-moe "
                "--page-size 1 "
                "--trust-remote-code "
                "--moe-dense-tp-size 1 "
                "--enable-dp-lm-head "
                "--disable-radix-cache "
                "--watchdog-timeout 1000000 "
                "--enable-two-batch-overlap "
                "--deepep-mode normal "
                "--mem-fraction-static 0.85 "
                "--deepep-config /configs/deepep.json "
                "--ep-num-redundant-experts 32 "
                "--ep-dispatch-algorithm dynamic "
                "--eplb-algorithm deepseek "
            ]
        }
    elif config_flag == "h100_sglang":
        return {
            "script": "python3 -m sglang.launch_server",
            "args": [
                "--model-path /model/ ",
                "--served-model-name deepseek-ai/DeepSeek-R1 ",
                "--disaggregation-transfer-backend nixl ",
                "--disaggregation-mode prefill ",
                f"--dist-init-addr {host_ip}:{port} ",
                f"--nnodes {total_nodes} ",
                f"--node-rank {rank} ",
                f"--tp-size {total_gpus} ",
                f"--dp-size {total_gpus} ",
                "--enable-dp-attention ",
                "--decode-log-interval 1 ",
                "--enable-deepep-moe ",
                "--page-size 1 ",
                "--host 0.0.0.0 ",
                "--trust-remote-code ",
                "--moe-dense-tp-size 1 ",
                "--enable-dp-lm-head ",
                "--disable-radix-cache ",
                "--watchdog-timeout 1000000 ",
                "--enable-two-batch-overlap ",
                "--deepep-mode normal ",
                "--mem-fraction-static 0.85 ",
                "--chunked-prefill-size 524288 ",
                "--max-running-requests 8192 ",
                "--max-total-tokens 131072 ",
                "--context-length 8192 ",
                "--init-expert-location /configs/prefill_in4096.json ",
                "--ep-num-redundant-experts 32 ",
                "--ep-dispatch-algorithm dynamic ",
                "--eplb-algorithm deepseek ",
                "--deepep-config /configs/deepep.json "
            ]
        }
    elif config_flag == "gb200_sglang":
        return {
            "script": "python3 -m sglang.launch_server",
            "args": [
                "--served-model-name deepseek-ai/DeepSeek-R1 ",
                "--model-path /model/ ",
                "--trust-remote-code ",
                "--disaggregation-mode prefill ",
                f"--dist-init-addr {host_ip}:{port} ",
                f"--nnodes {total_nodes} ",
                f"--node-rank {rank} ",
                f"--tp-size {total_gpus} ",
                f"--dp-size {total_gpus} ",
                "--enable-dp-attention ",
                "--host 0.0.0.0 ",
                "--decode-log-interval 1 ",
                "--max-running-requests 6144 ",
                "--context-length 2176 ",
                "--disable-radix-cache ",
                "--enable-deepep-moe ",
                "--deepep-mode low_latency ",
                "--moe-dense-tp-size 1 ",
                "--enable-dp-lm-head ",
                "--disable-shared-experts-fusion ",
                "--ep-num-redundant-experts 32 ",
                "--ep-dispatch-algorithm static ",
                "--eplb-algorithm deepseek ",
                "--attention-backend cutlass_mla ",
                "--watchdog-timeout 1000000 ",
                "--disable-cuda-graph ",
                "--chunked-prefill-size 16384 ",
                "--max-total-tokens 32768 ",
                "--mem-fraction-static 0.9 "
            ]
        }
    else:
        raise ValueError(f"Invalid config flag: {config_flag}")
    
def get_decode_command_args(config_flag: str, host_ip: str, port: int, total_nodes: int, rank: int, total_gpus: int) -> dict:
    """
    Get the command arguments for a specific config and worker type.
    
    Args:
        config_flag: One of "h100_dynamo", "h100_sglang", "gb200_dynamo", "gb200_sglang"
        worker_type: "prefill" or "decode"
        
    Returns:
        Dictionary with 'script' and 'args' keys
    """

    if config_flag == "h100_dynamo":
        return {
            "script": "python3 components/decode_worker.py",
            "args": [
                "--model-path /model/ ",
                "--served-model-name deepseek-ai/DeepSeek-R1 ",
                "--skip-tokenizer-init ",
                "--disaggregation-mode decode ",
                "--disaggregation-transfer-backend nixl ",
                "--disaggregation-bootstrap-port 30001 ",
                f"--dist-init-addr {host_ip}:{port} ",
                f"--nnodes {total_nodes} ",
                f"--node-rank {rank} ",
                f"--tp-size {total_gpus} ",
                f"--dp-size {total_gpus} ",
                "--enable-dp-attention ",
                "--decode-log-interval 1 "
                "--enable-deepep-moe "
                "--page-size 1 "
                "--trust-remote-code "
                "--moe-dense-tp-size 1 "
                "--enable-dp-lm-head "
                "--disable-radix-cache "
                "--watchdog-timeout 1000000 "
                "--enable-two-batch-overlap "
                "--deepep-mode low_latency "
                "--mem-fraction-static 0.835 "
                "--ep-num-redundant-experts 32 "
                "--cuda-graph-bs 256 "
            ]
        }
    elif config_flag == "h100_sglang":
        return {
            "script": "python3 -m sglang.launch_server",
            "args": [
                "--model-path /model/ ",
                "--disaggregation-transfer-backend nixl ",
                "--disaggregation-mode decode ",
                f"--dist-init-addr {host_ip}:{port} ",
                f"--nnodes {total_nodes} ",
                f"--node-rank {rank} ",
                f"--tp-size {total_gpus} ",
                f"--dp-size {total_gpus} ",
                "--enable-dp-attention ",
                "--decode-log-interval 1 ",
                "--enable-deepep-moe ",
                "--page-size 1 ",
                "--host 0.0.0.0 ",
                "--trust-remote-code ",
                "--moe-dense-tp-size 1 ",
                "--enable-dp-lm-head ",
                "--disable-radix-cache ",
                "--watchdog-timeout 1000000 ",
                "--enable-two-batch-overlap ",
                "--deepep-mode low_latency ",
                "--mem-fraction-static 0.835 ",
                "--max-running-requests 18432 ",
                "--context-length 4500 ",
                "--ep-num-redundant-experts 32 ",
                "--cuda-graph-bs 256 "
            ]
        }
    elif config_flag == "gb200_sglang":
        return {
            "script": "python3 -m sglang.launch_server",
            "args": [
                "--model-path /model/ ",
                "--trust-remote-code ",
                "--disaggregation-transfer-backend nixl ",
                "--disaggregation-mode decode ",
                f"--dist-init-addr {host_ip}:{port} ",
                f"--nnodes {total_nodes} ",
                f"--node-rank {rank} ",
                f"--tp-size {total_gpus} ",
                f"--dp-size {total_gpus} ",
                "--enable-dp-attention ",
                "--host 0.0.0.0 ",
                "--decode-log-interval 1 ",
                "--max-running-requests 36864 ",
                "--context-length 2176 ",
                "--disable-radix-cache ",
                "--enable-deepep-moe ",
                "--deepep-mode low_latency ",
                "--moe-dense-tp-size 1 ",
                "--enable-dp-lm-head ",
                "--cuda-graph-bs 768 ",
                "--disable-shared-experts-fusion ",
                "--ep-num-redundant-experts 32 ",
                "--ep-dispatch-algorithm static ",
                "--eplb-algorithm deepseek ",
                "--attention-backend cutlass_mla ",
                "--watchdog-timeout 1000000 ",
                "--chunked-prefill-size 36864 ",
                "--mem-fraction-static 0.82 "
            ]
        }
    