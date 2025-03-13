import argparse

from ._gpu_info import gpu_count, gpu_product_name


# FIXME: Remove unused args if any
def parse_known_args():
    parser = argparse.ArgumentParser(description="Run an example of the llm.")

    parser.add_argument("--dry-run", action="store_true", required=False)

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        choices=[
            "meta-llama/llama-3.1-8b-instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        ],
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="model",
    )

    parser.add_argument("--block-size", type=int, default=64, required=False)

    parser.add_argument("--max-model-len", type=int, default=16384)

    parser.add_argument(
        "--prefill-workers",
        type=int,
        required=False,
        default=0,
        dest="prefill_count",
        help="Number of prefill workers",
    )

    parser.add_argument(
        "--leader",
        type=str,
        required=False,
        default=0,
        help="Number of context workers",
    )

    parser.add_argument(
        "--leader-address",
        type=str,
        required=False,
        default=0,
        help="Number of context workers",
    )

    parser.add_argument(
        "--router",
        type=str,
        required=False,
        default="",
        choices=["", "random", "kv", "round-robin"],
        help="Number of context workers",
    )

    parser.add_argument(
        "--http-port",
        type=int,
        required=False,
        default=8181,
        help="Number of context workers",
    )

    parser.add_argument(
        "--backend",
        type=str,
        required=False,
        choices=["vllm", "tensorrtllm"],
        default="vllm",
        help="backend framework",
    )

    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        default=0,
        dest="worker_count",
        help="Number of decode / aggregate workers",
    )

    parser.add_argument(
        "--reuse-gpus",
        action="store_true",
        required=False,
        default=False,
        help="reuse gpu devices",
    )

    parser.add_argument(
        "--hf-hub-offline",
        action="store_true",
        required=False,
        default=False,
        help="set hf hub offline",
    )

    parser.add_argument(
        "--prefill-tp",
        type=int,
        default=1,
        help="Tensor parallel size of a context worker.",
    )

    parser.add_argument(
        "--worker-tp",
        type=int,
        default=1,
        help="Tensor parallel size of a generate worker.",
    )

    known_args, unknown_args = parser.parse_known_args()

    known_args.gpu_count = gpu_count()
    known_args.gpu_product_name = gpu_product_name()
    known_args._next_gpu = 0
    return known_args, unknown_args
