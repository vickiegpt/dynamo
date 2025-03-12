import argparse


# FIXME: Remove unused args if any
def parse_known_args():
    parser = argparse.ArgumentParser(description="Run an example of the llm.")

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
        help="Number of context workers",
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
        choices=["", "prefix"],
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
        type=int,
        required=False,
        default=0,
        help="Number of context workers",
    )

    parser.add_argument(
        "--decode-workers",
        type=int,
        required=False,
        default=0,
        help="Number of generate workers",
    )

    parser.add_argument(
        "--prefill-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size of a context worker.",
    )

    parser.add_argument(
        "--decode-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size of a generate worker.",
    )

    return parser.parse_known_args()
