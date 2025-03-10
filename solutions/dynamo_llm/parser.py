import argparse


# FIXME: Remove unused args if any
def parse_known_args():
    parser = argparse.ArgumentParser(description="Run an example of the llm.")

    parser.add_argument(
        "--prefill-workers",
        type=int,
        required=False,
        default=0,
        help="Number of context workers",
    )

    parser.add_argument(
        "--router",
        type=int,
        required=False,
        default=0,
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
