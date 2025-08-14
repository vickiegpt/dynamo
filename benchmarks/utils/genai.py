# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from typing import List

CONCURRENCIES: List[int] = [1, 2, 5, 10, 50, 100, 250]


def run_genai_perf(
    service_url: str,
    model_name: str,
    isl: int,
    osl: int,
    stddev: int,
    concurrency: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "genai-perf",
        "profile",
        "-m",
        model_name,
        "--endpoint-type",
        "chat",
        "--streaming",
        "-u",
        service_url.replace("http://", "").replace("https://", ""),
        "--synthetic-input-tokens-mean",
        str(isl),
        "--synthetic-input-tokens-stddev",
        str(stddev),
        "--concurrency",
        str(concurrency),
        "--output-tokens-mean",
        str(osl),
        "--extra-inputs",
        f"max_tokens:{osl}",
        "--extra-inputs",
        f"min_tokens:{osl}",
        "--extra-inputs",
        "ignore_eos:true",
        "--tokenizer",
        model_name,
        "--",
        "-v",
        "--max-threads=300",
    ]
    subprocess.run(cmd, cwd=output_dir, check=True)


def run_concurrency_sweep(
    service_url: str, model_name: str, isl: int, osl: int, stddev: int, output_dir: Path
) -> None:
    for c in CONCURRENCIES:
        run_genai_perf(
            service_url, model_name, isl, osl, stddev, c, output_dir / f"c{c}"
        )
