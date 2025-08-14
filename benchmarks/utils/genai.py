# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
        "--artifact-dir",
        output_dir,
        "--",
        "-v",
        "--max-threads=300",
    ]
    logger.info(
        f"Running genai-perf with isl {isl}, osl {osl}, concurrency {concurrency}"
    )
    gap_process = subprocess.Popen(
        cmd,
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = gap_process.communicate()
    if gap_process.returncode == 0:
        logger.info("Genai-perf profiling completed successfully")
        logger.info(stdout)
    else:
        logger.error(f"Genai-perf failed with error code: {gap_process.returncode}")
        logger.error(f"stderr: {stderr}")
        raise subprocess.CalledProcessError(gap_process.returncode, cmd)


def run_concurrency_sweep(
    service_url: str, model_name: str, isl: int, osl: int, stddev: int, output_dir: Path
) -> None:
    logger.info(
        f"Running concurrency sweep for {model_name} with input sequence length {isl} and output sequence length {osl} and standard deviation {stddev}"
    )
    for c in CONCURRENCIES:
        logger.info(f"Starting concurrency level {c}")
        run_genai_perf(
            service_url, model_name, isl, osl, stddev, c, output_dir / f"c{c}"
        )
