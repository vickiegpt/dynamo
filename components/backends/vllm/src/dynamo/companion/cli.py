# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for companion launcher."""

import argparse
import logging


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments for companion launcher."""
    parser = argparse.ArgumentParser(
        description="Launch vLLM companion coordinator for CUDA IPC weight sharing"
    )
    
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=None,
        help="Port for the companion coordinator to listen on. If not provided, defaults to 55800 (from CompanionConfig)."
    )
    
    parser.add_argument(
        "--companion-master-port",
        type=int,
        default=None,
        help="Master port for CPU group initialization during distributed setup. If not provided, a free port will be chosen."
    )
    
    args = parser.parse_args()
    
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    return args