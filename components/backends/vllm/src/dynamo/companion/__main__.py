# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main entry point for companion launcher."""

import logging
import signal
import sys

from .cli import parse_args
from .launcher import launch_companion

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    try:
        # Parse and validate arguments
        args = parse_args()
        
        # Launch companion
        launcher = launch_companion(
            coordinator_port=args.coordinator_port,
            companion_master_port=args.companion_master_port,
        )
        
        logger.info("Companion coordinator running at %s", launcher.get_coordinator_address())
        
        # Set up signal handlers
        def signal_handler(sig, frame):
            logger.info("Received signal %s, stopping launcher", sig)
            launcher.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Wait for coordinator to finish
        logger.info("Press Ctrl+C to stop.")
        launcher.wait()
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Failed to launch companion: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
