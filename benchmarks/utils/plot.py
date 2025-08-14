# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def generate_plots(base_output_dir: Path) -> None:
    # Placeholder: Walk result dirs and aggregate into plots
    summary = base_output_dir / "SUMMARY.txt"
    summary.write_text("Plots generation placeholder")
