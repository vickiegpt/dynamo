#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Wait for workers to startup
sleep 30

#TODO: Automatically generate the disagg_config.yaml file.
# For now, we manually create the file by looking at what nodes are running which workers.

# NOTE: This is a blocking call.
trtllm-serve disaggregated -c /mnt/multinode/baseline/disagg_config.yaml --server_start_timeout 360

