#!/bin/bash

srun \
    --job-name "test-trtllm-repro" \
    --container-mounts "/lustre:/lustre" \
    --container-image "/lustre/fsw/core_dlfw_ci/rihuo/dynamo-arrch64-trtllm-28590086.sqsh" \
    --nodes "4" \
    --no-container-mount-home \
    --mpi "pmix" \
    --label \
    bash -c 'env -u SLURM_JOBID -u SLURM_NODELIST python /lustre/fsw/core_dlfw_ci/rihuo/trtllm_repro.py'
