#!/bin/bash

#SBATCH --job-name "${account_name}-repro-trtllm-01" --account ${account_name} --partition 36x2-a01r --output ${log_file} --exclusive --nodes 1 --time 0:14400 --deadline "now+20hours" --parsable

srun \
    --job-name "test-trtllm-repro" \
    --container-mounts "/lustre:/lustre" \
    --container-image "gitlab-master.nvidia.com/dl/ai-dynamo/dynamo-ci:0d1fdf9c52b799d63baffc3365411b19962a8110-29735692-tensorrtllm-arm64" \
    --nodes "1" \
    --no-container-mount-home \
    --mpi "pmix" \
    --label \
    bash -c 'env -u SLURM_JOBID -u SLURM_NODELIST {path_to_run_benchmark.sh_in_lustre}/run_benchmark.sh'
