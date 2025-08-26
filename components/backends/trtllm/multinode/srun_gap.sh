#!/bin/bash
srun \
--overlap \
--container-image "${IMAGE}" \
--container-mounts "${MOUNTS}" \
--container-env SERVED_MODEL_NAME,MODEL_PATH \
--verbose \
-A "${ACCOUNT}" \
-J "${ACCOUNT}-dynamo.trtllm" \
--jobid "${SLURM_JOB_ID}" \
--nodes 1 \
--pty \
bash
