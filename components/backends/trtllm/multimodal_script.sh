cd dynamo/components/backends/trtllm
export MODALITY=${MODALITY:-"multimodal"}
export IMAGE="/lustre/fsw/core_dlfw_ci/kprashanth/dynamo33252043_trtllm_rc6.sqsh"
export MOUNTS="${PWD}/:/mnt,/lustre:/lustre"
export MODEL_PATH="/lustre/fsw/core_dlfw_ci/kprashanth/meta-llama_Llama-4-Maverick-17B-128E-Instruct"
export SERVED_MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
export NUM_PREFILL_NODES=2
export NUM_DECODE_NODES=2
export NUM_GPUS_PER_NODE=4
export PREFILL_ENGINE_CONFIG="/mnt/engine_configs/multimodal/llama4/prefill.yaml"
export DECODE_ENGINE_CONFIG="/mnt/engine_configs/multimodal/llama4/decode.yaml"
# ./multinode/srun_disaggregated.sh

