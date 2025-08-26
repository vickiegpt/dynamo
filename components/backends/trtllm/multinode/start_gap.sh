#!/bin/bash -x

genai-perf profile -m ${SERVED_MODEL_NAME} \
--endpoint-type=chat \
--synthetic-input-tokens-mean 128   \
--synthetic-input-tokens-stddev 0   \
--output-tokens-mean 100   \
--output-tokens-stddev 0   \
--url localhost:8000   \
--streaming   \
--request-count 10   \
--warmup-request-count 2 \
--tokenizer ${MODEL_PATH}


#genai-perf profile -m ${SERVED_MODEL_NAME} \
#--endpoint-type=chat \
#--synthetic-input-tokens-mean 20480   \
#--synthetic-input-tokens-stddev 5120   \
#--output-tokens-mean 1024   \
#--output-tokens-stddev 256  \
#--url localhost:8000   \
#--streaming   \
#--request-count 10   \
#--warmup-request-count 2 \
#--tokenizer ${MODEL_PATH}
