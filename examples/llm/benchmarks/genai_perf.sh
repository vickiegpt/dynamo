model=meta-llama/Llama-3.3-70B-Instruct
seed=42

isl=7000
osl=100

num_prompts=1

requests=100
conc=10

# Loop over different prefix ratios
for prefix_ratio in 0.2 0.5 0.9; do
  echo "Running benchmark with prefix_ratio=${prefix_ratio}, seed=${seed}"
  
  # Calculate prefix and synthetic input lengths based on prefix_ratio
  prefix_length=$(python3 -c "print(int($isl * $prefix_ratio))")
  synthetic_input_length=$(python3 -c "print(int($isl * (1 - $prefix_ratio)))")
  
  # NOTE: For Dynamo HTTP OpenAI frontend, use `nvext` for fields like
  # `ignore_eos` since they are not in the official OpenAI spec.
  genai-perf profile \
    --model ${model} \
    --tokenizer ${model} \
    --endpoint-type chat \
    --endpoint v1/chat/completions \
    --streaming \
    --url http://localhost:8888 \
    --synthetic-input-tokens-mean ${synthetic_input_length} \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean ${osl} \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:${osl} \
    --extra-inputs min_tokens:${osl} \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency ${conc} \
    --request-count ${requests} \
    --num-dataset-entries ${requests} \
    --random-seed ${seed} \
    --prefix-prompt-length ${prefix_length} \
    --num-prefix-prompts ${num_prompts} \
    -- \
    -v \
    --max-threads 256 \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'
  
  echo "Completed benchmark with prefix_ratio=${prefix_ratio}, seed=${seed}"
  
  # Increment seed for next iteration
  ((seed++))
done
