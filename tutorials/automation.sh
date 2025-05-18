#!/bin/bash
set -euo pipefail

SERVER_CMD_AGG="dynamo serve graphs.agg:Frontend --working-dir /workspace/examples/llm -f" 
SERVER_CMD_DISAGG="dynamo serve graphs.disagg:Frontend --working-dir /workspace/examples/llm -f"

# Configuration
#configs(agg_tp_1_dp_4.yaml, disagg_p_tp_1_dp_2_d_tp_1_dp_2.yaml, agg_tp_2_dp_2.yaml, disagg_p_tp_2_dp_1_d_tp_1_dp_2.yaml, agg_tp_4_dp_1.yaml, disagg_p_tp_1_dp_2_d_tp_2_dp_1.yaml, disagg_p_tp_2_dp_1_d_tp_2_dp_1.yaml)  # Replace with actual config files

configs=("disagg_p_tp_1_dp_2_d_tp_1_dp_2.yaml")  # Replace with actual config files

benchmark_command="./perf.sh"

HEALTH_CHECK_URL="http://localhost:8000/v1/chat/completions"
MAX_WAIT_SECONDS=60                             # Maximum time to wait for server startup
CHECK_INTERVAL=2                                # Seconds between health checks

HEALTH_CHECK_STRING="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# Health check payload
HEALTH_CHECK_DATA=$(cat <<EOF
{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user", 
        "content": "Hello"
    }
    ],
    "stream": false,
    "max_tokens": 30
}
EOF
)

# Globals for trap cleanup
server_pid=""
benchmark_pid=""

cleanup() {
    echo "Caught interrupt. Cleaning up..."
    if [[ -n "${benchmark_pid:-}" ]] && kill -0 "$benchmark_pid" 2>/dev/null; then
        echo "Killing benchmark process tree (PID $benchmark_pid)"
        pkill -P "$benchmark_pid" 2>/dev/null || true
        kill "$benchmark_pid" 2>/dev/null || true
        wait "$benchmark_pid" 2>/dev/null || true
    fi
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        echo "Killing server process tree (PID $server_pid)"
        pkill -P "$server_pid" 2>/dev/null || true
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    pkill -f .*circusd.*
    exit 1
}

trap cleanup SIGINT

# Process each configuration sequentially
for config in "${configs[@]}"; do
    base_name="${config%.*}"
    log_file="${base_name}.log"
    perf_log_file="${base_name}.genaiperf.log"
    server_pid=""

   if [[ "$config" == *"disagg"* ]]; then
        server_cmd="$SERVER_CMD_DISAGG"
        echo "=== Starting DISAGGREGATED server with ${config} ==="
    else
        server_cmd="$SERVER_CMD_AGG"
        echo "=== Starting AGGREGATED server with ${config} ==="
    fi
    
    # Start server with current config
    echo "=== Starting server with ${config} ==="
    ${server_cmd} "/workspace/examples/llm/configs/${config}" > "${log_file}" 2>&1 &
    server_pid=$!

    # Health check loop
    echo "Waiting for server to become responsive..."
    wait_time=0
    while true; do

	if curl --fail --max-time 5 -X POST -H "Content-Type: application/json" \
            -d "$HEALTH_CHECK_DATA" "$HEALTH_CHECK_URL" ; then
            echo "Server responded successfully to completion request"
            break
        fi
	
        # Check if server process is still running
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Error: Server process died during startup"
            exit 1
        fi
        
        # Check timeout
        if (( wait_time >= MAX_WAIT_SECONDS )); then
            echo "Error: Server did not become responsive within ${MAX_WAIT_SECONDS} seconds"
	    cleanup
            exit 1
        fi
        
        ((wait_time += CHECK_INTERVAL))
        sleep "$CHECK_INTERVAL"
    done

    
    # Give server a moment to start (adjust if needed)
    sleep 10
    
    # Run benchmark
    echo "=== Running benchmark against ${config} ==="
    export ARTIFACT_DIR="artifacts_${base_name}"
    
    (${benchmark_command} 2>&1 | tee "${perf_log_file}") &

    benchmark_pid=$!
    wait $benchmark_pid
    benchmark_pid=""
    
    # Cleanup
    echo "=== Benchmark complete - terminating server ==="
    if kill -0 "${server_pid}" 2>/dev/null; then
        echo "Killing server process tree (PID ${server_pid})"
        pkill -P "${server_pid}" 2>/dev/null || true  # Kill children first
        kill "${server_pid}" 2>/dev/null || true      # Then parent
        wait "${server_pid}" 2>/dev/null || true
	pkill -f .*circusd.* || true
    fi

    server_pid=""
    
    echo "=== Finished processing ${config} ==="
    echo
done



echo "All configurations processed"
