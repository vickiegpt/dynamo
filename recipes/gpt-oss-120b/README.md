Note:

- This recipe is for gpt-oss-120b in aggregated mode.

# Running the recipe
```bash
./run.sh --model gpt-oss-120b --framework trtllm agg
```

# Images

This recipe uses the following container images using custom commits. You might need to build the images to reproduce the benchmark.

* aiperf
 Based on commit [70af59489df24a601dba57604a7341966150b366](https://github.com/ai-dynamo/aiperf/commit/70af59489df24a601dba57604a7341966150b366)

* dynamo trtllm runtime for arm64
based on commit [7fdf50fec2cae9112224f5cea26cef3dde78506f](https://github.com/ai-dynamo/dynamo/commit/7fdf50fec2cae9112224f5cea26cef3dde78506f)
```
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:7fdf50fec2cae9112224f5cea26cef3dde78506f-35606896-trtllm-arm64
```