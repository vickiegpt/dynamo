# KV Cache Routing

KV Cache Routing is a core feature of Dynamo that intelligently directs requests to workers with the most relevant cached data, significantly improving performance by reducing redundant KV cache recomputation.

## Overview

KV-aware routing leverages the specific properties of Large Language Models (LLMs) to optimize request distribution. Instead of simple load balancing, it routes requests to workers with the highest KV cache hit rates, enabling immediate processing even under heavy load.

## Key Benefits

- **3x improvement in Time To First Token (TTFT)**
- **2x reduction in average request latency**
- **Eliminates unnecessary KV cache recomputation**
- **Maintains load balance through worker utilization metrics**

## Quick Start

Enable KV routing when starting the frontend:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

## Learn More

For comprehensive information about KV Cache Routing, including:
- Architecture and design principles
- Cost calculation algorithms
- Worker selection strategies
- Inter-router communication
- Performance analysis and benchmarks

See the detailed [KV Cache Routing Architecture](../../architecture/kv_cache_routing.md) documentation.

For practical configuration, tuning parameters, and CLI options, see the [Router Configuration Guide](README.md).