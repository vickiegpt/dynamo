# Dynamo Layout Transfer Benchmarking Tool

## Overview

`bench_layout_transfers` is a CLI tool for benchmarking memory layout transfer performance in Dynamo's block manager. It tracks and analyzes the performance characteristics of different memory layouts (`FullyContiguous` vs `LayerSeparate`) across various transfer paths (Host↔Device, Host↔Disk GDS/NIXL).

## Purpose

This tool helps answer questions like:
- How do `FullyContiguous` and `LayerSeparate` layouts compare in terms of transfer performance?
- What are the transfer size distributions for different layouts?
- How many transfers occur for different workloads?
- What is the bandwidth utilization for different transfer paths?

## Building

```bash
cd dynamo
cargo build -p dynamo-llm --features block-manager --bin bench_layout_transfers --target-dir /tmp_build
```

## Workflow

### 1. Initialize a Benchmark Session

Create a new benchmark session with a descriptive name:

```bash
cargo run -p dynamo-llm --features block-manager --bin bench_layout_transfers --target-dir /tmp_build -- init "my_test_session"
```

This will:
- Create a new benchmark session
- Save it to `~/.dynamo/bench_my_test_session.json`
- Set the `DYNAMO_BENCHMARK_SESSION` environment variable (you need to export this in your shell)

**Important:** Export the session name so your vLLM process can find it:
```bash
export DYNAMO_BENCHMARK_SESSION="my_test_session"
```

### 2. Run Your Workload

Start your vLLM server or application with the Dynamo connector. The benchmark hooks will automatically collect data as transfers occur:

```bash
# Example: Start vLLM with your connector
vllm serve <your-model> --enable-dynamo-kvbm-connector ...
```

As your workload runs, the benchmarking system will:
- Track every memory transfer
- Record transfer sizes, counts, and timing
- Categorize by transfer path (Host→Disk, Device→Host, etc.)
- Save data periodically to the session file

### 3. Check Status

While your workload is running (or after), check the current benchmark status:

```bash
cargo run -p dynamo-llm --features block-manager --bin bench_layout_transfers --target-dir /tmp_build -- status
```

Or for a specific session:
```bash
cargo run -p dynamo-llm --features block-manager --bin bench_layout_transfers --target-dir /tmp_build -- status --session "my_test_session"
```

This shows a quick summary:
- Number of transfers collected
- Total bytes transferred
- Number of unique transfer paths

### 4. Generate Report

Generate a detailed performance report:

```bash
cargo run -p dynamo-llm --features block-manager --bin bench_layout_transfers --target-dir /tmp_build -- report
```

Or for a specific session:
```bash
cargo run -p dynamo-llm --features block-manager --bin bench_layout_transfers --target-dir /tmp_build -- report --session "my_test_session"
```

The report includes:
- **Per-path statistics**: Min/Max/Avg transfer sizes
- **Total metrics**: Total transfers, blocks, bytes
- **Bandwidth**: Average MB/s (if timing data available)
- **Breakdown by transfer path**: Host→Disk (NIXL), Device→Host (CUDA), etc.

### 5. Export Data

Export raw benchmark data as JSON for further analysis:

```bash
cargo run -p dynamo-llm --features block-manager --bin bench_layout_transfers --target-dir /tmp_build -- export --session "my_test_session" --output results.json
```

### 6. Reset or Start Over

Reset the current session to start fresh:

```bash
cargo run -p dynamo-llm --features block-manager --bin bench_layout_transfers --target-dir /tmp_build -- reset
```

Or initialize a new session (which automatically creates a fresh state).

## Commands Reference

### `init <session_name>`
Initialize a new benchmark session with the given name.

**Example:**
```bash
bench_layout_transfers init "layerseparate_100req"
```

### `status [--session <name>]`
Show current benchmark status (quick summary).

**Options:**
- `--session <name>`: Check status of a specific session (default: active session)

**Example:**
```bash
bench_layout_transfers status --session "layerseparate_100req"
```

### `report [--session <name>]`
Generate detailed performance report.

**Options:**
- `--session <name>`: Generate report for a specific session (default: active session)

**Example:**
```bash
bench_layout_transfers report --session "layerseparate_100req"
```

### `export --session <name> --output <file>`
Export benchmark data as JSON.

**Options:**
- `--session <name>`: Session to export
- `--output <file>`: Output JSON file path

**Example:**
```bash
bench_layout_transfers export --session "test1" --output results.json
```

### `reset`
Reset the current active benchmark session.

**Example:**
```bash
bench_layout_transfers reset
```

### `test`
Run a quick test to verify the benchmarking system is working.

**Example:**
```bash
bench_layout_transfers test
```

### `enable` / `disable`
Enable or disable benchmark data collection (without losing existing data).

**Examples:**
```bash
bench_layout_transfers disable  # Stop collecting data
bench_layout_transfers enable   # Resume collecting data
```

## Transfer Paths

The tool tracks transfers across these paths:

| Path | Technology | Description |
|------|------------|-------------|
| **Host → Disk** | NIXL (GDS) | CPU memory → NVMe/disk via GPUDirect Storage |
| **Disk → Device** | NIXL (GDS) | NVMe/disk → GPU memory via GPUDirect Storage |
| **Host → Device** | CUDA | CPU memory → GPU memory via CUDA memcpy |
| **Device → Host** | CUDA | GPU memory → CPU memory via CUDA memcpy |
| **Device → Device** | CUDA | GPU memory → GPU memory |
| **Host → Host** | memcpy | CPU memory → CPU memory |

## Memory Layouts

### FullyContiguous
All KV cache data for a block is stored in a single contiguous memory region. Transfers happen as single large operations.

**Characteristics:**
- Fewer total transfers
- Larger transfer sizes
- `Total Transfers` ≈ `Total Blocks`

### LayerSeparate
KV cache data is organized by layer, with each layer stored separately. Transfers happen layer-by-layer.

**Characteristics:**
- More total transfers (one per layer per outer dimension)
- Smaller transfer sizes
- `Total Transfers` >> `Total Blocks`
- For a model with 80 layers and 2 outer dims: `Total Transfers` = `Total Blocks` × 80 × 2

## Understanding the Metrics

### Total Transfers
Number of individual transfer operations (e.g., CUDA memcpy calls, NIXL requests).

### Total Blocks
Number of logical blocks transferred. For `LayerSeparate`, this counts unique blocks, not individual layer transfers.

### Total Bytes
Sum of all bytes transferred across all operations.

### Min / Max / Avg Size
- **Min Size**: Smallest single transfer size (typically layer size for LayerSeparate)
- **Max Size**: Largest single transfer size (typically block size for FullyContiguous)
- **Avg Size**: `Total Bytes / Total Transfers`

### Bandwidth (MB/s)
Average transfer throughput if timing data is available.

## Troubleshooting

### "No benchmark session active"

Make sure you:
1. Ran `init` to create a session
2. Exported the `DYNAMO_BENCHMARK_SESSION` environment variable:
   ```bash
   export DYNAMO_BENCHMARK_SESSION="your_session_name"
   ```

### No data collected

Check that:
1. Your application is actually performing transfers (check logs)
2. Benchmarking is enabled (run `bench_layout_transfers enable`)
3. The session file exists: `ls ~/.dynamo/bench_*.json`
4. You're running with the correct features: `--features block-manager`

### Different numbers between `status` and `report`

Both commands reload data from the session file, so they should match. If they don't:
- Your application might still be writing data
- Try running `status` or `report` again after your workload completes

## Data Storage

Benchmark data is stored in:
```
~/.dynamo/bench_<session_name>.json
~/.dynamo/dynamo_benchmark_active_session.txt  # Tracks active session name
```

## Advanced Usage

### Comparing Layouts

To compare `FullyContiguous` vs `LayerSeparate`:

1. Run test with `FullyContiguous` layout:
   ```bash
   export DYNAMO_BENCHMARK_SESSION="test_contiguous"
   bench_layout_transfers init "test_contiguous"
   # Run your workload with FullyContiguous layout
   bench_layout_transfers report > contiguous_report.txt
   ```

2. Run test with `LayerSeparate` layout:
   ```bash
   export DYNAMO_BENCHMARK_SESSION="test_layerseparate"
   bench_layout_transfers init "test_layerseparate"
   # Run your workload with LayerSeparate layout
   bench_layout_transfers report > layerseparate_report.txt
   ```

3. Compare reports:
   ```bash
   diff -u contiguous_report.txt layerseparate_report.txt
   ```

### Filtering by Transfer Path

The report shows data broken down by transfer path. To focus on specific paths:
- **NIXL transfers only**: Look for "Pinned → Disk" or "Disk → Device" sections
- **CUDA transfers**: Look for "Device → Pinned" or "Pinned → Device" sections

## Implementation Details

The benchmarking system uses:
- **Non-intrusive hooks** injected at key transfer points
- **Atomic counters** for lock-free metric updates
- **File-based persistence** for data durability
- **Session management** for organizing multiple test runs
- **Transfer path categorization** based on source/destination storage types

Hooks are placed in:
- `dynamo/lib/llm/src/block_manager/block/transfer/cuda.rs` (CUDA transfers)
- `dynamo/lib/llm/src/block_manager/block/transfer/memcpy.rs` (CPU transfers)
- `dynamo/lib/llm/src/block_manager/block/transfer/nixl.rs` (GDS transfers)

Core benchmarking logic:
- `dynamo/lib/llm/src/block_manager/bench.rs` (metrics, storage)
- `dynamo/lib/llm/src/block_manager/bench/hooks.rs` (hook functions)
- `dynamo/lib/llm/src/block_manager/bench/reporter.rs` (report generation)