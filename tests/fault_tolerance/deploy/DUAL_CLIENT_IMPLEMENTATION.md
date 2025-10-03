# Dual Client Implementation for Fault Tolerance Tests

## Overview

This document describes the implementation of dual client support for fault tolerance tests, allowing tests to use either the **AI-Perf** client or the **legacy custom client**.

## Motivation

The PR #3289 replaced the custom client with AI-Perf for load generation. However, there was a requirement to support both clients simultaneously for:
- Comparing performance and results between implementations
- Gradual migration path from legacy to AI-Perf
- Supporting different use cases (AI-Perf for comprehensive metrics, legacy for simple testing)

## Architecture

The implementation uses a **factory pattern** to cleanly separate client implementations and parsers while providing a unified interface.

```
┌─────────────────────────────────────────────────────────────┐
│                    test_deployment.py                        │
│                     (Test Runner)                            │
└──────────────────────┬──────────────────────┬────────────────┘
                       │                      │
                       ├──────────────────────┤
                       │                      │
          ┌────────────▼─────────┐ ┌─────────▼──────────┐
          │  client_factory.py   │ │ parse_factory.py   │
          │  (Client Selection)  │ │ (Parser Selection) │
          └──────┬───────┬───────┘ └──────┬──────┬──────┘
                 │       │                │      │
         ┌───────▼───┐ ┌─▼──────────┐ ┌──▼──────▼───────────┐
         │ client.py │ │legacy_     │ │parse_   │legacy_    │
         │ (AI-Perf) │ │client.py   │ │results  │parse_     │
         └───────────┘ └────────────┘ │.py      │results.py │
                                      └─────────┴───────────┘
```

## File Structure

### New Files Created

1. **`legacy_client.py`** - Legacy custom HTTP client implementation
   - Direct HTTP requests with retry logic
   - JSONL output format
   - Rate limiting support
   - Per-request retry strategy

2. **`client_factory.py`** - Factory for client selection
   - `get_client_function(client_type)` - Returns appropriate client
   - `validate_client_type()` - Validates client type
   - `get_client_description()` - Returns client documentation

3. **`legacy_parse_results.py`** - Parser for legacy JSONL results
   - Parses `client_N.log.txt` files
   - Calculates before/after metrics
   - Recovery time calculation
   - Original table format output

4. **`parse_factory.py`** - Factory for parser selection
   - `detect_result_type()` - Auto-detects result format
   - `parse_test_results()` - Routes to correct parser
   - `get_result_info()` - Returns result metadata

### Modified Files

1. **`scenarios.py`**
   - Added `client_type` field to `Load` dataclass
   - Added `max_request_rate` field for legacy client
   - Added `create_aiperf_load()` helper function
   - Added `create_legacy_load()` helper function

2. **`test_deployment.py`**
   - Updated imports to use factory modules
   - Modified `_clients()` to accept `Load` config object
   - Uses `get_client_function()` for client selection
   - Uses `parse_test_results()` with auto-detection
   - Added logging for client type selection

## Usage

### 1. Using AI-Perf Client (Default)

```python
from tests.fault_tolerance.deploy.scenarios import Load, Scenario, create_aiperf_load

# Option 1: Use default Load (AI-Perf)
load = Load()

# Option 2: Use helper function with custom settings
load = create_aiperf_load(
    clients=20,
    requests_per_client=200,
    input_token_length=200,
    sla=2.0,
)

scenario = Scenario(
    deployment=deployment_spec,
    load=load,
    failures=failures,
    backend="vllm",
)
```

### 2. Using Legacy Client

```python
from tests.fault_tolerance.deploy.scenarios import create_legacy_load

# Use helper function
load = create_legacy_load(
    clients=10,
    requests_per_client=100,
    max_request_rate=1.0,  # Rate limiting
)

scenario = Scenario(
    deployment=deployment_spec,
    load=load,
    failures=failures,
    backend="vllm",
)
```

### 3. Manual Configuration

```python
from tests.fault_tolerance.deploy.scenarios import Load

load = Load(
    clients=15,
    requests_per_client=120,
    client_type="legacy",  # or "aiperf"
    max_request_rate=1.5,
    sla=1.5,
)
```

### 4. Running Tests

```bash
# Run tests with default AI-Perf client
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v

# Results are automatically parsed using the correct parser
```

### 5. Parsing Results Manually

```bash
# Auto-detect and parse
python tests/fault_tolerance/deploy/parse_factory.py test_fault_scenario[...]

# Force specific parser
python tests/fault_tolerance/deploy/parse_factory.py test_dir --force-parser legacy

# Get result info without parsing
python tests/fault_tolerance/deploy/parse_factory.py test_dir --info
```

## Client Comparison

| Feature | AI-Perf Client | Legacy Client |
|---------|---------------|---------------|
| **Implementation** | CLI tool (aiperf) | Custom HTTP loop |
| **Output Format** | JSON/CSV in `client_N/attempt_M/` | JSONL in `client_N.log.txt` |
| **Metrics** | P50/P90/P99, TTFT, ITL, Throughput | Success/Failure counts, Avg latency |
| **Retry Strategy** | Full test retry (3 attempts) | Per-request retry (1 attempt) |
| **Rate Limiting** | Concurrency control | max_request_rate |
| **Default Requests** | 150 per client | 100 per client |
| **Warmup** | Disabled (warmup-request-count=0) | N/A |
| **Streaming** | Supported (--streaming) | Not supported |
| **Token Generation** | Synthetic with mean/stddev | Random word selection |

## Parser Comparison

| Feature | AI-Perf Parser | Legacy Parser |
|---------|---------------|---------------|
| **Input Files** | `profile_export_aiperf.json` | `client_N.log.txt` (JSONL) |
| **Detection** | `client_N/attempt_M/` structure | `client_N.log.txt` files |
| **Metrics Output** | Total/Successful/Failed, Latencies, TTFT/ITL, Throughput | Success/Failed Before/After, Avg latency, Recovery time |
| **Table Format** | Single summary table per test | Before/After comparison table |
| **Multiple Attempts** | Supports retry attempts | Single attempt per request |
| **Recovery Time** | Container restart timestamps | Process log timestamps |

## Auto-Detection Logic

The `parse_factory.py` module automatically detects result type:

```python
def detect_result_type(log_dir):
    # Check for AI-Perf indicators
    if exists("client_N/attempt_M/profile_export_aiperf.json"):
        return "aiperf"
    
    # Check for legacy indicators
    if exists("client_N.log.txt"):
        return "legacy"
    
    # Mixed or unknown
    return None  # or prioritize AI-Perf
```

## Testing Strategy

### Phase 1: Verification ✓
- [x] Create legacy_client.py
- [x] Create client_factory.py
- [x] Create legacy_parse_results.py
- [x] Create parse_factory.py
- [x] Modify scenarios.py
- [x] Modify test_deployment.py

### Phase 2: Testing (Next)
- [ ] Run single test with AI-Perf client
- [ ] Run single test with legacy client
- [ ] Verify auto-detection works
- [ ] Compare results between both clients
- [ ] Test mixed scenarios (some AI-Perf, some legacy)

### Phase 3: Validation
- [ ] Verify all metrics are calculated correctly
- [ ] Test edge cases (failures during client execution)
- [ ] Performance comparison
- [ ] Documentation review

## Example Scenarios

### 1. Compare Both Clients on Same Test

```python
# In scenarios.py
scenarios = {
    "vllm-agg-tp-1-dp-1-frontend-aiperf": Scenario(
        deployment=deployment_spec,
        load=create_aiperf_load(clients=10),
        failures=[Failure(30, "Frontend", "dynamo.frontend")],
        backend="vllm",
    ),
    "vllm-agg-tp-1-dp-1-frontend-legacy": Scenario(
        deployment=deployment_spec,
        load=create_legacy_load(clients=10),
        failures=[Failure(30, "Frontend", "dynamo.frontend")],
        backend="vllm",
    ),
}
```

### 2. Mixed Test Suite

```python
# Different scenarios use different clients based on requirements
scenarios = {
    # Comprehensive metrics - use AI-Perf
    "detailed-test": Scenario(
        load=create_aiperf_load(clients=20, requests_per_client=200),
        ...
    ),
    
    # Quick smoke test - use legacy
    "smoke-test": Scenario(
        load=create_legacy_load(clients=5, requests_per_client=50),
        ...
    ),
}
```

## Key Design Decisions

1. **Factory Pattern**: Chosen for clean separation and extensibility
2. **Auto-Detection**: Automatic format detection reduces configuration burden
3. **Minimal Changes**: Existing AI-Perf code remains untouched
4. **Backward Compatible**: Default behavior uses AI-Perf client
5. **Configuration-Driven**: Client selection via `Load.client_type`
6. **Consistent Interface**: Both clients use same function signature

## Migration Path

1. **Current State**: All tests use AI-Perf by default
2. **Transition**: Can selectively use legacy for comparison
3. **Future**: Deprecate legacy client once AI-Perf is fully validated
4. **Long-term**: Single client implementation (likely AI-Perf)

## Troubleshooting

### Issue: Parser auto-detection fails
**Solution**: Use `force_parser` parameter:
```python
parse_test_results(log_dir=test_dir, force_parser="legacy")
```

### Issue: Wrong client being used
**Solution**: Check `Load.client_type` in scenario configuration

### Issue: Mixed result types in same directory
**Solution**: Parser prioritizes AI-Perf format. Clean directory between runs.

### Issue: Legacy client results not parsing correctly
**Solution**: Verify JSONL format in `client_N.log.txt` files

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `legacy_client.py` | 294 | Legacy HTTP client implementation |
| `client_factory.py` | 125 | Client selection factory |
| `legacy_parse_results.py` | 578 | Legacy result parser |
| `parse_factory.py` | 367 | Parser selection factory |
| `scenarios.py` | +45 | Added dual client support |
| `test_deployment.py` | ~30 changes | Integrated factories |

**Total New Code**: ~1,400 lines  
**Modified Code**: ~75 lines

## References

- PR #3289: Original AI-Perf migration
- AI-Perf Documentation: https://github.com/triton-inference-server/perf_analyzer
- Test Results: See `tests/fault_tolerance/deploy/README.md`

