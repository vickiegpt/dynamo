# Dynamo Companion Launcher

The Dynamo companion launcher starts the vLLM companion coordinator for CUDA IPC weight sharing. The coordinator acts as a message router between vLLM workers and companion servers.

## Usage

The companion launcher can be run as a module:

```bash
# Run with default ports (auto-selected)
python -m dynamo.companion

# Run with specific ports
python -m dynamo.companion --coordinator-port 12345 --companion-master-port 12346
```

## Arguments

- `--coordinator-port`: Port for the companion coordinator (defaults to 55800 from CompanionConfig if not provided)
- `--companion-master-port`: Master port for distributed init (auto-selected if not provided)

## Integration with vLLM Backend

The companion launcher must be started **before** launching vLLM workers. The workflow is:

1. Launch the companion coordinator:
   ```bash
   python -m dynamo.companion --coordinator-port 12345
   ```

2. Launch vLLM with companion process enabled:
   ```bash
   # Your vLLM launch command with --enable-companion-process
   ```

The companion coordinator:
1. Starts companion servers for all available GPUs
2. Routes messages between vLLM workers and companion servers
3. Ensures all companion servers complete handshake during startup

## Example

```python
from dynamo.companion.launcher import launch_companion

# Launch companion coordinator
launcher = launch_companion(
    coordinator_port=12345,  # Optional, auto-selected if not provided
    companion_master_port=12346,  # Optional, auto-selected if not provided
)

print(f"Coordinator running at: {launcher.get_coordinator_address()}")
# Now vLLM workers can connect to this address
```
