#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo System Information Checker

A comprehensive diagnostic tool that displays system configuration and Dynamo project status
in a hierarchical tree format. This script checks for:

- System resources (OS, CPU, memory, GPU)
- Service monitoring (etcd, NATS)
- File system (permissions and disk space, detailed with --thorough-check)
- Hugging Face model cache
- Development tools (Cargo/Rust, Maturin, Python)
- LLM frameworks (vllm, sglang, tensorrt_llm)
- Dynamo runtime and framework components
- Installation status and component availability

The output uses status indicators:
- ✅ Component found and working
- ❌ Component missing or error
- ⚠️ Warning condition
- ❓ Component not found (for optional items)

By default, the tool runs quickly by checking only directory permissions and skipping
size calculations. Use --thorough-check for detailed file-level permission analysis,
directory size information, and disk space checking.

Exit codes:
- 0: All critical components are present
- 1: One or more errors detected (❌ status)

Example output (default mode):

System info (hostname=keivenc-linux, IP=10.110.41.216)
├─ OS: Ubuntu 24.04.2 LTS (Noble Numbat) (Linux 6.14.0-29-generic x86_64), Memory=31.4/125.5 GiB, Cores=32
├─ User info: user=ubuntu, uid=1776734304, gid=748400513
├─ ✅ NVIDIA GPU: NVIDIA RTX 6000 Ada Generation, driver 570.133.07, CUDA 12.9, Power=26.30/300.00 W, Memory=10584/49140 MiB
├─ ✅ etcd: service running on localhost:2379
├─ ✅ NATS: service running on localhost:4222
├─ File System
│  ├─ ✅ Dynamo workspace ($HOME/dynamo): writable
│  ├─ ✅ Dynamo .git directory ($HOME/dynamo/.git): writable
│  ├─ ✅ Rustup home ($HOME/dynamo/.build/.rustup): writable
│  ├─ ✅ Cargo home ($HOME/dynamo/.build/.cargo): writable
│  ├─ ✅ Cargo target ($HOME/dynamo/.build/target): writable
│  └─ ✅ site-packages (/opt/dynamo/venv/lib/python3.12/site-packages): writable
├─ ✅ Hugging Face Cache: 5 models in $HOME/.cache/huggingface/hub
├─ ✅ Cargo: /.cargo/bin/cargo, cargo 1.90.0 (840b83a10 2025-07-30)
│  ├─ Cargo home directory: CARGO_HOME=$HOME/dynamo/.build/.cargo
│  └─ Cargo target directory: CARGO_TARGET_DIR=$HOME/dynamo/.build/target
├─ ✅ Maturin: /opt/dynamo/venv/bin/maturin, maturin 1.9.4
├─ ✅ Python: 3.12.3, /opt/dynamo/venv/bin/python
│  ├─ ✅ PyTorch: 2.8.0a0+5228986c39.nv25.06, ✅torch.cuda.is_available
│  └─ PYTHONPATH: $HOME/dynamo:$HOME/dynamo/components/metrics/src:$HOME/dynamo/components/frontend/src:...
├─ 🤖Framework
│  └─ ✅ TensorRT-LLM: 1.1.0rc5, module=/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/__init__.py
└─ Dynamo: $HOME/dynamo, SHA: 9d73be125, Date: 2025-09-26 16:01:01 PDT
   ├─ ✅ Runtime components: ai-dynamo-runtime 0.5.1
   │  │  /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo_runtime-0.5.1.dist-info: created=2025-09-26 15:55:58 PDT
   │  │  /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo_runtime.pth: modified=2025-09-26 15:55:58 PDT
   │  │  └─ →: $HOME/dynamo/lib/bindings/python/src
   │  ├─ ✅ dynamo._core            : $HOME/dynamo/lib/bindings/python/src/dynamo/_core.cpython-312-x86_64-linux-gnu.so, modified=2025-09-26 15:55:58 PDT
   │  ├─ ✅ dynamo.logits_processing: $HOME/dynamo/lib/bindings/python/src/dynamo/logits_processing/__init__.py
   │  ├─ ✅ dynamo.nixl_connect     : $HOME/dynamo/lib/bindings/python/src/dynamo/nixl_connect/__init__.py
   │  ├─ ✅ dynamo.llm              : $HOME/dynamo/lib/bindings/python/src/dynamo/llm/__init__.py
   │  └─ ✅ dynamo.runtime          : $HOME/dynamo/lib/bindings/python/src/dynamo/runtime/__init__.py
   └─ ✅ Framework components: ai-dynamo 0.5.1
      │  /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo-0.5.1.dist-info: created=2025-09-26 15:34:13 PDT
      ├─ ✅ dynamo.frontend : $HOME/dynamo/components/frontend/src/dynamo/frontend/__init__.py
      ├─ ✅ dynamo.llama_cpp: $HOME/dynamo/components/backends/llama_cpp/src/dynamo/llama_cpp/__init__.py
      ├─ ✅ dynamo.mocker   : $HOME/dynamo/components/backends/mocker/src/dynamo/mocker/__init__.py
      ├─ ✅ dynamo.planner  : $HOME/dynamo/components/planner/src/dynamo/planner/__init__.py
      ├─ ✅ dynamo.sglang   : $HOME/dynamo/components/backends/sglang/src/dynamo/sglang/__init__.py
      ├─ ✅ dynamo.trtllm   : $HOME/dynamo/components/backends/trtllm/src/dynamo/trtllm/__init__.py
      └─ ✅ dynamo.vllm     : $HOME/dynamo/components/backends/vllm/src/dynamo/vllm/__init__.py

Example output (--thorough-check mode):

In thorough mode, additional details are shown:
- Hugging Face Cache: Shows individual models with download dates and sizes
- etcd: Shows version, cluster members, and all keys with revision information
- NATS: Shows uptime, server name, connections, messages, JetStream streams, and Dynamo subjects
- File System: Shows detailed directory information and disk usage

├─ ✅ etcd: service running on localhost:2379
│  ├─ Version: 3.6.1
│  ├─ Cluster Members: 1
│  └─ Keys: 10 keys found
│     ├─ Key 1: dyn://dynamo/ports/10.110.41.216/28266 (rev:386, v:1)
│     ├─ Key 2: instances/dynamo/backend/clear_kv_blocks:694d995f0d9ed01b (rev:390, v:1)
│     ├─ Key 3: instances/dynamo/backend/generate:694d995f0d9ed01b (rev:389, v:1)
│     ├─ Key 4: instances/dynamo/backend/load_metrics:694d995f0d9ed01b (rev:387, v:1)
│     ├─ Key 5: mdc/qwen_qwen3-0_6b (rev:4, v:1)
│     ├─ Key 6: mdc/test-mdc-model (rev:100, v:1)
│     ├─ Key 7: mdc/tinyllama_tinyllama-1_1b-chat-v1_0 (rev:288, v:1)
│     ├─ Key 8: models/f5303ed3-ed16-4615-81c3-34e95a4eaad4 (rev:388, v:1)
│     ├─ Key 9: test_concurrent_bucket/concurrent_test_key_198126a9-e297-4f35-bded-edd63b72325f (rev:220, v:1)
│     └─ Key 10: test_concurrent_bucket/concurrent_test_key_45ddffcd-29e0-4185-9bf9-1443cf9f518b (rev:203, v:1)
├─ ✅ NATS: service running on localhost:4222
│  ├─ Server: NA4B6QC5MG5YT3ZY7UPBLNQYNG6QXZLQEGILXT3BQSETPZVMQ6EI5NKN
│  ├─ Uptime: 8d18h59m5s
│  ├─ Connections: 3
│  ├─ Messages: in=23638, out=25596
│  ├─ Subscriptions: 88 active
│  └─ Dynamo Subjects: 10 subjects found
│     ├─ Subject 1: $SRV.PING.dynamo_backend (msgs: 0, conn: 573)
│     ├─ Subject 2: $SRV.PING.dynamo_backend.OMZmmccbcIcvcwKF4JRoZB (msgs: 0, conn: 573)
│     ├─ Subject 3: $SRV.INFO.dynamo_backend (msgs: 0, conn: 573)
│     ├─ Subject 4: $SRV.INFO.dynamo_backend.OMZmmccbcIcvcwKF4JRoZB (msgs: 0, conn: 573)
│     ├─ Subject 5: $SRV.STATS.dynamo_backend (msgs: 124, conn: 573)
│     ├─ Subject 6: $SRV.STATS.dynamo_backend.OMZmmccbcIcvcwKF4JRoZB (msgs: 0, conn: 573)
│     ├─ Subject 7: dynamo_backend.load_metrics-694d995f0d9ed01b (msgs: 0, conn: 573)
│     ├─ Subject 8: dynamo_backend.generate-694d995f0d9ed01b (msgs: 0, conn: 573)
│     ├─ Subject 9: dynamo_backend.clear_kv_blocks-694d995f0d9ed01b (msgs: 0, conn: 573)
│     └─ Subject 10: namespace-dynamo-component-backend-kv-events.* (msgs: 0, conn: 372)
├─ File System
│  ├─ ✅ Dynamo workspace ($HOME/dynamo): writable, size=2.1 GB
│  ├─ ✅ Dynamo .git directory ($HOME/dynamo/.git): writable, size=1.2 GB
│  ├─ ✅ Rustup home ($HOME/dynamo/.build/.rustup): writable, size=45.2 MB
│  ├─ ✅ Cargo home ($HOME/dynamo/.build/.cargo): writable, size=12.8 MB
│  ├─ ✅ Cargo target ($HOME/dynamo/.build/target): writable, size=856.3 MB
│  └─ ✅ site-packages (/opt/dynamo/venv/lib/python3.12/site-packages): writable, size=3.2 GB
├─ ✅ Hugging Face Cache: 5 models in $HOME/.cache/huggingface/hub
│  ├─ Model 1: EricB/mistralrs_tests, downloaded=2025-07-23 17:46:49 PDT, size=20.6 MB
│  ├─ Model 2: Qwen/Qwen3-0.6B, downloaded=2025-07-26 10:03:25 PDT, size=1.41 GB
│  ├─ Model 3: TinyLlama/TinyLlama-1.1B-Chat-v1.0, downloaded=2025-09-16 09:47:01 PDT, size=2.05 GB
│  ├─ Model 4: deepseek-ai/DeepSeek-R1-Distill-Llama-8B, downloaded=2025-08-09 13:33:57 PDT, size=15.0 GB
│  └─ Model 5: llava-hf/llava-1.5-7b-hf, downloaded=2025-08-22 09:57:11 PDT, size=4.01 MB

Usage:
    python dynamo_check.py [--thorough-check] [--terse]

Options:
    --thorough-check  Enable thorough checking (file permissions, directory sizes, etc.)
    --terse           Enable terse output mode
"""

import datetime
import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ANSI color constants
class Colors:
    """ANSI color escape sequences for terminal output."""

    RESET = "\033[0m"
    BRIGHT_RED = "\033[38;5;196m"


class NodeStatus(Enum):
    """Status of a tree node"""

    OK = "ok"  # ✅ Success/available
    ERROR = "error"  # ❌ Error/not found
    WARNING = "warn"  # ⚠️ Warning
    INFO = "info"  # No symbol, just information
    NONE = "none"  # No status indicator
    UNKNOWN = "unknown"  # ❓ Unknown/not found


@dataclass
class NodeInfo:
    """Base class for all information nodes in the tree structure"""

    # Core properties
    label: str  # Main text/description
    desc: Optional[str] = None  # Primary value/description
    status: NodeStatus = NodeStatus.NONE  # Status indicator

    # Additional metadata as key-value pairs
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tree structure
    children: List["NodeInfo"] = field(default_factory=list)

    # Display control
    show_symbol: bool = True  # Whether to show status symbol

    # Class-level terse mode behavior (override in subclasses)
    _always_show_when_terse: bool = False

    def add_child(self, child: "NodeInfo") -> "NodeInfo":
        """Add a child node and return it for chaining"""
        self.children.append(child)
        return child

    def add_metadata(self, key: str, value: str) -> "NodeInfo":
        """Add metadata key-value pair"""
        self.metadata[key] = value
        return self

    def should_show_in_terse_mode(self) -> bool:
        """Determine if this node should be shown in terse mode"""
        return self._always_show_when_terse or self.status in [
            NodeStatus.ERROR,
            NodeStatus.WARNING,
        ]

    def add_child_with_terse_filtering(
        self, child: "NodeInfo", terse_mode: bool
    ) -> "NodeInfo":
        """Add child node with terse mode filtering"""
        if not terse_mode or child.should_show_in_terse_mode():
            self.add_child(child)
        return child

    def render(
        self, prefix: str = "", is_last: bool = True, is_root: bool = True
    ) -> List[str]:
        """Render the tree node and its children as a list of strings"""
        lines = []

        # Determine the connector
        if not is_root:
            # Check if this is a sub-category item
            if self.metadata and self.metadata.get("part_of_previous"):
                connector = "│"
            else:
                connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        # Build the line content
        line_parts = []

        # Add status symbol
        if self.show_symbol and self.status != NodeStatus.NONE:
            if self.status == NodeStatus.OK:
                line_parts.append("✅")
            elif self.status == NodeStatus.ERROR:
                line_parts.append("❌")
            elif self.status == NodeStatus.WARNING:
                line_parts.append("⚠️")
            elif self.status == NodeStatus.UNKNOWN:
                line_parts.append("❓")

        # Add label and value
        if self.desc:
            line_parts.append(f"{self.label}: {self.desc}")
        else:
            line_parts.append(self.label)

        # Add metadata inline - consistent format for all
        if self.metadata:
            metadata_items = []
            for k, v in self.metadata.items():
                # Skip internal metadata that shouldn't be displayed
                if k != "part_of_previous":
                    # Format all metadata consistently as "key=value"
                    metadata_items.append(f"{k}={v}")

            if metadata_items:
                # Use consistent separator (comma) for all metadata
                metadata_str = ", ".join(metadata_items)
                line_parts[-1] += f", {metadata_str}"

        # Construct the full line
        line_content = " ".join(line_parts)
        if current_prefix or line_content:
            lines.append(current_prefix + line_content)

        # Render children
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            if is_root:
                child_prefix = ""
            else:
                child_prefix = prefix + ("   " if is_last else "│  ")
            lines.extend(child.render(child_prefix, is_last_child, False))

        return lines

    def print_tree(self) -> None:
        """Print the tree to console"""
        for line in self.render():
            print(line)

    def has_errors(self) -> bool:
        """Check if this node or any of its children have errors"""
        # Check if this node has an error
        if self.status == NodeStatus.ERROR:
            return True

        # Recursively check all children
        for child in self.children:
            if child.has_errors():
                return True

        return False

    def _replace_home_with_var(self, path: str) -> str:
        """Replace home directory with $HOME in path."""
        home = os.path.expanduser("~")
        if path.startswith(home):
            return path.replace(home, "$HOME", 1)
        return path

    def _is_inside_container(self) -> bool:
        """Check if we're running inside a container."""
        # Check for common container indicators
        container_indicators = [
            # Docker
            os.path.exists("/.dockerenv"),
            # Podman/containerd
            os.path.exists("/run/.containerenv"),
            # Check if cgroup contains docker/containerd
            self._check_cgroup_for_container(),
            # Check environment variables
            os.environ.get("container") is not None,
            os.environ.get("DOCKER_CONTAINER") is not None,
        ]
        return any(container_indicators)

    def _check_cgroup_for_container(self) -> bool:
        """Check cgroup for container indicators."""
        try:
            with open("/proc/1/cgroup", "r") as f:
                content = f.read()
                return any(
                    indicator in content.lower()
                    for indicator in ["docker", "containerd", "podman", "lxc"]
                )
        except Exception:
            return False

    def _get_gpu_container_remedies(self) -> str:
        """Get remedies for GPU issues when running inside a container."""
        return "maybe try a docker restart?"

    def _format_timestamp_pdt(self, timestamp: float) -> str:
        """Format timestamp as PDT time string."""
        dt_utc = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        # Convert to PDT (UTC-7)
        dt_pdt = dt_utc - datetime.timedelta(hours=7)
        return dt_pdt.strftime("%Y-%m-%d %H:%M:%S PDT")


class SystemInfo(NodeInfo):
    """Root node for system information"""

    def __init__(
        self,
        hostname: Optional[str] = None,
        thorough_check: bool = False,
        terse: bool = False,
    ):
        self.thorough_check = thorough_check
        self.terse = terse
        if hostname is None:
            hostname = platform.node()

        # Get IP address
        ip_address = self._get_ip_address()

        # Format label with hostname and IP
        if ip_address:
            label = f"System info (hostname={hostname}, IP={ip_address})"
        else:
            label = f"System info (hostname={hostname})"

        super().__init__(label=label, status=NodeStatus.INFO)

        # Suppress Prometheus endpoint warnings from planner module
        self._suppress_planner_warnings()

        # Collect and add all system information
        # Always show: OS, User, GPU
        self.add_child(OSInfo())
        self.add_child(UserInfo())
        self.add_child(GPUInfo())

        # Add all components, filtering based on terse mode and component attributes
        self._add_components_with_terse_filtering()

        # Add Framework right before Dynamo
        self.add_child(FrameworkInfo())

        # Add Dynamo workspace info (always show, even if not found)
        self.add_child(DynamoInfo(thorough_check=self.thorough_check, terse=self.terse))

    def _get_ip_address(self) -> Optional[str]:
        """Get the primary IP address of the system."""
        try:
            import socket

            # Get hostname
            hostname = socket.gethostname()
            # Get IP address
            ip_address = socket.gethostbyname(hostname)
            # Filter out localhost
            if ip_address.startswith("127."):
                # Try to get external IP by connecting to a public DNS
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # Connect to Google DNS (doesn't actually send data)
                    s.connect(("8.8.8.8", 80))
                    ip_address = s.getsockname()[0]
                finally:
                    s.close()
            return ip_address
        except Exception:
            return None

    def _suppress_planner_warnings(self) -> None:
        """Suppress Prometheus endpoint warnings from planner module during import testing."""
        # The planner module logs a warning about Prometheus endpoint when imported
        # outside of a Kubernetes cluster. Suppress this for cleaner output.
        planner_logger = logging.getLogger("dynamo.planner.defaults")
        planner_logger.setLevel(logging.ERROR)
        # Also suppress the defaults._get_default_prometheus_endpoint logger
        defaults_logger = logging.getLogger("defaults._get_default_prometheus_endpoint")
        defaults_logger.setLevel(logging.ERROR)

    def _add_components_with_terse_filtering(self) -> None:
        """Add components based on terse mode and always_show_when_terse attribute"""
        # Create all components in the desired order
        components_to_check = [
            EtcdInfo(thorough_check=self.thorough_check),
            NatsInfo(thorough_check=self.thorough_check),
            FilePermissionsInfo(thorough_check=self.thorough_check),
            HuggingFaceInfo(thorough_check=self.thorough_check),
            CargoInfo(thorough_check=self.thorough_check),
            MaturinInfo(),
            PythonInfo(),
        ]

        for component in components_to_check:
            self.add_child_with_terse_filtering(component, self.terse)


class ServiceInfo(NodeInfo):
    """Base class for service monitoring (NATS, etcd, etc.)"""

    def __init__(self, service_name: str, thorough_check: bool = False):
        self.service_name = service_name
        self.thorough_check = thorough_check

        # Parse configuration and check service
        host, port = self._parse_service_config()
        is_running = self._check_service_connection(host, port)

        if is_running:
            super().__init__(
                label=service_name,
                desc=f"service running on {host}:{port}",
                status=NodeStatus.OK,
            )
            if thorough_check:
                self._add_service_details(host, port)
        else:
            super().__init__(
                label=service_name,
                desc=f"service not available on {host}:{port}",
                status=NodeStatus.WARNING,
            )

    def _parse_service_config(self) -> tuple:
        """Parse service configuration from environment variables. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _parse_service_config")

    def _check_service_connection(self, host: str, port: int) -> bool:
        """Check if service is running by attempting connection."""
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _add_service_details(self, host: str, port: int):
        """Add detailed service information. Override in subclasses."""
        pass

    def _make_http_request(self, url: str, timeout: int = 2) -> dict:
        """Make HTTP request with common error handling."""
        try:
            import json
            import urllib.request

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except Exception:
            return {}


class NatsInfo(ServiceInfo):
    """NATS service information"""

    def __init__(self, thorough_check: bool = False):
        super().__init__("NATS", thorough_check)

    def _parse_service_config(self) -> tuple:
        """Parse NATS server configuration from environment variables.

        Uses the same environment variables as the actual Dynamo codebase:
        - NATS_SERVER: Used in lib/runtime/src/transports/nats.rs (default_server function)
        - Also used in lib/llm/src/kv_router/publisher.rs and subscriber.rs
        - Set by components/backends/sglang/slurm_jobs/scripts/worker_setup.py
        """
        # Check for NATS_SERVER (primary environment variable used in Rust/Python code)
        # See: lib/runtime/src/transports/nats.rs:284 (default_server function)
        nats_server = os.environ.get("NATS_SERVER")
        if nats_server:
            try:
                # Parse URL like "nats://localhost:4222"
                import re

                match = re.match(r"nats://([^:]+):(\d+)", nats_server)
                if match:
                    host = match.group(1)
                    port = int(match.group(2))
                    return host, port
            except Exception:
                pass

        # Check for NATS_SERVER_PORT + NATS_SERVER_HOST (alternative format)
        if "NATS_SERVER_PORT" in os.environ:
            try:
                port = int(os.environ["NATS_SERVER_PORT"])
                host = os.environ.get("NATS_SERVER_HOST", "localhost")
                return host, port
            except ValueError:
                pass

        # Fallback to defaults (same as Rust code default: "nats://localhost:4222")
        # See: lib/runtime/src/transports/nats.rs:288
        return "localhost", 4222

    def _add_service_details(self, host: str, port: int):
        """Add NATS streams and connection details in thorough mode."""
        try:
            # Try to get NATS server info using HTTP monitoring endpoint
            nats_info = self._get_nats_server_info(host, port)
            if nats_info:
                # Add server info
                server = (
                    nats_info.get("server_id")
                    or nats_info.get("server_name")
                    or nats_info.get("name")
                )
                if server:
                    server_node = NodeInfo(
                        label="Server",
                        desc=server,
                        status=NodeStatus.INFO,
                    )
                    self.add_child(server_node)

                # Add uptime
                if "uptime" in nats_info:
                    uptime_str = self._format_uptime(nats_info["uptime"])
                    uptime_node = NodeInfo(
                        label="Uptime", desc=uptime_str, status=NodeStatus.INFO
                    )
                    self.add_child(uptime_node)

                # Add connection count
                if "connections" in nats_info:
                    conn_node = NodeInfo(
                        label="Connections",
                        desc=str(nats_info["connections"]),
                        status=NodeStatus.INFO,
                    )
                    self.add_child(conn_node)

                # Add message stats
                if "in_msgs" in nats_info and "out_msgs" in nats_info:
                    msgs_node = NodeInfo(
                        label="Messages",
                        desc=f"in={nats_info['in_msgs']}, out={nats_info['out_msgs']}",
                        status=NodeStatus.INFO,
                    )
                    self.add_child(msgs_node)

                # Add JetStream info if available
                if "jetstream" in nats_info:
                    js_info = nats_info["jetstream"] or {}
                    streams = (js_info.get("stats") or {}).get(
                        "streams"
                    ) or js_info.get("streams")
                    if streams is not None:
                        streams_node = NodeInfo(
                            label="JetStream Streams",
                            desc=str(streams),
                            status=NodeStatus.INFO,
                        )
                        self.add_child(streams_node)

                # Add subscription and subject information
                self._add_nats_subscriptions(host, port)
            else:
                # Fallback: just show that monitoring endpoint is not accessible
                monitor_node = NodeInfo(
                    label="Monitoring",
                    desc="HTTP monitoring endpoint not accessible",
                    status=NodeStatus.WARNING,
                )
                self.add_child(monitor_node)
        except Exception as e:
            error_node = NodeInfo(
                label="Details",
                desc=f"Could not retrieve NATS details: {str(e)[:50]}...",
                status=NodeStatus.WARNING,
            )
            self.add_child(error_node)

    def _get_nats_server_info(self, host: str, port: int) -> Optional[dict]:
        """Get NATS server info from HTTP monitoring endpoint."""
        try:
            import json
            import urllib.request

            # NATS monitoring port is typically client_port + 4000 (e.g., 4222 -> 8222)
            monitor_port = port + 4000  # Try 8222 for default 4222
            monitor_url = f"http://{host}:{monitor_port}/varz"

            # Try to get server info with a short timeout
            req = urllib.request.Request(monitor_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode())
                return data
        except Exception:
            # Try alternative monitoring port
            try:
                monitor_port = 8222  # Standard NATS monitoring port
                monitor_url = f"http://{host}:{monitor_port}/varz"

                req = urllib.request.Request(monitor_url)
                with urllib.request.urlopen(req, timeout=2) as response:
                    data = json.loads(response.read().decode())
                    return data
            except Exception:
                pass
        return None

    def _format_uptime(self, uptime_seconds) -> str:
        """Format uptime from seconds to human readable format."""
        try:
            # NATS uptime is typically in format like "1h2m3.456s" or just seconds
            if isinstance(uptime_seconds, str):
                # If it's already formatted (like "1h2m3s"), return as is
                if (
                    "h" in uptime_seconds
                    or "m" in uptime_seconds
                    or "s" in uptime_seconds
                ):
                    return uptime_seconds
                # Otherwise try to parse as seconds
                uptime_seconds = float(uptime_seconds.rstrip("s"))

            seconds = int(float(uptime_seconds))

            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                minutes = seconds // 60
                secs = seconds % 60
                return f"{minutes}m{secs}s"
            elif seconds < 86400:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours}h{minutes}m"
            else:
                days = seconds // 86400
                hours = (seconds % 86400) // 3600
                return f"{days}d{hours}h"
        except (ValueError, TypeError):
            return str(uptime_seconds)

    def _add_nats_subscriptions(self, host: str, port: int):
        """Add NATS subscription and subject information."""
        try:
            import json
            import urllib.request

            # NATS monitoring port is typically client_port + 4000 (e.g., 4222 -> 8222)
            monitor_port = port + 4000  # Try 8222 for default 4222

            # Get subscription statistics
            subsz_url = f"http://{host}:{monitor_port}/subsz"
            req = urllib.request.Request(subsz_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                subsz_data = json.loads(response.read().decode())

                if "num_subscriptions" in subsz_data:
                    subs_node = NodeInfo(
                        label="Subscriptions",
                        desc=f"{subsz_data['num_subscriptions']} active",
                        status=NodeStatus.INFO,
                    )
                    self.add_child(subs_node)

            # Get detailed subscription list to find Dynamo-related subjects
            subsz_detail_url = f"http://{host}:{monitor_port}/subsz?subs=1"
            req = urllib.request.Request(subsz_detail_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                subsz_detail = json.loads(response.read().decode())

                dynamo_subjects = []
                for sub in subsz_detail.get("subscriptions_list", []):
                    subject = sub.get("subject", "")
                    if any(
                        pattern in subject.lower() for pattern in ["dynamo", "backend"]
                    ):
                        msgs = sub.get("msgs", 0)
                        cid = sub.get("cid", 0)
                        dynamo_subjects.append((subject, msgs, cid))

                if dynamo_subjects:
                    subjects_node = NodeInfo(
                        label="Dynamo Subjects",
                        desc=f"{len(dynamo_subjects)} subjects found",
                        status=NodeStatus.INFO,
                    )
                    self.add_child(subjects_node)

                    # Show all Dynamo subjects
                    for i, (subject, msgs, cid) in enumerate(dynamo_subjects):
                        subject_node = NodeInfo(
                            label=f"Subject {i+1}",
                            desc=f"{subject} (msgs: {msgs}, conn: {cid})",
                            status=NodeStatus.INFO,
                        )
                        subjects_node.add_child(subject_node)
                else:
                    no_subjects_node = NodeInfo(
                        label="Dynamo Subjects",
                        desc="no Dynamo-related subjects found",
                        status=NodeStatus.INFO,
                    )
                    self.add_child(no_subjects_node)

        except Exception as e:
            error_node = NodeInfo(
                label="Subscriptions",
                desc=f"Could not retrieve subscription info: {str(e)[:50]}...",
                status=NodeStatus.WARNING,
            )
            self.add_child(error_node)


class EtcdInfo(ServiceInfo):
    """etcd service information"""

    def __init__(self, thorough_check: bool = False):
        super().__init__("etcd", thorough_check)

    def _parse_service_config(self) -> tuple:
        """Parse etcd configuration from environment variables.

        Uses the same environment variables as the actual Dynamo codebase:
        - ETCD_ENDPOINTS: Used in lib/runtime/src/transports/etcd.rs (default_servers function)
        - Set by components/backends/sglang/slurm_jobs/scripts/worker_setup.py
        - Also configured in deploy/helm/chart/values.yaml and docker-compose.yml
        """
        # Check for ETCD_ENDPOINTS first (primary environment variable used in Rust/Python code)
        # See: lib/runtime/src/transports/etcd.rs:499 (default_servers function)
        etcd_endpoints = os.environ.get("ETCD_ENDPOINTS")
        if etcd_endpoints:
            try:
                # Parse URL like "http://localhost:2379" (take first endpoint if comma-separated)
                import re

                first_endpoint = etcd_endpoints.split(",")[0].strip()
                match = re.match(r"https?://([^:]+):(\d+)", first_endpoint)
                if match:
                    host = match.group(1)
                    port = int(match.group(2))
                    return host, port
            except Exception:
                pass

        # Check for ETCD_LISTEN_CLIENT_URLS (alternative format)
        if "ETCD_LISTEN_CLIENT_URLS" in os.environ:
            try:
                import re

                urls = os.environ["ETCD_LISTEN_CLIENT_URLS"]
                # Parse URL like "http://localhost:2379" (take first URL if comma-separated)
                first_url = urls.split(",")[0].strip()
                match = re.match(r"https?://([^:]+):(\d+)", first_url)
                if match:
                    host = match.group(1)
                    if host == "0.0.0.0":
                        host = "localhost"  # Convert bind-all to localhost for client access
                    port = int(match.group(2))
                    return host, port
            except Exception:
                pass

        # Fallback to defaults (same as Rust code default: "http://localhost:2379")
        # See: lib/runtime/src/transports/etcd.rs:504
        return "localhost", 2379

    def _add_service_details(self, host: str, port: int):
        """Add etcd keys and cluster details in thorough mode."""
        try:
            etcd_ver = self._get_etcd_info(host, port)  # /version
            etcd_stats = self._get_etcd_stats(host, port) or {}
            if etcd_ver and "etcdserver" in etcd_ver:
                self.add_child(
                    NodeInfo(
                        label="Version",
                        desc=etcd_ver["etcdserver"],
                        status=NodeStatus.INFO,
                    )
                )
            # derive IDs from v3 APIs
            cid = (etcd_stats.get("header") or {}).get("cluster_id")
            mid = (etcd_stats.get("header") or {}).get("member_id")
            if cid:
                self.add_child(
                    NodeInfo(label="Cluster ID", desc=str(cid), status=NodeStatus.INFO)
                )
            if mid:
                self.add_child(
                    NodeInfo(label="Member ID", desc=str(mid), status=NodeStatus.INFO)
                )

            # Add uptime and member info from etcd_stats
            if etcd_stats:
                # Add uptime from leader stats
                if "uptime" in etcd_stats:
                    uptime_str = self._format_etcd_uptime(etcd_stats["uptime"])
                    uptime_node = NodeInfo(
                        label="Uptime", desc=uptime_str, status=NodeStatus.INFO
                    )
                    self.add_child(uptime_node)

                # Add member count from cluster stats
                if "members" in etcd_stats:
                    members_node = NodeInfo(
                        label="Cluster Members",
                        desc=str(etcd_stats["members"]),
                        status=NodeStatus.INFO,
                    )
                    self.add_child(members_node)

            # Try to get all keys from etcd
            all_keys = self._get_all_keys(host, port)
            if all_keys:
                keys_node = NodeInfo(
                    label="Keys",
                    desc=f"{len(all_keys)} keys found",
                    status=NodeStatus.INFO,
                )
                self.add_child(keys_node)

                # Show all keys with their metadata
                for i, (key, create_rev, mod_rev, version) in enumerate(all_keys):
                    if create_rev is not None:
                        # v3 API with revision info
                        desc = f"{key} (rev:{create_rev}"
                        if mod_rev != create_rev:
                            desc += f", mod:{mod_rev}"
                        if version and version != 1:
                            desc += f", v:{version}"
                        desc += ")"
                    else:
                        # v2 API or no revision info
                        desc = key

                    key_node = NodeInfo(
                        label=f"Key {i+1}", desc=desc, status=NodeStatus.INFO
                    )
                    keys_node.add_child(key_node)
            else:
                # Check if we can access etcd API at all
                api_accessible = self._test_etcd_api_access(host, port)
                if api_accessible:
                    # API works but no Dynamo keys found - this is normal for fresh etcd
                    keys_node = NodeInfo(
                        label="Dynamo Keys",
                        desc="no Dynamo keys found (empty database)",
                        status=NodeStatus.INFO,
                    )
                    self.add_child(keys_node)
                else:
                    # API is not accessible
                    api_node = NodeInfo(
                        label="API Access",
                        desc="etcd API not accessible",
                        status=NodeStatus.WARNING,
                    )
                    self.add_child(api_node)

        except Exception as e:
            error_node = NodeInfo(
                label="Details",
                desc=f"Could not retrieve etcd details: {str(e)[:50]}...",
                status=NodeStatus.WARNING,
            )
            self.add_child(error_node)

    def _get_etcd_info(self, host: str, port: int) -> Optional[dict]:
        """Get etcd server info from HTTP API."""
        try:
            import json
            import urllib.request

            # Try to get etcd version and cluster info
            version_url = f"http://{host}:{port}/version"

            req = urllib.request.Request(version_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode())
                return data
        except Exception:
            pass
        return None

    def _get_etcd_stats(self, host: str, port: int) -> Optional[Dict[str, Any]]:
        """Get etcd statistics including uptime and member info."""
        try:
            import json
            import urllib.request

            stats: Dict[str, Any] = {}

            # Try to get leader stats for uptime
            try:
                leader_url = f"http://{host}:{port}/v2/stats/leader"
                req = urllib.request.Request(leader_url)
                with urllib.request.urlopen(req, timeout=2) as response:
                    leader_data = json.loads(response.read().decode())
                    if "leader" in leader_data:
                        # Leader stats contain uptime information
                        leader_info = leader_data["leader"]
                        if "uptime" in leader_info:
                            stats["uptime"] = leader_info["uptime"]
            except Exception:
                pass

            # Try to get self stats for uptime (alternative method)
            try:
                self_url = f"http://{host}:{port}/v2/stats/self"
                req = urllib.request.Request(self_url)
                with urllib.request.urlopen(req, timeout=2) as response:
                    self_data = json.loads(response.read().decode())
                    if "startTime" in self_data:
                        # Calculate uptime from start time
                        import datetime

                        start_time = datetime.datetime.fromisoformat(
                            self_data["startTime"].replace("Z", "+00:00")
                        )
                        current_time = datetime.datetime.now(datetime.timezone.utc)
                        uptime_seconds = (current_time - start_time).total_seconds()
                        stats["uptime"] = uptime_seconds
            except Exception:
                pass

            # Try to get cluster member information and header (cluster_id)
            try:
                members_url = f"http://{host}:{port}/v2/members"
                req = urllib.request.Request(members_url)
                with urllib.request.urlopen(req, timeout=2) as response:
                    members_data = json.loads(response.read().decode())
                    if "members" in members_data:
                        stats["members"] = len(members_data["members"])
            except Exception:
                try:
                    members_url = f"http://{host}:{port}/v3/cluster/member/list"
                    data: dict[str, str] = {}
                    req = urllib.request.Request(
                        members_url,
                        data=json.dumps(data).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=2) as response:
                        members_data = json.loads(response.read().decode())
                        if "members" in members_data:
                            stats["members"] = len(members_data["members"])
                        if "header" in members_data:
                            stats["header"] = members_data["header"]
                except Exception:
                    pass

            return stats if stats else None
        except Exception:
            pass
        return None

    def _format_etcd_uptime(self, uptime) -> str:
        """Format etcd uptime to human readable format."""
        try:
            # uptime might be in different formats
            if isinstance(uptime, str):
                # Try to parse ISO format or duration string
                if "T" in uptime:  # ISO format
                    import datetime

                    start_time = datetime.datetime.fromisoformat(
                        uptime.replace("Z", "+00:00")
                    )
                    current_time = datetime.datetime.now(datetime.timezone.utc)
                    seconds = int((current_time - start_time).total_seconds())
                else:
                    # Try to parse as duration string or seconds
                    seconds = int(float(uptime))
            else:
                seconds = int(float(uptime))

            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                minutes = seconds // 60
                secs = seconds % 60
                return f"{minutes}m{secs}s"
            elif seconds < 86400:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours}h{minutes}m"
            else:
                days = seconds // 86400
                hours = (seconds % 86400) // 3600
                return f"{days}d{hours}h"
        except (ValueError, TypeError):
            return str(uptime)

    def _test_etcd_api_access(self, host: str, port: int) -> bool:
        """Test if etcd API is accessible by making a simple version call."""
        try:
            import json
            import urllib.request

            # Try a simple version call to test API access
            version_url = f"http://{host}:{port}/version"
            req = urllib.request.Request(version_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode())
                return "etcdserver" in data or "etcdcluster" in data
        except Exception:
            return False

    def _get_all_keys(self, host: str, port: int) -> List[tuple]:
        """Get all keys from etcd with metadata."""
        try:
            import json
            import urllib.request

            # Try to get all keys first
            keys_url = f"http://{host}:{port}/v2/keys/?recursive=true"

            req = urllib.request.Request(keys_url)
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode())

                keys = []

                def extract_keys(node):
                    """Recursively extract keys from etcd node structure."""
                    if "key" in node:
                        # v2 API doesn't provide revision info, just the key
                        keys.append((node["key"], None, None, None))

                    if "nodes" in node:
                        for child_node in node["nodes"]:
                            extract_keys(child_node)

                if "node" in data:
                    extract_keys(data["node"])

                return sorted(keys, key=lambda x: x[0])
        except Exception:
            # Try v3 API format - get all keys with metadata
            try:
                import base64

                keys_url = f"http://{host}:{port}/v3/kv/range"
                data = {
                    "key": "AA==",  # base64 for \x00 to get all keys
                    "range_end": "AA==",  # Same range_end to get all keys
                }

                req = urllib.request.Request(
                    keys_url,
                    data=json.dumps(data).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3) as response:
                    result = json.loads(response.read().decode())

                    keys = []
                    if "kvs" in result:
                        for kv in result["kvs"]:
                            if "key" in kv:
                                # Decode base64 key
                                import base64

                                key = base64.b64decode(kv["key"]).decode("utf-8")
                                create_rev = kv.get("create_revision", "unknown")
                                mod_rev = kv.get("mod_revision", "unknown")
                                version = kv.get("version", "unknown")
                                keys.append((key, create_rev, mod_rev, version))

                    return sorted(keys, key=lambda x: x[0])
            except Exception:
                pass

        return []


class HuggingFaceInfo(NodeInfo):
    """Hugging Face models cache information"""

    def __init__(self, thorough_check: bool = False):
        # Check default Hugging Face cache location
        hf_cache_path = os.path.expanduser("~/.cache/huggingface/hub")

        if os.path.exists(hf_cache_path):
            models = self._get_cached_models(hf_cache_path)
            if models:
                self._init_with_models(hf_cache_path, models, thorough_check)
            else:
                self._init_no_models_found(hf_cache_path)
        else:
            self._init_cache_not_available()

        # Add HF_TOKEN info if set (common to all cases)
        self._add_hf_token_info()

    def _init_with_models(
        self, hf_cache_path: str, models: List[tuple], thorough_check: bool
    ):
        """Initialize when models are found in cache."""
        model_count = len(models)
        display_path = self._replace_home_with_var(hf_cache_path)
        super().__init__(
            label="Hugging Face Cache",
            desc=f"{model_count} models in {display_path}",
            status=NodeStatus.OK,
        )

        # Only show detailed model list in thorough mode
        if thorough_check:
            self._add_model_details(models)

    def _init_no_models_found(self, hf_cache_path: str):
        """Initialize when cache exists but no models found."""
        display_path = self._replace_home_with_var(hf_cache_path)
        super().__init__(
            label="Hugging Face Cache",
            desc=f"directory exists but no models found in {display_path}",
            status=NodeStatus.WARNING,
        )

    def _init_cache_not_available(self):
        """Initialize when cache directory doesn't exist."""
        super().__init__(
            label="Hugging Face Cache",
            desc="~/.cache/huggingface/hub not available",
            status=NodeStatus.WARNING,
        )

    def _add_model_details(self, models: List[tuple]):
        """Add detailed model information as child nodes."""
        # Add first few models as children (limit to 5 for readability)
        for i, model_info in enumerate(models[:5]):
            model_name, download_date, size_str = model_info
            model_node = NodeInfo(
                label=f"Model {i+1}",
                desc=f"{model_name}, downloaded={download_date}, size={size_str}",
                status=NodeStatus.INFO,
            )
            self.add_child(model_node)

        if len(models) > 5:
            more_node = NodeInfo(
                label=f"... and {len(models) - 5} more models",
                status=NodeStatus.INFO,
            )
            self.add_child(more_node)

    def _add_hf_token_info(self):
        """Add HF_TOKEN information if the environment variable is set."""
        if os.environ.get("HF_TOKEN"):
            token_node = NodeInfo(
                label="HF_TOKEN",
                desc="<set>",
                status=NodeStatus.INFO,
            )
            self.add_child(token_node)

    def _get_cached_models(self, cache_path: str) -> List[tuple]:
        """Get list of cached Hugging Face models with metadata.

        Returns:
            List of tuples: (model_name, download_date, size_str)
        """
        models = []
        try:
            if os.path.exists(cache_path):
                for item in os.listdir(cache_path):
                    item_path = os.path.join(cache_path, item)
                    if os.path.isdir(item_path):
                        # Get model name
                        if item.startswith("models--"):
                            # Convert "models--org--model-name" to "org/model-name"
                            parts = item.split("--")
                            if len(parts) >= 3:
                                org = parts[1]
                                model_name = "--".join(
                                    parts[2:]
                                )  # Handle model names with dashes
                                display_name = f"{org}/{model_name}"
                            else:
                                display_name = item  # Fallback to raw name
                        elif not item.startswith("."):  # Skip hidden files/dirs
                            display_name = item
                        else:
                            continue  # Skip hidden directories

                        # Get download date (directory creation/modification time)
                        try:
                            stat_info = os.stat(item_path)
                            # Use the earlier of creation time or modification time
                            download_time = min(stat_info.st_ctime, stat_info.st_mtime)
                            download_date = self._format_timestamp_pdt(download_time)
                        except Exception:
                            download_date = "unknown"

                        # Get directory size
                        try:
                            size_bytes = self._get_directory_size_bytes(item_path)
                            size_str = self._format_size(size_bytes)
                        except Exception:
                            size_str = "unknown"

                        models.append((display_name, download_date, size_str))
        except Exception:
            pass

        # Sort by model name
        return sorted(models, key=lambda x: x[0])

    def _get_directory_size_bytes(self, directory: str) -> int:
        """Get the total size of a directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        if not os.path.islink(filepath):  # Skip symbolic links
                            total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass  # Skip files that can't be accessed
        except Exception:
            pass
        return total_size

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format."""
        if size_bytes == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_bytes)
        unit_index = 0

        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1

        # Format with appropriate precision
        if unit_index == 0:  # Bytes
            return f"{int(size)} {units[unit_index]}"
        elif size >= 100:
            return f"{size:.0f} {units[unit_index]}"
        elif size >= 10:
            return f"{size:.1f} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"


class UserInfo(NodeInfo):
    """User information"""

    _always_show_when_terse: bool = True

    def __init__(self):
        # Get user info
        username = os.getenv("USER") or os.getenv("LOGNAME") or "unknown"
        if username == "unknown":
            try:
                import pwd

                username = pwd.getpwuid(os.getuid()).pw_name
            except Exception:
                try:
                    import subprocess

                    result = subprocess.run(
                        ["whoami"], capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        username = result.stdout.strip()
                except Exception:
                    pass
        uid = os.getuid()
        gid = os.getgid()

        desc = f"user={username}, uid={uid}, gid={gid}"

        # Add warning if running as root
        status = NodeStatus.WARNING if uid == 0 else NodeStatus.INFO
        if uid == 0:
            desc += " ⚠️"

        super().__init__(label="User info", desc=desc, status=status)


class OSInfo(NodeInfo):
    """Operating system information"""

    _always_show_when_terse: bool = True

    def __init__(self):
        # Collect OS information
        uname = platform.uname()

        # Try to get distribution info
        distro = ""
        version = ""
        try:
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release", "r") as f:
                    for line in f:
                        if line.startswith("NAME="):
                            distro = line.split("=", 1)[1].strip().strip('"')
                        elif line.startswith("VERSION="):
                            version = line.split("=", 1)[1].strip().strip('"')
        except Exception:
            pass

        # Get memory info
        mem_used_gb = None
        mem_total_gb = None
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meminfo[k.strip()] = v.strip()

                if "MemTotal" in meminfo and "MemAvailable" in meminfo:
                    total_kb = float(meminfo["MemTotal"].split()[0])
                    avail_kb = float(meminfo["MemAvailable"].split()[0])
                    mem_used_gb = (total_kb - avail_kb) / (1024 * 1024)
                    mem_total_gb = total_kb / (1024 * 1024)
        except Exception:
            pass

        # Get CPU cores
        cores = os.cpu_count()

        # Build the value string
        if distro:
            value = f"{distro} {version} ({uname.system} {uname.release} {uname.machine})".strip()
        else:
            value = f"{uname.system} {uname.release} {uname.machine}"

        super().__init__(label="OS", desc=value, status=NodeStatus.INFO)

        # Add memory and cores as metadata
        if mem_used_gb is not None and mem_total_gb is not None:
            self.add_metadata("Memory", f"{mem_used_gb:.1f}/{mem_total_gb:.1f} GiB")
            if mem_total_gb > 0 and (mem_used_gb / mem_total_gb) >= 0.9:
                self.status = NodeStatus.WARNING
        if cores:
            self.add_metadata("Cores", str(cores))


class GPUInfo(NodeInfo):
    """NVIDIA GPU information"""

    _always_show_when_terse: bool = True

    def __init__(self):
        # Find nvidia-smi executable (check multiple paths)
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            # Check common paths if `which` fails
            for candidate in [
                "/usr/bin/nvidia-smi",
                "/usr/local/bin/nvidia-smi",
                "/usr/local/nvidia/bin/nvidia-smi",
            ]:
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    nvidia_smi = candidate
                    break

        if not nvidia_smi:
            super().__init__(
                label="NVIDIA GPU", desc="nvidia-smi not found", status=NodeStatus.ERROR
            )
            return

        try:
            # Get GPU list
            result = subprocess.run(
                [nvidia_smi, "-L"], capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                # Extract and process error message from stderr or stdout
                error_msg = "nvidia-smi failed"

                # Try stderr first, then stdout
                for output in [result.stderr, result.stdout]:
                    if output and output.strip():
                        error_lines = output.strip().splitlines()
                        if error_lines:
                            error_msg = error_lines[0].strip()
                            break

                # Handle NVML-specific errors
                if "Failed to initialize NVML" in error_msg:
                    error_msg = "No NVIDIA GPU detected (NVML initialization failed)"
                    # Add docker restart suggestion specifically for NVML failures in containers
                    if self._is_inside_container():
                        error_msg += " - maybe try a docker restart?"

                super().__init__(
                    label="NVIDIA GPU", desc=error_msg, status=NodeStatus.ERROR
                )
                return

            # Parse GPU names
            gpu_names = []
            lines = result.stdout.strip().splitlines()
            for line in lines:
                # Example: "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-...)"
                if ":" in line:
                    gpu_name = line.split(":", 1)[1].split("(")[0].strip()
                    gpu_names.append(gpu_name)

            # Check for zero GPUs
            if not gpu_names:
                # Get driver and CUDA even for zero GPUs
                driver, cuda = self._get_driver_cuda_versions(nvidia_smi)
                driver_cuda_str = ""
                if driver or cuda:
                    parts = []
                    if driver:
                        parts.append(f"driver {driver}")
                    if cuda:
                        parts.append(f"CUDA {cuda}")
                    driver_cuda_str = f", {', '.join(parts)}"
                super().__init__(
                    label="NVIDIA GPU",
                    desc=f"not detected{driver_cuda_str}",
                    status=NodeStatus.ERROR,
                )
                return

            # Get driver and CUDA versions
            driver, cuda = self._get_driver_cuda_versions(nvidia_smi)

            # Handle single vs multiple GPUs
            if len(gpu_names) == 1:
                # Single GPU - compact format
                value = gpu_names[0]
                if driver or cuda:
                    driver_cuda = []
                    if driver:
                        driver_cuda.append(f"driver {driver}")
                    if cuda:
                        driver_cuda.append(f"CUDA {cuda}")
                    value += f", {', '.join(driver_cuda)}"

                super().__init__(label="NVIDIA GPU", desc=value, status=NodeStatus.OK)

                # Add power and memory metadata for single GPU
                self._add_power_memory_info(nvidia_smi, 0)
            else:
                # Multiple GPUs - show count in main label
                value = f"{len(gpu_names)} GPUs"
                if driver or cuda:
                    driver_cuda = []
                    if driver:
                        driver_cuda.append(f"driver {driver}")
                    if cuda:
                        driver_cuda.append(f"CUDA {cuda}")
                    value += f", {', '.join(driver_cuda)}"

                super().__init__(label="NVIDIA GPU", desc=value, status=NodeStatus.OK)

                # Add each GPU as a child node
                for i, name in enumerate(gpu_names):
                    gpu_child = NodeInfo(
                        label=f"GPU {i}", desc=name, status=NodeStatus.OK
                    )
                    # Add power and memory for this specific GPU
                    power_mem = self._get_power_memory_string(nvidia_smi, i)
                    if power_mem:
                        gpu_child.add_metadata("Stats", power_mem)
                    self.add_child(gpu_child)

        except Exception:
            super().__init__(
                label="NVIDIA GPU", desc="detection failed", status=NodeStatus.ERROR
            )

    def _get_driver_cuda_versions(
        self, nvidia_smi: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get NVIDIA driver and CUDA versions using query method."""
        driver, cuda = None, None
        try:
            # Use query method for more reliable detection
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                driver = result.stdout.strip().splitlines()[0].strip()

            # Try to get CUDA version from nvidia-smi output
            result = subprocess.run(
                [nvidia_smi], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import re

                m = re.search(r"CUDA Version:\s*([0-9.]+)", result.stdout)
                if m:
                    cuda = m.group(1)
        except Exception:
            pass
        return driver, cuda

    def _add_power_memory_info(self, nvidia_smi: str, gpu_index: int = 0):
        """Add power and memory metadata for a specific GPU."""
        power_mem = self._get_power_memory_string(nvidia_smi, gpu_index)
        if power_mem:
            # Split into Power and Memory parts
            if "; " in power_mem:
                parts = power_mem.split("; ")
                for part in parts:
                    if part.startswith("Power:"):
                        self.add_metadata("Power", part.replace("Power: ", ""))
                    elif part.startswith("Memory:"):
                        self.add_metadata("Memory", part.replace("Memory: ", ""))

    def _get_power_memory_string(
        self, nvidia_smi: str, gpu_index: int = 0
    ) -> Optional[str]:
        """Get power and memory info string for a specific GPU."""
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=power.draw,power.limit,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().splitlines()
                if gpu_index < len(lines):
                    parts = lines[gpu_index].split(",")
                    if len(parts) >= 4:
                        power_draw = parts[0].strip()
                        power_limit = parts[1].strip()
                        mem_used = parts[2].strip()
                        mem_total = parts[3].strip()

                        info_parts = []
                        if power_draw and power_limit:
                            info_parts.append(f"Power: {power_draw}/{power_limit} W")

                        if mem_used and mem_total:
                            # Add warning if memory usage is 90% or higher
                            warning = ""
                            try:
                                if float(mem_used) / float(mem_total) >= 0.9:
                                    warning = " ⚠️"
                            except Exception:
                                pass
                            info_parts.append(
                                f"Memory: {mem_used}/{mem_total} MiB{warning}"
                            )

                        if info_parts:
                            return "; ".join(info_parts)
        except Exception:
            pass
        return None


class FilePermissionsInfo(NodeInfo):
    """File system check for development environment directories

    Checks writability of critical directories needed for:
    - Dynamo development (top-level dynamo directory)
    - Rust development (Cargo target directory + all files, RUSTUP_HOME, CARGO_HOME)
    - Python development (site-packages)

    In thorough mode, also checks disk space for the dynamo working directory
    and shows a warning if less than 10% free space is available.

    In fast mode, skips recursive file checking in Cargo target directory
    for improved performance on large target directories.
    """

    def __init__(self, thorough_check: bool = False):
        super().__init__(label="File System", status=NodeStatus.INFO)
        self.thorough_check = thorough_check

        # Check top-level dynamo directory
        self._check_dynamo_directory_permissions()

        # Check Rust toolchain directories (RUSTUP_HOME and CARGO_HOME)
        self._check_rust_toolchain_permissions()

        # Check Cargo target directory (with optional recursive file checking)
        self._check_cargo_target_permissions()

        # Check Python site-packages directory
        self._check_site_packages_permissions()

    def _check_permissions_unified(
        self,
        candidate_paths: List[str],
        label_prefix: str,
        recursive: bool = False,
        exclude_files: Optional[List[str]] = None,
    ) -> List[NodeInfo]:
        """Unified permission checking function

        Args:
            candidate_paths: List of paths to check, uses first available one
            label_prefix: Prefix for the node label
            recursive: If True, check all files recursively; if False, check directory only
            exclude_files: List of filenames to exclude from file checking (e.g., ['.git'])

        Returns:
            List of NodeInfo objects for the results
        """
        exclude_files = exclude_files or []
        results = []

        # Find first available path
        selected_path = None
        for path in candidate_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                selected_path = expanded_path
                break

        if not selected_path:
            # No paths exist
            path_list = ", ".join(candidate_paths)
            results.append(
                NodeInfo(
                    label=f"{label_prefix} (tried: {path_list})",
                    desc="No candidate paths exist",
                    status=NodeStatus.ERROR,
                )
            )
            return results

        try:
            # Check if it's actually a directory
            if not os.path.isdir(selected_path):
                results.append(
                    NodeInfo(
                        label=f"{label_prefix} ({self._replace_home_with_var(selected_path)})",
                        desc="Path is not a directory",
                        status=NodeStatus.ERROR,
                    )
                )
                return results

            # Check if directory is effectively writable
            if not self._is_effectively_writable(selected_path):
                results.append(
                    NodeInfo(
                        label=f"{label_prefix} ({self._replace_home_with_var(selected_path)})",
                        desc="Directory not writable",
                        status=NodeStatus.ERROR,
                    )
                )
                return results

            if not recursive:
                # Just check directory writability
                # Check if running as root but directory is not owned by root
                is_root = os.getuid() == 0
                is_root_owned = False
                warning_symbol = ""
                desc_text = "writable"
                owner_name = None

                if is_root:
                    try:
                        stat_info = os.stat(selected_path)
                        is_root_owned = stat_info.st_uid == 0
                        if not is_root_owned:
                            warning_symbol = " ⚠️"
                            # Get the owner name
                            try:
                                import pwd

                                owner_name = pwd.getpwuid(stat_info.st_uid).pw_name
                            except Exception:
                                owner_name = f"uid={stat_info.st_uid}"
                        desc_text = f"writable (owned by {owner_name or 'root'})"
                    except Exception:
                        desc_text = "writable (owned by unknown)"

                # Add disk space info in thorough mode
                status = NodeStatus.OK  # Default status
                if self.thorough_check:
                    disk_space, disk_warning = self._format_disk_space(selected_path)
                    desc_text += disk_space
                    # Override status if disk space is low
                    if disk_warning:
                        status = disk_warning

                results.append(
                    NodeInfo(
                        label=f"{label_prefix} ({self._replace_home_with_var(selected_path)}){warning_symbol}",
                        desc=desc_text,
                        status=status,
                    )
                )
            else:
                # Check files recursively
                (
                    total_files,
                    non_writable_files,
                    non_writable_list,
                ) = self._count_writable_files(
                    selected_path, recursive=True, exclude_files=exclude_files
                )

                # Create description based on results
                desc, status = self._create_file_count_description(
                    total_files, non_writable_files, "files"
                )

                # Check if running as root but directory is not owned by root
                is_root = os.getuid() == 0
                is_root_owned = False
                warning_symbol = ""
                owner_name = None

                if is_root:
                    try:
                        stat_info = os.stat(selected_path)
                        is_root_owned = stat_info.st_uid == 0
                        if not is_root_owned:
                            warning_symbol = " ⚠️"
                            # Get the owner name
                            try:
                                import pwd

                                owner_name = pwd.getpwuid(stat_info.st_uid).pw_name
                            except Exception:
                                owner_name = f"uid={stat_info.st_uid}"
                        # Modify description to indicate ownership
                        if "writable" in desc:
                            desc = desc.replace(
                                "writable",
                                f"writable (owned by {owner_name or 'root'})",
                            )
                    except Exception:
                        # Modify description to indicate ownership
                        if "writable" in desc:
                            desc = desc.replace(
                                "writable", "writable (owned by unknown)"
                            )

                # Add disk space info in thorough mode
                if self.thorough_check:
                    disk_space, disk_warning = self._format_disk_space(selected_path)
                    desc += disk_space
                    # Override status if disk space is low
                    if disk_warning:
                        status = disk_warning

                results.append(
                    NodeInfo(
                        label=f"{label_prefix} ({self._replace_home_with_var(selected_path)}){warning_symbol}",
                        desc=desc,
                        status=status,
                    )
                )

                # Add details for non-writable files if there are any (limit to first 10)
                if non_writable_files > 0:
                    details_label = (
                        f"Non-writable files (showing first 10 of {non_writable_files})"
                    )
                    if non_writable_files <= 10:
                        details_label = f"Non-writable files ({non_writable_files})"

                    details_node = NodeInfo(
                        label=details_label,
                        desc="; ".join(non_writable_list[:10]),
                        status=NodeStatus.WARNING,
                    )
                    results.append(details_node)

        except Exception as e:
            results.append(
                NodeInfo(
                    label=f"{label_prefix} ({self._replace_home_with_var(selected_path)})",
                    desc=f"Permission check failed: {str(e)}",
                    status=NodeStatus.ERROR,
                )
            )

        return results

    def _is_effectively_writable(self, file_path: str) -> bool:
        """Check if a file is effectively writable

        A file is considered effectively writable if:
        1. It's already writable (os.access check)
        2. We own the file (can chmod it)
        3. We are root (can do anything) - but only if os.access confirms write access
           Note: Root may still be denied write access on NFS mounts due to root squashing
        """
        try:
            # First check if it's already writable - this works for all cases including NFS
            if os.access(file_path, os.W_OK):
                return True

            # Check if we own the file (and can therefore chmod it)
            stat_info = os.stat(file_path)
            if stat_info.st_uid == os.getuid():
                return True

            # For root, we still need to respect the os.access result
            # Root privileges don't guarantee write access on NFS mounts
            # If os.access(W_OK) returned False above, respect that even for root
            return False
        except Exception:
            # If we can't stat the file, assume it's not writable
            return False

    def _count_writable_files(
        self,
        directory: str,
        recursive: bool = False,
        exclude_files: Optional[List[str]] = None,
    ) -> Tuple[int, int, List[str]]:
        """Count total files and non-writable files in directory

        Returns:
            Tuple of (total_files, non_writable_files, non_writable_list)
        """
        exclude_files = exclude_files or []
        total_files = 0
        non_writable_files = 0
        non_writable_list = []

        if recursive:
            # Walk through all files in the directory tree recursively
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Skip symbolic links
                    if os.path.islink(file_path):
                        continue
                    total_files += 1
                    if not self._is_effectively_writable(file_path):
                        non_writable_files += 1
                        rel_path = os.path.relpath(file_path, directory)
                        non_writable_list.append(rel_path)
        else:
            # Only check files in the immediate directory (non-recursive)
            for item in os.listdir(directory):
                if item in exclude_files:
                    continue
                item_path = os.path.join(directory, item)
                # Skip symbolic links and only check regular files
                if os.path.isfile(item_path) and not os.path.islink(item_path):
                    total_files += 1
                    try:
                        if not self._is_effectively_writable(item_path):
                            non_writable_files += 1
                            non_writable_list.append(item)
                    except Exception:
                        non_writable_files += 1
                        non_writable_list.append(item)

        return total_files, non_writable_files, non_writable_list

    def _create_file_count_description(
        self, total_files: int, non_writable_files: int, context: str = "files"
    ) -> Tuple[str, NodeStatus]:
        """Create description and status for file count results"""
        if total_files == 0:
            return f"writable, no {context} found", NodeStatus.INFO
        elif non_writable_files == 0:
            return f"writable, all {total_files} {context} writable", NodeStatus.OK
        else:
            return (
                f"writable, {non_writable_files} of {total_files} {context} not writable",
                NodeStatus.WARNING,
            )

    def _get_cargo_target_path_candidates(self) -> List[str]:
        """Get candidate paths for cargo target directory"""
        candidates = []

        # Try to get target directory from cargo metadata (most accurate)
        try:
            result = subprocess.run(
                ["cargo", "metadata", "--format-version=1", "--no-deps"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=".",
            )
            if result.returncode == 0:
                import json

                metadata = json.loads(result.stdout)
                target_path = metadata.get("target_directory")
                if target_path:
                    candidates.append(target_path)
        except Exception:
            pass

        # Add fallback candidates
        cargo_target = os.environ.get("CARGO_TARGET_DIR")
        if cargo_target:
            candidates.append(cargo_target)

        candidates.append("~/.cargo/target")
        return candidates

    def _check_dynamo_directory_permissions(self):
        """Check top-level dynamo directory and key files writability"""
        # Use the existing workspace detection logic
        dynamo_root = DynamoInfo.find_workspace()

        if not dynamo_root:
            self.add_child(
                NodeInfo(
                    label="Dynamo workspace",
                    desc="workspace not found",
                    status=NodeStatus.ERROR,
                )
            )
            return

        if not DynamoInfo.is_dynamo_workspace(dynamo_root):
            self.add_child(
                NodeInfo(
                    label="Dynamo workspace",
                    desc="not a valid dynamo workspace",
                    status=NodeStatus.ERROR,
                )
            )
            return

        # Check dynamo root directory and files (exclude .git)
        recursive = self.thorough_check
        results = self._check_permissions_unified(
            [dynamo_root],
            "Dynamo workspace",
            recursive=recursive,
            exclude_files=[".git"],
        )
        for result in results:
            self.add_child(result)

        # Check .git directory separately
        git_dir = os.path.join(dynamo_root, ".git")
        if os.path.exists(git_dir):
            git_results = self._check_permissions_unified(
                [git_dir], "Dynamo .git directory", recursive=recursive
            )
            for result in git_results:
                self.add_child(result)
        else:
            self.add_child(
                NodeInfo(
                    label="Dynamo .git directory",
                    desc="not available",
                    status=NodeStatus.WARNING,
                )
            )

    def _check_site_packages_permissions(self):
        """Check site-packages directory writability"""
        try:
            import site

            # Get all candidate site-packages directories
            site_packages_dirs = site.getsitepackages()
            user_site = site.getusersitepackages()
            if user_site:
                site_packages_dirs.append(user_site)

            # Check each existing site-packages directory
            recursive = self.thorough_check
            for site_dir in site_packages_dirs:
                if os.path.exists(site_dir):
                    results = self._check_permissions_unified(
                        [site_dir], "site-packages", recursive=recursive
                    )
                    for result in results:
                        self.add_child(result)

        except Exception as e:
            self.add_child(
                NodeInfo(
                    label="Python site-packages",
                    desc=f"Permission check failed: {str(e)}",
                    status=NodeStatus.ERROR,
                )
            )

    def _check_cargo_target_permissions(self):
        """Check Cargo target directory writability and file permissions"""
        candidates = self._get_cargo_target_path_candidates()
        recursive = self.thorough_check
        results = self._check_permissions_unified(
            candidates, "Cargo target", recursive=recursive
        )

        if not results or (
            len(results) == 1
            and results[0].status == NodeStatus.ERROR
            and results[0].desc is not None
            and "No candidate paths exist" in results[0].desc
        ):
            # No paths exist - show warning instead of error
            self.add_child(
                NodeInfo(
                    label="Cargo target",
                    desc="Path does not exist",
                    status=NodeStatus.WARNING,
                )
            )
        else:
            for result in results:
                self.add_child(result)

    def _check_rust_toolchain_permissions(self):
        """Check RUSTUP_HOME and CARGO_HOME directory writability

        These directories need recursive checking because:
        - RUSTUP_HOME: rustup needs to write toolchain files, documentation, etc.
        - CARGO_HOME: cargo needs to write registry cache, git repos, binaries, etc.
        """
        # Check RUSTUP_HOME
        rustup_env = os.environ.get("RUSTUP_HOME")
        rustup_candidates = [rustup_env] if rustup_env is not None else []
        rustup_candidates.append("~/.rustup")

        recursive = self.thorough_check
        rustup_results = self._check_permissions_unified(
            rustup_candidates, "Rustup home", recursive=recursive
        )
        for result in rustup_results:
            self.add_child(result)

        # Check CARGO_HOME
        cargo_env = os.environ.get("CARGO_HOME")
        cargo_candidates = [cargo_env] if cargo_env is not None else []
        cargo_candidates.append("~/.cargo")

        cargo_results = self._check_permissions_unified(
            cargo_candidates, "Cargo home", recursive=recursive
        )
        for result in cargo_results:
            self.add_child(result)

    def _format_disk_space(self, path: str) -> Tuple[str, Optional[NodeStatus]]:
        """Format disk space information for a given path

        Returns:
            Tuple of (formatted_string, warning_status_if_low_space)
        """
        try:
            # Get disk usage statistics
            statvfs = os.statvfs(path)

            # Calculate sizes in bytes
            total_bytes = statvfs.f_frsize * statvfs.f_blocks
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            used_bytes = total_bytes - free_bytes

            # Convert to human readable format
            def format_bytes(bytes_val):
                """Convert bytes to human readable format"""
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.1f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.1f} PB"

            # Calculate percentage used
            percent_used = (used_bytes / total_bytes) * 100
            percent_free = 100 - percent_used

            formatted_string = f", {format_bytes(used_bytes)}/{format_bytes(total_bytes)} ({percent_used:.1f}% used)"

            # Return warning status if less than 10% free space
            warning_status = NodeStatus.WARNING if percent_free < 10 else None

            return formatted_string, warning_status

        except Exception:
            return "", None


class CargoInfo(NodeInfo):
    """Cargo tool information"""

    def __init__(self, thorough_check: bool = False):
        self.thorough_check = thorough_check
        cargo_path = shutil.which("cargo")
        cargo_version = None

        # Get cargo version
        if cargo_path:
            try:
                result = subprocess.run(
                    ["cargo", "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    cargo_version = result.stdout.strip()
            except Exception:
                pass

        if not cargo_path and not cargo_version:
            super().__init__(
                label="Cargo",
                desc="not found, install Rust toolchain to see cargo target directory",
                status=NodeStatus.ERROR,
            )
            return

        # Initialize with cargo path and version
        value = ""
        if cargo_path:
            value = self._replace_home_with_var(cargo_path)
        if cargo_version:
            value += f", {cargo_version}" if value else cargo_version

        super().__init__(label="Cargo", desc=value, status=NodeStatus.OK)

        # Get cargo home directory from the environment (may not exist, which is OK)
        cargo_home_env = os.environ.get("CARGO_HOME")
        if cargo_home_env:
            cargo_home = cargo_home_env
            home_value = f"CARGO_HOME={self._replace_home_with_var(cargo_home)}"
        else:
            cargo_home = os.path.expanduser("~/.cargo")
            home_value = (
                f"CARGO_HOME=<not set>, using {self._replace_home_with_var(cargo_home)}"
            )

        if cargo_home and os.path.exists(cargo_home):
            status = NodeStatus.INFO
        else:
            home_value += " (directory does not exist)"
            status = NodeStatus.WARNING

        home_node = NodeInfo(
            label="Cargo home directory", desc=home_value, status=status
        )
        self.add_child(home_node)

        # Get cargo target directory
        cargo_target_env = os.environ.get("CARGO_TARGET_DIR")
        cargo_target = self._get_cargo_target_directory()

        # Calculate total directory size (only if thorough check and directory exists)
        size_str = ""
        if cargo_target and os.path.exists(cargo_target) and self.thorough_check:
            total_size_gb = self._get_directory_size_gb(cargo_target)
            size_str = f", {total_size_gb:.1f} GB" if total_size_gb is not None else ""

        # Format the display value
        if cargo_target_env:
            display_cargo_target = (
                self._replace_home_with_var(cargo_target) if cargo_target else "unknown"
            )
            target_value = f"CARGO_TARGET_DIR={display_cargo_target}{size_str}"
        else:
            display_cargo_target = (
                self._replace_home_with_var(cargo_target) if cargo_target else "unknown"
            )
            target_value = (
                f"CARGO_TARGET_DIR=<not set>, using {display_cargo_target}{size_str}"
            )

        # Check directory existence and set status
        if cargo_target and os.path.exists(cargo_target):
            status = NodeStatus.INFO
            target_node = NodeInfo(
                label="Cargo target directory",
                desc=target_value,
                status=status,
            )
            self.add_child(target_node)
            # Add debug/release/binary info as children of target directory
            self._add_build_info(target_node, cargo_target)
        else:
            target_value += " (directory does not exist)"
            status = NodeStatus.WARNING if cargo_target_env else NodeStatus.INFO
            target_node = NodeInfo(
                label="Cargo target directory",
                desc=target_value,
                status=status,
            )
            self.add_child(target_node)

    def _get_directory_size_gb(self, directory: str) -> Optional[float]:
        """Get the size of a directory in GB."""
        try:
            # Use du command to get directory size in bytes
            result = subprocess.run(
                ["du", "-sb", directory], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                # Parse output: "size_in_bytes\tdirectory_path"
                size_bytes = int(result.stdout.split()[0])
                # Convert to GB
                size_gb = size_bytes / (1024**3)
                return size_gb
        except Exception:
            pass
        return None

    def _get_cargo_target_directory(self) -> Optional[str]:
        """Get cargo target directory using cargo metadata."""
        try:
            # Use DynamoInfo's static method to find workspace
            workspace_dir = DynamoInfo.find_workspace()

            # Run cargo metadata command to get target directory
            cmd_args = ["cargo", "metadata", "--format-version=1", "--no-deps"]
            kwargs: Dict[str, Any] = {
                "capture_output": True,
                "text": True,
                "timeout": 10,
            }

            # Add cwd if workspace_dir was found
            if workspace_dir and os.path.isdir(workspace_dir):
                kwargs["cwd"] = workspace_dir

            result = subprocess.run(cmd_args, **kwargs)

            if result.returncode == 0:
                # Parse JSON output to extract target_directory
                metadata = json.loads(result.stdout)
                return metadata.get("target_directory")
        except Exception:
            pass
        return None

    def _add_build_info(self, parent_node: NodeInfo, cargo_target: str):
        """Add debug/release/binary information as children of target directory."""
        debug_dir = os.path.join(cargo_target, "debug")
        release_dir = os.path.join(cargo_target, "release")

        # Check debug directory
        if os.path.exists(debug_dir):
            display_debug = self._replace_home_with_var(debug_dir)
            debug_value = display_debug

            # Add size (only if thorough check)
            if self.thorough_check:
                debug_size_gb = self._get_directory_size_gb(debug_dir)
                if debug_size_gb is not None:
                    debug_value += f", {debug_size_gb:.1f} GB"

            try:
                debug_mtime = os.path.getmtime(debug_dir)
                debug_time = self._format_timestamp_pdt(debug_mtime)
                debug_value += f", modified={debug_time}"
            except Exception:
                debug_value += " (unable to read timestamp)"

            debug_node = NodeInfo(
                label="Debug", desc=debug_value, status=NodeStatus.INFO
            )
            parent_node.add_child(debug_node)

        # Check release directory
        if os.path.exists(release_dir):
            display_release = self._replace_home_with_var(release_dir)
            release_value = display_release

            # Add size (only if thorough check)
            if self.thorough_check:
                release_size_gb = self._get_directory_size_gb(release_dir)
                if release_size_gb is not None:
                    release_value += f", {release_size_gb:.1f} GB"

            try:
                release_mtime = os.path.getmtime(release_dir)
                release_time = self._format_timestamp_pdt(release_mtime)
                release_value += f", modified={release_time}"
            except Exception:
                release_value += " (unable to read timestamp)"

            release_node = NodeInfo(
                label="Release", desc=release_value, status=NodeStatus.INFO
            )
            parent_node.add_child(release_node)

        # Find *.so file
        so_file = self._find_so_file(cargo_target)
        if so_file:
            display_so = self._replace_home_with_var(so_file)
            so_value = display_so

            # Add file size (only if thorough check)
            if self.thorough_check:
                try:
                    file_size_bytes = os.path.getsize(so_file)
                    file_size_mb = file_size_bytes / (1024**2)
                    so_value += f", {file_size_mb:.1f} MB"
                except Exception:
                    pass

            try:
                so_mtime = os.path.getmtime(so_file)
                so_time = self._format_timestamp_pdt(so_mtime)
                so_value += f", modified={so_time}"
            except Exception:
                so_value += " (unable to read timestamp)"

            binary_node = NodeInfo(
                label="Binary", desc=so_value, status=NodeStatus.INFO
            )
            parent_node.add_child(binary_node)

    def _find_so_file(self, target_directory: str) -> Optional[str]:
        """Find the compiled *.so file in target directory."""
        # Check common locations for .so files
        search_dirs = [
            os.path.join(target_directory, "debug"),
            os.path.join(target_directory, "release"),
            target_directory,
        ]

        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue

            # Walk through directory looking for .so files
            try:
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".so"):
                            return os.path.join(root, file)
                    # Don't recurse too deep
                    if root.count(os.sep) - search_dir.count(os.sep) > 2:
                        dirs[:] = []  # Stop recursion
            except Exception:
                pass

        return None


class MaturinInfo(NodeInfo):
    """Maturin tool information (Python-Rust build tool)"""

    def __init__(self):
        maturin_path = shutil.which("maturin")
        if not maturin_path:
            super().__init__(label="Maturin", desc="not found", status=NodeStatus.ERROR)
            # Add installation hint as a child node
            install_hint = NodeInfo(
                label="Install with",
                desc="uv pip install maturin[patchelf]",
                status=NodeStatus.INFO,
            )
            self.add_child(install_hint)
            return

        try:
            result = subprocess.run(
                ["maturin", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                # Include the maturin binary path like Cargo and Git do
                display_maturin_path = self._replace_home_with_var(maturin_path)
                super().__init__(
                    label="Maturin",
                    desc=f"{display_maturin_path}, {version}",
                    status=NodeStatus.OK,
                )
                return
        except Exception:
            pass

        super().__init__(label="Maturin", desc="not found", status=NodeStatus.ERROR)


class PythonInfo(NodeInfo):
    """Python installation information"""

    def __init__(self):
        py_version = platform.python_version()
        py_exec = sys.executable or "python"
        display_py_exec = self._replace_home_with_var(py_exec)

        super().__init__(
            label="Python",
            desc=f"{py_version}, {display_py_exec}",
            status=NodeStatus.OK if os.path.exists(py_exec) else NodeStatus.ERROR,
        )

        # Check for PyTorch (optional)
        try:
            torch = __import__("torch")
            version = getattr(torch, "__version__", "installed")

            # Check CUDA availability
            cuda_status = None
            if hasattr(torch, "cuda"):
                try:
                    cuda_available = torch.cuda.is_available()
                    cuda_status = (
                        "✅torch.cuda.is_available"
                        if cuda_available
                        else "❌torch.cuda.is_available"
                    )
                except Exception:
                    pass

            # Get installation path
            install_path = None
            if hasattr(torch, "__file__") and torch.__file__:
                file_path = torch.__file__
                if "site-packages" in file_path:
                    parts = file_path.split(os.sep)
                    for i, part in enumerate(parts):
                        if part == "site-packages":
                            install_path = os.sep.join(parts[: i + 1])
                            break
                elif file_path:
                    install_path = os.path.dirname(file_path)

                if install_path:
                    install_path = self._replace_home_with_var(install_path)

            package_info = PythonPackageInfo(
                package_name="PyTorch",
                version=version,
                cuda_status=cuda_status,
                install_path=install_path,
                is_framework=False,
            )
            self.add_child(package_info)
        except ImportError:
            pass  # PyTorch is optional, don't show if not installed

        # Add PYTHONPATH
        pythonpath = os.environ.get("PYTHONPATH", "")
        self.add_child(PythonPathInfo(pythonpath))


class FrameworkInfo(NodeInfo):
    """LLM Framework information"""

    _always_show_when_terse: bool = True

    def __init__(self):
        super().__init__(label="🤖Framework", status=NodeStatus.INFO)

        # Check for framework packages (mandatory to show)
        frameworks_to_check = [
            ("vllm", "vLLM"),
            ("sglang", "SGLang"),
            ("tensorrt_llm", "TensorRT-LLM"),
        ]

        frameworks_found = 0

        for module_name, display_name in frameworks_to_check:
            # Regular import for all frameworks
            try:
                module = __import__(module_name)
                version = getattr(module, "__version__", "installed")
                frameworks_found += 1

                # Get module path
                module_path = None
                if hasattr(module, "__file__") and module.__file__:
                    module_path = self._replace_home_with_var(module.__file__)

                # Get executable path
                exec_path = None
                exec_path_raw = shutil.which(module_name)
                if exec_path_raw:
                    exec_path = self._replace_home_with_var(exec_path_raw)

                package_info = PythonPackageInfo(
                    package_name=display_name,
                    version=version,
                    module_path=module_path,
                    exec_path=exec_path,
                    is_framework=True,
                    is_installed=True,
                )
                self.add_child(package_info)
            except (ImportError, Exception) as e:
                # Framework import failed - check if it's installed but can't be imported
                if module_name == "tensorrt_llm":
                    # TensorRT-LLM might be installed but fail to import due to GPU issues
                    installed_path = self._check_tensorrt_llm_installation()
                    if installed_path:
                        # Show as installed but with import error
                        package_info = PythonPackageInfo(
                            package_name=display_name,
                            version="installed (import failed)",
                            module_path=installed_path,
                            exec_path=None,
                            is_framework=True,
                            is_installed=True,
                            import_error=str(e),
                        )
                        self.add_child(package_info)
                        frameworks_found += 1
                    # If not installed, don't add it (same as before)
                else:
                    # For other frameworks, don't add if import fails (same as before)
                    pass

        # If no frameworks found, set status to ERROR (X) and show what's missing
        if frameworks_found == 0:
            self.status = NodeStatus.ERROR
            # List all the frameworks that were checked but not found
            missing_frameworks = []
            for module_name, display_name in frameworks_to_check:
                missing_frameworks.append(f"no {module_name}")
            missing_text = ", ".join(missing_frameworks)
            self.desc = missing_text

    def _check_tensorrt_llm_installation(self) -> Optional[str]:
        """Check if TensorRT-LLM is installed by looking for its directory."""
        try:
            import site

            # Check in site-packages directories
            for site_dir in site.getsitepackages():
                # Look for tensorrt_llm directory
                tensorrt_path = os.path.join(site_dir, "tensorrt_llm")
                if os.path.exists(tensorrt_path) and os.path.isdir(tensorrt_path):
                    return self._replace_home_with_var(tensorrt_path)

                # Also check for dist-info directories
                import glob

                dist_pattern = os.path.join(site_dir, "*tensorrt*llm*.dist-info")
                matches = glob.glob(dist_pattern)
                if matches:
                    # Found dist-info, look for the actual package
                    for match in matches:
                        parent_dir = os.path.dirname(match)
                        tensorrt_path = os.path.join(parent_dir, "tensorrt_llm")
                        if os.path.exists(tensorrt_path):
                            return self._replace_home_with_var(tensorrt_path)

            # Check user site-packages
            try:
                user_site = site.getusersitepackages()
                if user_site:
                    tensorrt_path = os.path.join(user_site, "tensorrt_llm")
                    if os.path.exists(tensorrt_path) and os.path.isdir(tensorrt_path):
                        return self._replace_home_with_var(tensorrt_path)
            except Exception:
                pass

        except Exception:
            pass

        return None


class PythonPackageInfo(NodeInfo):
    """Python package information"""

    def __init__(
        self,
        package_name: str,
        version: str,
        cuda_status: Optional[str] = None,
        module_path: Optional[str] = None,
        exec_path: Optional[str] = None,
        install_path: Optional[str] = None,
        is_framework: bool = False,
        is_installed: bool = True,
        import_error: Optional[str] = None,
    ):
        # Build display value
        display_value = version

        # Determine status based on whether package is installed
        if not is_installed or version == "-":
            # Framework not found - show with "-" and use UNKNOWN status for ❓ symbol
            display_value = "-"
            status = NodeStatus.UNKNOWN  # Show ❓ for not found frameworks
        elif import_error:
            # Package installed but import failed - show as warning
            status = NodeStatus.WARNING
            parts = [version]
            if module_path:
                parts.append(f"module={module_path}")
            if import_error:
                # Truncate long error messages
                error_msg = (
                    import_error[:100] + "..."
                    if len(import_error) > 100
                    else import_error
                )
                parts.append(f"error={error_msg}")
            display_value = ", ".join(parts)
        else:
            status = NodeStatus.OK

            # Add CUDA status for PyTorch
            if cuda_status:
                display_value = f"{version}, {cuda_status}"
                # Don't add install path for PyTorch with CUDA status
            # For frameworks, add module and exec paths
            elif is_framework and (module_path or exec_path):
                parts = [version]
                if module_path:
                    parts.append(f"module={module_path}")
                if exec_path:
                    parts.append(f"exec={exec_path}")
                display_value = ", ".join(parts)
            # For regular packages, add install path
            elif install_path:
                display_value = f"{version} ({install_path})"

        super().__init__(label=package_name, desc=display_value, status=status)


class PythonPathInfo(NodeInfo):
    """PYTHONPATH environment variable information"""

    def __init__(self, pythonpath: str):
        if pythonpath:
            # Split by colon and replace home in each path
            paths = pythonpath.split(":")
            display_paths = []
            has_invalid_paths = False

            for p in paths:
                display_path = self._replace_home_with_var(p)
                # Check if path exists and is accessible
                if not os.path.exists(p) or not os.access(p, os.R_OK):
                    display_paths.append(
                        f"{Colors.BRIGHT_RED}{display_path}{Colors.RESET}"
                    )  # Bright red path
                    has_invalid_paths = True
                else:
                    display_paths.append(display_path)

            display_pythonpath = ":".join(display_paths)
            status = NodeStatus.WARNING if has_invalid_paths else NodeStatus.INFO
        else:
            display_pythonpath = "not set"
            status = (
                NodeStatus.INFO
            )  # PYTHONPATH not set is fine with editable installs

        super().__init__(label="PYTHONPATH", desc=display_pythonpath, status=status)


class DynamoRuntimeInfo(NodeInfo):
    """Dynamo runtime components information"""

    def __init__(self, workspace_dir: str, thorough_check: bool = False):
        self.thorough_check = thorough_check
        # Try to get package version
        import importlib.metadata

        try:
            version = importlib.metadata.version("ai-dynamo-runtime")
            runtime_value = f"ai-dynamo-runtime {version}"
            is_installed = True
        except Exception:
            runtime_value = "ai-dynamo-runtime - Not installed"
            is_installed = False

        super().__init__(
            label="Runtime components",
            desc=runtime_value,
            status=NodeStatus.INFO,  # Will update based on components found
        )

        # Add package info if installed
        if is_installed:
            # Add dist-info directory
            dist_info = self._find_dist_info()
            if dist_info:
                self.add_child(dist_info)

            # Add .pth file
            pth_file = self._find_pth_file()
            if pth_file:
                self.add_child(pth_file)

        # Discover runtime components from source
        components = self._discover_runtime_components(workspace_dir)

        # Find where each component actually is and add them
        if components:
            # Calculate max width for alignment
            max_len = max(len(comp) for comp in components)

            components_found = False
            for component in components:
                try:
                    # Try to import to find actual location
                    module = __import__(component, fromlist=[""])
                    module_path = getattr(module, "__file__", None)

                    if module_path:
                        # Add timestamp for .so files
                        timestamp_str = ""
                        if module_path.endswith(".so"):
                            try:
                                stat = os.stat(module_path)
                                timestamp = self._format_timestamp_pdt(stat.st_mtime)
                                timestamp_str = f", modified={timestamp}"
                            except Exception:
                                pass

                        display_path = self._replace_home_with_var(module_path)
                        padded_name = f"{component:<{max_len}}"
                        module_node = NodeInfo(
                            label=f"✅ {padded_name}",
                            desc=f"{display_path}{timestamp_str}",
                            status=NodeStatus.NONE,
                        )
                        self.add_child(module_node)
                        components_found = True
                except ImportError as e:
                    # Module not importable - show as error
                    padded_name = f"{component:<{max_len}}"
                    error_msg = str(e) if str(e) else "Import failed"
                    module_node = NodeInfo(
                        label=padded_name, desc=error_msg, status=NodeStatus.ERROR
                    )
                    self.add_child(module_node)
                    # Don't set components_found to True for failed imports

            # Update status and value based on whether we found components
            if components_found:
                self.status = NodeStatus.OK
                # If not installed but components work via PYTHONPATH, update the message
                if not is_installed:
                    self.desc = "ai-dynamo-runtime (via PYTHONPATH)"
            else:
                self.status = NodeStatus.ERROR
        else:
            # No components discovered at all
            self.status = NodeStatus.ERROR

        # Final check: if no children at all (no components found), ensure it's an error
        if not self.children:
            self.status = NodeStatus.ERROR

    def _discover_runtime_components(self, workspace_dir: str) -> list:
        """Discover ai-dynamo-runtime components from filesystem.

        Returns:
            List of runtime component module names
            Example: ['dynamo._core', 'dynamo.nixl_connect', 'dynamo.llm', 'dynamo.runtime']

        Note: Always includes 'dynamo._core' (compiled Rust module), then scans
              lib/bindings/python/src/dynamo/ for additional components.
        """
        components = ["dynamo._core"]  # Always include compiled Rust module

        if not workspace_dir:
            return components

        # Scan runtime components (llm, runtime, nixl_connect, etc.)
        runtime_path = os.path.join(workspace_dir, "lib/bindings/python/src/dynamo")
        if not os.path.exists(runtime_path):
            return components

        for item in os.listdir(runtime_path):
            item_path = os.path.join(runtime_path, item)
            if os.path.isdir(item_path) and os.path.exists(
                os.path.join(item_path, "__init__.py")
            ):
                components.append(f"dynamo.{item}")

        return components

    def _find_dist_info(self) -> Optional[NodeInfo]:
        """Find the dist-info directory for ai-dynamo-runtime."""
        import site

        for site_dir in site.getsitepackages():
            pattern = os.path.join(site_dir, "ai_dynamo_runtime*.dist-info")
            matches = glob.glob(pattern)
            if matches:
                path = matches[0]
                display_path = self._replace_home_with_var(path)
                try:
                    stat = os.stat(path)
                    timestamp = self._format_timestamp_pdt(stat.st_ctime)
                    return NodeInfo(
                        label=f" {display_path}",
                        desc=f"created={timestamp}",
                        status=NodeStatus.INFO,
                        metadata={"part_of_previous": True},
                    )
                except Exception:
                    return NodeInfo(
                        label=f" {display_path}",
                        status=NodeStatus.INFO,
                        metadata={"part_of_previous": True},
                    )
        return None

    def _find_pth_file(self) -> Optional[NodeInfo]:
        """Find the .pth file for ai-dynamo-runtime."""
        import site

        for site_dir in site.getsitepackages():
            pth_path = os.path.join(site_dir, "ai_dynamo_runtime.pth")
            if os.path.exists(pth_path):
                display_path = self._replace_home_with_var(pth_path)
                try:
                    stat = os.stat(pth_path)
                    timestamp = self._format_timestamp_pdt(stat.st_mtime)
                    node = NodeInfo(
                        label=f" {display_path}",
                        desc=f"modified={timestamp}",
                        status=NodeStatus.INFO,
                        metadata={"part_of_previous": True},
                    )

                    # Read where it points to
                    with open(pth_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            display_content = self._replace_home_with_var(content)
                            points_to = NodeInfo(
                                label="→", desc=display_content, status=NodeStatus.INFO
                            )
                            node.add_child(points_to)

                    return node
                except Exception:
                    return NodeInfo(label=display_path, status=NodeStatus.INFO)
        return None


class DynamoFrameworkInfo(NodeInfo):
    """Dynamo framework components information"""

    def __init__(self, workspace_dir: str, thorough_check: bool = False):
        self.thorough_check = thorough_check
        # Try to get package version
        import importlib.metadata

        try:
            version = importlib.metadata.version("ai-dynamo")
            framework_value = f"ai-dynamo {version}"
            is_installed = True
        except Exception:
            framework_value = "ai-dynamo - Not installed"
            is_installed = False

        super().__init__(
            label="Framework components",
            desc=framework_value,
            status=NodeStatus.INFO,  # Will update based on components found
        )

        # Add package info if installed
        if is_installed:
            import glob
            import site

            for site_dir in site.getsitepackages():
                # Look specifically for ai_dynamo (not ai_dynamo_runtime)
                dist_pattern = os.path.join(site_dir, "ai_dynamo-*.dist-info")
                matches = glob.glob(dist_pattern)
                if matches:
                    path = matches[0]
                    display_path = self._replace_home_with_var(path)
                    try:
                        stat = os.stat(path)
                        timestamp = self._format_timestamp_pdt(stat.st_ctime)
                        dist_node = NodeInfo(
                            label=f" {display_path}",
                            desc=f"created={timestamp}",
                            status=NodeStatus.INFO,
                            metadata={"part_of_previous": True},
                        )
                        self.add_child(dist_node)
                    except Exception:
                        dist_node = NodeInfo(
                            label=f" {display_path}",
                            status=NodeStatus.INFO,
                            metadata={"part_of_previous": True},
                        )
                        self.add_child(dist_node)
                    break

        # Discover framework components from source
        components = self._discover_framework_components(workspace_dir)

        # Find where each component actually is and add them
        if components:
            # Sort components for consistent output
            components.sort()

            # Calculate max width for alignment
            max_len = max(len(comp) for comp in components)

            components_found = False
            for component in components:
                try:
                    # Try to import to find actual location
                    module = __import__(component, fromlist=[""])
                    module_path = getattr(module, "__file__", None)

                    if module_path:
                        display_path = self._replace_home_with_var(module_path)
                        padded_name = f"{component:<{max_len}}"
                        component_node = NodeInfo(
                            label=f"✅ {padded_name}",
                            desc=display_path,
                            status=NodeStatus.NONE,
                        )
                        self.add_child(component_node)
                        components_found = True
                except ImportError as e:
                    # Module not importable - show as error
                    padded_name = f"{component:<{max_len}}"
                    error_msg = str(e) if str(e) else "Import failed"
                    component_node = NodeInfo(
                        label=padded_name, desc=error_msg, status=NodeStatus.ERROR
                    )
                    self.add_child(component_node)
                    # Don't set components_found to True for failed imports

            # Update status and value based on whether we found components
            if components_found:
                self.status = NodeStatus.OK
                # If not installed but components work via PYTHONPATH, update the message
                if not is_installed:
                    self.desc = "ai-dynamo (via PYTHONPATH)"
            else:
                self.status = NodeStatus.ERROR
        else:
            # No components discovered at all
            self.status = NodeStatus.ERROR

    def _discover_framework_components(self, workspace_dir: str) -> list:
        """Discover ai-dynamo framework components from filesystem.

        Returns:
            List of framework component module names
            Example: ['dynamo.frontend', 'dynamo.planner', 'dynamo.vllm', 'dynamo.sglang', 'dynamo.llama_cpp']

        Note: Scans components/src/dynamo/... directory for modules with __init__.py files.
        """
        components: List[str] = []

        if not workspace_dir:
            return components

        # Scan the components/src/dynamo/... Python directory for __init__.py files
        components_path = os.path.join(workspace_dir, "components", "src", "dynamo")
        if os.path.exists(components_path):
            for item in os.listdir(components_path):
                item_path = os.path.join(components_path, item)
                if os.path.isdir(item_path):
                    # Check for dynamo module in src
                    module_path = os.path.join(item_path, "__init__.py")
                    if os.path.exists(module_path):
                        components.append(f"dynamo.{item}")

        return components


class DynamoInfo(NodeInfo):
    """Dynamo workspace information"""

    _always_show_when_terse: bool = True

    def __init__(self, thorough_check: bool = False, terse: bool = False):
        self.thorough_check = thorough_check
        self.terse = terse

        # Find workspace directory
        workspace_dir = DynamoInfo.find_workspace()

        if not workspace_dir:
            # Show error when workspace is not found
            super().__init__(
                label="Dynamo",
                desc="workspace not found - cannot detect Runtime and Framework components",
                status=NodeStatus.ERROR,
            )
            # Add helpful information about where we looked
            search_paths = NodeInfo(
                label="Searched in",
                desc="current dir, ~/dynamo, DYNAMO_HOME, /workspace",
                status=NodeStatus.INFO,
            )
            self.add_child(search_paths)
            hint = NodeInfo(
                label="Hint",
                desc="Run from a Dynamo workspace directory or set DYNAMO_HOME",
                status=NodeStatus.INFO,
            )
            self.add_child(hint)
            return

        # Get git info
        sha, date = self._get_git_info(workspace_dir)

        # Build main label
        display_workspace = self._replace_home_with_var(workspace_dir)
        if sha and date:
            value = f"{display_workspace}, SHA: {sha}, Date: {date}"
        else:
            value = display_workspace

        super().__init__(label="Dynamo", desc=value, status=NodeStatus.INFO)

        # Always add runtime components
        runtime_info = DynamoRuntimeInfo(
            workspace_dir, thorough_check=self.thorough_check
        )
        self.add_child(runtime_info)

        # Always add framework components
        framework_info = DynamoFrameworkInfo(
            workspace_dir, thorough_check=self.thorough_check
        )
        self.add_child(framework_info)

    def _get_git_info(self, workspace_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """Get git SHA and date for the workspace."""
        try:
            # Get short SHA
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=workspace_dir,
                timeout=5,
            )
            sha = result.stdout.strip() if result.returncode == 0 else None

            # Get commit date
            result = subprocess.run(
                ["git", "show", "-s", "--format=%ci", "HEAD"],
                capture_output=True,
                text=True,
                cwd=workspace_dir,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Convert to PDT format
                date_str = result.stdout.strip()
                # Parse and format as PDT
                try:
                    # Parse the git date (format: 2025-08-30 23:22:29 +0000)
                    import datetime as dt_module

                    # Split off timezone info
                    date_part = date_str.rsplit(" ", 1)[0]
                    dt = dt_module.datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
                    # Convert to PDT (UTC-7)
                    dt_pdt = dt - dt_module.timedelta(hours=7)
                    date = dt_pdt.strftime("%Y-%m-%d %H:%M:%S PDT")
                except Exception:
                    date = date_str
            else:
                date = None

            return sha, date
        except Exception:
            return None, None

    @staticmethod
    def find_workspace() -> Optional[str]:
        """Find dynamo workspace directory."""
        candidates = []

        # Check DYNAMO_HOME environment variable first
        dynamo_home = os.environ.get("DYNAMO_HOME")
        if dynamo_home:
            candidates.append(dynamo_home)

        # Then check common locations
        candidates.extend(
            [
                ".",  # Current directory
                os.path.expanduser("~/dynamo"),
                "/workspace",
            ]
        )

        for candidate in candidates:
            if DynamoInfo.is_dynamo_workspace(candidate):
                return os.path.abspath(candidate)
        return None

    @staticmethod
    def is_dynamo_workspace(path: str) -> bool:
        """Check if directory is a dynamo workspace."""
        if not os.path.exists(path):
            return False

        # Check for indicators of a dynamo workspace
        indicators = [
            "README.md",
            "components",
            "lib/bindings/python",
            "lib/runtime",
            "Cargo.toml",
        ]

        # Require at least 3 indicators to be confident
        found = 0
        for indicator in indicators:
            check_path = os.path.join(path, indicator)
            if os.path.exists(check_path):
                found += 1

        return found >= 3


def has_framework_errors(tree: NodeInfo) -> bool:
    """Check if there are framework component errors in the tree"""
    # Find the Dynamo node
    for child in tree.children:
        if child.label and "Dynamo" in child.label:
            # Find the Framework components node
            for dynamo_child in child.children:
                if dynamo_child.label and "Framework components" in dynamo_child.label:
                    # Use the has_errors() method to check the entire subtree
                    return dynamo_child.has_errors()
    return False


def show_installation_recommendation():
    """Show installation recommendations for missing components."""
    print("\nTo install missing components for development (not production):")
    print("  Runtime:   (cd lib/bindings/python && maturin develop)")
    print("  Framework: uv pip install -e .")
    print("             or export PYTHONPATH=$DYNAMO_HOME/components/src\n")


def main():
    """Main function - collect and display system information"""
    import argparse
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Display system information for Dynamo project"
    )
    parser.add_argument(
        "--thorough-check",
        action="store_true",
        help="Enable thorough checking (file permissions, directory sizes, disk space, etc.)",
    )
    parser.add_argument(
        "--terse",
        action="store_true",
        help="Show only essential information (OS, User, GPU, Framework, Dynamo) and errors",
    )
    args = parser.parse_args()

    # Validate mutual exclusion
    if args.thorough_check and args.terse:
        parser.error("--thorough-check and --terse cannot be used together")

    # Simply create a SystemInfo instance - it collects everything in its constructor
    tree = SystemInfo(thorough_check=args.thorough_check, terse=args.terse)
    tree.print_tree()

    # Check if there are framework component errors and show installation recommendation
    if has_framework_errors(tree):
        show_installation_recommendation()

    # Exit with non-zero status if there are any errors
    if tree.has_errors():
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
