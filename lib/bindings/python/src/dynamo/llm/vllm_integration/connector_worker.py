# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext


# from dynamo.llm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
# from dynamo.llm.vllm_integration.rust import BlockManager
# from dynamo.llm.vllm_integration.rust import (
#     KvConnectorMetadata as RustKvConnectorMetadata,
#     KvConnectorWorker as RustKvConnectorWorker,
# )

from dynamo.llm.vllm_integration.kv_cache_utils import (
    find_and_set_available_port_from_env,
)
from dynamo.llm.vllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker
from dynamo.runtime import DistributedRuntime


class DynamoConnectorMetadata(KVConnectorMetadata):
    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


class KvConnectorWorker:
    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        drt = kwargs.get("drt", None)
        if drt is None:
            # this is needed to avoid metrics port conflict with KVBM leader side DRT if metrics is enabled
            find_and_set_available_port_from_env("DYN_SYSTEM_PORT")
            self.drt = DistributedRuntime.detached()
        else:
            self.drt = drt

        self.vllm_config = vllm_config
        self._connector = RustKvConnectorWorker(self.drt, engine_id)

    # Worker

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        print(
            f"KvConnectorWorker.register_kv_caches called with {len(kv_caches)} kv_caches"
        )
        cache_config = self.vllm_config.cache_config

        # Create ordered list of (layer_name, tensor) tuples sorted by layer index
        ordered_kv_caches = [
            (layer_name, tensor)
            for layer_name, tensor in sorted(
                kv_caches.items(), key=lambda item: extract_layer_index(item[0])
            )
        ]

        events = [
            torch.cuda.Event(enable_timing=False, interprocess=False)
            for _ in range(len(ordered_kv_caches))
        ]

        # events are lazy, if we don't record them once here, the raw handles we pass to rust will be null
        for event in events:
            event.record(torch.cuda.current_stream())

        raw_event_handles = [event.cuda_event for event in events]

        self.events = {
            layer_name: event
            for (layer_name, _tensor), event in zip(ordered_kv_caches, events)
        }

        # Get first tensor to extract common properties
        first_tensor = ordered_kv_caches[0][1]
        shape = first_tensor.shape

        # Validate all tensors have same shape
        if not all(t.shape == shape for t in kv_caches.values()):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        # Extract parameters
        # TODO: Assume the block dimension is within the first 2. This will break if you're doing something weird like having 1 or 2 device blocks.
        num_device_blocks = max(shape[0], shape[1])
        page_size = cache_config.block_size
        device_id = first_tensor.device.index

        # Determine cache dtype
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = self.vllm_config.model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        self.kv_caches = kv_caches
        self._kv_dump_thread = None
        self.start_kv_dump_watcher()

        # Register with connector using ordered data
        self._connector.register_kv_caches(
            num_device_blocks,
            page_size,
            device_id,
            kv_cache_dtype.itemsize,
            ordered_kv_caches,
            raw_event_handles,
        )

    def bind_connector_metadata(self, data: bytes) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector.bind_connector_metadata(data)

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._connector.clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.events[layer_name].record(torch.cuda.current_stream())
        self._connector.save_kv_layer(layer_name, kv_layer)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        # finished_ids = [id for id in finished_req_ids]
        # return set(sending_ids), set(receiving_ids)
        return self._connector.get_finished(finished_req_ids)

    def start_kv_dump_watcher(
        self,
        trigger_file: str = "/tmp/trigger_dump",
        out_dir: str = "/LMBenchmark/dump_kv",
        poll_interval: float = 0.5,
    ):
        """Start a background thread that waits for trigger_file, then dumps once."""

        def _watcher():
            print(f"[kvdump] Watcher started. Waiting for: {trigger_file}")
            try:
                while True:
                    if os.path.exists(trigger_file):
                        print(f"[kvdump] Trigger detected. Dumping to: {out_dir}")
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        snapshot_fp8_kv_caches(self.kv_caches, out_dir)
                        print(
                            f"[kvdump] Done. Manifest: {os.path.join(out_dir, 'manifest.json')}"
                        )
                        break
                    time.sleep(poll_interval)
            except Exception as e:
                import traceback

                print(f"[kvdump] ERROR: {e}\n{traceback.format_exc()}")

        self._kv_dump_thread = threading.Thread(
            target=_watcher, name="kv_dump_watcher", daemon=True
        )
        self._kv_dump_thread.start()
        print("[kvdump] started in the background.")


def _dump_bytes(t: torch.Tensor, path: str, chunk_bytes: int = 512 * 1024 * 1024):
    """Stream raw bytes from t to path, handling CUDA/CPU tensors safely."""
    elsize = t.element_size()
    flat = t.reshape(-1)
    numel = flat.numel()
    step = max(1, chunk_bytes // elsize)

    with open(path, "wb") as f:
        off = 0
        while off < numel:
            end = min(numel, off + step)
            piece = flat[off:end]
            # 1) Move to CPU (no pin_memory arg here)
            if piece.is_cuda:
                cpu_chunk = piece.detach().to(
                    torch.device("cpu"), non_blocking=False, copy=True
                )
            else:
                cpu_chunk = piece.detach().contiguous()
            # 2) (Optional) pin *after* it's on CPU if you really want pinned memory
            # cpu_chunk = cpu_chunk.pin_memory()  # not required for a blocking write
            mv = memoryview(cpu_chunk.numpy().data)
            f.write(mv)
            off = end


def snapshot_fp8_kv_caches(kv_caches: Dict[str, torch.Tensor], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "layers"), exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    manifest = {
        "version": "fp8_kv_snapshot_v1",
        "format": "fp8_only",
        "devices": sorted(
            {int(t.device.index) for t in kv_caches.values() if t.is_cuda}
        ),
        "layers": {},
        "aliases": {},
    }

    def key(t: torch.Tensor):
        return (
            int(t.device.index) if t.is_cuda else -1,
            int(t.data_ptr()),
            tuple(t.size()),
            tuple(t.stride()),
            str(t.dtype),
        )

    seen = {}

    for layer, t in kv_caches.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{layer} is not a tensor")
        fmt = "e4m3fn"

        info = {
            "shape": list(t.size()),
            "stride": list(t.stride()),
            "device": f"cuda:{int(t.device.index)}" if t.is_cuda else "cpu",
            "fp8_format": fmt,
            "numel": t.numel(),
            "num_bytes": t.numel() * t.element_size(),  # == numel
        }

        k = key(t)
        if k in seen:
            owner, rel = seen[k]
            manifest["aliases"][layer] = owner
            manifest["layers"][layer] = {**info, "path": rel, "alias_of": owner}
        else:
            fname = f"{layer.replace('/', '_').replace('.', '_')}.bin"
            dst = os.path.join(out_dir, "layers", fname)
            _dump_bytes(t, dst)
            rel = os.path.relpath(dst, out_dir)
            manifest["layers"][layer] = {**info, "path": rel}
            seen[k] = (layer, rel)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
