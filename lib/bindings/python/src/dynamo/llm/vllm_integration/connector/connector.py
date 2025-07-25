from dataclasses import field
from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic.dataclasses import dataclass
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.cache_engine import CacheEngine

from dynamo._core import _vllm_connector_integration
from dynamo.llm import BlockManager, KvbmLeader, KvbmWorker

KvbmConnectorLeader = _vllm_connector_integration.KvbmConnectorLeader

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class RequestState:
    """Track state for each request across prefill and decode phases"""

    request_id: str
    block_ids: list[int]
    initial_tokens: int  # Number of tokens in prefill
    last_computed_tokens: int = 0  # Track progress to detect new full blocks


@dataclass
class BlockOffloadInfo:
    """Information about a block ready for offload"""

    block_id: int
    request_id: str
    tokens_in_block: int
    is_full: bool  # True when block has 16+ tokens and ready for offload
    device_id: int = 0


@dataclass
class DynamoKvbmConnectorMetadata(KVConnectorMetadata):
    """Metadata instructing workers what to offload/load"""

    # Requests that just finished prefill (ready to start caching)
    prefill_completed_requests: list[str] = field(default_factory=list)

    # Blocks ready for offload to KVBM host storage
    blocks_to_offload: list[BlockOffloadInfo] = field(default_factory=list)

    # Step information for worker coordination
    step_type: str = "idle"  # "prefill_complete", "decode", "idle"

    # Total scheduled tokens this step
    total_tokens_scheduled: int = 0


class DynamoKvbmConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole"):
        super().__init__(vllm_config, role)

        # State tracking for requests across prefill and decode phases
        self._request_states: dict[str, RequestState] = {}

        # We can immediately initialize our leader here.
        if role == KVConnectorRole.SCHEDULER:
            world_size = self._vllm_config.parallel_config.world_size

            bytes_per_block = CacheEngine.get_cache_block_size(
                self._vllm_config.cache_config,
                self._vllm_config.model_config,
                self._vllm_config.parallel_config,
            )

            total_bytes = bytes_per_block * world_size

            leader = KvbmLeader(total_bytes, world_size)

            block_manager = BlockManager(
                0,
                leader,
                self._vllm_config.cache_config.block_size,
            )

            self.leader = KvbmConnectorLeader(block_manager)

    @property
    def role(self) -> "KVConnectorRole":
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        assert self.role == KVConnectorRole.WORKER

        # Sophisticated logging of received metadata
        self._print_worker_metadata(connector_metadata)

        self._connector_metadata = connector_metadata

    def _print_worker_metadata(self, metadata: KVConnectorMetadata) -> None:
        """Print detailed analysis of metadata received by worker"""
        print("🔄" + "=" * 80)
        print("🤖 WORKER RECEIVED METADATA FROM SCHEDULER")
        print("🔄" + "=" * 80)

        if isinstance(metadata, DynamoKvbmConnectorMetadata):
            print(f"📋 Metadata Type: {type(metadata).__name__}")
            print(f"📊 Step Type: {metadata.step_type}")
            print(f"🔢 Total Tokens Scheduled: {metadata.total_tokens_scheduled}")

            # Prefill completion analysis
            if metadata.prefill_completed_requests:
                print(
                    f"✅ Prefill Completed Requests: {len(metadata.prefill_completed_requests)}"
                )
                for req_id in metadata.prefill_completed_requests:
                    print(f"   └── 🎯 {req_id}")
                print("   💡 Action: Prefill phase finished, prepare for caching")
            else:
                print("✅ Prefill Completed Requests: None")

            # Block offload analysis
            if metadata.blocks_to_offload:
                print(f"📦 Blocks to Offload: {len(metadata.blocks_to_offload)}")
                print(
                    "   ┌─────────────────────────────────────────────────────────────────"
                )
                print(
                    "   │ Block ID │ Request ID (first 8 chars) │ Tokens │ Status │ Device"
                )
                print(
                    "   ├─────────────────────────────────────────────────────────────────"
                )

                for block_info in metadata.blocks_to_offload:
                    req_short = (
                        block_info.request_id[:8] + "..."
                        if len(block_info.request_id) > 8
                        else block_info.request_id
                    )
                    status = "FULL" if block_info.is_full else "PARTIAL"
                    print(
                        f"   │ {block_info.block_id:8d} │ {req_short:23s} │ {block_info.tokens_in_block:6d} │ {status:6s} │ {block_info.device_id:6d}"
                    )

                print(
                    "   └─────────────────────────────────────────────────────────────────"
                )
                print("   💡 Action: Start vLLM → KVBM host offload for these blocks")

                # Offload instructions
                self._print_offload_instructions(metadata.blocks_to_offload)
            else:
                print("📦 Blocks to Offload: None")
                print("   💡 Action: No offload required this step")

            # Step-specific worker actions
            self._print_worker_actions(metadata)

        else:
            print(f"⚠️  Unknown metadata type: {type(metadata)}")
            print(f"📋 Raw metadata: {metadata}")

        print("🔄" + "=" * 80)

    def _print_offload_instructions(
        self, blocks_to_offload: list[BlockOffloadInfo]
    ) -> None:
        """Print detailed offload instructions for worker"""
        if not blocks_to_offload:
            return

        print("   🔧 OFFLOAD INSTRUCTIONS:")

        # Group by request for cleaner instructions
        by_request = {}
        for block in blocks_to_offload:
            if block.request_id not in by_request:
                by_request[block.request_id] = []
            by_request[block.request_id].append(block)

        for req_id, blocks in by_request.items():
            req_short = req_id[:12] + "..." if len(req_id) > 12 else req_id
            block_ids = [str(b.block_id) for b in blocks]
            total_tokens = sum(b.tokens_in_block for b in blocks)

            print(f"   ├── Request {req_short}:")
            print(f"   │   ├── Blocks: [{', '.join(block_ids)}]")
            print(f"   │   ├── Total tokens: {total_tokens}")
            print(
                f"   │   └── Action: Copy {len(blocks)} blocks from GPU → Host memory"
            )

    def _print_worker_actions(self, metadata: DynamoKvbmConnectorMetadata) -> None:
        """Print recommended actions based on step type"""
        print("   🎯 RECOMMENDED WORKER ACTIONS:")

        if metadata.step_type == "prefill":
            print("   ├── 🟡 PREFILL PHASE")
            print("   │   ├── Monitor block allocation")
            print("   │   ├── Prepare for potential caching")
            print("   │   └── Wait for prefill completion")

        elif metadata.step_type == "prefill_complete":
            print("   ├── 🟢 PREFILL COMPLETED")
            print("   │   ├── Prefill phase finished")
            print("   │   ├── Start monitoring for full blocks")
            if metadata.blocks_to_offload:
                print("   │   └── ⚡ EXECUTE OFFLOAD: Copy ready blocks to KVBM")
            else:
                print("   │   └── Wait for blocks to fill (< 16 tokens)")

        elif metadata.step_type == "decode":
            print("   ├── 🔵 DECODE PHASE")
            print("   │   ├── Continue token generation")
            if metadata.blocks_to_offload:
                print("   │   └── ⚡ EXECUTE OFFLOAD: New blocks ready")
            else:
                print("   │   └── Monitor for block completion")

        elif metadata.step_type == "idle":
            print("   ├── ⚪ IDLE")
            print("   │   └── No active scheduling")

        else:
            print(f"   ├── ❓ UNKNOWN: {metadata.step_type}")

        # Memory pressure recommendations
        if metadata.blocks_to_offload and len(metadata.blocks_to_offload) > 3:
            print("   ├── ⚠️  HIGH OFFLOAD VOLUME")
            print("   │   ├── Consider batch offload for efficiency")
            print("   │   └── Monitor KVBM host memory pressure")

        print("   └── ✅ End of actions")

    def _print_scheduler_metadata(self, metadata: DynamoKvbmConnectorMetadata) -> None:
        """Print detailed analysis of metadata being prepared by scheduler"""
        print("📡" + "=" * 80)
        print("📡 SCHEDULER PREPARING METADATA FOR WORKERS")
        print("📡" + "=" * 80)

        print("🏗️  Metadata Creation Summary:")
        print(f"   ├── Type: {type(metadata).__name__}")
        print(f"   ├── Step Classification: {metadata.step_type}")
        print(f"   ├── Total Tokens This Step: {metadata.total_tokens_scheduled}")
        print(f"   └── Active Request States: {len(self._request_states)}")

        # State tracking overview
        if self._request_states:
            print("📚 Request State Tracking:")
            print(
                "   ┌──────────────────────────────────────────────────────────────────────"
            )
            print(
                "   │ Request ID (first 12)    │ Blocks │ Init Tokens │ Last Computed │ Status"
            )
            print(
                "   ├──────────────────────────────────────────────────────────────────────"
            )

            for req_id, state in self._request_states.items():
                req_short = req_id[:12] + "..." if len(req_id) > 12 else req_id
                blocks_str = f"[{','.join(map(str, state.block_ids[:3]))}{'...' if len(state.block_ids) > 3 else ''}]"
                status = "DECODE" if state.last_computed_tokens > 0 else "PREFILL"
                print(
                    f"   │ {req_short:20s} │ {blocks_str:6s} │ {state.initial_tokens:11d} │ {state.last_computed_tokens:13d} │ {status:6s}"
                )

            print(
                "   └──────────────────────────────────────────────────────────────────────"
            )
        else:
            print("📚 Request State Tracking: Empty")

        # Prefill analysis
        if metadata.prefill_completed_requests:
            print(
                f"✅ Prefill Completions Detected: {len(metadata.prefill_completed_requests)}"
            )
            for req_id in metadata.prefill_completed_requests:
                req_short = req_id[:16] + "..." if len(req_id) > 16 else req_id
                print(f"   ├── 🎯 {req_short}")
                if req_id in self._request_states:
                    state = self._request_states[req_id]
                    print(f"   │   ├── Initial tokens: {state.initial_tokens}")
                    print(f"   │   ├── Allocated blocks: {len(state.block_ids)}")
                    print("   │   └── Transition: PREFILL → DECODE")
            print(
                "   💌 Instruction to Workers: Prefill finished, start cache monitoring"
            )
        else:
            print("✅ Prefill Completions: None")

        # Block offload analysis
        if metadata.blocks_to_offload:
            print(f"📦 Block Offload Instructions: {len(metadata.blocks_to_offload)}")

            # Group by request for analysis
            by_request = {}
            total_offload_tokens = 0

            for block in metadata.blocks_to_offload:
                if block.request_id not in by_request:
                    by_request[block.request_id] = []
                by_request[block.request_id].append(block)
                total_offload_tokens += block.tokens_in_block

            print(
                "   ┌────────────────────────────────────────────────────────────────────"
            )
            print("   │ Request (first 12)   │ Block IDs    │ Tokens │ Action")
            print(
                "   ├────────────────────────────────────────────────────────────────────"
            )

            for req_id, blocks in by_request.items():
                req_short = req_id[:12] + "..." if len(req_id) > 12 else req_id
                block_ids = [str(b.block_id) for b in blocks]
                block_str = (
                    f"[{','.join(block_ids[:3])}{'...' if len(block_ids) > 3 else ''}]"
                )
                total_tokens = sum(b.tokens_in_block for b in blocks)
                action = f"Offload {len(blocks)} blocks"

                print(
                    f"   │ {req_short:20s} │ {block_str:12s} │ {total_tokens:6d} │ {action}"
                )

            print(
                "   └────────────────────────────────────────────────────────────────────"
            )
            print(
                f"   📊 Total Offload Volume: {len(metadata.blocks_to_offload)} blocks, {total_offload_tokens} tokens"
            )
            print(
                "   💌 Instruction to Workers: Execute vLLM → KVBM host memory transfers"
            )

            # Performance analysis
            if len(metadata.blocks_to_offload) > 3:
                print("   ⚠️  High Offload Volume - Workers should batch operations")

        else:
            print("📦 Block Offload Instructions: None")
            print("   💌 Instruction to Workers: No offload required this step")

        # Step-specific scheduler analysis
        self._print_scheduler_step_analysis(metadata)

        # Performance recommendations
        self._print_scheduler_recommendations(metadata)

        print("📡" + "=" * 80)

    def _print_scheduler_step_analysis(
        self, metadata: DynamoKvbmConnectorMetadata
    ) -> None:
        """Print scheduler's analysis of the current step"""
        print("🧠 Scheduler Step Analysis:")

        if metadata.step_type == "prefill":
            print("   ├── 🟡 PREFILL PHASE DETECTED")
            print("   │   ├── New requests are being processed")
            print("   │   ├── Block allocation in progress")
            print("   │   └── State tracking initiated for new requests")

        elif metadata.step_type == "prefill_complete":
            print("   ├── 🟢 PREFILL → DECODE TRANSITION")
            print("   │   ├── Prefill phase completed")
            print("   │   ├── Workers can start cache operations")
            if metadata.blocks_to_offload:
                print("   │   └── Some blocks already ready for offload")
            else:
                print("   │   └── Waiting for blocks to reach 16-token threshold")

        elif metadata.step_type == "decode":
            print("   ├── 🔵 ONGOING DECODE PHASE")
            print("   │   ├── Token generation in progress")
            print("   │   ├── Monitoring block completion")
            if metadata.blocks_to_offload:
                print("   │   └── New full blocks detected - triggering offload")
            else:
                print("   │   └── No new full blocks this step")

        elif metadata.step_type == "idle":
            print("   ├── ⚪ IDLE STATE")
            print("   │   ├── No requests currently scheduled")
            print("   │   └── Workers can optimize existing cache")

        else:
            print(f"   ├── ❓ UNKNOWN STATE: {metadata.step_type}")

    def _print_scheduler_recommendations(
        self, metadata: DynamoKvbmConnectorMetadata
    ) -> None:
        """Print scheduler's recommendations for optimization"""
        print("💡 Scheduler Recommendations:")

        # Memory pressure analysis
        active_requests = len(self._request_states)
        total_blocks_tracked = sum(
            len(state.block_ids) for state in self._request_states.values()
        )

        if active_requests > 5:
            print("   ├── ⚠️  HIGH REQUEST LOAD")
            print(f"   │   ├── {active_requests} active requests")
            print(f"   │   ├── {total_blocks_tracked} total blocks tracked")
            print("   │   └── Consider aggressive offloading")

        if metadata.blocks_to_offload and len(metadata.blocks_to_offload) > 2:
            print("   ├── 🚀 OFFLOAD OPTIMIZATION")
            print("   │   ├── High block offload volume")
            print("   │   └── Workers should batch transfers")

        if not metadata.blocks_to_offload and active_requests > 0:
            print("   ├── 📈 CACHE BUILDING PHASE")
            print("   │   ├── Requests generating tokens")
            print("   │   └── Monitor for block completion")

        print("   └── ✅ Analysis complete")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert (
            self.role == KVConnectorRole.WORKER
        ), "Only worker role can register KV caches"

        cache_config = self._vllm_config.cache_config

        shape = list(kv_caches.values())[0].shape

        if not all(t.shape == shape for t in kv_caches.values()):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        # TODO: Assume the block dimension is within the first 2. This will break if you're doing something weird like having 1 or 2 device blocks.
        num_device_blocks = max(shape[0], shape[1])
        page_size = cache_config.block_size
        tensors = list(kv_caches.values())

        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = self._vllm_config.model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        device_id = tensors[0].device.index

        # TODO: We can actually just initialize our connection to the leader from the kv transfer params port argument.
        self.worker = KvbmWorker(
            num_device_blocks,
            page_size,
            tensors,
            device_id=device_id,
            worker_id=device_id,
            dtype_width_bytes=kv_cache_dtype.itemsize,
        )

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

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.
        Args:
            layer_name: the name of that layer
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:  # type: ignore
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
        print("=" * 60)
        print("WORKER/KVBM SAVE OPERATION")
        print("=" * 60)
        print(layer_name)
        print("=" * 60)
        connector_metadata = self._get_connector_metadata()
        if connector_metadata.blocks_to_offload:
            for block in connector_metadata.blocks_to_offload:
                print("=" * 60)
                print("BLOCK TO OFFLOAD")
                print("=" * 60)
                print(block)
                print("=" * 60)
        else:
            print("No blocks to offload")

        if layer_name == "model.layers.27.self_attn.attn":
            print("=" * 60)
            print("KV LAYER ANALYSIS")
            print("=" * 60)

            if isinstance(kv_layer, torch.Tensor):
                print(f"📊 KV Layer Tensor Shape: {kv_layer.shape}")
                print(f"📊 KV Layer Tensor dtype: {kv_layer.dtype}")
                print(f"📊 KV Layer Tensor device: {kv_layer.device}")
                print(
                    f"📊 KV Layer Tensor size (bytes): {kv_layer.numel() * kv_layer.element_size()}"
                )

            print("=" * 60)

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.
        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

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
        return None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.
        """
        _tokens = request.all_token_ids
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.
        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        pass

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> "KVConnectorMetadata":
        """
        Build the connector metadata for this step.
        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        # print("=" * 60)
        # print("LEADER/SCHEDULER METADATA PREPARATION")
        # print("=" * 60)

        def print_object_structure(
            obj, name="object", max_depth=3, current_depth=0, max_items=5
        ):
            indent = "  " * current_depth

            if current_depth >= max_depth:
                print(f"{indent}{name}: <max_depth_reached> {type(obj).__name__}")
                return

            if obj is None:
                print(f"{indent}{name}: None")
            elif isinstance(obj, (str, int, float, bool)):
                print(f"{indent}{name}: {type(obj).__name__} = {repr(obj)}")
            elif isinstance(obj, (list, tuple)):
                print(f"{indent}{name}: {type(obj).__name__}[{len(obj)}]")
                for i, item in enumerate(obj[:max_items]):
                    print_object_structure(
                        item, f"[{i}]", max_depth, current_depth + 1, max_items
                    )
                if len(obj) > max_items:
                    print(f"{indent}  ... {len(obj) - max_items} more items")
            elif isinstance(obj, dict):
                print(f"{indent}{name}: dict[{len(obj)}]")
                for i, (key, value) in enumerate(list(obj.items())[:max_items]):
                    print_object_structure(
                        value, f"'{key}'", max_depth, current_depth + 1, max_items
                    )
                if len(obj) > max_items:
                    print(f"{indent}  ... {len(obj) - max_items} more keys")
            elif hasattr(obj, "__dict__"):
                attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
                print(f"{indent}{name}: {type(obj).__name__}")
                for i, (attr, value) in enumerate(list(attrs.items())[:max_items]):
                    print_object_structure(
                        value, f".{attr}", max_depth, current_depth + 1, max_items
                    )
                if len(attrs) > max_items:
                    print(f"{indent}  ... {len(attrs) - max_items} more attributes")
            elif hasattr(obj, "shape"):  # Tensor-like objects
                print(
                    f"{indent}{name}: {type(obj).__name__} shape={getattr(obj, 'shape', 'unknown')}"
                )
            else:
                # For other objects, try to show useful info
                attrs = []
                for attr_name in ["shape", "size", "dtype", "device"]:
                    if hasattr(obj, attr_name):
                        attrs.append(f"{attr_name}={getattr(obj, attr_name)}")
                attrs_str = " ".join(attrs)
                print(f"{indent}{name}: {type(obj).__name__} {attrs_str}")

        # print("=" * 60)
        # print("SCHEDULER OUTPUT STRUCTURE:")
        # print("=" * 60)
        # print_object_structure(scheduler_output, "scheduler_output", max_depth=10, max_items=999)
        # print("=" * 60)

        assert self.role == KVConnectorRole.SCHEDULER

        # Analyze scheduler output to build metadata
        prefill_completed = []
        blocks_to_offload = []
        step_type = "idle"

        # Block size from cache config (assuming 16 tokens per block)
        BLOCK_SIZE = 16

        # Case 1: Process new prefill requests - store their state and check for immediate offload
        if len(scheduler_output.scheduled_new_reqs) > 0:
            step_type = "prefill"

            for new_req in scheduler_output.scheduled_new_reqs:
                request_id = new_req.req_id
                initial_tokens = len(new_req.prompt_token_ids)

                # Extract block IDs from the new request
                block_ids = []
                if new_req.block_ids and len(new_req.block_ids) > 0:
                    # block_ids is a tuple of sequences, get the first sequence
                    if len(new_req.block_ids[0]) > 0:
                        block_ids = list(new_req.block_ids[0])

                # Store state for this request
                self._request_states[request_id] = RequestState(
                    request_id=request_id,
                    block_ids=block_ids,
                    initial_tokens=initial_tokens,
                    last_computed_tokens=0,
                )

                # 🚀 NEW: Check for full blocks during prefill and mark for immediate offload
                full_blocks_during_prefill = initial_tokens // BLOCK_SIZE
                if full_blocks_during_prefill > 0:
                    print(
                        f"⚡ Prefill has {full_blocks_during_prefill} full blocks ready for immediate offload"
                    )

                    for block_idx in range(full_blocks_during_prefill):
                        if block_idx < len(block_ids):
                            blocks_to_offload.append(
                                BlockOffloadInfo(
                                    block_id=block_ids[block_idx],
                                    request_id=request_id,
                                    tokens_in_block=BLOCK_SIZE,
                                    is_full=True,
                                    device_id=0,
                                )
                            )
                            print(
                                f"📦 Block {block_ids[block_idx]} marked for prefill offload (request {request_id})"
                            )

                    # Update the tracking state to reflect what we've already accounted for
                    self._request_states[request_id].last_computed_tokens = (
                        full_blocks_during_prefill * BLOCK_SIZE
                    )

                # print(f"📝 Stored state for prefill request {request_id}: {len(block_ids)} blocks, {initial_tokens} tokens")

        # Case 2: Process cached requests (decode phase) - use stored state
        if len(scheduler_output.scheduled_cached_reqs.req_ids) > 0:
            if step_type == "idle":  # No new requests, this is decode
                step_type = "decode"

            for i, cached_req in enumerate(
                scheduler_output.scheduled_cached_reqs.req_ids
            ):
                computed_tokens = (
                    scheduler_output.scheduled_cached_reqs.num_computed_tokens[i]
                    if i
                    < len(scheduler_output.scheduled_cached_reqs.num_computed_tokens)
                    else 0
                )

                # Get stored state for this request
                if cached_req in self._request_states:
                    req_state = self._request_states[cached_req]

                    # Detect if this is first decode step (prefill just completed)
                    if req_state.last_computed_tokens == 0 and computed_tokens > 0:
                        prefill_completed.append(cached_req)
                        if step_type == "decode":
                            step_type = "prefill_complete"

                    # Calculate newly completed blocks since last step
                    previous_full_blocks = req_state.last_computed_tokens // BLOCK_SIZE
                    current_full_blocks = computed_tokens // BLOCK_SIZE

                    # Check if we have new full blocks to offload
                    if current_full_blocks > previous_full_blocks:
                        # Add newly completed blocks to offload list
                        for block_idx in range(
                            previous_full_blocks, current_full_blocks
                        ):
                            if block_idx < len(req_state.block_ids):
                                blocks_to_offload.append(
                                    BlockOffloadInfo(
                                        block_id=req_state.block_ids[block_idx],
                                        request_id=cached_req,
                                        tokens_in_block=BLOCK_SIZE,
                                        is_full=True,
                                        device_id=0,
                                    )
                                )
                                print(
                                    f"🔄 Block {req_state.block_ids[block_idx]} ready for offload (request {cached_req})"
                                )

                    # Update tracking state
                    req_state.last_computed_tokens = computed_tokens
                else:
                    # Request not in our state - this shouldn't happen in normal flow
                    print(
                        f"⚠️  Warning: Request {cached_req} not found in stored states"
                    )

        # Cleanup finished requests from our state tracking
        if (
            hasattr(scheduler_output, "finished_req_ids")
            and scheduler_output.finished_req_ids
        ):
            for finished_req_id in scheduler_output.finished_req_ids:
                if finished_req_id in self._request_states:
                    del self._request_states[finished_req_id]
                    print(
                        f"🗑️  Cleaned up state for finished request {finished_req_id}"
                    )

        # print(f"🔄 Connector Analysis: step_type={step_type}, prefill_completed={prefill_completed}, blocks_to_offload={len(blocks_to_offload)}")
        # print(f"📊 Active request states: {len(self._request_states)}")

        # Create metadata
        metadata = DynamoKvbmConnectorMetadata(
            prefill_completed_requests=prefill_completed,
            blocks_to_offload=blocks_to_offload,
            step_type=step_type,
            total_tokens_scheduled=scheduler_output.total_num_scheduled_tokens,
        )

        # Sophisticated logging of prepared metadata
        self._print_scheduler_metadata(metadata)

        return metadata

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.
        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return False, None
