import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config import LoadConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.distributed.parallel_state import DRY_RUN_BACKEND

from .companion_messages import CUDATensorRebuildInfo


logger = init_logger(__name__)


def override_vllm_config(
    vllm_config: VllmConfig,
    device_id: int,
) -> VllmConfig:
    # NOTE: If we deepcopy, there's a bunch of other stuff that gets copied that we don't want
    new_vllm_config = VllmConfig(
        model_config=vllm_config.model_config,
        parallel_config=vllm_config.parallel_config,
        cache_config=vllm_config.cache_config,
        device_config=vllm_config.device_config,
        load_config=vllm_config.load_config,
    )
    # Override load config for dry run on SPECIFIED device
    # NOTE: we use device_id and not local_rank because we assume CUDA_VISIBLE_DEVICES is not set
    # local_rank would come from the client view which may not be correct
    new_vllm_config.load_config.device = torch.device(f"cuda:{device_id}")

    # We want to load the model for real here
    new_vllm_config.load_config.enable_ipc_loading = False

    return new_vllm_config


class ModelInstanceManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device_id: int,
        local_rank: int,
        global_rank: int,
        world_size: int,
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        self.vllm_config = override_vllm_config(
            vllm_config,
            device_id,
        )

        torch.cuda.set_device(self.vllm_config.load_config.device)

        self._model = None
        self._model_parameters_ipc_info: dict[str, CUDATensorRebuildInfo] = {}

        with set_current_vllm_config(self.vllm_config):
            self._load_model()

    def _initialize_dry_distributed_environment(self):
        """Initialize distributed environment in dry mode for correct weight loading."""
        # Set environment variables for torch.distributed if not set
        logger.warning("Initializing dry distributed environment...")
        # Initialize distributed environment in dry mode
        parallel_state.init_distributed_environment(
            world_size=self.world_size,
            rank=self.global_rank,
            distributed_init_method="env://",
            local_rank=self.local_rank,
            backend=DRY_RUN_BACKEND,
        )
        logger.warning("Dry distributed environment initialized!")

        # Initialize model parallel groups
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.vllm_config.parallel_config.tensor_parallel_size,
            pipeline_model_parallel_size=self.vllm_config.parallel_config.pipeline_parallel_size,
        )

        logger.info(
            "Initialized dry distributed environment: local rank=%d, global rank=%d, world_size=%d, "
            "TP=%d, PP=%d, DP=%d",
            self.local_rank,
            self.global_rank,
            self.world_size,
            self.vllm_config.parallel_config.tensor_parallel_size,
            self.vllm_config.parallel_config.pipeline_parallel_size,
            self.vllm_config.parallel_config.data_parallel_size,
        )

    def _load_model(self):
        # Initialize dry distributed environment first
        self._initialize_dry_distributed_environment()

        # Create model loader
        load_config = LoadConfig()
        default_loader = DefaultModelLoader(load_config=load_config)

        logger.info("Loading model %s...", self.vllm_config.model_config.model)

        self._model = default_loader.load_model(
            self.vllm_config, self.vllm_config.model_config
        )

        # Make sure model is in eval mode
        assert self._model.training is False

        logger.info("Model loaded! Getting IPC rebuild info...")

        # Extract parameters and create IPC rebuild info
        self._model_parameters_ipc_info = {}
        for name, param in self._model.named_parameters():
            if not isinstance(param, torch.nn.Parameter):
                continue

            # Make sure parameter is on the current device ID
            assert param.device == self.vllm_config.load_config.device

            # Get the underlying tensor
            tensor = param.data

            # Ensure tensor is on CUDA
            if tensor.device.type != "cuda":
                logger.warning("Parameter %s is not on CUDA, skipping", name)
                continue

            # Get rebuild info for IPC sharing
            _, rebuild_args = reduce_tensor(tensor)
            self._model_parameters_ipc_info[name] = (
                CUDATensorRebuildInfo.from_rebuild_args(rebuild_args)
            )

        logger.info(
            "Model initialized with %d parameters ready for IPC sharing",
            len(self._model_parameters_ipc_info),
        )

    def get_model_parameters_ipc_info(self) -> dict[str, CUDATensorRebuildInfo]:
        """Get IPC rebuild info for all model parameters."""
        return self._model_parameters_ipc_info
