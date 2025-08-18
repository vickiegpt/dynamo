# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging

import torch

import dynamo.nixl_connect as nixl_connect
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.request_handlers.handler_base import (
    DisaggregationMode,
    DisaggregationStrategy,
    HandlerBase,
    RequestHandlerConfig,
)

configure_dynamo_logging()


class RequestHandlerFactory:
    def __init__(self):
        self.handlers = {
            "prefill": PrefillHandler,
            "decode": DecodeHandler,
            "encode": EncodeHandler,
            "prefill_and_decode": AggregatedHandler,
        }

    def _validate_config(self, config: RequestHandlerConfig):
        if config.disaggregation_mode.value not in self.handlers:
            raise ValueError(
                f"Invalid disaggregation_mode '{config.disaggregation_mode.value}'"
            )

        if not config.next_client:
            if (
                config.disaggregation_mode == DisaggregationMode.PREFILL
                and config.disaggregation_strategy
                == DisaggregationStrategy.PREFILL_FIRST
            ):
                raise ValueError(
                    "Next client is required for the main worker when disaggregation_mode='prefill' and disaggregation_strategy='prefill_first'."
                )
            if (
                config.disaggregation_mode == DisaggregationMode.DECODE
                and config.disaggregation_strategy
                == DisaggregationStrategy.DECODE_FIRST
            ):
                raise ValueError(
                    "Next client is required for the decode worker when disaggregation_mode='decode' and disaggregation_strategy='decode_first'."
                )

    def get_request_handler(self, config: RequestHandlerConfig) -> HandlerBase:
        self._validate_config(config)
        return self.handlers[config.disaggregation_mode.value](config)


def get_request_handler(config: RequestHandlerConfig) -> HandlerBase:
    return RequestHandlerFactory().get_request_handler(config)


class AggregatedHandler(HandlerBase):
    """
    Handler for the aggregated mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def generate(self, request: dict):
        # Implement all steps locally.
        async for res in self.generate_locally(request):
            yield res


class EncodeHandler(HandlerBase):
    """
    Handler for the encode mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        logging.info(f"EncodeHandler initialized with config: {config}")
        super().__init__(config)
        self.encodings = None

    async def generate(self, request: dict):
        logging.info(f"EncodeHandler received request: {request}")

        if self.connector and "nixl_metadata" in request:
            # The prefill worker has requested that we write the encodings
            # to a shared memory region.
            metadata = nixl_connect.RdmaMetadata.model_validate(
                request["nixl_metadata"]
            )
            messages = request.get("messages", [])
            _, _, embedding_paths = self.multimodal_processor.extract_prompt_and_media(
                messages
            )
            if embedding_paths:
                self.encodings = self.multimodal_processor.load_tensor_from_path_or_url(
                    embedding_paths[0]
                )
            else:
                # Placeholder for TRTLLM Encoder to be called
                # TRTLLM Encoder will return a memory handler on the the encoder GPU with the encodings
                logging.warning(
                    "No embedding paths found, NIXL transfer for image urls not supported by TRTLLM Encoder yet"
                )
                yield {}
                return

            descriptor = nixl_connect.Descriptor(self.encodings)
            write_op = await self.connector.begin_write(descriptor, metadata)
            with write_op:
                await write_op.wait_for_completion()
            logging.info("EncodeHandler completed write to shared memory.")
            # Yield back an empty response to signal completion.
            yield {}
            return

        if not request.get("streaming", False):
            yield request
            return

        yield request


class PrefillHandler(HandlerBase):
    """
    Handler for the prefill mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        logging.info(f"PrefillHandler initialized with config: {config}")
        super().__init__(config)

    async def remote_encode_with_nixl(self, request: dict):
        # 1. Allocate a tensor for the encodings.
        shape = self.get_embeddings_shape()
        encodings_tensor = torch.zeros(*shape, dtype=torch.float32)
        logging.info(f"PrefillHandler encodings before write: {encodings_tensor}")
        # 2. Create a descriptor for the tensor.
        descriptor = nixl_connect.Descriptor(encodings_tensor)

        # 3. Create a writable operation.
        with self.connector.create_writable(descriptor) as writable_op:
            # 4. Get the metadata from the operation.
            op_metadata = writable_op.metadata()

            # 5. Send the metadata to the encode worker.
            request["nixl_metadata"] = op_metadata.model_dump()
            logging.info(f"PrefillHandler sending nixl metadata: {request}")
            response_received = False
            async for res in await self.encode_client.round_robin(request):
                response_received = True

            if not response_received:
                raise RuntimeError("Did not receive a response from the encode worker.")

            # 6. Wait for the encode worker to complete the write operation.
            await writable_op.wait_for_completion()
            logging.info(
                f"PrefillHandler received encodings from encode worker: {encodings_tensor}"
            )

        # 7. Return the modifed encodings.
        return encodings_tensor

    async def remote_encode(self, request: dict):
        logging.info(f"PrefillHandler.remote_encode sending request: {request}")
        response_received = False
        async for res in await self.encode_client.round_robin(request):
            logging.info(f"PrefillHandler.remote_encode received response: {res}")
            response_received = True
            yield res.data()

        if not response_received:
            logging.warning("No response received from the encode client.")
            yield request

    async def remote_decode(self, request: dict):
        async for res in await self.next_client.round_robin(request):
            yield res.data()

    async def generate(self, request: dict):
        logging.info(f"PrefillHandler.generate received request: {request}")
        embeddings_tensor = None

        # STATE 1: If an encoder is configured and the request needs encoding, call it.
        if self.encode_client and "encodings" not in request:
            if self.connector:
                logging.info("PrefillHandler calling remote_encode_with_nixl")
                embeddings_tensor = await self.remote_encode_with_nixl(request)
            else:
                logging.info("PrefillHandler calling remote_encode")
                async for res in self.remote_encode(request):
                    # The encoder returns the modified request. Adopt it as our new state.
                    request = res
                    logging.info(f"Encoded request: {request}")
                    break  # The encoder only returns one response.

        # STATE 2: Request is ready for prefill.
        # Generate the prefill response locally
        prefill_request = copy.deepcopy(request)
        prefill_response = None
        response_count = 0
        logging.info(f"Prefill request: {prefill_request}")
        async for res in self.generate_locally(prefill_request, embeddings_tensor):
            prefill_response = res
            response_count += 1
            if response_count > 1:
                raise ValueError("Prefill response should be generated only once.")

        if prefill_response is None:
            logging.warning(
                "Prefill response is None. It's possible that the generation resulted in an empty response."
            )

        if (
            self.disaggregation_strategy == DisaggregationStrategy.PREFILL_FIRST
            and not self.check_error(prefill_response)
        ):
            # If operating under prefill_first strategy, the prefill handler needs to trigger
            # the decode handler.
            if prefill_response is not None:
                request["disaggregated_params"] = prefill_response[
                    "disaggregated_params"
                ]
            async for res in self.remote_decode(request):
                yield res
        else:
            # Return response to the decode handler.
            yield prefill_response


class DecodeHandler(HandlerBase):
    """
    Handler for the decode mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        logging.info(f"DecodeHandler initialized with config: {config}")
        super().__init__(config)

    async def remote_prefill(self, request: dict):
        async for res in await self.next_client.round_robin(request):
            yield res

    async def generate(self, request: dict):
        logging.info(f"DecodeHandler.generate received request: {request}")
        if self.disaggregation_strategy == DisaggregationStrategy.DECODE_FIRST:
            prefill_response = None
            # If operating under decode_first strategy, the decode handler needs to trigger
            # the prefill handler.
            response_count = 0
            # Do not yield the prefill response directly.
            # Instead, capture it and extract the state.
            async for res in self.remote_prefill(request):
                prefill_response = res
                response_count += 1
                if response_count > 1:
                    raise ValueError("Prefill response should be generated only once.")

            response_data = (
                prefill_response.data() if prefill_response is not None else None
            )
            if prefill_response is not None and self.check_error(response_data):
                yield response_data
                return

            if prefill_response is not None and response_data is not None:
                request["disaggregated_params"] = response_data["disaggregated_params"]

        async for res in self.generate_locally(request):
            yield res
