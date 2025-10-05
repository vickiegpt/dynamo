# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import logging
import os
import socket
import sys
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from sglang.srt.server_args import ServerArgs

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang import __version__

configure_dynamo_logging()

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"

DYNAMO_ARGS: Dict[str, Dict[str, Any]] = {
    "endpoint": {
        "flags": ["--endpoint"],
        "type": str,
        "help": f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Example: {DEFAULT_ENDPOINT}",
    },
    "migration-limit": {
        "flags": ["--migration-limit"],
        "type": int,
        "default": 0,
        "help": "Maximum number of times a request may be migrated to a different engine worker",
    },
    "tool-call-parser": {
        "flags": ["--dyn-tool-call-parser"],
        "type": str,
        "default": None,
        "choices": get_tool_parser_names(),
        "help": "Tool call parser name for the model.",
    },
    "reasoning-parser": {
        "flags": ["--dyn-reasoning-parser"],
        "type": str,
        "default": None,
        "choices": get_reasoning_parser_names(),
        "help": "Reasoning parser name for the model. If not specified, no reasoning parsing is performed.",
    },
    "custom-jinja-template": {
        "flags": ["--custom-jinja-template"],
        "type": str,
        "default": None,
        "help": "Path to a custom Jinja template file to override the model's default chat template. This template will take precedence over any template found in the model repository. This template will be applied by Dynamo's preprocessor and cannot be used with --use-sglang-tokenizer.",
    },
    "use-sglang-tokenizer": {
        "flags": ["--use-sglang-tokenizer"],
        "action": "store_true",
        "default": False,
        "help": "Use SGLang's tokenizer. This will skip tokenization of the input and output and only v1/chat/completions will be available when using the dynamo frontend. Cannot be used with --custom-jinja-template.",
    },
    "multimodal-processor": {
        "flags": ["--multimodal-processor"],
        "action": "store_true",
        "default": False,
        "help": "Run as multimodal processor component for handling multimodal requests",
    },
    "multimodal-encode-worker": {
        "flags": ["--multimodal-encode-worker"],
        "action": "store_true",
        "default": False,
        "help": "Run as multimodal encode worker component for processing images/videos",
    },
    "multimodal-worker": {
        "flags": ["--multimodal-worker"],
        "action": "store_true",
        "default": False,
        "help": "Run as multimodal worker component for LLM inference with multimodal data",
    },
}


@dataclass
class DynamoArgs:
    namespace: str
    component: str
    endpoint: str
    migration_limit: int

    # tool and reasoning parser options
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    custom_jinja_template: Optional[str] = None

    # preprocessing options
    use_sglang_tokenizer: bool = False

    # multimodal options
    multimodal_processor: bool = False
    multimodal_encode_worker: bool = False
    multimodal_worker: bool = False


class DisaggregationMode(Enum):
    AGGREGATED = "agg"
    PREFILL = "prefill"
    DECODE = "decode"


class Config:
    def __init__(self, server_args: ServerArgs, dynamo_args: DynamoArgs) -> None:
        self.server_args = server_args
        self.dynamo_args = dynamo_args
        self.serving_mode = self._set_serving_strategy()

    def _set_serving_strategy(self):
        if self.server_args.disaggregation_mode == "null":
            return DisaggregationMode.AGGREGATED
        elif self.server_args.disaggregation_mode == "prefill":
            return DisaggregationMode.PREFILL
        elif self.server_args.disaggregation_mode == "decode":
            return DisaggregationMode.DECODE
        else:
            return DisaggregationMode.AGGREGATED


def _set_parser(
    sglang_str: Optional[str],
    dynamo_str: Optional[str],
    arg_name: str = "tool-call-parser",
) -> Optional[str]:
    # If both are present, give preference to dynamo_str
    if sglang_str is not None and dynamo_str is not None:
        logging.warning(
            f"--dyn-{arg_name} and --{arg_name} are both set. Giving preference to --dyn-{arg_name}"
        )
        return dynamo_str
    # If dynamo_str is not set, use try to use sglang_str if it matches with the allowed parsers
    elif sglang_str is not None:
        logging.warning(f"--dyn-{arg_name} is not set. Using --{arg_name}.")
        if arg_name == "tool-call-parser" and sglang_str not in get_tool_parser_names():
            raise ValueError(
                f"--{arg_name} is not a valid tool call parser. Valid parsers are: {get_tool_parser_names()}"
            )
        elif (
            arg_name == "reasoning-parser"
            and sglang_str not in get_reasoning_parser_names()
        ):
            raise ValueError(
                f"--{arg_name} is not a valid reasoning parser. Valid parsers are: {get_reasoning_parser_names()}"
            )
        return sglang_str
    else:
        return dynamo_str


def parse_args(args: list[str]) -> Config:
    """
    Parse all arguments and return Config with server_args and dynamo_args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version", action="version", version=f"Dynamo Backend SGLang {__version__}"
    )

    # Dynamo args
    for info in DYNAMO_ARGS.values():
        kwargs = {
            "default": info["default"] if "default" in info else None,
            "help": info["help"],
        }
        if "type" in info:
            kwargs["type"] = info["type"]
        if "choices" in info:
            kwargs["choices"] = info["choices"]
        if "action" in info:
            kwargs["action"] = info["action"]

        parser.add_argument(*info["flags"], **kwargs)

    # SGLang args
    bootstrap_port = _reserve_disaggregation_bootstrap_port()
    ServerArgs.add_cli_args(parser)

    parsed_args = parser.parse_args(args)

    # Auto-set bootstrap port if not provided
    if not any(arg.startswith("--disaggregation-bootstrap-port") for arg in args):
        args_dict = vars(parsed_args)
        args_dict["disaggregation_bootstrap_port"] = bootstrap_port
        parsed_args = Namespace(**args_dict)

    # Dynamo argument processing
    # If an endpoint is provided, validate and use it
    # otherwise fall back to default endpoints
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")

    endpoint = parsed_args.endpoint
    if endpoint is None:
        if (
            hasattr(parsed_args, "disaggregation_mode")
            and parsed_args.disaggregation_mode == "prefill"
        ):
            endpoint = f"dyn://{namespace}.prefill.generate"
        elif parsed_args.multimodal_processor:
            endpoint = f"dyn://{namespace}.processor.generate"
        elif parsed_args.multimodal_encode_worker:
            endpoint = f"dyn://{namespace}.encoder.generate"
        elif (
            parsed_args.multimodal_worker
            and parsed_args.disaggregation_mode == "prefill"
        ):
            endpoint = f"dyn://{namespace}.prefill.generate"
        else:
            endpoint = f"dyn://{namespace}.backend.generate"

    # Always parse the endpoint (whether auto-generated or user-provided)
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logging.error(
            f"Invalid endpoint format: '{endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    tool_call_parser = _set_parser(
        parsed_args.tool_call_parser,
        parsed_args.dyn_tool_call_parser,
        "tool-call-parser",
    )
    reasoning_parser = _set_parser(
        parsed_args.reasoning_parser,
        parsed_args.dyn_reasoning_parser,
        "reasoning-parser",
    )

    if parsed_args.custom_jinja_template and parsed_args.use_sglang_tokenizer:
        logging.error(
            "Cannot use --custom-jinja-template and --use-sglang-tokenizer together. "
            "--custom-jinja-template requires Dynamo's preprocessor to apply the template, "
            "while --use-sglang-tokenizer bypasses Dynamo's preprocessor entirely."
            "If you want to use the SGLang tokenizer with a custom chat template, "
            "please use the --chat-template argument from SGLang."
        )
        sys.exit(1)

    # Replaces any environment variables or home dir (~) to get absolute path
    expanded_template_path = None
    if parsed_args.custom_jinja_template:
        expanded_template_path = os.path.expandvars(
            os.path.expanduser(parsed_args.custom_jinja_template)
        )

    dynamo_args = DynamoArgs(
        namespace=parsed_namespace,
        component=parsed_component_name,
        endpoint=parsed_endpoint_name,
        migration_limit=parsed_args.migration_limit,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        custom_jinja_template=expanded_template_path,
        use_sglang_tokenizer=parsed_args.use_sglang_tokenizer,
        multimodal_processor=parsed_args.multimodal_processor,
        multimodal_encode_worker=parsed_args.multimodal_encode_worker,
        multimodal_worker=parsed_args.multimodal_worker,
    )
    logging.debug(f"Dynamo args: {dynamo_args}")

    server_args = ServerArgs.from_cli_args(parsed_args)

    if parsed_args.use_sglang_tokenizer:
        logging.info(
            "Using SGLang's built in tokenizer. Setting skip_tokenizer_init to False"
        )
        server_args.skip_tokenizer_init = False
    else:
        logging.info(
            "Using dynamo's built in tokenizer. Setting skip_tokenizer_init to True"
        )
        server_args.skip_tokenizer_init = True

    return Config(server_args, dynamo_args)


@contextlib.contextmanager
def reserve_free_port(host: str = "localhost"):
    """
    Find and reserve a free port until context exits.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        _, port = sock.getsockname()
        yield port
    finally:
        sock.close()


def parse_endpoint(endpoint: str) -> List[str]:
    """Parse endpoint string into namespace, component, and endpoint parts."""
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        error_msg = (
            f"Invalid endpoint format: '{endpoint}'. "
            f"Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    return endpoint_parts


def _reserve_disaggregation_bootstrap_port():
    """
    Each worker requires a unique port for disaggregation_bootstrap_port.
    We use an existing utility function that reserves a free port on your
    machine to avoid collisions.
    """
    with reserve_free_port() as port:
        return port
