# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Tuple

# import from __init__.py in the same directory
from __init__ import BaseReasoningParser


class BasicReasoningParser(BaseReasoningParser):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
    ):
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self._in_reasoning = False
        self.stream_reasoning = True

        self._buffer = ""
        self.stripped_think_start = False

    def detect_and_parse_reasoning(
        self, text: str, _token_ids: Sequence[int]
    ) -> Tuple[str, str]:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return (text, "")

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # Assume reasoning was truncated before `</think>` token
            return ("", processed_text)

        # Extract reasoning content
        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0]
        normal_text = splits[1].strip()

        return (normal_text, reasoning_text)

    def parse_reasoning_streaming_incremental(
        self, new_text: str, _token_ids: Sequence[int]
    ) -> Tuple[str, str]:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        if any(
            token.startswith(current_text) and token != current_text
            for token in [self.think_start_token, self.think_end_token]
        ):
            return ("", "")

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return (normal_text, reasoning_text.rstrip())

        # Continue with reasoning content
        if self._in_reasoning:
            if self.stream_reasoning:
                self._buffer = ""
                return ("", current_text)
            else:
                return ("", "")

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return (current_text, "")

        return ("", "")
