// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::base_parser::BasicReasoningParser;
use crate::ParserResult;
use crate::ReasoningParser;

#[derive(Default, Debug, Clone)]
pub struct DeepseekR1ReasoningParser {
    base: BasicReasoningParser,
}

impl DeepseekR1ReasoningParser {
    pub fn new(vocab: &HashMap<String, u32>) -> Self {
        Self {
            base: BasicReasoningParser::new(
                "<think>".to_string(),
                "</think>".to_string(),
                true,
                true,
                vocab,
            ),
        }
    }
}

impl ReasoningParser for DeepseekR1ReasoningParser {
    fn parse_reasoning_streaming_incremental(&mut self, token_ids: &[u32]) -> ParserResult {
        self.base.parse_reasoning_streaming_incremental(token_ids)
    }

    fn detect_and_parse_reasoning(&self, token_ids: &[u32]) -> ParserResult {
        self.base.detect_and_parse_reasoning(token_ids)
    }
}
