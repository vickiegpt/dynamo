// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod base_parser;
mod deepseek_r1_parser;

use std::collections::HashMap;

// Re-export main types and functions for convenience
pub use base_parser::BasicReasoningParser;
pub use deepseek_r1_parser::DeepseekR1ReasoningParser;

#[derive(Debug, Clone, Default)]
pub struct ParserResult {
    /// The normal text outside of reasoning blocks.
    pub normal_token_ids: Vec<u32>,
    /// The extracted reasoning text from within reasoning blocks.
    pub reasoning_token_ids: Vec<u32>,
}

// impl ParserResult {
//     pub fn get_some_reasoning(&self) -> Option<Vec<u32>> {
//         if self.reasoning_token_ids.is_empty() {
//             None
//         } else {
//             Some(self.reasoning_token_ids.clone())
//         }
//     }

//     pub fn get_some_normal_text(&self) -> Option<Vec<u32>> {
//         if self.normal_token_ids.is_empty() {
//             None
//         } else {
//             Some(self.normal_token_ids.clone())
//         }
//     }
// }

pub trait ReasoningParser: Send + std::fmt::Debug {
    /// Parses a standalone, non-streaming input chunk. Implementations may reset or ignore
    /// internal streaming state and should return the split of normal vs reasoning text for
    /// this complete input. Marker tokens must not be included in either output.
    fn detect_and_parse_reasoning(&self, token_ids: &[u32]) -> ParserResult;

    /// Parses a streaming chunk and updates internal state. The return value should be the
    /// delta: only the newly discovered normal and reasoning text attributable to this chunk
    /// (not the cumulative totals). Marker tokens must not be included in either output.
    fn parse_reasoning_streaming_incremental(&mut self, token_ids: &[u32]) -> ParserResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningParserType {
    DeepseekR1,
    Basic,
}

#[derive(std::fmt::Debug)]
pub struct ReasoningParserWrapper {
    parser: Box<dyn ReasoningParser>,
}

impl ReasoningParser for ReasoningParserWrapper {
    fn detect_and_parse_reasoning(&self, token_ids: &[u32]) -> ParserResult {
        self.parser.detect_and_parse_reasoning(token_ids)
    }

    fn parse_reasoning_streaming_incremental(&mut self, token_ids: &[u32]) -> ParserResult {
        self.parser.parse_reasoning_streaming_incremental(token_ids)
    }
}

impl ReasoningParserType {
    pub fn get_reasoning_parser(self, vocab: &HashMap<String, u32>) -> ReasoningParserWrapper {
        match self {
            ReasoningParserType::DeepseekR1 => ReasoningParserWrapper {
                parser: Box::new(DeepseekR1ReasoningParser::new(vocab)),
            },
            ReasoningParserType::Basic => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "<think>".into(),
                    "</think>".into(),
                    false,
                    true,
                    vocab,
                )),
            },
        }
    }
}
