// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Deref;

use crate::ParserResult;
use crate::ReasoningParser;

use openai_harmony::chat::TextContent;
use openai_harmony::StreamableParser;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName, HarmonyEncoding, chat::Role};

#[derive(Debug)]
pub struct GptOssReasoningParser {
    enc: HarmonyEncoding,
}

impl GptOssReasoningParser {
    pub fn new() -> Self {
        let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        Self { enc }
    }
}

impl GptOssReasoningParser {
    fn reason_parsing_wrapper(&self, token_ids: &[u32]) -> ParserResult {
        let mut parser = StreamableParser::new(self.enc.clone(), Some(Role::Assistant)).unwrap();
        for token_id in token_ids {
            parser.process(*token_id).unwrap();
        }
        let output_msgs = parser.messages();
        // let mut reasoning_token_ids = vec![];
        // let mut normal_token_ids = vec![];
        match output_msgs.len() {
            0 => {
                return ParserResult {
                    normal_token_ids: vec![], // No normal text in this example
                    reasoning_token_ids: self.enc.tokenizer().encode_with_special_tokens(parser.current_content().unwrap().deref()),
                }
            },
            1 => {
                let mut reasoning_token_ids = vec![];
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) = output_msgs[0].content.first() {
                    reasoning_token_ids.extend(self.enc.tokenizer().encode_with_special_tokens(text));
                }
                return ParserResult {
                    normal_token_ids: self.enc.tokenizer().encode_with_special_tokens(parser.current_content().unwrap().deref()),
                    reasoning_token_ids: reasoning_token_ids,
                };
            },
            _ => {
                let mut reasoning_token_ids = vec![];
                let mut normal_token_ids = vec![];

                // Loop until second last message
                for i in 0..(output_msgs.len() - 1) {
                    let parse_msg = &output_msgs[i]; 
                    if let Some(openai_harmony::chat::Content::Text(TextContent { text })) = parse_msg.content.first() {
                        reasoning_token_ids.extend(self.enc.tokenizer().encode_with_special_tokens(text));
                    }
                }

                let last_msg = &output_msgs[output_msgs.len() - 1];

                // Handle the last message
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) = last_msg.content.first() {
                    normal_token_ids.extend(self.enc.tokenizer().encode_with_special_tokens(text));
                }

                return ParserResult {
                    normal_token_ids,
                    reasoning_token_ids,
                };
            }
        }
    }
}

impl ReasoningParser for  GptOssReasoningParser {

    fn detect_and_parse_reasoning(&self, token_ids: &[u32]) -> ParserResult {
        self.reason_parsing_wrapper(token_ids)
    }

    fn parse_reasoning_streaming_incremental(&mut self, token_ids: &[u32]) -> ParserResult {
        self.reason_parsing_wrapper(token_ids)
    }
    
}