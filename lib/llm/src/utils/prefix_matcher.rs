// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Efficient multi-pattern marker detection with partial suffix matching
//!
//! This module provides utilities for detecting complete and partial marker patterns
//! in streaming text, with support for detecting markers split across chunk boundaries.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use std::collections::HashMap;

/// Result of processing a chunk with potential marker detection
#[derive(Debug, Clone, PartialEq)]
pub enum MatchResult {
    /// Complete marker found
    Complete {
        /// Content before the marker (safe to emit)
        prefix: String,
        /// The complete marker matched
        marker: String,
        /// Start position of the marker in the input
        marker_start: usize,
        /// Remaining content after the marker
        suffix: String,
    },
    /// Partial marker at end of chunk
    Partial {
        /// Content before the partial (safe to emit)
        prefix: String,
        /// The partial match to hold
        partial: String,
        /// Which patterns this could match
        possible_patterns: Vec<String>,
    },
    /// No markers detected
    None {
        /// All content is safe to emit
        content: String,
    },
}

/// Efficient multi-pattern matcher with partial suffix detection
pub struct MarkerMatcher {
    /// All patterns we're looking for
    patterns: Vec<String>,
    /// Aho-Corasick matcher for complete patterns
    complete_matcher: AhoCorasick,
    /// Trie for partial matching
    prefix_trie: PrefixTrie,
    /// Maximum pattern length (for buffer limits)
    max_pattern_len: usize,
}

impl MarkerMatcher {
    /// Create a new matcher with the given patterns
    pub fn new(patterns: Vec<String>) -> Result<Self, String> {
        if patterns.is_empty() {
            return Err("Cannot create MarkerMatcher with empty patterns".to_string());
        }

        let complete_matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostFirst)
            .build(&patterns)
            .map_err(|e| format!("Failed to build Aho-Corasick matcher: {}", e))?;

        let max_pattern_len = patterns.iter().map(|p| p.len()).max().unwrap_or(0);
        let prefix_trie = PrefixTrie::new(&patterns);

        Ok(Self {
            patterns,
            complete_matcher,
            prefix_trie,
            max_pattern_len,
        })
    }

    /// Get the maximum pattern length
    pub fn max_pattern_len(&self) -> usize {
        self.max_pattern_len
    }

    /// Process a chunk with an optional partial buffer from previous chunk
    pub fn process_chunk(&self, chunk: &str, partial_buffer: &str) -> MatchResult {
        // Combine buffer with new chunk
        let combined = if partial_buffer.is_empty() {
            chunk.to_string()
        } else {
            format!("{}{}", partial_buffer, chunk)
        };

        // First check for complete markers
        if let Some(mat) = self.complete_matcher.find(&combined) {
            let marker = &self.patterns[mat.pattern().as_usize()];
            return MatchResult::Complete {
                prefix: combined[..mat.start()].to_string(),
                marker: marker.clone(),
                marker_start: mat.start(),
                suffix: combined[mat.end()..].to_string(),
            };
        }

        // No complete match - check for partial at ANY suffix position
        // This is the key: check "n<T" → finds "<T" as partial
        if let Some((partial_start, partial, patterns)) = self.find_partial_suffix(&combined) {
            return MatchResult::Partial {
                prefix: combined[..partial_start].to_string(),
                partial: partial.to_string(),
                possible_patterns: patterns,
            };
        }

        // No matches at all
        MatchResult::None { content: combined }
    }

    /// Find the longest partial match in any suffix of the input
    ///
    /// This scans from left to right to find the EARLIEST partial match,
    /// ensuring we emit as much content as possible while holding only the minimal partial.
    fn find_partial_suffix<'a>(&self, text: &'a str) -> Option<(usize, &'a str, Vec<String>)> {
        // Start from the beginning to find the EARLIEST partial match
        // This ensures we emit as much as possible
        for i in 0..text.len() {
            let suffix = &text[i..];
            if let Some(patterns) = self.prefix_trie.find_prefix_match(suffix) {
                // This suffix is a prefix of one or more patterns
                return Some((i, suffix, patterns));
            }
        }
        None
    }
}

/// Trie structure for efficient prefix matching
struct PrefixTrie {
    root: TrieNode,
}

#[derive(Debug)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    /// Patterns that have this exact prefix
    matching_patterns: Vec<String>,
    /// Is this node a complete pattern?
    is_complete: bool,
}

impl PrefixTrie {
    fn new(patterns: &[String]) -> Self {
        let mut root = TrieNode {
            children: HashMap::new(),
            matching_patterns: Vec::new(),
            is_complete: false,
        };

        // Build trie
        for pattern in patterns {
            let mut current = &mut root;
            let chars: Vec<char> = pattern.chars().collect();

            for (i, &ch) in chars.iter().enumerate() {
                current = current.children.entry(ch).or_insert(TrieNode {
                    children: HashMap::new(),
                    matching_patterns: Vec::new(),
                    is_complete: false,
                });

                // Add this pattern to all prefix nodes
                if !current.matching_patterns.contains(pattern) {
                    current.matching_patterns.push(pattern.clone());
                }

                // Mark complete if we're at the end
                if i == chars.len() - 1 {
                    current.is_complete = true;
                }
            }
        }

        PrefixTrie { root }
    }

    /// Check if text is a prefix of any pattern (but not a complete pattern)
    fn find_prefix_match(&self, text: &str) -> Option<Vec<String>> {
        let mut current = &self.root;

        for ch in text.chars() {
            if let Some(node) = current.children.get(&ch) {
                current = node;
            } else {
                // Not a prefix of any pattern
                return None;
            }
        }

        // If we matched the entire text and it's a prefix of something (but not complete)
        if !current.matching_patterns.is_empty() && !current.is_complete {
            Some(current.matching_patterns.clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_match() {
        let patterns = vec!["<TOOLCALL>".to_string(), "<tool_call>".to_string()];
        let matcher = MarkerMatcher::new(patterns).unwrap();

        let result = matcher.process_chunk("<TOOLCALL>data", "");

        if let MatchResult::Complete {
            prefix,
            marker,
            suffix,
            ..
        } = result
        {
            assert_eq!(prefix, "");
            assert_eq!(marker, "<TOOLCALL>");
            assert_eq!(suffix, "data");
        } else {
            panic!("Expected complete match");
        }
    }

    #[test]
    fn test_partial_match_suffix() {
        let patterns = vec!["<TOOLCALL>".to_string()];
        let matcher = MarkerMatcher::new(patterns).unwrap();

        // Test the key case: "n<T" should detect "<T" as partial
        let result = matcher.process_chunk("n<T", "");

        if let MatchResult::Partial {
            prefix,
            partial,
            possible_patterns,
        } = result
        {
            assert_eq!(prefix, "n");
            assert_eq!(partial, "<T");
            assert_eq!(possible_patterns, vec!["<TOOLCALL>"]);
        } else {
            panic!("Expected partial match, got: {:?}", result);
        }
    }

    #[test]
    fn test_no_false_positive() {
        let patterns = vec!["<TOOLCALL>".to_string()];
        let matcher = MarkerMatcher::new(patterns).unwrap();

        // Test case: "n < 5" should not trigger partial match
        let result = matcher.process_chunk("n < 5", "");

        if let MatchResult::None { content } = result {
            assert_eq!(content, "n < 5");
        } else {
            panic!("Expected no match, got: {:?}", result);
        }
    }

    #[test]
    fn test_partial_buffer_combination() {
        let patterns = vec!["<TOOLCALL>".to_string()];
        let matcher = MarkerMatcher::new(patterns).unwrap();

        // First chunk: partial "<"
        let result1 = matcher.process_chunk("<", "");
        let partial = if let MatchResult::Partial { partial, .. } = result1 {
            partial
        } else {
            panic!("Expected partial match");
        };

        // Second chunk: "TOOLCALL>" completes the pattern
        let result2 = matcher.process_chunk("TOOLCALL>", &partial);

        if let MatchResult::Complete { marker, .. } = result2 {
            assert_eq!(marker, "<TOOLCALL>");
        } else {
            panic!("Expected complete match, got: {:?}", result2);
        }
    }

    #[test]
    fn test_prefix_with_content() {
        let patterns = vec!["<TOOLCALL>".to_string()];
        let matcher = MarkerMatcher::new(patterns).unwrap();

        let result = matcher.process_chunk("text before <TOOLCALL> after", "");

        if let MatchResult::Complete {
            prefix,
            marker,
            suffix,
            ..
        } = result
        {
            assert_eq!(prefix, "text before ");
            assert_eq!(marker, "<TOOLCALL>");
            assert_eq!(suffix, " after");
        } else {
            panic!("Expected complete match");
        }
    }

    #[test]
    fn test_empty_patterns() {
        let result = MarkerMatcher::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_patterns() {
        let patterns = vec![
            "<TOOLCALL>".to_string(),
            "[TOOL_CALLS]".to_string(),
            "<tool_call>".to_string(),
        ];
        let matcher = MarkerMatcher::new(patterns).unwrap();

        // Test different patterns
        let result1 = matcher.process_chunk("[TOOL_CALLS]", "");
        if let MatchResult::Complete { marker, .. } = result1 {
            assert_eq!(marker, "[TOOL_CALLS]");
        } else {
            panic!("Expected complete match for [TOOL_CALLS]");
        }

        // Test partial for different pattern
        let result2 = matcher.process_chunk("text<to", "");
        if let MatchResult::Partial {
            partial,
            possible_patterns,
            ..
        } = result2
        {
            assert_eq!(partial, "<to");
            assert!(possible_patterns.contains(&"<tool_call>".to_string()));
        } else {
            panic!("Expected partial match for <tool_call>");
        }
    }
}
