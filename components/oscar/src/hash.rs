// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use blake3::Hash;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Content-addressable hash for objects using BLAKE3.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Create a new hash from bytes.
    pub fn new(hash: Hash) -> Self {
        Self(*hash.as_bytes())
    }

    /// Get the hash as bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hex string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Result<Self, hex::FromHexError> {
        let bytes = hex::decode(hex)?;
        if bytes.len() != 32 {
            return Err(hex::FromHexError::InvalidStringLength);
        }
        let mut array = [0u8; 32];
        array.copy_from_slice(&bytes);
        Ok(Self(array))
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl Serialize for ContentHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let hex_str = String::deserialize(deserializer)?;
        Self::from_hex(&hex_str).map_err(serde::de::Error::custom)
    }
}

/// BLAKE3 hasher for objects.
pub struct ObjectHasher;

impl ObjectHasher {
    /// Hash the given data.
    pub fn hash(data: &[u8]) -> ContentHash {
        ContentHash::new(blake3::hash(data))
    }

    /// Hash data from multiple chunks.
    pub fn hash_chunks<I>(chunks: I) -> ContentHash
    where
        I: IntoIterator<Item = bytes::Bytes>,
    {
        let mut hasher = blake3::Hasher::new();
        for chunk in chunks {
            hasher.update(&chunk);
        }
        ContentHash::new(hasher.finalize())
    }
}