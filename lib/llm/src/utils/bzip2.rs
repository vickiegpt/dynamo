// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bzip2 decompression utilities for handling compressed files in temporary directories.
//!
//! This module provides a clean API for decompressing bzip2 files to temporary directories
//! with automatic cleanup through RAII. The primary use case is for compressed files stored
//! in git-lfs that need to be decompressed for processing by code expecting file paths.
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use dynamo_llm::utils::bzip2::Bzip2Extractor;
//!
//! // Simple extraction with default filename
//! let extraction = Bzip2Extractor::builder()
//!     .source_path("data.json.bz2")
//!     .extract()?;
//!
//! let file_path = extraction.file_path();
//! process_file(file_path)?;
//! // Automatic cleanup when `extraction` goes out of scope
//!
//! // Custom target filename
//! let extraction = Bzip2Extractor::builder()
//!     .source_path("foo-x-y-z.json.bz2")
//!     .target_filename("foo.json")
//!     .extract()?;
//! ```

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Builder for configuring bzip2 extraction operations.
///
/// This struct uses the Builder pattern to configure extraction options.
/// The `extract()` method performs the actual decompression operation.
#[derive(Debug, Clone, Default)]
pub struct Bzip2ExtractorBuilder {
    /// Path to the source bzip2 compressed file
    source_path: Option<PathBuf>,
    /// Optional custom filename for the extracted file in temp directory
    target_filename: Option<String>,
}

/// Configuration for bzip2 extraction operations.
#[derive(Debug, Clone)]
pub struct Bzip2Extractor {
    /// Path to the source bzip2 compressed file
    source_path: PathBuf,
    /// Optional custom filename for the extracted file in temp directory
    /// If None, derives name from source_path by removing .bz2 extension
    target_filename: Option<String>,
}

/// Result of a successful bzip2 extraction operation.
///
/// This struct provides RAII cleanup of the temporary directory and
/// access to the extracted file path. The temporary directory and its
/// contents are automatically removed when this struct is dropped.
#[derive(Debug)]
pub struct Bzip2Extraction {
    /// RAII temporary directory - kept private to prevent misuse
    _temp_dir: TempDir,
    /// Path to the extracted file within the temporary directory
    file_path: PathBuf,
}

impl Bzip2ExtractorBuilder {
    /// Set the source path for the bzip2 file to extract.
    pub fn source_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.source_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set a custom target filename for the extracted file.
    pub fn target_filename<S: Into<String>>(mut self, filename: S) -> Self {
        self.target_filename = Some(filename.into());
        self
    }

    /// Build the extractor configuration and immediately perform the extraction.
    ///
    /// This method validates the configuration, performs the bzip2 decompression,
    /// and returns a `Bzip2Extraction` handle for accessing the extracted file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Source file doesn't exist or can't be read
    /// - Source file is not valid bzip2 format
    /// - Target filename is invalid
    /// - Temporary directory creation fails
    /// - Decompression operation fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let extraction = Bzip2Extractor::builder()
    ///     .source_path("data.json.bz2")
    ///     .target_filename("extracted.json")
    ///     .extract()?;
    /// ```
    pub fn extract(self) -> Result<Bzip2Extraction> {
        let source_path = self
            .source_path
            .ok_or_else(|| anyhow::anyhow!("source_path is required"))?;

        let extractor = Bzip2Extractor {
            source_path,
            target_filename: self.target_filename,
        };

        extractor.perform_extraction()
    }
}

impl Bzip2Extractor {
    /// Create a new builder for configuring bzip2 extraction.
    pub fn builder() -> Bzip2ExtractorBuilder {
        Bzip2ExtractorBuilder::default()
    }

    /// Perform the actual bzip2 extraction operation.
    ///
    /// This method creates a temporary directory, decompresses the source file,
    /// and returns a handle to the extracted file with RAII cleanup.
    fn perform_extraction(self) -> Result<Bzip2Extraction> {
        // Validate source file exists
        if !self.source_path.exists() {
            anyhow::bail!("Source file does not exist: {}", self.source_path.display());
        }

        // Create temporary directory
        let temp_dir = tempfile::tempdir().context("Failed to create temporary directory")?;

        // Determine target filename
        let target_filename = match &self.target_filename {
            Some(filename) => filename.clone(),
            None => {
                // Derive filename from source by removing .bz2 extension
                let source_filename = self
                    .source_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| anyhow::anyhow!("Invalid source filename"))?;

                if source_filename.ends_with(".bz2") {
                    source_filename.strip_suffix(".bz2").unwrap().to_string()
                } else {
                    format!("{}.extracted", source_filename)
                }
            }
        };

        let target_path = temp_dir.path().join(&target_filename);

        // Open source file and create decoder
        let source_file = File::open(&self.source_path).with_context(|| {
            format!("Failed to open source file: {}", self.source_path.display())
        })?;

        let mut decoder = BzDecoder::new(source_file);

        // Create target file and decompress
        let mut target_file = File::create(&target_path)
            .with_context(|| format!("Failed to create target file: {}", target_path.display()))?;

        std::io::copy(&mut decoder, &mut target_file).with_context(|| {
            format!(
                "Failed to decompress {} to {}",
                self.source_path.display(),
                target_path.display()
            )
        })?;

        target_file.flush().context("Failed to flush target file")?;

        Ok(Bzip2Extraction {
            _temp_dir: temp_dir,
            file_path: target_path,
        })
    }
}

impl Bzip2Extraction {
    /// Get the path to the extracted file.
    ///
    /// The file exists within a temporary directory that will be cleaned up
    /// when this `Bzip2Extraction` instance is dropped.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Get the path to the temporary directory containing the extracted file.
    ///
    /// The directory will be cleaned up when this `Bzip2Extraction` instance is dropped.
    pub fn temp_dir_path(&self) -> &Path {
        self._temp_dir.path()
    }

    /// Read the entire extracted file contents as a UTF-8 string.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The file contents are not valid UTF-8
    pub fn read_to_string(&self) -> Result<String> {
        std::fs::read_to_string(&self.file_path).with_context(|| {
            format!(
                "Failed to read file as string: {}",
                self.file_path.display()
            )
        })
    }

    /// Read the entire extracted file contents as raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub fn read_to_bytes(&self) -> Result<Vec<u8>> {
        std::fs::read(&self.file_path)
            .with_context(|| format!("Failed to read file as bytes: {}", self.file_path.display()))
    }

    /// Validate the extracted file against a BLAKE3 hash.
    ///
    /// # Arguments
    ///
    /// * `expected_hash` - The expected BLAKE3 hash as a hexadecimal string
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The computed hash doesn't match the expected hash
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let extraction = Bzip2Extractor::builder()
    ///     .source_path("data.json.bz2")
    ///     .extract()?;
    ///
    /// extraction.validate_blake3_hash("c61f943c9f3266a60a7e00e815591061f17564f297dd84433a101fb43eb15608")?;
    /// ```
    pub fn validate_blake3_hash(&self, expected_hash: &str) -> Result<()> {
        let file_contents = self.read_to_bytes()?;
        let computed_hash = blake3::hash(&file_contents);
        let computed_hash_hex = computed_hash.to_hex();

        if computed_hash_hex.as_str() == expected_hash {
            Ok(())
        } else {
            anyhow::bail!(
                "BLAKE3 hash mismatch for file {}: expected {}, got {}",
                self.file_path.display(),
                expected_hash,
                computed_hash_hex
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::write::BzEncoder;
    use bzip2::Compression;
    use std::io::Write;

    /// Create a test bzip2 file with known content for testing
    fn create_test_bzip2_file(content: &str) -> Result<tempfile::NamedTempFile> {
        let mut temp_file = tempfile::NamedTempFile::with_suffix(".bz2")?;

        // Compress the test data
        let mut encoder = BzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(content.as_bytes())?;
        let compressed_data = encoder.finish()?;

        // Write compressed data to temp file
        temp_file.write_all(&compressed_data)?;
        temp_file.flush()?;

        Ok(temp_file)
    }

    #[test]
    fn test_basic_extraction() -> Result<()> {
        let test_content = "Hello, World!\nThis is a test file for bzip2 compression.\n";
        let temp_file = create_test_bzip2_file(test_content)?;

        let extraction = Bzip2Extractor::builder()
            .source_path(temp_file.path())
            .extract()?;

        // Verify the extracted file exists and has correct content
        assert!(extraction.file_path().exists());
        let content = extraction.read_to_string()?;
        assert_eq!(content, test_content);

        Ok(())
    }

    #[test]
    fn test_custom_target_filename() -> Result<()> {
        let test_content = "Custom filename test content";
        let temp_file = create_test_bzip2_file(test_content)?;

        let extraction = Bzip2Extractor::builder()
            .source_path(temp_file.path())
            .target_filename("custom.txt")
            .extract()?;

        // Verify the file has the custom name
        assert_eq!(extraction.file_path().file_name().unwrap(), "custom.txt");
        let content = extraction.read_to_string()?;
        assert_eq!(content, test_content);

        Ok(())
    }

    #[test]
    fn test_automatic_filename_derivation() -> Result<()> {
        let test_content = "Automatic filename test";

        // Create a temp file with a .bz2 extension
        let mut temp_file = tempfile::NamedTempFile::with_suffix(".json.bz2")?;
        let mut encoder = BzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(test_content.as_bytes())?;
        let compressed_data = encoder.finish()?;
        temp_file.write_all(&compressed_data)?;
        temp_file.flush()?;

        let extraction = Bzip2Extractor::builder()
            .source_path(temp_file.path())
            .extract()?;

        // Should derive filename by removing .bz2 extension
        let expected_filename = temp_file
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_suffix(".bz2")
            .unwrap();

        assert_eq!(
            extraction
                .file_path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            expected_filename
        );
        let content = extraction.read_to_string()?;
        assert_eq!(content, test_content);

        Ok(())
    }

    #[test]
    fn test_missing_source_path_error() {
        let result = Bzip2Extractor::builder().extract();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("source_path is required"));
    }

    #[test]
    fn test_nonexistent_file_error() {
        let result = Bzip2Extractor::builder()
            .source_path("/nonexistent/file.bz2")
            .extract();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_raii_cleanup() -> Result<()> {
        let test_content = "RAII cleanup test";
        let temp_file = create_test_bzip2_file(test_content)?;
        let temp_dir_path;

        {
            let extraction = Bzip2Extractor::builder()
                .source_path(temp_file.path())
                .extract()?;

            temp_dir_path = extraction.temp_dir_path().to_path_buf();
            assert!(temp_dir_path.exists());
            assert!(extraction.file_path().exists());
        } // extraction goes out of scope here

        // Temporary directory should be cleaned up
        assert!(!temp_dir_path.exists());

        Ok(())
    }

    #[test]
    fn test_builder_validation() {
        // Test that builder validates required fields
        let result = Bzip2Extractor::builder().extract();
        assert!(result.is_err());

        // Test that builder accepts valid configuration
        let temp_file = tempfile::NamedTempFile::with_suffix(".bz2").unwrap();
        let result = Bzip2Extractor::builder()
            .source_path(temp_file.path())
            .target_filename("test.txt")
            .extract();

        // This should fail because the file is empty/invalid bzip2, but not due to validation
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_filename_patterns() -> Result<()> {
        let test_content = "Complex filename test";

        // Test various filename patterns
        let test_cases = vec![
            ("data.json.bz2", "data.json"),
            ("file.txt.bz2", "file.txt"),
            ("archive.tar.bz2", "archive.tar"),
            ("simple.bz2", "simple"),
            ("no-extension", "no-extension.extracted"),
        ];

        for (input_filename, expected_output) in test_cases {
            // Create temp directory and file with exact name we want to test
            let temp_dir = tempfile::tempdir()?;
            let temp_file_path = temp_dir.path().join(input_filename);

            // Create compressed file with the exact name
            let mut encoder = BzEncoder::new(Vec::new(), Compression::best());
            encoder.write_all(test_content.as_bytes())?;
            let compressed_data = encoder.finish()?;

            std::fs::write(&temp_file_path, &compressed_data)?;

            let extraction = Bzip2Extractor::builder()
                .source_path(&temp_file_path)
                .extract()?;

            assert_eq!(
                extraction
                    .file_path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap(),
                expected_output,
                "Failed for input: {}",
                input_filename
            );
        }

        Ok(())
    }

    #[test]
    fn test_validate_specific_tokenizer_file() -> Result<()> {
        // This test validates the specific file mentioned in the requirements
        let tokenizer_file_path = "dynamo/lib/llm/tests/data/replays/deepseek-r1-distill-llama-8b/tokenizer-deepseek-r1-distill-llama-8b.json.bz2";
        let expected_hash = "c61f943c9f3266a60a7e00e815591061f17564f297dd84433a101fb43eb15608";

        // Check if the test file exists first
        if !std::path::Path::new(tokenizer_file_path).exists() {
            println!(
                "Test file {} does not exist, skipping validation test",
                tokenizer_file_path
            );
            return Ok(());
        }

        let extraction = Bzip2Extractor::builder()
            .source_path(tokenizer_file_path)
            .target_filename("tokenizer.json")
            .extract()?;

        // Verify the file was extracted with the correct name
        assert_eq!(
            extraction
                .file_path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            "tokenizer.json"
        );

        // Validate against the expected BLAKE3 hash
        extraction.validate_blake3_hash(expected_hash)?;

        println!(
            "Successfully validated tokenizer file against BLAKE3 hash: {}",
            expected_hash
        );

        Ok(())
    }
}
