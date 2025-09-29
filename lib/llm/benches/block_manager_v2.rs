// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block Manager V2 Performance Benchmarks
//!
//! Benchmarks for performance-sensitive operations in the block manager v2:
//! - Block registration and deregistration
//! - Block lookup and matching
//! - Drop implementations and cleanup
//! - Block allocation and reuse

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use dynamo_llm::block_manager::v2::manager::BlockManager;
use dynamo_llm::tokens::{SequenceHash, TokenBlockSequence};

/// Test metadata for benchmarks
#[derive(Debug, Clone, PartialEq)]
struct BenchData {
    value: u64,
}

/// Create a token block for benchmarking
fn create_bench_token_block(start: u32, size: usize) -> dynamo_llm::tokens::TokenBlock {
    let tokens: Vec<u32> = (start..start + size as u32).collect();
    let token_sequence = TokenBlockSequence::from_slice(&tokens, size as u32, Some(42));
    token_sequence
        .blocks()
        .first()
        .cloned()
        .expect("Should have at least one block for the given token sequence")
}

/// Setup a manager for benchmarking
fn create_bench_manager(block_count: usize, block_size: usize) -> BlockManager<BenchData> {
    BlockManager::<BenchData>::builder()
        .block_count(block_count)
        .block_size(block_size)
        .with_lru_backend()
        .build()
        .expect("Should build manager")
}

/// Generate sequence hashes for lookup benchmarks
fn generate_sequence_hashes(count: usize, block_size: usize) -> Vec<SequenceHash> {
    (0..count)
        .map(|i| create_bench_token_block(i as u32 * 100, block_size).sequence_hash())
        .collect()
}

// =============================================================================
// REGISTRATION BENCHMARKS
// =============================================================================

fn registration_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("registration");

    // Single block registration
    group.bench_function("register_single_block", |b| {
        b.iter_batched(
            || {
                let manager = create_bench_manager(1000, 4);
                let token_block = create_bench_token_block(100, 4);
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete");
                (manager, complete_block)
            },
            |(manager, complete_block)| {
                black_box(manager.register_blocks(vec![complete_block]));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Bulk registration
    for size in &[10, 100, 1000] {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("register_bulk_blocks", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let manager = create_bench_manager(size + 100, 4);
                        let mut complete_blocks = Vec::new();

                        for i in 0..size {
                            let token_block = create_bench_token_block(i as u32 * 10, 4);
                            let mutable_blocks =
                                manager.allocate_blocks(1).expect("Should allocate");
                            let complete_block = mutable_blocks
                                .into_iter()
                                .next()
                                .unwrap()
                                .complete(token_block)
                                .expect("Should complete");
                            complete_blocks.push(complete_block);
                        }

                        (manager, complete_blocks)
                    },
                    |(manager, complete_blocks)| {
                        black_box(manager.register_blocks(complete_blocks));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // Registration with duplicates (deduplication overhead)
    group.bench_function("register_with_duplicates", |b| {
        b.iter_batched(
            || {
                let manager = create_bench_manager(100, 4);
                let token_block = create_bench_token_block(500, 4); // Same block
                let mut complete_blocks = Vec::new();

                // Create 10 identical blocks
                for _ in 0..10 {
                    let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
                    let complete_block = mutable_blocks
                        .into_iter()
                        .next()
                        .unwrap()
                        .complete(token_block.clone())
                        .expect("Should complete");
                    complete_blocks.push(complete_block);
                }

                (manager, complete_blocks)
            },
            |(manager, complete_blocks)| {
                black_box(manager.register_blocks(complete_blocks));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// LOOKUP/MATCHING BENCHMARKS
// =============================================================================

fn lookup_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup");

    // Single block lookup
    group.bench_function("find_single_block", |b| {
        b.iter_batched(
            || {
                let manager = create_bench_manager(1000, 4);
                let token_block = create_bench_token_block(100, 4);
                let seq_hash = token_block.sequence_hash();

                // Register the block
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete");
                manager.register_blocks(vec![complete_block]);

                (manager, vec![seq_hash])
            },
            |(manager, hashes)| {
                black_box(manager.match_blocks(&hashes));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Multiple block lookup
    for size in &[10, 100, 1000] {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("find_multiple_blocks", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let manager = create_bench_manager(size + 100, 4);
                        let mut hashes = Vec::new();
                        let mut complete_blocks = Vec::new();

                        // Create and register blocks
                        for i in 0..size {
                            let token_block = create_bench_token_block(i as u32 * 10, 4);
                            hashes.push(token_block.sequence_hash());

                            let mutable_blocks =
                                manager.allocate_blocks(1).expect("Should allocate");
                            let complete_block = mutable_blocks
                                .into_iter()
                                .next()
                                .unwrap()
                                .complete(token_block)
                                .expect("Should complete");
                            complete_blocks.push(complete_block);
                        }

                        manager.register_blocks(complete_blocks);
                        (manager, hashes)
                    },
                    |(manager, hashes)| {
                        black_box(manager.match_blocks(&hashes));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // Lookup miss (blocks not found)
    group.bench_function("find_miss", |b| {
        b.iter_batched(
            || {
                let manager = create_bench_manager(100, 4);
                let nonexistent_hashes = generate_sequence_hashes(10, 4);
                (manager, nonexistent_hashes)
            },
            |(manager, hashes)| {
                black_box(manager.match_blocks(&hashes));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Partial match (stops on first miss)
    group.bench_function("find_partial_match", |b| {
        b.iter_batched(
            || {
                let manager = create_bench_manager(100, 4);

                // Register first 5 blocks
                let mut complete_blocks = Vec::new();
                let mut hashes = Vec::new();

                for i in 0..5 {
                    let token_block = create_bench_token_block(i as u32 * 10, 4);
                    hashes.push(token_block.sequence_hash());

                    let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
                    let complete_block = mutable_blocks
                        .into_iter()
                        .next()
                        .unwrap()
                        .complete(token_block)
                        .expect("Should complete");
                    complete_blocks.push(complete_block);
                }

                manager.register_blocks(complete_blocks);

                // Add nonexistent hash in the middle to trigger partial match
                hashes.insert(2, 99999); // This hash won't exist
                (manager, hashes)
            },
            |(manager, hashes)| {
                black_box(manager.match_blocks(&hashes));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// DROP/CLEANUP BENCHMARKS
// =============================================================================

fn drop_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("drop");

    // MutableBlock drop (return to ResetPool)
    group.bench_function("mutable_block_drop", |b| {
        b.iter_batched(
            || {
                let manager = create_bench_manager(100, 4);
                manager.allocate_blocks(10).expect("Should allocate")
            },
            |mutable_blocks| {
                // Drop all blocks, measuring cleanup time
                black_box(drop(mutable_blocks));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // RegisteredBlock drop (return to InactivePool)
    group.bench_function("registered_block_drop", |b| {
        b.iter_batched(
            || {
                let manager = create_bench_manager(100, 4);
                let mut complete_blocks = Vec::new();

                for i in 0..10 {
                    let token_block = create_bench_token_block(i as u32 * 10, 4);
                    let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
                    let complete_block = mutable_blocks
                        .into_iter()
                        .next()
                        .unwrap()
                        .complete(token_block)
                        .expect("Should complete");
                    complete_blocks.push(complete_block);
                }

                manager.register_blocks(complete_blocks)
            },
            |registered_blocks| {
                // Drop all registered blocks, measuring cleanup time
                black_box(drop(registered_blocks));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// ALLOCATION BENCHMARKS
// =============================================================================

fn allocation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation");

    // Allocate from ResetPool
    for size in &[1, 10, 100] {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("allocate_from_reset", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || create_bench_manager(size + 10, 4),
                    |manager| {
                        black_box(manager.allocate_blocks(size));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // Allocate from InactivePool (reuse)
    for size in &[1, 10, 100] {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("allocate_from_inactive", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let manager = create_bench_manager(size + 10, 4);

                        // Pre-populate inactive pool by registering and dropping blocks
                        let mut complete_blocks = Vec::new();
                        for i in 0..size {
                            let token_block = create_bench_token_block(i as u32 * 10, 4);
                            let mutable_blocks =
                                manager.allocate_blocks(1).expect("Should allocate");
                            let complete_block = mutable_blocks
                                .into_iter()
                                .next()
                                .unwrap()
                                .complete(token_block)
                                .expect("Should complete");
                            complete_blocks.push(complete_block);
                        }

                        let registered_blocks = manager.register_blocks(complete_blocks);
                        drop(registered_blocks); // Puts blocks in inactive pool

                        manager
                    },
                    |manager| {
                        // Try to allocate from inactive pool (should reuse blocks)
                        black_box(manager.allocate_blocks(size));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    benches,
    registration_benchmarks,
    lookup_benchmarks,
    drop_benchmarks,
    allocation_benchmarks
);
criterion_main!(benches);
