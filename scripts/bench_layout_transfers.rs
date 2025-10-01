#!/usr/bin/env cargo

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Layout Transfer Benchmarking Utility 🚀
//!
//! This script provides a command-line interface for benchmarking and comparing
//! transfer performance between different layout configurations.
//!
//! ## Usage
//!
//! ```bash
//! # Initialize benchmarking for a session
//! cargo run --bin bench_layout_transfers -- init "fc_vs_ls_comparison"
//!
//! # Generate a report from current session
//! cargo run --bin bench_layout_transfers -- report
//!
//! # Compare two benchmark sessions
//! cargo run --bin bench_layout_transfers -- compare before.json after.json
//!
//! # Export metrics to CSV
//! cargo run --bin bench_layout_transfers -- export csv output.csv
//!
//! # Reset benchmark data
//! cargo run --bin bench_layout_transfers -- reset
//! ```

use clap::{Parser, Subcommand};
use dynamo_llm::block_manager::bench::{
    BenchmarkReport, TransferPath, global_benchmark, hooks::*, init_benchmark,
    load_benchmark, reset_global_benchmark,
};
use dynamo_llm::block_manager::layout::LayoutType;
use std::fs;
use std::path::PathBuf;

use serde_json;

#[derive(Parser)]
#[command(name = "bench_layout_transfers")]
#[command(about = "Benchmark and compare layout transfer performance")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize benchmarking session
    Init {
        /// Session name for this benchmark run
        session_name: String,
    },
    /// Generate report from current session
    Report {
        /// Output format (text, json, csv)
        #[arg(short, long, default_value = "text")]
        format: String,
        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Session name (defaults to current active session)
        #[arg(long)]
        session: Option<String>,
    },
    /// Compare two benchmark sessions
    Compare {
        /// Path to 'before' benchmark JSON file
        before: PathBuf,
        /// Path to 'after' benchmark JSON file
        after: PathBuf,
        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Export current metrics
    Export {
        /// Export format (json, csv)
        format: String,
        /// Output file path
        output: PathBuf,
    },
    /// Show current benchmark status
    Status {
        /// Session name (defaults to current active session)
        #[arg(short, long)]
        session: Option<String>,
    },
    /// Reset benchmark data
    Reset,
    /// Enable benchmarking
    Enable,
    /// Disable benchmarking
    Disable,
    /// Run a simple test to generate sample data
    Test {
        /// Number of test transfers to simulate
        #[arg(short, long, default_value = "100")]
        count: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Init { session_name } => match init_benchmark(session_name.clone()) {
            Ok(()) => {
                println!("🚀 Initialized benchmark session: '{}'", session_name);
                println!("Benchmarking is now active and will collect transfer metrics.");
                println!();
                println!("💡 To ensure your vLLM process uses this session, run:");
                println!("   export DYNAMO_BENCHMARK_SESSION=\"{}\"", session_name);
                println!("   # then start your vLLM process");
            }
            Err(e) => {
                eprintln!("❌ Failed to initialize benchmark: {}", e);
                std::process::exit(1);
            }
        },

        Commands::Report { format, output, session } => {
            let benchmark = if let Some(session_name) = session {
                match load_benchmark(session_name) {
                    Ok(b) => Some(b),
                    Err(e) => {
                        eprintln!("❌ Failed to load session '{}': {}", session_name, e);
                        std::process::exit(1);
                    }
                }
            } else {
                // Load from file to get latest saved state
                match global_benchmark() {
                    Some(b) => {
                        // Reload from file to get latest saved state
                        match load_benchmark(&b.session_name) {
                            Ok(loaded) => Some(loaded),
                            Err(_) => Some(b)
                        }
                    }
                    None => None
                }
            };

            if let Some(benchmark) = benchmark {
                let report = BenchmarkReport::from_benchmark(&benchmark);

                let content = match format.as_str() {
                    "json" => report.to_json()?,
                    "csv" => report.to_csv(),
                    "text" | _ => report.generate_report(),
                };

                if let Some(output_path) = output {
                    fs::write(output_path, content)?;
                    println!("Report written to: {}", output_path.display());
                } else {
                    println!("{}", content);
                }
            } else {
                eprintln!("❌ No benchmark session active. Run 'init' first.");
                std::process::exit(1);
            }
        }

        Commands::Compare {
            before,
            after,
            output,
        } => {
            let before_content = fs::read_to_string(before)?;
            let after_content = fs::read_to_string(after)?;

            let before_report: BenchmarkReport = serde_json::from_str(&before_content)?;
            let after_report: BenchmarkReport = serde_json::from_str(&after_content)?;

            let comparison = after_report.compare_with(&before_report);

            if let Some(output_path) = output {
                fs::write(output_path, &comparison)?;
                println!("Comparison report written to: {}", output_path.display());
            } else {
                println!("{}", comparison);
            }
        }

        Commands::Export { format, output } => {
            if let Some(benchmark) = global_benchmark() {
                let report = BenchmarkReport::from_benchmark(&benchmark);

                let content = match format.as_str() {
                    "json" => report.to_json()?,
                    "csv" => report.to_csv(),
                    _ => {
                        eprintln!(
                            "❌ Unsupported export format: {}. Use 'json' or 'csv'.",
                            format
                        );
                        std::process::exit(1);
                    }
                };

                fs::write(output, content)?;
                println!("Metrics exported to: {}", output.display());
            } else {
                eprintln!("❌ No benchmark session active. Run 'init' first.");
                std::process::exit(1);
            }
        }

        Commands::Status { session } => {
            let (benchmark, is_active) = if let Some(session_name) = session {
                match load_benchmark(session_name) {
                    Ok(b) => {
                        // Check if this is also the active session
                        let is_active = global_benchmark()
                            .map(|active| active.session_name == b.session_name)
                            .unwrap_or(false);
                        (Some(b), is_active)
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to load session '{}': {}", session_name, e);
                        std::process::exit(1);
                    }
                }
            } else {
                // Load from file instead of using in-memory instance to avoid race conditions
                match global_benchmark() {
                    Some(b) => {
                        // Reload from file to get latest saved state
                        match load_benchmark(&b.session_name) {
                            Ok(loaded) => (Some(loaded), true),
                            Err(_) => (Some(b), true)
                        }
                    }
                    None => (None, false)
                }
            };

            if let Some(benchmark) = benchmark {
                let summary = benchmark.summary();
                println!("📊 {}", summary);
                println!("Session enabled: {}", benchmark.is_enabled());
                if is_active {
                    println!("✅ This is the currently active session (collecting data)");
                } else {
                    println!("ℹ️  Viewing archived session (not currently active)");
                }
            } else {
                println!("❌ No benchmark session active. Run 'init' first.");
            }
        }

        Commands::Reset => match reset_global_benchmark() {
            Ok(()) => println!("🗑️ Benchmark data has been reset."),
            Err(e) => eprintln!("❌ Failed to reset benchmark: {}", e),
        },

        Commands::Enable => {
            set_benchmarking_enabled(true);
            println!("✅ Benchmarking enabled");
        }

        Commands::Disable => {
            set_benchmarking_enabled(false);
            println!("⏸️ Benchmarking disabled");
        }

        Commands::Test { count } => {
            if global_benchmark().is_none() {
                if let Err(e) = init_benchmark("test_session") {
                    eprintln!("❌ Failed to initialize test benchmark: {}", e);
                    std::process::exit(1);
                }
                println!("🚀 Initialized test benchmark session");
            }

            println!("🧪 Generating {} test transfer records...", count);

            // Simulate various transfer scenarios
            let paths = [
                TransferPath::HostToDisk,
                TransferPath::HostToDevice,
                TransferPath::DeviceToHost,
            ];
            let layouts = [
                LayoutType::FullyContiguous,
                LayoutType::LayerSeparate {
                    outer_contiguous: false,
                },
                LayoutType::LayerSeparate {
                    outer_contiguous: true,
                },
            ];

            for i in 0..*count {
                let path = paths[i % paths.len()];
                let layout = layouts[i % layouts.len()];

                // Simulate different transfer sizes
                let base_size = match layout {
                    LayoutType::FullyContiguous => 8192, // Larger transfers for FC
                    LayoutType::LayerSeparate { .. } => 2048, // Smaller for LS
                };

                let size_variation = (i % 10) as u64 * 512;
                let total_bytes = base_size + size_variation;
                let blocks = 1 + (i % 5);

                dynamo_llm::block_manager::bench::record_transfer(
                    path,
                    layout,
                    total_bytes,
                    blocks,
                );
            }

            println!("✅ Generated {} test transfers", count);

            // Show a quick summary
            if let Some(summary) = get_benchmark_summary() {
                println!("📊 {}", summary);
            }
        }
    }

    Ok(())
}

/// Helper function to load a benchmark report from JSON file
#[allow(dead_code)]
fn load_benchmark_report(path: &PathBuf) -> anyhow::Result<BenchmarkReport> {
    let content = fs::read_to_string(path)?;
    let report: BenchmarkReport = serde_json::from_str(&content)?;
    Ok(report)
}
