// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::runtime::Handle;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
use tokio::{fs, io::AsyncWriteExt};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::metrics::MetricsRegistry;
use prometheus::IntCounter;

#[derive(Debug)]
struct FlusherInner {
    cancel: CancellationToken,
    handle: Option<JoinHandle<()>>,
}

#[derive(Clone, Debug)]
struct Flusher(Arc<Mutex<FlusherInner>>);

impl Flusher {
    pub fn new(cancel: CancellationToken, handle: JoinHandle<()>) -> Self {
        Self(Arc::new(Mutex::new(FlusherInner {
            cancel,
            handle: Some(handle),
        })))
    }

    pub async fn shutdown(&self) {
        let mut inner = self.0.lock().await;
        inner.cancel.cancel();
        if let Some(h) = inner.handle.take() {
            drop(inner); // release lock before await
            let _ = h.await;
        }
    }
}

impl Drop for Flusher {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.0.try_lock() {
            inner.cancel.cancel();
            if let Some(h) = inner.handle.take() {
                h.abort();
            }
        } else {
            // couldn't lock; at least request cancel
            self.0.try_lock().ok();
        }
    }
}

#[derive(Clone, Debug)]
pub struct KvbmMetrics {
    // number of offload requests
    pub offload_requests: IntCounter,

    // number of blocks offloaded from device to host
    pub offload_blocks_d2h: IntCounter,

    // number of onboard requests
    pub onboard_requests: IntCounter,

    // number of blocks onboarded from host to device
    pub onboard_blocks_h2d: IntCounter,

    // number of blocks onboarded from disk to device
    pub onboard_blocks_d2d: IntCounter,

    // number of save kv layer requests
    pub save_kv_layer_requests: IntCounter,

    // number of matched tokens from KVBM
    pub matched_tokens: IntCounter,

    pub kvbm_stats: Arc<Stats>,

    flusher: Arc<Flusher>,
}

/// A struct for storing timing data for some action.
/// Generally, we'd expect that the time to perform some action on a set of blocks
/// is composed of a fixed latency, as well as some per-block latency, hence the blocks parameter.
#[derive(Debug, Clone)]
pub struct EventStats {
    request_id: String,
    start_instant: Instant, // for precise elapsed measurement
    start_time: SystemTime, // wall-clock timestamp for logs/plots
    time_elapsed: Duration,
    num_blocks: usize,
}

impl EventStats {
    pub fn new(request_id: String, num_blocks: usize) -> Self {
        Self {
            request_id,
            start_instant: Instant::now(),
            start_time: SystemTime::now(),
            time_elapsed: Duration::from_secs(0),
            num_blocks,
        }
    }

    pub fn event_complete(&mut self) {
        self.time_elapsed = self.start_instant.elapsed();
    }

    pub fn match_blocks_complete(&mut self, num_blocks: usize) {
        self.num_blocks = num_blocks;
        self.time_elapsed = self.start_instant.elapsed();
    }

    /// Helper: Unix epoch millis for start_time (safe for CSV)
    pub fn start_time_millis(&self) -> u128 {
        self.start_time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }
}

fn metrics_dump_dir() -> PathBuf {
    env::var("KVBM_METRICS_DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./kvbm_metrics"))
}

#[derive(Debug, Default)]
pub struct Stats {
    pub host_match_latency: Arc<Mutex<HashMap<String, EventStats>>>,
    pub host_offload_latency: Arc<Mutex<HashMap<String, EventStats>>>,
    pub host_onboard_latency: Arc<Mutex<HashMap<String, EventStats>>>,
    pub disk_match_latency: Arc<Mutex<HashMap<String, EventStats>>>,
    pub disk_onboard_latency: Arc<Mutex<HashMap<String, EventStats>>>,
}

impl Stats {
    fn new() -> Self {
        Self {
            host_match_latency: Arc::new(Mutex::new(HashMap::new())),
            host_offload_latency: Arc::new(Mutex::new(HashMap::new())),
            host_onboard_latency: Arc::new(Mutex::new(HashMap::new())),
            disk_match_latency: Arc::new(Mutex::new(HashMap::new())),
            disk_onboard_latency: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn spawn_periodic_flush(
        self: &Arc<Self>,
        rt: &Handle,
        interval: Duration,
        cancel: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let this = Arc::clone(self);
        let output_dir = metrics_dump_dir();

        rt.spawn(async move {
            let _ = tokio::fs::create_dir_all(&output_dir).await;
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        let _ = this.flush_once(&output_dir).await; // final flush
                        break;
                    }
                    _ = tokio::time::sleep(interval) => {
                        if let Err(e) = this.flush_once(&output_dir).await {
                            eprintln!("[Stats::flush] error: {e}");
                        }
                    }
                }
            }
        })
    }

    pub async fn flush_once(&self, output_dir: &Path) -> anyhow::Result<()> {
        // (name, map) pairs, used to generate filenames and iterate uniformly
        let maps: &[(&str, &Arc<Mutex<HashMap<String, EventStats>>>)] = &[
            ("host_match_latency", &self.host_match_latency),
            ("host_offload_latency", &self.host_offload_latency),
            ("host_onboard_latency", &self.host_onboard_latency),
            ("disk_match_latency", &self.disk_match_latency),
            ("disk_onboard_latency", &self.disk_onboard_latency),
        ];

        for (name, map) in maps {
            self.flush_one_map(name, map, output_dir).await?;
        }
        Ok(())
    }

    async fn flush_one_map(
        &self,
        name: &str,
        map: &Arc<Mutex<HashMap<String, EventStats>>>,
        output_dir: &Path,
    ) -> anyhow::Result<()> {
        // Collect completed entries (time_elapsed > 0) without holding the lock during I/O
        let mut completed: Vec<(String, EventStats)> = {
            let guard = map.lock().await;
            guard
                .iter()
                .filter_map(|(k, v)| {
                    if v.time_elapsed > Duration::from_millis(0) {
                        Some((k.clone(), v.clone()))
                    } else {
                        None
                    }
                })
                .collect()
        };

        if completed.is_empty() {
            return Ok(());
        }

        // File path is "<output_dir>/<field_name>.csv"
        let mut path = output_dir.to_path_buf();
        path.push(format!("{name}.csv"));

        // Create the file if missing and write header once
        let new_file = fs::metadata(&path).await.is_err();
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        if new_file {
            file.write_all(b"request_id,num_blocks,time_elapsed_ms,start_time_ms\n")
                .await?;
        }

        // Write all lines
        for (_, ev) in &completed {
            let line = format!(
                "{},{},{},{}\n",
                ev.request_id,
                ev.num_blocks,
                ev.time_elapsed.as_millis(),
                ev.start_time_millis()
            );
            file.write_all(line.as_bytes()).await?;
        }
        file.flush().await?;

        // After successful write, remove only those completed entries
        {
            let mut guard = map.lock().await;
            for (id, _) in completed.drain(..) {
                // Remove if still present and still completed
                if let std::collections::hash_map::Entry::Occupied(e) = guard.entry(id)
                    && e.get().time_elapsed > Duration::from_millis(0)
                {
                    e.remove();
                }
            }
        }

        Ok(())
    }
}

impl KvbmMetrics {
    pub fn new(mr: &dyn MetricsRegistry, rt: &Handle) -> Self {
        let offload_requests = mr
            .create_intcounter("offload_requests", "The number of offload requests", &[])
            .unwrap();
        let offload_blocks_d2h = mr
            .create_intcounter(
                "offload_blocks_d2h",
                "The number of offload blocks from device to host",
                &[],
            )
            .unwrap();
        let onboard_requests = mr
            .create_intcounter("onboard_requests", "The number of onboard requests", &[])
            .unwrap();
        let onboard_blocks_h2d = mr
            .create_intcounter(
                "onboard_blocks_h2d",
                "The number of onboard blocks from host to device",
                &[],
            )
            .unwrap();
        let onboard_blocks_d2d = mr
            .create_intcounter(
                "onboard_blocks_d2d",
                "The number of onboard blocks from disk to device",
                &[],
            )
            .unwrap();
        let save_kv_layer_requests = mr
            .create_intcounter(
                "save_kv_layer_requests",
                "The number of save kv layer requests",
                &[],
            )
            .unwrap();
        let matched_tokens = mr
            .create_intcounter("matched_tokens", "The number of matched tokens", &[])
            .unwrap();
        let kvbm_stats = Arc::new(Stats::new());

        let interval = Duration::from_secs(10);

        let cancel = CancellationToken::new();
        let handle = kvbm_stats.spawn_periodic_flush(rt, interval, cancel.clone());

        let flusher = Arc::new(Flusher::new(cancel, handle));

        Self {
            offload_requests,
            offload_blocks_d2h,
            onboard_requests,
            onboard_blocks_h2d,
            onboard_blocks_d2d,
            save_kv_layer_requests,
            matched_tokens,
            kvbm_stats,
            flusher,
        }
    }
}
