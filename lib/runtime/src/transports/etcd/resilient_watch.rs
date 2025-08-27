// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Result, error};
use etcd_client::{
    Client, GetOptions, KeyValue, WatchOptions, WatchResponse, WatchStream, Watcher,
};
use futures::StreamExt;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::{interval, sleep};
use tracing::{debug, info, trace, warn};
use tracing::error as tracing_error;

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (None for unlimited)
    pub max_retries: Option<u32>,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: None, // Unlimited retries by default
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Create a retry config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        if let Ok(max_retries) = std::env::var("ETCD_WATCH_MAX_RETRIES") {
            if let Ok(max) = max_retries.parse::<u32>() {
                config.max_retries = Some(max);
            }
        }
        
        if let Ok(initial_ms) = std::env::var("ETCD_WATCH_INITIAL_BACKOFF_MS") {
            if let Ok(ms) = initial_ms.parse::<u64>() {
                config.initial_backoff = Duration::from_millis(ms);
            }
        }
        
        if let Ok(max_ms) = std::env::var("ETCD_WATCH_MAX_BACKOFF_MS") {
            if let Ok(ms) = max_ms.parse::<u64>() {
                config.max_backoff = Duration::from_millis(ms);
            }
        }
        
        config
    }
}

/// Events emitted by the resilient watch stream for observability
#[derive(Debug, Clone)]
pub enum ReconnectEvent {
    /// Watch stream disconnected
    Disconnected { 
        endpoint: String, 
        error: String,
        last_revision: i64,
    },
    /// Attempting to reconnect
    Reconnecting { 
        endpoint: String, 
        attempt: u32,
        resuming_from: i64,
    },
    /// Successfully reconnected
    Reconnected { 
        endpoint: String, 
        resumed_from: i64 
    },
    /// Compaction detected (old revisions were removed)
    CompactionDetected { 
        required_revision: i64,
        compacted_revision: i64,
    },
}

/// A wrapper around etcd watch that automatically handles reconnection
pub struct ResilientWatchStream {
    /// The prefix being watched
    prefix: String,
    /// The etcd client
    client: Client,
    /// List of etcd endpoints for failover
    endpoints: Vec<String>,
    /// Index of current endpoint
    current_endpoint_idx: usize,
    /// Last seen revision for resumption
    last_revision: Arc<AtomicI64>,
    /// Retry configuration
    retry_config: RetryConfig,
    /// Channel for sending watch events
    event_tx: mpsc::Sender<WatchResponse>,
    /// Optional channel for reconnection events
    reconnect_events_tx: Option<mpsc::Sender<ReconnectEvent>>,
    /// Watch options
    watch_options: WatchOptions,
}

impl ResilientWatchStream {
    /// Create a new resilient watch stream
    pub fn new(
        prefix: String,
        client: Client,
        endpoints: Vec<String>,
        initial_revision: i64,
        event_tx: mpsc::Sender<WatchResponse>,
        watch_options: WatchOptions,
    ) -> Self {
        let retry_config = RetryConfig::from_env();
        
        Self {
            prefix,
            client,
            endpoints,
            current_endpoint_idx: 0,
            last_revision: Arc::new(AtomicI64::new(initial_revision)),
            retry_config,
            event_tx,
            reconnect_events_tx: None,
            watch_options,
        }
    }
    
    /// Set a channel to receive reconnection events
    pub fn set_reconnect_events_tx(&mut self, tx: mpsc::Sender<ReconnectEvent>) {
        self.reconnect_events_tx = Some(tx);
    }
    
    /// Update the retry configuration
    pub fn set_retry_config(&mut self, config: RetryConfig) {
        self.retry_config = config;
    }
    
    /// Get the current endpoint being used
    fn current_endpoint(&self) -> String {
        self.endpoints.get(self.current_endpoint_idx)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    }
    
    /// Send a reconnection event if channel is configured
    async fn send_reconnect_event(&self, event: ReconnectEvent) {
        if let Some(tx) = &self.reconnect_events_tx {
            let _ = tx.send(event).await;
        }
    }
    
    /// Calculate backoff duration for given attempt
    fn calculate_backoff(&self, attempt: u32) -> Duration {
        let mut backoff = self.retry_config.initial_backoff.as_millis() as f64;
        for _ in 1..attempt {
            backoff *= self.retry_config.backoff_multiplier;
        }
        let backoff_ms = backoff.min(self.retry_config.max_backoff.as_millis() as f64) as u64;
        Duration::from_millis(backoff_ms)
    }
    
    /// Create a watch stream with the current revision
    async fn create_watch(&mut self) -> Result<(Watcher, WatchStream)> {
        let start_revision = self.last_revision.load(Ordering::SeqCst);
        
        // Clone watch options and set the start revision
        let options = self.watch_options.clone()
            .with_start_revision(start_revision + 1);
        
        debug!(
            "Creating watch for prefix '{}' starting from revision {}",
            self.prefix,
            start_revision + 1
        );
        
        self.client
            .watch(self.prefix.clone(), Some(options))
            .await
            .map_err(|e| error!("Failed to create watch: {}", e))
    }
    
    /// Handle a compaction error by resetting to current revision
    async fn handle_compaction(&mut self) -> Result<()> {
        warn!(
            "Handling compaction for prefix '{}', fetching current state",
            self.prefix
        );
        
        // Get current state and revision
        let response = self.client
            .get(self.prefix.clone(), Some(GetOptions::new().with_prefix()))
            .await
            .map_err(|e| error!("Failed to get current state after compaction: {}", e))?;
        
        let current_revision = response
            .header()
            .and_then(|h| Some(h.revision()))
            .unwrap_or(0);
        
        let old_revision = self.last_revision.load(Ordering::SeqCst);
        
        // Send compaction event
        self.send_reconnect_event(ReconnectEvent::CompactionDetected {
            required_revision: old_revision,
            compacted_revision: current_revision,
        }).await;
        
        // Update last revision
        self.last_revision.store(current_revision, Ordering::SeqCst);
        
        info!(
            "Reset watch for prefix '{}' to revision {} after compaction",
            self.prefix, current_revision
        );
        
        Ok(())
    }
    
    /// Try the next endpoint in round-robin fashion
    fn select_next_endpoint(&mut self) {
        if !self.endpoints.is_empty() {
            self.current_endpoint_idx = (self.current_endpoint_idx + 1) % self.endpoints.len();
            debug!("Switched to endpoint: {}", self.current_endpoint());
        }
    }
    
    /// Main watch loop with automatic reconnection
    pub async fn run(mut self, cancel_token: tokio_util::sync::CancellationToken) {
        let mut retry_attempt = 0u32;
        let mut consecutive_failures = 0u32;
        
        loop {
            // Check cancellation
            if cancel_token.is_cancelled() {
                info!("Resilient watch for prefix '{}' cancelled", self.prefix);
                break;
            }
            
            // Check max retries
            if let Some(max) = self.retry_config.max_retries {
                if retry_attempt >= max {
                    tracing_error!(
                        "Max retries ({}) reached for watch prefix '{}', stopping",
                        max, self.prefix
                    );
                    break;
                }
            }
            
            // Create watch stream
            let watch_result = self.create_watch().await;
            
            match watch_result {
                Ok((mut watcher, mut stream)) => {
                    let last_revision = self.last_revision.load(Ordering::SeqCst);
                    
                    // Send reconnected event if this is a retry
                    if retry_attempt > 0 {
                        info!(
                            "Successfully reconnected watch for prefix '{}' at revision {}",
                            self.prefix, last_revision + 1
                        );
                        self.send_reconnect_event(ReconnectEvent::Reconnected {
                            endpoint: self.current_endpoint(),
                            resumed_from: last_revision + 1,
                        }).await;
                        
                        // Reset retry counter on successful reconnection
                        retry_attempt = 0;
                        consecutive_failures = 0;
                    }
                    
                    // Process watch events
                    loop {
                        tokio::select! {
                            _ = cancel_token.cancelled() => {
                                info!("Resilient watch for prefix '{}' cancelled", self.prefix);
                                return;
                            }
                            
                            maybe_msg = stream.message() => {
                                match maybe_msg {
                                    Ok(Some(response)) => {
                                        // Update last revision if present
                                        if let Some(header) = response.header() {
                                            let revision = header.revision();
                                            self.last_revision.store(revision, Ordering::SeqCst);
                                            trace!(
                                                "Updated last revision for prefix '{}' to {}",
                                                self.prefix, revision
                                            );
                                        }
                                        
                                        // Forward the event
                                        if self.event_tx.send(response).await.is_err() {
                                            warn!("Event receiver dropped for prefix '{}'", self.prefix);
                                            return;
                                        }
                                    }
                                    Ok(None) => {
                                        // Stream ended normally
                                        let last_rev = self.last_revision.load(Ordering::SeqCst);
                                        warn!(
                                            "Watch stream ended for prefix '{}' at revision {}",
                                            self.prefix, last_rev
                                        );
                                        
                                        self.send_reconnect_event(ReconnectEvent::Disconnected {
                                            endpoint: self.current_endpoint(),
                                            error: "Stream ended".to_string(),
                                            last_revision: last_rev,
                                        }).await;
                                        
                                        break; // Break inner loop to reconnect
                                    }
                                    Err(e) => {
                                        let last_rev = self.last_revision.load(Ordering::SeqCst);
                                        let error_msg = e.to_string();
                                        
                                        // Check if it's a compaction error
                                        if error_msg.contains("mvcc: required revision has been compacted") {
                                            warn!(
                                                "Compaction detected for prefix '{}': {}",
                                                self.prefix, error_msg
                                            );
                                            
                                            // Handle compaction
                                            if let Err(e) = self.handle_compaction().await {
                                                tracing_error!(
                                                    "Failed to handle compaction for prefix '{}': {}",
                                                    self.prefix, e
                                                );
                                            }
                                        } else {
                                            tracing_error!(
                                                "Watch error for prefix '{}' at revision {}: {}",
                                                self.prefix, last_rev, error_msg
                                            );
                                            
                                            self.send_reconnect_event(ReconnectEvent::Disconnected {
                                                endpoint: self.current_endpoint(),
                                                error: error_msg,
                                                last_revision: last_rev,
                                            }).await;
                                        }
                                        
                                        break; // Break inner loop to reconnect
                                    }
                                }
                            }
                        }
                    }
                    
                    // Cancel the watcher before reconnecting
                    watcher.cancel().await.ok();
                }
                Err(e) => {
                    consecutive_failures += 1;
                    let last_rev = self.last_revision.load(Ordering::SeqCst);
                    
                    tracing_error!(
                        "Failed to create watch for prefix '{}' (attempt {}, consecutive failures {}): {}",
                        self.prefix, retry_attempt + 1, consecutive_failures, e
                    );
                    
                    self.send_reconnect_event(ReconnectEvent::Disconnected {
                        endpoint: self.current_endpoint(),
                        error: e.to_string(),
                        last_revision: last_rev,
                    }).await;
                    
                    // Try next endpoint after multiple failures on current one
                    if consecutive_failures >= 3 && self.endpoints.len() > 1 {
                        warn!(
                            "Multiple failures on endpoint {}, trying next endpoint",
                            self.current_endpoint()
                        );
                        self.select_next_endpoint();
                        consecutive_failures = 0;
                    }
                }
            }
            
            // Calculate backoff and wait
            retry_attempt += 1;
            let backoff = self.calculate_backoff(retry_attempt);
            
            let resuming_from = self.last_revision.load(Ordering::SeqCst) + 1;
            info!(
                "Reconnecting watch for prefix '{}' in {:?} (attempt {}, resuming from revision {})",
                self.prefix, backoff, retry_attempt, resuming_from
            );
            
            self.send_reconnect_event(ReconnectEvent::Reconnecting {
                endpoint: self.current_endpoint(),
                attempt: retry_attempt,
                resuming_from,
            }).await;
            
            sleep(backoff).await;
        }
        
        info!("Resilient watch for prefix '{}' stopped", self.prefix);
    }
}

/// Helper function to create a resilient watch stream
pub async fn create_resilient_watch(
    mut client: Client,
    prefix: impl Into<String>,
    options: Option<WatchOptions>,
    endpoints: Vec<String>,
    cancel_token: tokio_util::sync::CancellationToken,
) -> Result<mpsc::Receiver<WatchResponse>> {
    let prefix = prefix.into();
    let options = options.unwrap_or_default().with_prefix().with_prev_key();
    
    // Get initial state and revision
    let response = client
        .get(prefix.clone(), Some(GetOptions::new().with_prefix()))
        .await
        .map_err(|e| error!("Failed to get initial state: {}", e))?;
    
    let start_revision = response
        .header()
        .and_then(|h| Some(h.revision()))
        .unwrap_or(0);
    
    debug!(
        "Starting resilient watch for prefix '{}' from revision {}",
        prefix, start_revision
    );
    
    // Create channel for events
    let (event_tx, event_rx) = mpsc::channel(100);
    
    // Create resilient watch stream
    let resilient_stream = ResilientWatchStream::new(
        prefix.clone(),
        client,
        endpoints,
        start_revision,
        event_tx,
        options,
    );
    
    // Spawn the resilient watch task
    tokio::spawn(resilient_stream.run(cancel_token));
    
    Ok(event_rx)
}