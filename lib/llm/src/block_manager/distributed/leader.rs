// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_runtime::active_message::create_core_system_handlers;
use utils::*;

use anyhow::{Context, Result, bail};
use derive_builder::Builder;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::active_message::{
    client::ActiveMessageClient,
    cohort::{CohortType, LeaderWorkerCohort},
    zmq::ZmqActiveMessageManager,
};

#[derive(Builder, Clone, Debug, Default)]
pub struct KvbmLeaderNumBlocksConfig {
    #[builder(default = "0.0")]
    pub cache_size_in_gb: f64,

    #[builder(default = "0")]
    pub num_blocks_overriden: usize,
}

fn compute_num_blocks(
    num_blocks_config: &KvbmLeaderNumBlocksConfig,
    bytes_per_block: usize,
) -> usize {
    if num_blocks_config.num_blocks_overriden > 0 {
        num_blocks_config.num_blocks_overriden
    } else {
        ((num_blocks_config.cache_size_in_gb * 1_000_000_000.0) / bytes_per_block as f64) as usize
    }
}

#[derive(Builder, Clone, Debug)]
pub struct KvbmLeaderConfig {
    /// The world size.
    #[builder(default = "1")]
    world_size: usize,

    /// The leader-worker init connection timeout seconds.
    #[builder(default = "120")]
    leader_init_timeout_secs: u64,

    #[builder(default = "5555")]
    leader_port: u16,

    #[builder(default = "KvbmLeaderNumBlocksConfig::default()")]
    host_blocks_config: KvbmLeaderNumBlocksConfig,

    #[builder(default = "KvbmLeaderNumBlocksConfig::default()")]
    disk_blocks_config: KvbmLeaderNumBlocksConfig,
}

impl KvbmLeaderConfig {
    pub fn builder() -> KvbmLeaderConfigBuilder {
        KvbmLeaderConfigBuilder::default()
    }

    pub fn sanity_check(&self) -> anyhow::Result<()> {
        let cpu = &self.host_blocks_config;
        let disk = &self.disk_blocks_config;
        if cpu.num_blocks_overriden == 0 && cpu.cache_size_in_gb == 0.0 {
            if disk.num_blocks_overriden == 0 && disk.cache_size_in_gb == 0.0 {
                panic!(
                    "KVBM Configuration Error: No CPU memory configured.\n\
                    \n\
                    To fix this, set one of the following environment variables:\n\
                    • DYN_KVBM_CPU_CACHE_GB=<size_in_gb>     (e.g., DYN_KVBM_CPU_CACHE_GB=4)\n\
                    • DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=<num_blocks>  (e.g., DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=1000)\n\
                    \n\
                    Example: export DYN_KVBM_CPU_CACHE_GB=4"
                );
            } else {
                panic!(
                    "KVBM Configuration Error: CPU memory must be configured before disk memory.\n\
                    \n\
                    To fix this, set one of the following environment variables:\n\
                    • DYN_KVBM_CPU_CACHE_GB=<size_in_gb>     (e.g., DYN_KVBM_CPU_CACHE_GB=4)\n\
                    • DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=<num_blocks>  (e.g., DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=1000)\n\
                    \n\
                    Example: export DYN_KVBM_CPU_CACHE_GB=4"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct KvbmLeaderState {
    pub num_device_blocks: Arc<AtomicUsize>,
    pub num_host_blocks: Arc<AtomicUsize>,
    pub num_disk_blocks: Arc<AtomicUsize>,
    pub workers_allocation_ready: Arc<AtomicBool>,
    pub workers_ready_notify: Arc<Notify>,
}

/// The leader of the KVBM.
///
/// This is responsible for:
/// - Establishing a ZMQ connection with workers.
/// - Syncing the leader barrier with workers.
/// - Sending messages to workers.
pub struct KvbmLeader {
    state: Arc<KvbmLeaderState>,
    config: KvbmLeaderConfig,
    am_manager: Arc<ZmqActiveMessageManager>,
    cohort: Arc<LeaderWorkerCohort>,
}

impl KvbmLeader {
    pub async fn new(config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        let cancel_token = CancellationToken::new();

        let leader_sub_endpoint = format!("tcp://0.0.0.0:{}", config.leader_port);
        let am_manager = Arc::new(
            ZmqActiveMessageManager::new(leader_sub_endpoint, cancel_token.clone()).await?,
        );
        let leader_client = am_manager.client();

        tracing::debug!("Leader listening on: {}", leader_client.endpoint());
        tracing::debug!("Leader instance ID: {}", leader_client.instance_id());

        // Create cohort with {world_size} expected workers
        let cohort: Arc<LeaderWorkerCohort> = Arc::new(LeaderWorkerCohort::new(
            leader_client.clone(),
            CohortType::FixedSize(config.world_size),
        ));

        let leader = Self {
            state: Arc::new(KvbmLeaderState::default()),
            config,
            am_manager,
            cohort,
        };

        let system_handlers = create_core_system_handlers(
            leader_client.clone_as_arc(),
            tokio_util::task::TaskTracker::new(),
        );

        for (name, dispatcher) in system_handlers {
            leader.am_manager.register_handler(name, dispatcher).await?;
        }

        leader.spawn_background_initialization(cancel_token);

        Ok(leader)
    }

    fn spawn_background_initialization(&self, cancel_token: CancellationToken) {
        let cohort = self.cohort.clone();
        let state = self.state.clone();
        let cfg = self.config.clone();
        // clone if you need am_manager inside; not necessary below:
        // let am_manager = self.am_manager.clone();

        tokio::spawn(async move {
            if let Err(e) =
                Self::do_background_initialization(cohort, state, cfg, cancel_token.clone()).await
            {
                tracing::error!(error = %e, "Leader background initialization failed");
                // decide your policy: cancel runtime so callers can detect failure
                cancel_token.cancel();
            } else {
                tracing::info!("Leader background initialization completed successfully.");
            }
        });
    }

    async fn do_background_initialization(
        cohort: Arc<LeaderWorkerCohort>,
        state: Arc<KvbmLeaderState>,
        cfg: KvbmLeaderConfig,
        _cancel_token: CancellationToken, // available if you want cooperative checks
    ) -> Result<()> {
        // (1) Wait for workers to join
        let mut attempts = 0;
        let max_attempts = cfg.leader_init_timeout_secs;
        loop {
            if cohort.is_full().await {
                break;
            }
            attempts += 1;
            if attempts >= max_attempts {
                bail!("Cohort not ready - timeout waiting for workers");
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        // (1b) Ensure workers have the metadata handler
        let ok = cohort
            .await_handler_on_all_workers(AM_MSG_WORKER_METADATA, Some(Duration::from_secs(30)))
            .await
            .is_ok();
        if !ok {
            bail!(
                "Workers missing required handler: {}",
                AM_MSG_WORKER_METADATA
            );
        }

        // Gather worker metadata
        let worker_metas: Vec<KvbmWorkerData> = cohort
            .par_broadcast_responses(AM_MSG_WORKER_METADATA, (), Duration::from_secs(30))
            .await
            .context("collect worker metadata via par_broadcast_responses")?;

        let min_device_blocks = worker_metas
            .iter()
            .map(|m| m.num_device_blocks)
            .min()
            .ok_or_else(|| anyhow::anyhow!("empty worker metadata vector"))?;

        let bytes_per_block_sum: usize = worker_metas.iter().map(|m| m.bytes_per_block).sum();
        if bytes_per_block_sum == 0 {
            bail!("bytes_per_block sum computed as 0");
        }

        // (2) Compute host/disk blocks
        let num_host_blocks = compute_num_blocks(&cfg.host_blocks_config, bytes_per_block_sum);
        let num_disk_blocks = compute_num_blocks(&cfg.disk_blocks_config, bytes_per_block_sum);

        // Store in state
        state
            .num_device_blocks
            .store(min_device_blocks, Ordering::Release);
        state
            .num_host_blocks
            .store(num_host_blocks, Ordering::Release);
        state
            .num_disk_blocks
            .store(num_disk_blocks, Ordering::Release);

        let leader_metadata = KvbmLeaderData {
            num_host_blocks,
            num_disk_blocks,
        };

        // (3) Ensure leader-metadata handler, then broadcast and collect ACKs
        let ok = cohort
            .await_handler_on_all_workers(AM_MSG_LEADER_METADATA, Some(Duration::from_secs(30)))
            .await
            .is_ok();
        if !ok {
            bail!(
                "Workers missing required handler: {}",
                AM_MSG_LEADER_METADATA
            );
        }

        let ack_results = cohort
            .par_broadcast_acks(
                AM_MSG_LEADER_METADATA,
                leader_metadata,
                Duration::from_secs(30),
            )
            .await
            .context("broadcast leader metadata and collect ACKs")?;

        let mut ready_failures = 0usize;
        for (wid, res) in &ack_results {
            if let Err(e) = res {
                ready_failures += 1;
                tracing::warn!(worker_id=%wid, err=%e, "Leader metadata READY ack failed");
            }
        }
        if ready_failures > 0 {
            bail!("{ready_failures} worker(s) failed to ACK leader metadata READY");
        }

        // (4) Verify block transfer handler is registered on all workers
        let ok = cohort
            .await_handler_on_all_workers(
                AM_MSG_TRANSFER_BLOCKS,
                Some(Duration::from_secs(cfg.leader_init_timeout_secs)),
            )
            .await
            .is_ok();
        if !ok {
            bail!(
                "Workers missing required handler: {}",
                AM_MSG_TRANSFER_BLOCKS
            );
        }

        // (5) Signal ready
        state
            .workers_allocation_ready
            .store(true, Ordering::Release);
        state.workers_ready_notify.notify_waiters();

        Ok(())
    }

    pub async fn transfer_blocks_request(
        &self,
        request: BlockTransferRequest,
    ) -> anyhow::Result<oneshot::Receiver<()>> {
        let (tx, rx) = oneshot::channel();
        let cohort = self.cohort.clone();

        tokio::spawn(async move {
            let result = cohort
                .par_broadcast_acks(AM_MSG_TRANSFER_BLOCKS, request, Duration::from_secs(15))
                .await;

            match result {
                Ok(acks) => {
                    // Succeed only if every worker ACKed Ok(())
                    let mut failures = 0usize;
                    for (wid, res) in acks {
                        if let Err(err) = res {
                            failures += 1;
                            tracing::warn!(%wid, %err, "transfer_blocks: worker failed to ACK");
                        }
                    }

                    if failures == 0 {
                        // Signal success to caller
                        let _ = tx.send(());
                    } else {
                        // On any failure, drop sender so receiver gets Canceled
                        tracing::error!("transfer_blocks: {} worker ACK failures", failures);
                    }
                }
                Err(e) => {
                    // par_broadcast_acks itself failed (e.g., client error)
                    tracing::error!("transfer_blocks: broadcast failed: {e:#}");
                    // Drop sender so receiver gets Canceled
                }
            }
        });

        Ok(rx)
    }

    pub fn num_device_blocks(&self) -> usize {
        self.state.num_device_blocks.load(Ordering::Acquire)
    }

    pub fn num_host_blocks(&self) -> usize {
        self.state.num_host_blocks.load(Ordering::Acquire)
    }

    pub fn num_disk_blocks(&self) -> usize {
        self.state.num_disk_blocks.load(Ordering::Acquire)
    }

    pub async fn wait_worker_sync_ready(&self) -> bool {
        // fast path
        if self.state.workers_allocation_ready.load(Ordering::Acquire) {
            return true;
        }

        let notified = self.state.workers_ready_notify.notified();

        // Double-check after creating future to avoid lost-notify race.
        if self.state.workers_allocation_ready.load(Ordering::Acquire) {
            return true;
        }

        // bounded wait using the leader's configured timeout
        tokio::select! {
            _ = notified => {
                self.state.workers_allocation_ready.load(Ordering::Acquire)
            }
            _ = tokio::time::sleep(Duration::from_secs(self.config.leader_init_timeout_secs)) => false,
        }
    }
}
