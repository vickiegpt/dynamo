// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;
use std::thread::spawn;
use tokio::sync::mpsc;

use crate::block_manager::block::{BlockMetadata, MutableBlock};
use crate::block_manager::storage::Storage;

use anyhow::Result;
use cudarc::driver::CudaEvent;

pub struct PendingOffload<S: Storage, M: BlockMetadata> {
    _block: Arc<MutableBlock<S, M>>,
    event: CudaEvent,
}

impl<S: Storage, M: BlockMetadata> PendingOffload<S, M> {
    pub fn new(block: Arc<MutableBlock<S, M>>, event: CudaEvent) -> Self {
        Self {
            _block: block,
            event,
        }
    }
}

// TODO: Parameterize this.
const MAX_OFFLOAD_STREAM_DEPTH: usize = 4;

pub struct PendingOffloadManager<S: Storage, M: BlockMetadata> {
    pending_offload_q: mpsc::Sender<PendingOffload<S, M>>,
}

impl<S: Storage, M: BlockMetadata> PendingOffloadManager<S, M> {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel::<PendingOffload<S, M>>(MAX_OFFLOAD_STREAM_DEPTH);

        spawn(move || {
            while let Some(pending_offload) = rx.blocking_recv() {
                // Wait for the event, then drop the struct (including the block).
                pending_offload.event.synchronize()?;
                drop(pending_offload);
            }
            Ok::<(), anyhow::Error>(())
        });

        Self {
            pending_offload_q: tx,
        }
    }

    pub async fn handle_pending_offload(
        &self,
        pending_offload: PendingOffload<S, M>,
    ) -> Result<()> {
        self.pending_offload_q.send(pending_offload).await?;

        Ok(())
    }
}
