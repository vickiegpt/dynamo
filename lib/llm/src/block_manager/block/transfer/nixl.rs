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

use super::*;

use anyhow::Result;
use nixl_sys::{MemoryRegion, NixlDescriptor, XferDescList, MemType};
use std::future::Future;
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct XferDesc {
    addr: usize,
    len: usize,
    dev_id: u64,
}

struct XferDescListWrapper<'a> {
    list: XferDescList<'a>,
    descs: Vec<XferDesc>,
    mem_type: MemType,
}

impl<'a> XferDescListWrapper<'a> {
    fn new(mem_type: MemType) -> Result<Self> {
        let list = XferDescList::new(mem_type, true)?;
        Ok(Self {
            list,
            descs: Vec::new(),
            mem_type,
        })
    }

    fn add_desc(&mut self, addr: usize, len: usize, dev_id: u64) -> Result<()> {
        self.list.add_desc(addr, len, dev_id)?;
        self.descs.push(XferDesc { addr, len, dev_id });
        Ok(())
    }

}

fn append_xfer_request<Source, Destination>(
    src: &Arc<Source>,
    dst: &mut Destination,
    src_dl: &mut XferDescListWrapper,
    dst_dl: &mut XferDescListWrapper,
) -> Result<()>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = src.block_data(private::PrivateToken);
    let dst_data = dst.block_data_mut(private::PrivateToken);

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_desc = src_data.block_view()?.as_nixl_descriptor();
        let dst_desc = dst_data.block_view_mut()?.as_nixl_descriptor_mut();

        unsafe {
            src_dl.add_desc(src_desc.as_ptr() as usize, src_desc.size(), src_desc.device_id())?;
            dst_dl.add_desc(dst_desc.as_ptr() as usize, dst_desc.size(), dst_desc.device_id())?;
        }

        Ok(())
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                debug_assert_eq!(src_view.size(), dst_view.size());

                let src_desc = src_view.as_nixl_descriptor();
                let dst_desc = dst_view.as_nixl_descriptor_mut();

                unsafe {
                    src_dl.add_desc(
                        src_desc.as_ptr() as usize,
                        src_desc.size(),
                        src_desc.device_id(),
                    )?;

                    dst_dl.add_desc(
                        dst_desc.as_ptr() as usize,
                        dst_desc.size(),
                        dst_desc.device_id(),
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Xfer {
    timestamp: u128,
    source_mem_type: MemType,
    target_mem_type: MemType,
    source_descs: Vec<XferDesc>,
    target_descs: Vec<XferDesc>,
}

impl Xfer {
    fn new(sources: &XferDescListWrapper, targets: &XferDescListWrapper) -> Self {
        Self {
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
            source_mem_type: sources.mem_type,
            target_mem_type: targets.mem_type,
            source_descs: sources.descs.clone(),
            target_descs: targets.descs.clone(),
        }
    }
}

lazy_static::lazy_static! {
    static ref NIXL_LOG_FILE: Option<Arc<Mutex<File>>> = {
        if let Ok(log_file) = std::env::var("NIXL_LOG_FILE") {
            Some(Arc::new(Mutex::new(File::create(log_file).unwrap())))
        } else {
            None
        }
    };
}

fn log_xfer_descs(src_dl: &XferDescListWrapper, dst_dl: &XferDescListWrapper) {
    if let Some(log_file) = NIXL_LOG_FILE.as_ref() {
        let mut log_file = log_file.lock().unwrap();
        let xfer = Xfer::new(src_dl, dst_dl);
        let serialized = serde_json::to_string(&xfer).unwrap();
        writeln!(log_file, "{}", serialized).unwrap();
    }
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn write_blocks_to<Source, Destination>(
    src: &[Arc<Source>],
    dst: &mut [Destination],
    ctx: &Arc<TransferContext>,
    transfer_type: NixlTransfer,
) -> Result<Box<dyn Future<Output = ()> + Send + Sync + Unpin>>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    if src.is_empty() || dst.is_empty() {
        return Ok(Box::new(std::future::ready(())));
    }
    assert_eq!(src.len(), dst.len());

    let nixl_agent_arc = ctx.as_ref().nixl_agent();
    let nixl_agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .expect("NIXL agent not found");

    let src_mem_type = src
        .first()
        .unwrap()
        .block_data(private::PrivateToken)
        .storage_type()
        .nixl_mem_type();
    let dst_mem_type = dst
        .first()
        .unwrap()
        .block_data(private::PrivateToken)
        .storage_type()
        .nixl_mem_type();

    let mut src_dl = XferDescListWrapper::new(src_mem_type)?;
    let mut dst_dl = XferDescListWrapper::new(dst_mem_type)?;

    for (src, dst) in src.iter().zip(dst.iter_mut()) {
        append_xfer_request(src, dst, &mut src_dl, &mut dst_dl)?;
    }

    debug_assert!(!src_dl.list.has_overlaps()? && !dst_dl.list.has_overlaps()?);

    log_xfer_descs(&src_dl, &dst_dl);

    let xfer_req = nixl_agent.create_xfer_req(
        transfer_type.as_xfer_op(),
        &src_dl.list,
        &dst_dl.list,
        &nixl_agent.name(),
        None,
    )?;

    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

    if still_pending {
        Ok(Box::new(Box::pin(async move {
            let nixl_agent = nixl_agent_arc
                .as_ref()
                .as_ref()
                .expect("NIXL agent not found");

            loop {
                match nixl_agent.get_xfer_status(&xfer_req) {
                    Ok(false) => break, // Transfer is complete.
                    Ok(true) => tokio::time::sleep(std::time::Duration::from_millis(5)).await, // Transfer is still in progress.
                    Err(e) => {
                        tracing::error!("Error getting transfer status: {}", e);
                        break;
                    }
                }
            }
        })))
    } else {
        Ok(Box::new(std::future::ready(())))
    }
}
